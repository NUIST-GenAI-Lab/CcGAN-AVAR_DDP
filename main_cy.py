import timeit
from datetime import datetime
import glob
import os
import timeit
from datetime import datetime

import torch.backends.cudnn as cudnn
import torch.nn as nn
from accelerate.utils import set_seed
from torch.nn import SyncBatchNorm

from dataset_cy import LoadDataSet
from evaluation.evaluator import Evaluator
from ipc_util import register_signal_handler
from label_embedding_cy import LabelEmbed
from models import sagan_generator, sagan_discriminator, sngan_generator, sngan_discriminator, biggan_generator, \
    biggan_discriminator, biggan_deep_generator, biggan_deep_discriminator, resnet18_aux_regre
from opts import parse_opts
from trainer_cy import Trainer
from utils import *

##############################################
''' Settings '''
plt.switch_backend('agg')
args = parse_opts()
register_signal_handler()

accelerator = init_accelerator(args)

# seeds (use accelerate lib # set the seed in `random`, `numpy`, `torch`)
set_seed(args.seed)

torch.backends.cudnn.deterministic = True
cudnn.benchmark = False

accelerator.print(
    "\n===================================================================================================")

#######################################################################################
'''                                Output folders                                  '''
#######################################################################################
path_to_output = os.path.join(args.root_path, 'output/{}_{}'.format(args.data_name, args.img_size))
save_setting_folder = os.path.join(path_to_output, "{}".format(args.setting_name))
setting_log_file = os.path.join(save_setting_folder, 'setting_info.txt')
save_results_folder = os.path.join(save_setting_folder, 'results')
path_to_fake_data = os.path.join(save_results_folder, 'fake_data')
if accelerator.is_main_process:
    os.makedirs(path_to_output, exist_ok=True)
    os.makedirs(save_setting_folder, exist_ok=True)
    if not os.path.isfile(setting_log_file):
        logging_file = open(setting_log_file, "w")
        logging_file.close()
    with open(setting_log_file, 'a') as logging_file:
        logging_file.write(
            "\n===================================================================================================")
        accelerator.print(args, file=logging_file)
    os.makedirs(save_results_folder, exist_ok=True)
    os.makedirs(path_to_fake_data, exist_ok=True)

#######################################################################################
'''                                Make dataset                                     '''
#######################################################################################

dataset = LoadDataSet(data_name=args.data_name, data_path=args.data_path, min_label=args.min_label,
                      max_label=args.max_label, img_size=args.img_size,
                      max_num_img_per_label=args.max_num_img_per_label,
                      num_img_per_label_after_replica=args.num_img_per_label_after_replica,
                      imbalance_type=args.imb_type)

train_images, train_labels, train_labels_norm = dataset.load_train_data()
num_classes = dataset.num_classes

_, _, eval_labels = dataset.load_evaluation_data()

#######################################################################################
'''                           Compute Vicinal Params                                '''
#######################################################################################

unique_labels_norm = np.sort(np.array(list(set(train_labels_norm))))

if args.kernel_sigma < 0:
    std_label = np.std(train_labels_norm)
    args.kernel_sigma = 1.06 * std_label * (len(train_labels_norm)) ** (-1 / 5)

    accelerator.print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")
    accelerator.print(
        "\r The std of {} labels is {:.4f} so the kernel sigma is {:.4f}".format(len(train_labels_norm), std_label,
                                                                                 args.kernel_sigma))
##end if

if args.kappa < 0:
    n_unique = len(unique_labels_norm)

    diff_list = []
    for i in range(1, n_unique):
        diff_list.append(unique_labels_norm[i] - unique_labels_norm[i - 1])
    kappa_base = np.abs(args.kappa) * np.max(np.array(diff_list))

    if args.threshold_type == "hard":
        args.kappa = kappa_base
    else:
        args.kappa = 1 / kappa_base ** 2
##end if

accelerator.print("\r Kappa:{:.4f}".format(args.kappa))

vicinal_params = {
    "kernel_sigma": args.kernel_sigma,
    "kappa": args.kappa,
    "threshold_type": args.threshold_type,
    "nonzero_soft_weight_threshold": args.nonzero_soft_weight_threshold,
    "use_ada_vic": args.use_ada_vic,
    "ada_vic_type": args.ada_vic_type,
    "min_n_per_vic": args.min_n_per_vic,
    "ada_eps": 1e-5,
    "use_symm_vic": args.use_symm_vic,
}

#######################################################################################
'''                             label embedding method                              '''
#######################################################################################

dataset_embed = LoadDataSet(data_name=args.data_name, data_path=args.data_path, min_label=args.min_label,
                            max_label=args.max_label, img_size=args.img_size,
                            max_num_img_per_label=args.max_num_img_per_label, num_img_per_label_after_replica=0,
                            imbalance_type=args.imb_type)

label_embedding = LabelEmbed(dataset=dataset_embed, path_y2h=path_to_output + '/model_y2h',
                             path_y2cov=path_to_output + '/model_y2cov', y2h_type="resnet", y2cov_type="sinusoidal",
                             h_dim=args.dim_y, cov_dim=args.img_size ** 2 * args.num_channels, nc=args.num_channels)
fn_y2h = label_embedding.fn_y2h

#######################################################################################
'''                                 Model Config                                    '''
#######################################################################################

if args.ch_multi_g is not None:
    ch_multi_g = (args.ch_multi_g).split("_")
    ch_multi_g = [int(dim) for dim in ch_multi_g]
if args.ch_multi_d is not None:
    ch_multi_d = (args.ch_multi_d).split("_")
    ch_multi_d = [int(dim) for dim in ch_multi_d]

if args.net_name.lower() == "sngan":
    netG = sngan_generator(dim_z=args.dim_z, dim_y=args.dim_y, nc=args.num_channels, img_size=args.img_size,
                           gene_ch=args.gene_ch, ch_multi=args.ch_multi_g)
    netD = sngan_discriminator(dim_y=args.dim_y, nc=args.num_channels, img_size=args.img_size, disc_ch=args.disc_ch,
                               ch_multi=args.ch_multi_d, use_aux_reg=args.use_aux_reg_branch,
                               use_aux_dre=args.use_dre_reg, dre_head_arch=args.dre_head_arch)

    if accelerator.num_processes > 1 and args.use_sync_bn:
        print("==================================== Use sync-bn ====================================")
        netG = SyncBatchNorm.convert_sync_batchnorm(netG)
        netD = SyncBatchNorm.convert_sync_batchnorm(netD)


elif args.net_name.lower() == "sagan":
    netG = sagan_generator(dim_z=args.dim_z, dim_y=args.dim_y, nc=args.num_channels, img_size=args.img_size,
                           gene_ch=args.gene_ch, ch_multi=args.ch_multi_g)
    netD = sagan_discriminator(dim_y=args.dim_y, nc=args.num_channels, img_size=args.img_size, disc_ch=args.disc_ch,
                               ch_multi=args.ch_multi_d, use_aux_reg=args.use_aux_reg_branch,
                               use_aux_dre=args.use_dre_reg, dre_head_arch=args.dre_head_arch)
elif args.net_name.lower() == "biggan":
    netG = biggan_generator(dim_z=args.dim_z, dim_y=args.dim_y, img_size=args.img_size, nc=args.num_channels,
                            gene_ch=args.gene_ch, ch_multi=args.ch_multi_g, use_sn=args.use_sn, use_attn=args.use_attn,
                            g_init="ortho")
    netD = biggan_discriminator(dim_y=args.dim_y, img_size=args.img_size, nc=args.num_channels, disc_ch=args.disc_ch,
                                ch_multi=args.ch_multi_d, use_sn=args.use_sn, use_attn=args.use_attn, d_init="ortho",
                                use_aux_reg=args.use_aux_reg_branch, use_aux_dre=args.use_dre_reg,
                                dre_head_arch=args.dre_head_arch)
elif args.net_name.lower() == "biggan-deep":
    netG = biggan_deep_generator(dim_z=args.dim_z, dim_y=args.dim_y, img_size=args.img_size, nc=args.num_channels,
                                 gene_ch=args.gene_ch, ch_multi=args.ch_multi_g, use_sn=args.use_sn,
                                 use_attn=args.use_attn, g_init="ortho")
    netD = biggan_deep_discriminator(dim_y=args.dim_y, img_size=args.img_size, nc=args.num_channels,
                                     disc_ch=args.disc_ch, ch_multi=args.ch_multi_d, use_sn=args.use_sn,
                                     use_attn=args.use_attn, d_init="ortho", use_aux_reg=args.use_aux_reg_branch,
                                     use_aux_dre=args.use_dre_reg, dre_head_arch=args.dre_head_arch)
elif args.net_name.lower() == "dcgan":
    netG = sngan_generator(dim_z=args.dim_z, dim_y=args.dim_y, nc=args.num_channels, img_size=args.img_size,
                           gene_ch=args.gene_ch)
    netD = sngan_discriminator(dim_y=args.dim_y, nc=args.num_channels, img_size=args.img_size, disc_ch=args.disc_ch,
                               use_aux_reg=args.use_aux_reg_branch, use_aux_dre=args.use_dre_reg,
                               dre_head_arch=args.dre_head_arch)
else:
    raise ValueError("Not Supported Network!")

accelerator.print('\r netG size:', get_parameter_number(netG))
accelerator.print('\r netD size:', get_parameter_number(netD))

## independent auxiliary regressor
if args.use_aux_reg_model:
    aux_reg_net = resnet18_aux_regre(nc=args.num_channels)
    path_to_ckpt = os.path.join(path_to_output, "aux_reg_model")
    if args.data_name in ["RC-49_imb"]:
        path_to_ckpt += "/{}".format(args.imb_type)
    path_to_ckpt += "/ckpt_resnet18_epoch_200.pth"
    checkpoint = torch.load(path_to_ckpt, weights_only=True)
    aux_reg_net.load_state_dict(checkpoint['net_state_dict'])
    aux_reg_net.eval()
else:
    aux_reg_net = None

#######################################################################################
'''                                  Training                                      '''
#######################################################################################

aux_loss_params = {
    # ===========================
    # auxiliary regression loss for both D and G
    "use_aux_reg_branch": args.use_aux_reg_branch,
    "use_aux_reg_model": args.use_aux_reg_model,
    "aux_reg_loss_type": args.aux_reg_loss_type,
    "aux_reg_loss_ei_hinge_factor": args.aux_reg_loss_ei_hinge_factor,
    "aux_reg_loss_huber_delta": args.aux_reg_loss_huber_delta,
    "aux_reg_loss_huber_quantile": args.aux_reg_loss_huber_quantile,
    "weight_d_aux_reg_loss": args.weight_d_aux_reg_loss,
    "weight_g_aux_reg_loss": args.weight_g_aux_reg_loss,
    "aux_reg_net": aux_reg_net,
    # ===========================
    # density density ratio model training and auxiliary penalty for G
    "use_dre_reg": args.use_dre_reg,
    "dre_lambda": args.dre_lambda,
    "weight_d_aux_dre_loss": args.weight_d_aux_dre_loss,
    "weight_g_aux_dre_loss": args.weight_g_aux_dre_loss,
    "do_dre_ft": args.do_dre_ft,  # finetuning dre branch after the GAN training
    "dre_ft_niters": args.dre_ft_niters,
    "dre_ft_lr": args.dre_ft_lr,
    "dre_ft_batch_size": args.dre_ft_batch_size,
}

trainer = Trainer(
    data_name=args.data_name,
    train_images=train_images,
    train_labels=train_labels_norm,
    eval_labels=dataset.fn_normalize_labels(eval_labels),
    net_name=args.net_name,
    netG=netG,
    netD=netD,
    fn_y2h=fn_y2h,
    vicinal_params=vicinal_params,
    aux_loss_params=aux_loss_params,
    img_size=args.img_size,
    img_ch=args.num_channels,
    results_folder=save_results_folder,
    dim_z=args.dim_z,
    niters=args.niters,
    resume_iter=args.resume_iter,
    num_D_steps=args.num_D_steps,
    batch_size_disc=args.batch_size_disc,
    batch_size_gene=args.batch_size_gene,
    lr_g=args.lr_g,
    lr_d=args.lr_d,
    loss_type=args.loss_type,
    save_freq=args.save_freq,
    sample_freq=args.sample_freq,
    num_grad_acc_d=args.num_grad_acc_d,
    num_grad_acc_g=args.num_grad_acc_g,
    max_grad_norm=args.max_grad_norm,
    nrow_visual=10,
    use_amp=args.use_amp,
    mixed_precision_type=args.mixed_precision_type,
    adam_betas=(0.5, 0.999),
    use_ema=args.use_ema,
    ema_update_after_step=args.ema_update_after_step,
    ema_update_every=args.ema_update_every,
    ema_decay=args.ema_decay,
    use_diffaug=args.use_diffaug,
    diffaug_policy=args.diffaug_policy,
    exp_seed=args.seed,
    num_workers=None,
)

start = timeit.default_timer()
accelerator.print("\n")
accelerator.print("Begin Training:")
if not args.eval_only:
    trainer.train()
stop = timeit.default_timer()
accelerator.print("End training; Time elapses: {}s. \n".format(stop - start))

if args.do_dre_ft:
    trainer.finetune_cdre()

#######################################################################################
'''                         Sampling and evaluation                                 '''
#######################################################################################

def load_evaluated_ckpts(registry_path: str):
    """读取已经评估过的 ckpt 文件名集合"""
    done = set()
    if os.path.isfile(registry_path):
        with open(registry_path, "r") as f:
            for line in f:
                name = line.strip()
                if name:
                    done.add(name)
    return done

def append_evaluated_ckpt(registry_path: str, ckpt_name: str):
    """把新评估完成的 ckpt 名字追加写入索引文件"""
    with open(registry_path, "a") as f:
        f.write(ckpt_name + "\n")

if args.do_eval:
    accelerator.print("\n[Eval] 开始评估 ...")

    # ============ 准备 evaluator 需要的评估网络类 ============
    if args.data_name in ["RC-49", "RC-49_imb"]:
        eval_data_name = "RC49"
    else:
        eval_data_name = args.data_name

    conduct_import_codes = (
        "from evaluation.eval_models.{}.metrics_{}x{} import "
        "ResNet34_class_eval, ResNet34_regre_eval, encoder"
    ).format(eval_data_name, args.img_size, args.img_size)
    accelerator.print("\r" + conduct_import_codes)
    exec(conduct_import_codes, globals())

    # for FID
    PreNetFID = encoder(dim_bottleneck=512)
    PreNetFID = nn.DataParallel(PreNetFID)
    # for Diversity
    if args.data_name in ["UTKFace", "RC-49", "RC-49_imb", "SteeringAngle"]:
        PreNetDiversity = ResNet34_class_eval(
            num_classes=num_classes,
            ngpu=torch.cuda.device_count()
        )
    else:
        PreNetDiversity = None
    # for LS
    PreNetLS = ResNet34_regre_eval(ngpu=torch.cuda.device_count())

    # ============ eval_only：遍历 results 目录下所有 checkpoint 逐个评估 ============
    if args.eval_only:
        results_dir = save_results_folder   # 就是 Trainer 保存 checkpoint 的目录
        os.makedirs(results_dir, exist_ok=True)

        # 索引文件：记录哪些 pth 已经评估过
        registry_path = os.path.join(results_dir, "eval_index.txt")
        evaluated_ckpts = load_evaluated_ckpts(registry_path)

        # 找出当前目录下所有 .pth
        ckpt_paths = sorted(glob.glob(os.path.join(results_dir, "*.pth")))

        print(f"在 {results_dir} 中找到 {len(ckpt_paths)} 个 ckpt 文件。")
        print(f"其中已有 {len(evaluated_ckpts)} 个记录为已评估。")


        def extract_iter(fname: str) -> int:
            import re
            nums = re.findall(r"\d+", fname)
            if not nums:
                return -1
            return int(nums[-1])


        summary_path = os.path.join(results_dir, "all_eval_results.txt")
        for ckpt_path in ckpt_paths:
            ckpt_name = os.path.basename(ckpt_path)

            # 已评估过就跳过
            if ckpt_name in evaluated_ckpts:
                print(f"[跳过] {ckpt_name} 已在 eval_index.txt 中记录，略过评估。")
                continue

            print(f"[评估] 开始评估 {ckpt_name} ...")

            # 调用 Trainer 内部自己的 load(iter) 逻辑，确保和训练时保存的格式一致
            iter_num = extract_iter(ckpt_name)
            trainer.load(iter_num)

            # 为当前 checkpoint 重新构造 Evaluator（它会重新采样 fake）
            evaluator = Evaluator(
                dataset=dataset,
                trainer=trainer,
                args=args,
                device=trainer.device
            )

            # 为当前 iter 单独建一个 eval 输出目录
            eval_results_path = os.path.join(
                save_setting_folder, f"eval_iter_{iter_num}"
            )
            os.makedirs(eval_results_path, exist_ok=True)

            # dump fake data in h5 files
            if args.dump_fake_for_h5:
                path_to_h5files = os.path.join(eval_results_path, 'h5')
                os.makedirs(path_to_h5files, exist_ok=True)
                evaluator.dump_h5_files(output_path=path_to_h5files)

            # dump for niqe computation
            if args.dump_fake_for_niqe:
                if args.niqe_dump_path == "None":
                    dump_fake_images_folder = os.path.join(eval_results_path, 'png')
                else:
                    dump_fake_images_folder = os.path.join(
                        args.niqe_dump_path, f'fake_images_iter_{iter_num}'
                    )
                os.makedirs(dump_fake_images_folder, exist_ok=True)
                evaluator.dump_png_images(output_path=dump_fake_images_folder)

            # === 计算指标 ===
            evaluator.compute_metrics(
                eval_results_path,
                PreNetFID,
                PreNetDiversity,
                PreNetLS
            )

            # === 从刚刚生成的 eval_results_*.txt 中解析数值，拼成一个总表 ===
            import re, glob

            # 找到 eval_results_*.txt 中“最新的那一个”
            pattern = os.path.join(
                eval_results_path, "eval_results_*.txt"
            )
            txt_files = glob.glob(pattern)
            if not txt_files:
                continue
            latest_txt = max(txt_files, key=os.path.getmtime)

            with open(latest_txt, "r") as f:
                content = f.read()

            # 解析 SFID, LS, Diversity, FID, IS
            m_sfid = re.search(r"SFID:\s*([0-9eE+\-\.]+)\s*\(([0-9eE+\-\.]+)\)", content)
            m_ls = re.search(r"LS:\s*([0-9eE+\-\.]+)\s*\(([0-9eE+\-\.]+)\)", content)
            m_div = re.search(r"Diversity:\s*([0-9eE+\-\.]+)\s*\(([0-9eE+\-\.]+)\)", content)
            m_fid = re.search(r"FID:\s*([0-9eE+\-\.]+)", content)
            m_is = re.search(r"IS \(STD\):\s*([0-9eE+\-\.]+)\s*\(([0-9eE+\-\.]+)\)", content)

            with open(summary_path, "a") as sf:
                sf.write(f"\nCheckpoint: {ckpt_name} (iter={iter_num})\n")
                if m_sfid:
                    sf.write(
                        f"  SFID: {m_sfid.group(1)} ({m_sfid.group(2)})\n"
                    )
                if m_ls:
                    sf.write(
                        f"  LS: {m_ls.group(1)} ({m_ls.group(2)})\n"
                    )
                if m_div:
                    sf.write(
                        f"  Diversity: {m_div.group(1)} ({m_div.group(2)})\n"
                    )
                if m_fid:
                    sf.write(f"  FID: {m_fid.group(1)}\n")
                if m_is:
                    sf.write(
                        f"  IS: {m_is.group(1)} ({m_is.group(2)})\n"
                    )
            # 当前ckpt评估成功后，记录到索引文件里
            append_evaluated_ckpt(registry_path, ckpt_name)
            print(f"[完成] {ckpt_name} 评估结束，已写入 eval_index.txt")


        accelerator.print(
            f"\n[Eval] 所有 checkpoint 的汇总结果已写入: {summary_path}"
        )

    # ============ 非 eval_only：保持原来单次评估最后模型的逻辑 ============
    else:
        accelerator.print("\n Start sampling fake images from the model >>>")

        # 单 checkpoint：用训练结束时的 trainer 直接评估一次
        evaluator = Evaluator(
            dataset=dataset,
            trainer=trainer,
            args=args,
            device=trainer.device
        )

        # dump fake data in h5 files
        if args.dump_fake_for_h5:
            path_to_h5files = os.path.join(path_to_fake_data, 'h5')
            os.makedirs(path_to_h5files, exist_ok=True)
            evaluator.dump_h5_files(output_path=path_to_h5files)

        # dump for niqe computation
        if args.dump_fake_for_niqe:
            if args.niqe_dump_path == "None":
                dump_fake_images_folder = os.path.join(path_to_fake_data, 'png')
            else:
                dump_fake_images_folder = args.niqe_dump_path + '/fake_images'
            os.makedirs(dump_fake_images_folder, exist_ok=True)
            evaluator.dump_png_images(output_path=dump_fake_images_folder)

        # start computing evaluation metrics
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        eval_results_path = os.path.join(save_setting_folder, "eval_{}".format(time_str))
        os.makedirs(eval_results_path, exist_ok=True)
        evaluator.compute_metrics(eval_results_path, PreNetFID, PreNetDiversity, PreNetLS)

    accelerator.print(
        "\n==================================================================================================="
    )