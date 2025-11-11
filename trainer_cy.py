"""
The trainer class for training CcGANs.

Support adaptive vicinity but the left and right vicinity may be asymmetric!

"""


import copy
import glob
import os
import re
import timeit
import torch
import warnings

import numpy as np
import torch.nn.functional as F
from accelerate.utils import set_seed
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from DiffAugment_pytorch import DiffAugment
from ema_pytorch import EMA
from ipc_util import get_s1, get_s2, switch_s1, switch_s2
from utils import get_accelerator, SimpleProgressBar, normalize_images, random_hflip, random_rotate, random_vflip, \
    exists, divisible_by


class Trainer(object):
    def __init__(
        self,
        data_name,
        train_images,
        train_labels,
        eval_labels,
        net_name,
        netG,
        netD,
        fn_y2h,
        vicinal_params,
        aux_loss_params,
        img_size,
        img_ch,
        results_folder,
        *,
        dim_z = 128,
        niters = 10000,
        resume_iter = 0,
        num_D_steps = 1, 
        batch_size_disc = 16,
        batch_size_gene = 16,
        lr_g = 1e-4,
        lr_d = 1e-4,
        loss_type = "hinge",
        save_freq = 1000,
        sample_freq = 1000,
        num_grad_acc_d = 1,
        num_grad_acc_g = 1,
        max_grad_norm = 1.,
        nrow_visual = 10,
        use_amp = False,
        mixed_precision_type = 'fp16',
        adam_betas = (0.5, 0.999),
        use_ema = False,
        ema_update_after_step = 1e30,
        ema_update_every = 10,
        ema_decay = 0.999,
        use_diffaug = False,
        diffaug_policy = 'color,translation,cutout',
        exp_seed = 123,
        num_workers = None,
    ):
        super().__init__()

        # accelerator
        self.accelerator = get_accelerator()

        # path
        self.results_folder = results_folder
        os.makedirs(self.results_folder, exist_ok=True)
        self.save_images_folder = self.results_folder + "/imgs_in_train"
        os.makedirs(self.save_images_folder, exist_ok=True)
        # self.save_niqe_nda_images_folder = self.results_folder + "/niqe_nda_imgs_in_train"
        # os.makedirs(self.save_niqe_nda_images_folder, exist_ok=True)
        
        # dataset
        self.data_name = data_name
        self.train_images = train_images # training images are not normalized here !!!
        self.train_labels = train_labels # training labels are normalized to [0,1]
        # self.unique_train_labels = np.sort(np.array(list(set(train_labels))))
        self.unique_train_labels, self.counts_train_elements = np.unique(train_labels, return_counts=True) 
        self.min_abs_diff = np.min(np.abs(np.diff(np.sort(self.unique_train_labels))))  # Compute the minimum absolute difference between adjacent elements.
        #counts_train_elements: number of samples for each unique label
        assert train_images.max()>1.0
        assert train_labels.min()>=0 and train_labels.max()<=1.0
        self.accelerator.print("\n Training labels' range is [{},{}].".format(train_labels.min(), train_labels.max()))
        # self.accelerator.print(self.counts_train_elements)
        
        self.eval_labels = eval_labels #evaluation labels are normalized to [0,1]
        assert self.eval_labels.min()>=0 and self.eval_labels.max()<=1.0
        
        self.img_size = img_size #image size
        self.img_ch = img_ch #number of channels
        
        self.num_workers = num_workers
        
        # model
        self.net_name = net_name
        self.dim_z = dim_z
        self.netG = netG
        self.netD = netD
        self.fn_y2h = fn_y2h

        # cause our sampling does not depend on idx, we must use different seed in DDP mode.
        set_seed(exp_seed+self.accelerator.process_index)
        
        # training
        self.niters = niters
        self.resume_iter = resume_iter
        self.num_D_steps = num_D_steps #number of D update steps for each iteration
        
        self.batch_size_disc = batch_size_disc
        self.batch_size_gene = batch_size_gene
        self.num_grad_acc_d = num_grad_acc_d
        self.num_grad_acc_g = num_grad_acc_g
        
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.loss_type = loss_type
        
        self.save_freq = save_freq
        
        self.max_grad_norm = max_grad_norm
        
        ## vicinal params
        self.vicinal_params = vicinal_params

        ## auxiliary loss params
        self.aux_loss_params = aux_loss_params
        if self.aux_loss_params["use_aux_reg_model"]:
            self.aux_reg_net = self.aux_loss_params["aux_reg_net"].to(self.device)
            self.aux_reg_net.eval()
        
        if self.aux_loss_params["use_dre_reg"]:
            self.dre_lambda = self.aux_loss_params["dre_lambda"]
        
        ## visualize
        self.sample_freq = sample_freq
        self.nrow_visual = nrow_visual
        
        # printed images with labels between the 5-th quantile and 95-th quantile of training labels
        self.z_visual = torch.randn(nrow_visual*nrow_visual, dim_z, dtype=torch.float).to(self.accelerator.device)
        start_label = np.quantile(train_labels, 0.05)
        end_label = np.quantile(train_labels, 0.95)
        selected_labels = np.linspace(start_label, end_label, num=nrow_visual)
        y_visual = np.zeros(nrow_visual*nrow_visual)
        for i in range(nrow_visual):
            curr_label = selected_labels[i]
            for j in range(nrow_visual):
                y_visual[i*nrow_visual+j] = curr_label
        self.accelerator.print(y_visual)
        self.y_visual = torch.from_numpy(y_visual).type(torch.float).to(self.accelerator.device)

        ## diffaugment
        self.use_diffaug = use_diffaug
        self.diffaug_policy = diffaug_policy

        ## optimizer
        self.optG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=adam_betas)
        self.optD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=adam_betas)

        ## prepare model, dataloader, optimizer with accelerator
        # self.netG = nn.parallel.DistributedDataParallel(self.netG, find_unused_parameters=True)
        self.netG, self.netD, self.optG, self.optD = self.accelerator.prepare(self.netG, self.netD, self.optG, self.optD)

        # self.netD, self.optG, self.optD = self.accelerator.prepare(self.netD, self.optG, self.optD)
        # from torch.nn.parallel import DistributedDataParallel as DDP
        # self.netG = DDP(
        #     self.netG,
        #     device_ids=[self.accelerator.local_process_index],
        #     output_device=self.accelerator.local_process_index,
        #     find_unused_parameters=True  # 关键参数
        # )
        # self.netG = nn.parallel.DistributedDataParallel(self.netG, find_unused_parameters=True)

        ## EMA
        self.use_ema = use_ema
        self.ema_update_after_step = ema_update_after_step
        self.ema_update_every = ema_update_every
        self.ema_decay = ema_decay
        
        if self.accelerator.is_main_process:
            if self.use_ema:
                self.ema_g = EMA(netG, update_after_step=ema_update_after_step, beta = ema_decay, update_every = ema_update_every)
                # self.ema_g.to(self.device) #before
                self.ema_g = self.ema_g.to(self.device)
            
        # step counter state
        self.step = 0
        
        # # resume training
        # if self.resume_iter>0:
        #     self.load(self.resume_iter)

        # resume max ckpt
        ckpt_dir = self.results_folder
        ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.pth"))

        max_iter = None
        for f in ckpt_files:
            fname = os.path.basename(f)
            m = re.findall(r"\d+", fname)
            if not m:
                continue
            it = int(m[-1])
            if (max_iter is None) or (it > max_iter):
                max_iter = it

        if max_iter is not None:
            self.accelerator.print(f"[Trainer] Auto resume from iter {max_iter}")
            self.load(max_iter)
        else:
            self.accelerator.print("[Trainer] No checkpoint found, start from scratch")



        self.ft_dre_flag = False #By default, the dre branch is not finetuned.
    
    
    
    ########################################################################################      
    @property
    def device(self):
        return self.accelerator.device
    
    
    
    ############################################################################################################################ 
    ######################################################################################## 
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'netG': self.accelerator.get_state_dict(self.netG),
            'netD': self.accelerator.get_state_dict(self.netD),
            'optG': self.optG.state_dict(),
            'optD': self.optD.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }
        if self.use_ema:
            data['ema_g'] = self.ema_g.state_dict()
        torch.save(data, self.results_folder + "/ckpt_niter_{}.pth".format(milestone))

      
    ############################################################################################################################ 
    ######################################################################################## 
    def load(self, milestone, return_ema=False):
        device = self.accelerator.device
        data = torch.load(self.results_folder + "/ckpt_niter_{}.pth".format(milestone), map_location=device, weights_only=True)
        self.netG = self.accelerator.unwrap_model(self.netG)
        self.netG.load_state_dict(data['netG'])
        self.netD = self.accelerator.unwrap_model(self.netD)
        self.netD.load_state_dict(data['netD'])
        self.step = data['step']
        self.optG.load_state_dict(data['optG'])
        self.optD.load_state_dict(data['optD'])
        if self.accelerator.is_main_process:
            if self.use_ema:
                self.ema_g.load_state_dict(data["ema_g"])
                if return_ema:
                    return self.ema_g
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    

    ############################################################################################################################ 
    ######################################################################################## 
    ## random data augmentation for a batch of real images
    def fn_transform(self, batch_real_images):
        assert isinstance(batch_real_images, np.ndarray)
        if self.data_name == "UTKFace":
            batch_real_images = random_hflip(batch_real_images)
        if self.data_name[0:7] == "Cell200":
            batch_real_images = random_rotate(batch_real_images)
            batch_real_images = random_hflip(batch_real_images)
            batch_real_images = random_vflip(batch_real_images)
        return batch_real_images

    
    ########################################################################################  
    ## adversarial loss for training discriminator
    def fn_disc_adv_loss(self, real_adv_out, fake_adv_out, real_weights=None, fake_weights=None, eps=1e-20):
        if self.loss_type.lower() =="vanilla":
            real_adv_out = torch.sigmoid(real_adv_out)
            fake_adv_out = torch.sigmoid(fake_adv_out)
            d_loss_real = - torch.log(real_adv_out+eps)
            d_loss_fake = - torch.log(1-fake_adv_out+eps)
        elif self.loss_type.lower() == "hinge":
            d_loss_real = F.relu(1.0 - real_adv_out)
            d_loss_fake = F.relu(1.0 + fake_adv_out)
        else:
           raise ValueError('Not supported loss type!!!') 
        if real_weights is None:
            real_weights = torch.ones(len(d_loss_real), dtype=torch.float).to(self.device)
        if fake_weights is None:
            fake_weights = torch.ones(len(d_loss_fake), dtype=torch.float).to(self.device)
        loss = torch.mean(real_weights.view(-1) * d_loss_real.view(-1)) + torch.mean(fake_weights.view(-1) * d_loss_fake.view(-1)) 
        return loss
    
    ## adversarial loss for training generator
    def fn_gene_adv_loss(self, adv_out, eps=1e-20):
        if self.loss_type.lower() =="vanilla":
            adv_out = torch.sigmoid(adv_out)
            g_loss = - torch.mean(torch.log(adv_out+eps))
        elif self.loss_type.lower() == "hinge":
            g_loss = - adv_out.mean()
        else:
            raise ValueError('Not supported loss type!!!') 
        return g_loss
    

    ############################################################################################################################ 
    ######################################################################################## 
    ## Auxiliary regression loss for better label consistency
    def adaptive_huber_loss(self, y_pred, y_true, quantile=0.9, delta=None, reduction="mean"):
        #Small delta:​​ Close to MAE (Mean Absolute Error), more robust to outliers.
        #Large delta:​​ Closer to MSE, resulting in smoother optimization.
        residuals = torch.abs(y_true - y_pred)
        if delta is None or delta<0 or delta>1:
            delta = torch.quantile(residuals, quantile)  #compute delta dynamicly
        else:
            assert isinstance(delta, float) and (delta>=0.0 and delta<=1.0)
        loss = torch.where(
            residuals < delta,
            0.5 * residuals ** 2,
            delta * (residuals - 0.5 * delta)
        )
        if reduction=="sum":
            return loss.sum()
        elif reduction=="mean":
            return loss.mean()
        else:
            return loss


    def fn_disc_aux_reg_loss(self, real_gt_labels, real_pred_labels, fake_gt_labels, fake_pred_labels, epsilon = 0):
        if self.aux_loss_params['aux_reg_loss_type'].lower() in ['mse']:
            reg_loss = torch.mean( (real_pred_labels.view(-1) - real_gt_labels.view(-1))**2 ) + torch.mean( (fake_pred_labels.view(-1) - fake_gt_labels.view(-1))**2 )
        elif self.aux_loss_params['aux_reg_loss_type'].lower() in ['ei_hinge']: #epsilon-insensitive hinge loss 
            if isinstance(epsilon, np.ndarray):
                epsilon = torch.from_numpy(epsilon).type(torch.float).to(self.device)
            real_abs_diff = torch.abs(real_pred_labels.view(-1) - real_gt_labels.view(-1))
            fake_abs_diff = torch.abs(fake_pred_labels.view(-1) - fake_gt_labels.view(-1))
            reg_loss = torch.mean(torch.clamp(real_abs_diff - epsilon*self.aux_loss_params['aux_reg_loss_ei_hinge_factor'], min=0)) + torch.mean(torch.clamp(fake_abs_diff - epsilon*self.aux_loss_params['aux_reg_loss_ei_hinge_factor'], min=0)) 
        elif self.aux_loss_params['aux_reg_loss_type'].lower() in ['huber']:
            huber_loss_real = self.adaptive_huber_loss(real_pred_labels.view(-1), real_gt_labels.view(-1), quantile=self.aux_loss_params["aux_reg_loss_huber_quantile"], delta=self.aux_loss_params["aux_reg_loss_huber_delta"], reduction="mean")
            huber_loss_fake = self.adaptive_huber_loss(fake_pred_labels.view(-1), fake_gt_labels.view(-1), quantile=self.aux_loss_params["aux_reg_loss_huber_quantile"], delta=self.aux_loss_params["aux_reg_loss_huber_delta"], reduction="mean")
            reg_loss =  huber_loss_real + huber_loss_fake
        else:
            raise ValueError('Not supported loss type!!!') 
        return reg_loss


    def fn_gene_aux_reg_loss(self, fake_gt_labels, fake_pred_labels):
        if self.aux_loss_params['aux_reg_loss_type'].lower() in ['mse','huber']:
            reg_loss = torch.mean( (fake_pred_labels.view(-1) - fake_gt_labels.view(-1))**2 )
        elif self.aux_loss_params['aux_reg_loss_type'].lower() in ['ei_hinge']:
            fake_abs_diff = torch.abs(fake_pred_labels.view(-1) - fake_gt_labels.view(-1))
            reg_loss = torch.mean(fake_abs_diff)
        else:
            raise ValueError('Not supported loss type!!!') 
        # reg_loss = torch.mean( (fake_pred_labels.view(-1) - fake_gt_labels.view(-1))**2 )
        return reg_loss

    
    ############################################################################################################################ 
    ######################################################################################## 
    ## DRE-based subsampling
    def penalized_softplus_loss(self, dr_real, dr_fake):
        softplus_fn = torch.nn.Softplus(beta=1,threshold=20)
        sigmoid_fn = torch.nn.Sigmoid()
        SP_div = torch.mean(sigmoid_fn(dr_fake) * dr_fake) - torch.mean(softplus_fn(dr_fake)) - torch.mean(sigmoid_fn(dr_real))
        penalty = self.dre_lambda * (torch.mean(dr_fake) - 1)**2
        dre_loss = SP_div + penalty
        return dre_loss
        

    ############################################################################################################################ 
    ######################################################################################## 
    def train(self):
        device = self.accelerator.device

        # ====== 构建 NeighborSamplerDataset + DataLoader ======
        neighbor_dataset = VicinitySamplerDataset(
            images=self.train_images,
            labels_norm=self.train_labels,
            unique_labels_norm=self.unique_train_labels,
            vicinal_params=self.vicinal_params,
        )
        neighbor_loader = neighbor_dataset.get_dataloader(
            batch_size=self.batch_size_disc,
        )
        neighbor_loader = self.accelerator.prepare(neighbor_loader)
        neighbor_iter = iter(neighbor_loader)

        def compute_vicinal_weights_from_batch(
                batch_real_labels: torch.Tensor,
                batch_fake_labels: torch.Tensor,
                batch_target_labels: torch.Tensor,
                kappa_l: torch.Tensor,
                kappa_r: torch.Tensor,
                vicinal_params: dict,
                device=None,
        ):
            """
            输入张量形状 (B,)
            返回:
                real_weights: (B,)
                fake_weights: (B,)
            """
            if device is None:
                device = batch_real_labels.device

            threshold_type = vicinal_params["threshold_type"].lower()
            ada_vic_type = vicinal_params["ada_vic_type"].lower()

            if threshold_type == "soft" or ada_vic_type == "hybrid":
                nu_l_all = 1.0 / (kappa_l ** 2)
                nu_r_all = 1.0 / (kappa_r ** 2)

                indx_left_real = torch.where((batch_real_labels - batch_target_labels) <= 0)[0]
                indx_right_real = torch.where((batch_real_labels - batch_target_labels) > 0)[0]
                indx_left_fake = torch.where((batch_fake_labels - batch_target_labels) <= 0)[0]
                indx_right_fake = torch.where((batch_fake_labels - batch_target_labels) > 0)[0]

                real_weights = torch.zeros_like(nu_l_all, dtype=torch.float, device=device)

                # 兼容原始CcGAN的邻域权重计算, 这里只是分为left和right两部分了, 其实一样
                real_weights[indx_left_real] = torch.exp(
                    -nu_l_all[indx_left_real] *
                    (batch_real_labels[indx_left_real] - batch_target_labels[indx_left_real]) ** 2
                )
                real_weights[indx_right_real] = torch.exp(
                    -nu_r_all[indx_right_real] *
                    (batch_real_labels[indx_right_real] - batch_target_labels[indx_right_real]) ** 2
                )

                fake_weights = torch.zeros_like(nu_l_all, dtype=torch.float, device=device)
                fake_weights[indx_left_fake] = torch.exp(
                    -nu_l_all[indx_left_fake] *
                    (batch_fake_labels[indx_left_fake] - batch_target_labels[indx_left_fake]) ** 2
                )
                fake_weights[indx_right_fake] = torch.exp(
                    -nu_r_all[indx_right_fake] *
                    (batch_fake_labels[indx_right_fake] - batch_target_labels[indx_right_fake]) ** 2
                )

                # if ada_vic_type == "hybrid":
                #     nz_thr = vicinal_params["nonzero_soft_weight_threshold"]
                #     real_weights = torch.maximum(real_weights, torch.tensor(nz_thr, device=device))
                #     fake_weights = torch.maximum(fake_weights, torch.tensor(nz_thr, device=device))

            elif threshold_type == "hard":
                real_weights = torch.ones_like(batch_real_labels, dtype=torch.float, device=device)
                fake_weights = torch.ones_like(batch_fake_labels, dtype=torch.float, device=device)
            else:
                raise ValueError("Not supported vicinal weight type!!!")

            return real_weights, fake_weights


        log_filename = os.path.join(self.results_folder, 'log_loss_niters{}.txt'.format(self.niters))
        if not os.path.isfile(log_filename):
            logging_file = open(log_filename, "w")
            logging_file.close()
        with open(log_filename, 'a') as file:
            file.write(
                "\n===================================================================================================\n")

        start_time = timeit.default_timer()

        while self.step < self.niters:

            d_reg_loss_val = 0.0
            d_dre_loss_val = 0.0
            g_reg_loss_val = 0.0
            g_dre_loss_val = 0.0

            ########################################################
            ### Train Discriminator

            for _ in range(self.num_D_steps):

                self.netD.train()

                for accumulation_index in range(self.num_grad_acc_d):

                    # ====== NEW: 从 NeighborSamplerDataLoader 里取一个 batch ======
                    (batch_real_images_raw,
                     batch_real_labels,
                     batch_fake_labels,
                     batch_target_labels,
                     kappa_l_all, kappa_r_all) = next(neighbor_iter)

                    # batch_real_images_raw: (B, H, W, C) 或 (B, C, H, W)，取决于 self.train_images
                    # DataLoader 对 numpy 会自动转成 torch.Tensor，这里统一转回 numpy 给原来的 fn_transform 用
                    if isinstance(batch_real_images_raw, torch.Tensor):
                        imgs_np = batch_real_images_raw.cpu().numpy()
                    else:
                        imgs_np = np.asarray(batch_real_images_raw)

                    # 保持你原来的 preprocessing 流程
                    batch_real_images = self.fn_transform(imgs_np)
                    batch_real_images = torch.from_numpy(
                        normalize_images(batch_real_images, to_neg_one_to_one=True)
                    ).type(torch.float).to(device)

                    # ====== 批量计算 vicinal weights ======
                    real_weights, fake_weights = compute_vicinal_weights_from_batch(
                        batch_real_labels=batch_real_labels,
                        batch_fake_labels=batch_fake_labels,
                        batch_target_labels=batch_target_labels,
                        kappa_l=kappa_l_all,
                        kappa_r=kappa_r_all,
                        vicinal_params=self.vicinal_params,
                        device=device,
                    )

                    with torch.no_grad():
                        with self.accelerator.autocast():
                            # 生成 fake images
                            z = torch.randn(self.batch_size_disc, self.dim_z, dtype=torch.float).to(device)
                            batch_fake_images = self.netG(z, self.fn_y2h(batch_fake_labels))

                    with (self.accelerator.autocast()):

                        # forward pass for D
                        h_target = self.fn_y2h(batch_target_labels).to(device)

                        if self.use_diffaug:
                            real_disc_out_dict = self.netD(
                                DiffAugment(batch_real_images, policy=self.diffaug_policy),h_target)
                            fake_disc_out_dict = self.netD(
                                DiffAugment(batch_fake_images.detach(), policy=self.diffaug_policy),h_target)
                        else:
                            real_disc_out_dict = self.netD(batch_real_images, h_target)
                            fake_disc_out_dict = self.netD(batch_fake_images.detach(), h_target)

                        # adversarial loss (带权重)
                        d_adv_loss = self.fn_disc_adv_loss(
                            real_adv_out=real_disc_out_dict['adv_output'],
                            fake_adv_out=fake_disc_out_dict['adv_output'],
                            real_weights=real_weights,
                            fake_weights=fake_weights,
                        )
                        d_adv_loss_val = d_adv_loss.item()
                        d_loss = d_adv_loss

                        # auxiliary regression loss
                        if self.aux_loss_params["weight_d_aux_reg_loss"] > 0 \
                            and self.aux_loss_params["use_aux_reg_branch"]:
                            if self.aux_loss_params["use_aux_reg_model"]:
                                fake_gt_labels = self.aux_reg_net(batch_fake_images).detach()
                            else:
                                # actually, the labels used as condition are unlikely to be the gt labels of fake images due to label inconsistency.
                                # fake_gt_labels = batch_target_labels
                                fake_gt_labels = batch_fake_labels

                            d_reg_loss = self.fn_disc_aux_reg_loss(
                                real_gt_labels=batch_target_labels,
                                real_pred_labels=real_disc_out_dict['reg_output'],
                                fake_gt_labels=fake_gt_labels,
                                fake_pred_labels=fake_disc_out_dict['reg_output'],
                                epsilon=np.maximum(kappa_l_all, kappa_r_all),
                            )
                            d_loss += self.aux_loss_params['weight_d_aux_reg_loss'] * d_reg_loss
                            d_reg_loss_val = d_reg_loss.item()

                        # auxiliary dre loss
                        if self.aux_loss_params["use_dre_reg"]:
                            dr_real = real_disc_out_dict["dre_output"]
                            dr_fake = fake_disc_out_dict["dre_output"]
                            d_dre_loss = self.penalized_softplus_loss(dr_real=dr_real, dr_fake=dr_fake)
                            d_loss += self.aux_loss_params["weight_d_aux_dre_loss"] * d_dre_loss
                            d_dre_loss_val = d_dre_loss.item()

                        d_loss /= float(self.num_grad_acc_d)

                    self.accelerator.backward(d_loss)
                # end for accumulation

                self.accelerator.clip_grad_norm_(self.netD.parameters(), self.max_grad_norm)
                self.accelerator.wait_for_everyone()
                self.optD.step()
                self.optD.zero_grad()
                self.accelerator.wait_for_everyone()

            ########################################################
            ### Train Generator

            self.netG.train()

            for _ in range(self.num_grad_acc_g):

                # generate fake images
                batch_target_labels_in_dataset = np.random.choice(
                    self.unique_train_labels,
                    size=self.batch_size_gene,
                    replace=True
                )
                batch_epsilons = np.random.normal(0, self.vicinal_params["kernel_sigma"],self.batch_size_gene)
                batch_target_labels = batch_target_labels_in_dataset + batch_epsilons
                batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(device)

                z = torch.randn(self.batch_size_gene, self.dim_z, dtype=torch.float).to(device)

                with self.accelerator.autocast():
                    batch_fake_images = self.netG(z, self.fn_y2h(batch_target_labels))

                    if self.use_diffaug:
                        disc_out_dict = self.netD(
                            DiffAugment(batch_fake_images, policy=self.diffaug_policy),
                            self.fn_y2h(batch_target_labels))
                    else:
                        disc_out_dict = self.netD(batch_fake_images, self.fn_y2h(batch_target_labels))

                    # adv loss
                    g_adv_loss = self.fn_gene_adv_loss(adv_out=disc_out_dict['adv_output'])
                    g_adv_loss_val = g_adv_loss.item()
                    g_loss = g_adv_loss

                    # auxiliary regression
                    if self.aux_loss_params["weight_g_aux_reg_loss"] > 0 and (
                            self.aux_loss_params["use_aux_reg_branch"] or self.aux_loss_params["use_aux_reg_model"]):
                        if self.aux_loss_params["use_aux_reg_branch"]:
                            fake_pred_labels = disc_out_dict['reg_output']
                        elif self.aux_loss_params["use_aux_reg_model"]:
                            fake_pred_labels = self.aux_reg_net(batch_fake_images)
                        g_reg_loss = self.fn_gene_aux_reg_loss(
                            fake_gt_labels=batch_target_labels,
                            fake_pred_labels=fake_pred_labels
                        )
                        g_loss += self.aux_loss_params["weight_g_aux_reg_loss"] * g_reg_loss
                        g_reg_loss_val = g_reg_loss.item()

                    # auxiliary dre penalty
                    if self.aux_loss_params["use_dre_reg"]:
                        g_dre_loss = (disc_out_dict["dre_output"].mean() - 1) ** 2
                        g_loss += self.aux_loss_params["weight_g_aux_dre_loss"] * g_dre_loss
                        g_dre_loss_val = g_dre_loss.item()

                    g_loss /= float(self.num_grad_acc_g)

                    self.accelerator.backward(g_loss)
            ##end for

            self.accelerator.clip_grad_norm_(self.netG.parameters(), self.max_grad_norm)
            self.accelerator.wait_for_everyone()
            self.optG.step()
            self.optG.zero_grad()
            self.accelerator.wait_for_everyone()

            self.step += 1

            if divisible_by(self.step, 20):
                print("\nGPU-%d CcGAN,%s,%s: [Iter %d/%d] [D loss: %.3f/%.3f/%.3f] [G loss: %.3f/%.3f/%.3f] [Time: %.3f]"
                      % (self.accelerator.local_process_index,self.net_name, self.loss_type, self.step, self.niters,
                         d_adv_loss_val, d_reg_loss_val, d_dre_loss_val,
                         g_adv_loss_val, g_reg_loss_val, g_dre_loss_val,
                         timeit.default_timer() - start_time))

            if self.accelerator.is_main_process:

                if self.use_ema:
                    self.ema_g.update()

                # if divisible_by(self.step, 20):
                #     print("\n CcGAN,%s,%s: [Iter %d/%d] [D loss: %.3f/%.3f/%.3f] [G loss: %.3f/%.3f/%.3f] [Time: %.3f]"
                #           % (self.net_name, self.loss_type, self.step, self.niters,
                #              d_adv_loss_val, d_reg_loss_val, d_dre_loss_val,
                #              g_adv_loss_val, g_reg_loss_val, g_dre_loss_val,
                #              timeit.default_timer() - start_time))

                if divisible_by(self.step, 500):
                    with open(log_filename, 'a') as file:
                        file.write(
                            "CcGAN,%s,%s: [Iter %d/%d] [D loss: %.3f/%.3f/%.3f] [G loss: %.3f/%.3f/%.3f] [Time: %.3f] \n"
                            % (self.net_name, self.loss_type, self.step, self.niters,
                               d_adv_loss_val, d_reg_loss_val, d_dre_loss_val,
                               g_adv_loss_val, g_reg_loss_val, g_dre_loss_val,
                               timeit.default_timer() - start_time))

                if self.step != 0 and (divisible_by(self.step, self.sample_freq) or get_s1()):
                    if self.use_ema:
                        self.ema_g.ema_model.eval()
                    else:
                        self.netG.eval()
                    with torch.inference_mode():
                        if self.use_ema:
                            gen_imgs = self.ema_g.ema_model(self.z_visual.to(device),
                                                            self.fn_y2h(self.y_visual).to(device))
                        else:
                            gen_imgs = self.netG(self.z_visual.to(device), self.fn_y2h(self.y_visual).to(device))
                        gen_imgs = gen_imgs.detach().cpu()
                        save_image(gen_imgs.data,
                                   self.save_images_folder + '/{}.png'.format(self.step),
                                   nrow=self.nrow_visual,
                                   normalize=True)
                    switch_s1(0)

                if self.step != 0 and (divisible_by(self.step, self.save_freq) or get_s2()):
                    milestone = self.step
                    self.save(milestone)
                    switch_s2(0)

        self.accelerator.print('training complete \n')

    ##end def train
    
    
    
    
    
    ############################################################################################################################ 
    ######################################################################################## 
    def sample_given_labels(self, given_labels, batch_size, denorm=True, to_numpy=False, verbose=False):
        """
        Generate samples based on given labels
        :given_labels: normalized labels
        :fn_y2h: label embedding function
        """
        
        device = self.accelerator.device
        
        assert isinstance(given_labels, np.ndarray)
        assert given_labels.min()>=-0.5 and given_labels.max()<=1.5 #labels may have noise, resulting values outside of [0,1]
        nfake = len(given_labels)

        if batch_size>nfake:
            batch_size = nfake
        
        fake_images = []
        fake_labels = np.concatenate((given_labels, given_labels[0:batch_size]))
        
        if verbose:
            pb = SimpleProgressBar()
        
        if self.use_ema:
            self.ema_g.ema_model.eval()
        else:
            self.netG.eval()
        with torch.inference_mode():
            n_img_got = 0
            while n_img_got < nfake:
                z = torch.randn(batch_size, self.dim_z, dtype=torch.float).to(device)
                y = torch.from_numpy(fake_labels[n_img_got:(n_img_got+batch_size)]).type(torch.float).view(-1,1).to(device)
                if self.use_ema:
                    batch_fake_images = self.ema_g.ema_model(z, self.fn_y2h(y))
                else:
                    batch_fake_images = self.netG(z, self.fn_y2h(y))
                if torch.isnan(batch_fake_images).any():
                    warnings.warn("NaN values detected in generated images!")
                    batch_fake_images = torch.nan_to_num(batch_fake_images, nan=torch.nanmean(batch_fake_images).item())
                if denorm: #denorm imgs to save memory
                    assert batch_fake_images.max().item()<=1.0 and batch_fake_images.min().item()>=-1.0
                    batch_fake_images = batch_fake_images*0.5+0.5
                    batch_fake_images = batch_fake_images*255.0
                    batch_fake_images = batch_fake_images.type(torch.uint8)
                    # assert batch_fake_images.max().item()>1
                fake_images.append(batch_fake_images.cpu())
                n_img_got += batch_size
                if verbose:
                    pb.update(min(float(n_img_got)/nfake, 1)*100)
            ##end while
            
        fake_images = torch.cat(fake_images, dim=0)
        #remove extra entries
        fake_images = fake_images[0:nfake]
        fake_labels = fake_labels[0:nfake]

        if to_numpy:
            fake_images = fake_images.numpy()
        else:
            fake_labels = torch.from_numpy(fake_labels).type(torch.float)

        return fake_images, fake_labels
                
                
        
    
    ##########################################################################################################  
    ########################################################################################  
    # Enhanced sampler based on the trained DR model
    '''
    (0) Finetuning the density ratio branch of the discriminator after the GAN training
    (1) Generate num_burnin_per_label fake images for each evaluation label
    (2) Compute density ratios for all generated (fake) images and determine the maximum density ratio per evaluation. Store these values in a dictionary, where evaluation labels serve as keys and the corresponding maximum density ratios as values.
    (3) Use rejection sampling and esitmated density ratios to sample from the trained GAN.
    ''' 
    def finetune_cdre(self):
        assert self.aux_loss_params["use_dre_reg"] and self.aux_loss_params["do_dre_ft"]
        assert self.step == self.niters #make sure the GAN training is complete
        
        self.ft_dre_flag = True
        
        dre_lambda = self.aux_loss_params["dre_lambda"]
        dre_ft_niters = self.aux_loss_params["dre_ft_niters"]
        dre_ft_lr = self.aux_loss_params["dre_ft_lr"]
        dre_ft_batch_size = self.aux_loss_params["dre_ft_batch_size"]      
         
        self.dre_net = copy.deepcopy(self.netD)
        
        # fix the parameters in dre_net except the final dre branch
        for param in self.dre_net.parameters():
            param.requires_grad = False
        for param in self.dre_net.dre_linear.parameters():
            param.requires_grad = True  # only train the dre linear branch

        device = self.accelerator.device
        
        optDRE = torch.optim.Adam(self.dre_net.dre_linear.parameters(), lr=dre_ft_lr, weight_decay=1e-4)

        self.dre_net, optDRE = self.accelerator.prepare(self.dre_net, optDRE)

        path_to_ckpt = self.results_folder + "/ckpt_niter_{}_dre_niter_{}.pth".format(self.step, dre_ft_niters)
        self.accelerator.print("\n dre ckpt path:", path_to_ckpt)
        
        if os.path.isfile(path_to_ckpt):
            checkpoint = torch.load(path_to_ckpt, map_location=device, weights_only=True)
            self.dre_net.load_state_dict(checkpoint['dre_net_state_dict'])
            return

        if dre_ft_niters == 0:
            return

        self.netG.eval()
        self.dre_net.eval()
        self.dre_net.dre_linear.train()
        for step in range(dre_ft_niters):
            ## randomly draw batch_size_disc y's from unique_train_labels
            batch_real_indx = np.random.choice(np.arange(len(self.train_labels)), size=dre_ft_batch_size, replace=True)

            ## draw real image/label batch from the training set
            batch_real_images = self.fn_transform(self.train_images[batch_real_indx])
            batch_real_images = torch.from_numpy(normalize_images(batch_real_images, to_neg_one_to_one=True)).type(torch.float).to(device)
            batch_target_labels = torch.from_numpy(self.train_labels[batch_real_indx]).type(torch.float).to(device)
            
            ## generate the fake image batch
            z = torch.randn(dre_ft_batch_size, self.dim_z, dtype=torch.float).to(device)
            batch_fake_images = self.netG(z, self.fn_y2h(batch_target_labels))
    
            ## density ratios for real images
            DR_real = self.dre_net(batch_real_images, self.fn_y2h(batch_target_labels))["dre_output"]
            ## density ratios for fake images
            DR_fake = self.dre_net(batch_fake_images, self.fn_y2h(batch_target_labels))["dre_output"]
    
            ## Softplus loss
            softplus_fn = torch.nn.Softplus(beta=1,threshold=20)
            sigmoid_fn = torch.nn.Sigmoid()
            SP_div = torch.mean(sigmoid_fn(DR_fake) * DR_fake) - torch.mean(softplus_fn(DR_fake)) - torch.mean(sigmoid_fn(DR_real))
            penalty = dre_lambda * (torch.mean(DR_fake) - 1)**2
            dre_loss = SP_div + penalty
            
            dre_loss_val = dre_loss.item()
            
            self.dre_accelerator.backward(dre_loss)
            self.dre_accelerator.clip_grad_norm_(self.dre_net.dre_linear.parameters(), self.max_grad_norm)
            self.dre_accelerator.wait_for_everyone()
            optDRE.step()
            optDRE.zero_grad()
            self.dre_accelerator.wait_for_everyone()
            
            self.accelerator.print("\n Step:{}/{}; Finetune DRE loss: {:.3f}; DR real: {:.3f}; DR fake: {:.3f}".format(step+1, dre_ft_niters, dre_loss_val, DR_real.mean().item(), DR_fake.mean().item()))
        
            # ## debugging
            if divisible_by(step+1, 100):
                self.dre_net.dre_linear.eval()
                with torch.no_grad():
                    DR_real2 = self.dre_net(batch_real_images, self.fn_y2h(batch_target_labels))["dre_output"]
                    DR_fake2 = self.dre_net(batch_fake_images, self.fn_y2h(batch_target_labels))["dre_output"]
                    self.accelerator.print("\n Debug DR real (train): {:.3f}; DR fake (train): {:.3f}".format(DR_real.mean().item(), DR_fake.mean().item()))
                    self.accelerator.print("\n Debug DR real (eval): {:.3f}; DR fake (eval): {:.3f}".format(DR_real2.mean().item(), DR_fake2.mean().item()))
                self.dre_net.dre_linear.train()
        ##end for step 
        # store model
        torch.save({
            'dre_net_state_dict': self.dre_net.state_dict(),
        }, path_to_ckpt)
        self.accelerator.print("\n End finetuning.")       
        

    
    ## compute density ratios given images and their labels
    def compute_density_ratio(self, images, labels, batch_size=100, to_numpy=False):
        
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert images.max().item()>1.0 ##make sure all images are not normalized
        assert labels.max().item()<=1.0 and labels.min().item()>=0 ##make sure all labels are normalized to [0,1]
        
        if self.ft_dre_flag:
            device = self.dre_accelerator.device
            dre_net = self.dre_net
        else:
            device = self.accelerator.device
            dre_net = self.netD
        
        #imgs: a torch tensor
        n_imgs = len(images)
        if batch_size>n_imgs:
            batch_size = n_imgs

        ##make sure the last iteration has enough samples
        images = torch.cat((images, images[0:batch_size]), dim=0)
        labels = torch.cat((labels, labels[0:batch_size]), dim=0)
        
        dre_net.eval()
        density_ratios = []
        with torch.inference_mode():
            n_imgs_got = 0
            while n_imgs_got < n_imgs:
                batch_images = images[n_imgs_got:(n_imgs_got+batch_size)]
                batch_images = normalize_images(batch_images, to_neg_one_to_one=True)
                batch_labels = labels[n_imgs_got:(n_imgs_got+batch_size)]
                batch_images = batch_images.type(torch.float).to(device)
                batch_labels = batch_labels.type(torch.float).view(-1,1).to(device)
                batch_ratios = dre_net(batch_images, self.fn_y2h(batch_labels))["dre_output"]
                density_ratios.append(batch_ratios.cpu().detach())
                n_imgs_got += batch_size
        ### while n_imgs_got
        density_ratios = torch.cat(density_ratios)
        density_ratios = density_ratios[0:n_imgs].cpu()
        if to_numpy:
            density_ratios = density_ratios.numpy()
        return density_ratios
        
    ## rejection sampling for one label
    def rejection_sampling_given_label(self, given_label, nfake, nburnin, batch_size=100, verbose=False):
        # given_label is a value    
        
        if batch_size>nfake:
            batch_size = nfake
        ## Burn-in Stage
        burnin_labels = np.ones(nburnin)*given_label
        burnin_images, _ = self.sample_given_labels(given_labels=burnin_labels, batch_size=np.min([batch_size, nburnin]), denorm=True, to_numpy=True, verbose=False)
        ### get the maximum density ratio
        burnin_density_ratios = self.compute_density_ratio(images=burnin_images, labels=burnin_labels, batch_size=np.min([batch_size, nburnin]), to_numpy=True)
        self.accelerator.print((burnin_density_ratios.min(), np.median(burnin_density_ratios), np.mean(burnin_density_ratios), burnin_density_ratios.max()))
        M_bar = np.max(burnin_density_ratios)
        
        ## Rejection sampling
        if verbose:
            pb = SimpleProgressBar()
        enhanced_imgs = []
        num_imgs = 0
        while num_imgs < nfake:
            batch_images, batch_labels = self.sample_given_labels(given_labels=np.ones(batch_size)*given_label, batch_size=batch_size, denorm=True, to_numpy=True, verbose=False)
            batch_ratios = self.compute_density_ratio(images=batch_images, labels=batch_labels, batch_size=batch_size, to_numpy=True)
            M_bar = np.max([M_bar, np.max(batch_ratios)])
            #threshold
            batch_p = batch_ratios/M_bar
            batch_psi = np.random.uniform(size=batch_size).reshape(-1,1)
            indx_accept = np.where(batch_psi<=batch_p)[0]
            if len(indx_accept)>0:
                enhanced_imgs.append(batch_images[indx_accept])
            num_imgs+=len(indx_accept)
            if verbose:
                pb.update(np.min([float(num_imgs)*100/nfake,100]))
        enhanced_imgs = np.concatenate(enhanced_imgs, axis=0)
        enhanced_imgs = enhanced_imgs[0:nfake]
        return enhanced_imgs, given_label*np.ones(nfake)         
    
    ## rejection sampling for some labels
    def rejection_sampling_given_labels(self, given_labels, nburnin_per_label, batch_size=100, verbose=False):
        # given_labels is an array
        self.accelerator.print("\n Start rejection sampling...")
        assert isinstance(given_labels, np.ndarray)
        assert given_labels.min()>=0 and given_labels.max()<=1.0
        unique_labels, counts_elements = np.unique(given_labels, return_counts=True) 
        fake_images = []
        fake_labels = []
        start = timeit.default_timer()
        for i in range(len(unique_labels)):
            label_i = unique_labels[i]
            nfake_i = counts_elements[i]
            self.accelerator.print("\n [{}/{}] Start generating {} fake images for label {}.".format(i+1, len(unique_labels), nfake_i, label_i))
            fake_images_i, fake_labels_i = self.rejection_sampling_given_label(given_label=label_i, nfake=nfake_i, nburnin=nburnin_per_label, batch_size=batch_size, verbose=verbose)
            assert isinstance(fake_images_i, np.ndarray) and isinstance(fake_labels_i, np.ndarray)
            assert fake_images_i.max()>1 and fake_images_i.max()<=255.0 and fake_labels_i.min()>=0 and fake_labels_i.max()<=1
            fake_images.append(fake_images_i)
            fake_labels.append(fake_labels_i.reshape(-1))
            self.accelerator.print("\n [{}/{}] Finish generating {} fake images for label {}. Time elapses: {}".format(i+1, len(unique_labels), nfake_i, label_i, timeit.default_timer()-start))
        ##end for i
        fake_images = np.concatenate(fake_images, axis=0)
        fake_labels = np.concatenate(fake_labels, axis=0)
        self.accelerator.print('\n End generating fake data!')
        self.accelerator.print("\n We got {} fake images.".format(len(fake_images)))
        return fake_images, fake_labels


class VicinitySamplerDataset(Dataset):
    """
    Dataset for adaptive or fixed vicinity-based sampling.
    """

    def __init__(self, images, labels_norm, unique_labels_norm, vicinal_params):

        self.accelerator = get_accelerator()
        self.images = images  # 原始 Dataset (带 __getitem__)
        self.labels_norm = labels_norm
        self.unique_labels_norm = unique_labels_norm

        # vicinal_params
        self.vicinal_params = vicinal_params

        # return the sorted unique elements of an array, and the number of times each unique value comes up in the input array
        self.unique_labels_norm_sorted, self.counts_train_elements = np.unique(self.unique_labels_norm, return_counts=True)

    def __len__(self):
        # 只要让 DataLoader 有持续的 batch，就可以虚拟化一个大长度
        return 10 ** 9  # 无限流式采样（或指定任意较大值）

    def __getitem__(self, idx):

        labels_norm = self.labels_norm  # (N,)
        unique_sorted = self.unique_labels_norm_sorted  # (K,)
        counts_sorted = self.counts_train_elements  # (K,)

        kernel_sigma = self.vicinal_params["kernel_sigma"]
        kappa = self.vicinal_params["kappa"]
        threshold_type = self.vicinal_params["threshold_type"].lower()
        nonzero_soft_weight_threshold =self.vicinal_params["nonzero_soft_weight_threshold"]
        use_ada_vic = self.vicinal_params["use_ada_vic"]
        ada_vic_type = self.vicinal_params["ada_vic_type"].lower()
        min_n_per_vic = self.vicinal_params["min_n_per_vic"]
        ada_eps = self.vicinal_params["ada_eps"]
        use_symm_vic = self.vicinal_params["use_symm_vic"]

        # Fetch a single sample
        target_label_in_dataset = np.random.choice(self.unique_labels_norm)
        eps = np.random.normal(0, kernel_sigma)
        target_y = target_label_in_dataset + eps

        if not use_ada_vic:

            soft_thr = -np.log(nonzero_soft_weight_threshold) / kappa

            while True:
                if threshold_type == "hard":
                    indx_real_in_vicinity = np.nonzero(np.abs(labels_norm - target_y) <= kappa)[0]
                else:
                    indx_real_in_vicinity = np.nonzero((labels_norm - target_y) ** 2 <= soft_thr)[0]
                if len(indx_real_in_vicinity) >= 1:
                    break
                eps = np.random.normal(0, kernel_sigma)
                target_y = target_label_in_dataset + eps

            # Select random real image from vicinity
            real_idx = np.random.choice(indx_real_in_vicinity)
            real_y = labels_norm[real_idx]

            # Generate fake labels within the vicinity
            if threshold_type == "hard":
                lb = target_y - kappa
                ub = target_y + kappa
            else:
                radius = np.sqrt(soft_thr)
                lb = target_y - radius
                ub = target_y + radius
            lb = max(0.0, lb)
            ub = min(ub, 1.0)
            fake_y = np.random.uniform(lb, ub, size=1)[0]

            kappa_l,kappa_r = kappa, kappa

        else:

            # ---------------------------
            # 2. 根据 target_y 动态扩展邻域, 求 kappa_l, kappa_r
            # ---------------------------

            # 在 unique_labels 里找到 target_y 的插入位置
            idx_y = np.searchsorted(unique_sorted, target_y, side='left')
            kappa_l, kappa_r = ada_eps, ada_eps
            n_got = 0

            # case 1: target_y target_y <= unique_y[0]. Only move toward right
            if idx_y <= 0:
                idx_l, idx_r = 0, 0
                n_got = counts_sorted[idx_r]
                kappa_r = np.abs(target_y - unique_sorted[idx_r]) + ada_eps

                loop_counter = 0
                # 不断往右扩，直到拿到足够样本或到头
                while n_got < min_n_per_vic and idx_r < len(unique_sorted) - 1:
                    idx_r += 1
                    n_got += counts_sorted[idx_r]
                    kappa_r = np.abs(target_y - unique_sorted[idx_r])
                    loop_counter += 1
                    if loop_counter > 1e20:
                        self.accelerator.print("\n Detected a potential infinite loop (case 1)")
                        break

            # case 2: target_y >= unique_y[-1]. Only move toward left
            elif idx_y >= len(unique_sorted) - 1:
                idx_l, idx_r = len(unique_sorted) - 1,len(unique_sorted) - 1
                n_got = counts_sorted[idx_l]
                kappa_l = np.abs(target_y - unique_sorted[idx_l]) + ada_eps

                loop_counter = 0
                while n_got < min_n_per_vic and idx_l > 0:
                    idx_l -= 1
                    n_got += counts_sorted[idx_l]
                    kappa_l = abs(target_y - unique_sorted[idx_l])
                    loop_counter += 1
                    if loop_counter > 1e20:
                        self.accelerator.print("\n Detected a potential infinite loop (case 2)")
                        break

            # case 3: between
            else:
                if target_y in unique_sorted:  # target_y 正好在 unique 里
                    idx_l, idx_r = idx_y - 1, idx_y + 1
                    n_got = counts_sorted[idx_y]
                else:
                    idx_l, idx_r = idx_y - 1, idx_y
                    n_got = 0

                # 初始的左右距离
                dist2left = np.abs(target_y - unique_sorted[idx_l])
                dist2right = np.abs(target_y - unique_sorted[idx_r])

                loop_counter = 0
                while n_got < min_n_per_vic:
                    if dist2left < dist2right:  # If closer to the left label, expand to the left.
                        kappa_l = dist2left
                        n_got += counts_sorted[idx_l]
                        idx_l -= 1
                    elif dist2left > dist2right:  #If closer to the right label, expand to the right.
                        kappa_r = dist2right
                        n_got += counts_sorted[idx_r]
                        idx_r += 1
                    else:  # When the distances on both sides are equal, expand in both directions.
                        kappa_l = dist2left
                        kappa_r = dist2right
                        n_got += counts_sorted[idx_l] + counts_sorted[idx_r]
                        idx_l -= 1
                        idx_r += 1

                    if idx_l < 0:
                        dist2left = 1e30
                    else:
                        dist2left = np.abs(target_y - unique_sorted[idx_l])

                    if idx_r > len(unique_sorted) - 1:
                        dist2right = 1e30
                    else:
                        dist2right = np.abs(target_y - unique_sorted[idx_r])

                    if dist2left > 1e10 and dist2right > 1e10:  # 两边都扩到底了
                        break

                    loop_counter += 1
                    if loop_counter > 1e20:
                        self.accelerator.print("\n Detected a potential infinite loop (case 3)")
                        break

            # 对称邻域: 左右取 max 或 min
            if use_symm_vic:
                max_kappa = np.max([kappa_l, kappa_r])
                kappa_l, kappa_r = max_kappa, max_kappa
                # 如需取较小的可改为 min

            # ---------------------------
            # 3. 基于 kappa_l/kappa_r 选取邻域内真实样本索引
            # ---------------------------
            # 硬邻域
            cond_hard = (labels_norm >= (target_y - kappa_l)) & (labels_norm <= (target_y + kappa_r))
            indx_real_in_hard_vicinity = np.nonzero(cond_hard)[0]

            # 软邻域 (左/右分开)
            nu_l = 1.0 / (kappa_l ** 2)
            nu_r = 1.0 / (kappa_r ** 2)
            soft_thr = -np.log(nonzero_soft_weight_threshold)

            indx_left = np.nonzero(labels_norm <= target_y)[0]
            indx_right = np.nonzero(labels_norm > target_y)[0]

            indx_soft_left = np.nonzero((labels_norm - target_y) ** 2 <= soft_thr / nu_l)[0]
            indx_soft_left = np.intersect1d(indx_soft_left, indx_left)

            indx_soft_right = np.nonzero((labels_norm - target_y) ** 2 <= soft_thr / nu_r)[0]
            indx_soft_right = np.intersect1d(indx_soft_right, indx_right)

            indx_real_in_soft_vicinity = np.concatenate([indx_soft_left, indx_soft_right])

            if ada_vic_type == "vanilla":
                if threshold_type == "hard":
                    indx_real_in_vicinity = indx_real_in_hard_vicinity
                else:
                    indx_real_in_vicinity = indx_real_in_soft_vicinity
            elif ada_vic_type == "hybrid":  # 硬/软交集, 变小
                indx_real_in_vicinity = np.intersect1d(
                    indx_real_in_hard_vicinity,
                    indx_real_in_soft_vicinity
                )
            else:
                raise ValueError(f"Not supported ada vicinity type: {ada_vic_type} ! ")

            assert len(indx_real_in_vicinity) >= 1, ("There are no real samples in the neighborhood. "
                                                     "Please check the data and parameter settings.")

            # 在邻域中随机取一个真实样本索引
            real_idx = int(np.random.choice(indx_real_in_vicinity, size=1)[0])
            real_y = labels_norm[real_idx]

            # ---------------------------
            # 4. 生成 fake_label_norm (同一邻域内均匀采样)
            # ---------------------------
            # 硬邻域的上下界
            lb_hard = target_y - kappa_l
            ub_hard = target_y + kappa_r
            lb_hard = max(0.0, lb_hard)
            ub_hard = min(1.0, ub_hard)
            assert 0.0 <= lb_hard <= 1.0 and 0.0 <= ub_hard <= 1.0 and lb_hard <= ub_hard

            # 软邻域的上下界
            lb_soft = target_y - np.sqrt(soft_thr / nu_l)
            ub_soft = target_y + np.sqrt(soft_thr / nu_r)
            lb_soft = max(0.0, lb_soft)
            ub_soft = min(1.0, ub_soft)
            assert 0.0 <= lb_soft <= 1.0 and 0.0 <= ub_soft <= 1.0 and lb_soft <= ub_soft

            if ada_vic_type == "vanilla":
                if threshold_type == "hard":
                    lb, ub = lb_hard, ub_hard
                else:
                    lb, ub = lb_soft, ub_soft
            elif ada_vic_type == "hybrid":  # 取交集, 更小的范围
                lb = max(lb_hard, lb_soft)
                ub = min(ub_hard, ub_soft)
            else:
                raise ValueError("Not supported vicinity type!!!")

            fake_y = np.random.uniform(lb, ub, size=1)[0]

        return (
            self.images[real_idx],
            torch.tensor(real_y, dtype=torch.float32),
            torch.tensor(fake_y, dtype=torch.float32),
            torch.tensor(target_y, dtype=torch.float32),
            torch.tensor(kappa_l, dtype=torch.float32),
            torch.tensor(kappa_r, dtype=torch.float32),
        )


    def get_dataloader(self, batch_size, num_workers=4, pin_memory=True, prefetch_factor=4, persistent_workers=True):
        return DataLoader(self, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers,
                          pin_memory=pin_memory,
                          prefetch_factor=prefetch_factor,
                          persistent_workers=persistent_workers)
