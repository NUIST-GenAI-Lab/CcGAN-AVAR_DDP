#!/bin/bash

# 如果命令行参数里包含 --eval_only，就不要改 CUDA_VISIBLE_DEVICES
if [[ " $* " == *" --eval_only"* ]]; then
    echo "[INFO] --eval_only detected, keep existing CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
else
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4
    echo "[INFO] training mode, set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
export PYTHONUNBUFFERED=1

ROOT_PATH="/home/cy/nuist-lab/CcGAN-AVAR"
DATA_PATH="/home/shared/CCGM"
NIQE_PATH="<YOUR PATH>"

DATA_NAME="UTKFace"
SETTING="hav_cy_4_01234_sync"

SEED=2025
MIN_LABEL=1
MAX_LABEL=60
IMG_SIZE=128

BATCH_SIZE_G=64
BATCH_SIZE_D=64
NUM_D_STEPS=2
SIGMA=-1
KAPPA=-1
LR_G=1e-4
LR_D=1e-4
NUM_ACC_D=1
NUM_ACC_G=1

NET_NAME="SNGAN"
LOSS_TYPE="hinge"
THRESH_TYPE="soft"
MIN_N_PER_VIC=400

DIM_GAN=256
DIM_Y=128

NITERS=60000
RESUME_STEP=0

LOGFILE="output_${DATA_NAME}_${IMG_SIZE}_${SETTING}.txt"

nohup setsid accelerate launch main_cy.py \
    --setting_name "$SETTING" --data_name "$DATA_NAME" \
    --root_path "$ROOT_PATH" --data_path "$DATA_PATH" --seed "$SEED" \
    --min_label "$MIN_LABEL" --max_label "$MAX_LABEL" --img_size "$IMG_SIZE" \
    --net_name "$NET_NAME" --dim_z "$DIM_GAN" --dim_y "$DIM_Y" \
    --gene_ch 64 --disc_ch 48 \
    --niters "$NITERS" --resume_iter "$RESUME_STEP" --loss_type "$LOSS_TYPE" --num_D_steps "$NUM_D_STEPS" \
    --save_freq 5000 --sample_freq 1000 \
    --batch_size_disc "$BATCH_SIZE_D" --batch_size_gene "$BATCH_SIZE_G" \
    --lr_g "$LR_G" --lr_d "$LR_D" \
    --num_grad_acc_d "$NUM_ACC_D" --num_grad_acc_g "$NUM_ACC_G" \
    --kernel_sigma "$SIGMA" --threshold_type "$THRESH_TYPE" --kappa "$KAPPA" \
    --use_diffaug --diffaug_policy color,translation,cutout \
    --use_ema --use_amp --max_grad_norm 1.0 \
    --use_ada_vic --ada_vic_type hybrid --min_n_per_vic "$MIN_N_PER_VIC" --use_symm_vic \
    --aux_reg_loss_type ei_hinge --weight_d_aux_reg_loss 1.0 --weight_g_aux_reg_loss 1.0 \
    --use_dre_reg --dre_head_arch "MLP3_dropout" --dre_lambda 1e-2 --weight_d_aux_dre_loss 0.5 --weight_g_aux_dre_loss 0.5 \
    --do_eval \
    --samp_batch_size 200 --eval_batch_size 200 \
    --use_sync_bn \
    "$@" \
    >> "$LOGFILE" 2>&1 &

PID=$!
echo
echo ">>> 训练已在后台启动，PID=${PID}, 日志: ${LOGFILE}"
echo
echo ">>> help:"
echo ">>> 结束进程: kill ${PID}"
echo ">>> 日志跟踪：tail -n 1000 -f ${LOGFILE}"
echo

tail -n 1000 -f "$LOGFILE"