#!/bin/bash
timestamp=$(date "+%Y_%m%d_%H%M.%S")
which python
python train_robust.py \
    --dset_id 42178 \
    --batchsize 32 \
    --attention_heads 16 \
    --task 'binary' \
    --attentiontype 'col' \
    --pretrain \
    --pt_tasks 'contrastive' 'denoising' \
    --pt_aug 'mixup' 'cutmix' \
    --ssl_samples 16 \
    --embedding_size 32 \
    --pretrain_save_path ./pretrain_model/${timestamp}.pt \
    --pretrain_epochs 50 \
    --epochs 200 \
    --lr 0.00001 \
    