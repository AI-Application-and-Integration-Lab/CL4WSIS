#!/bin/bash

port=$(python get_free_port.py)
echo ${port}
alias exp='python -m torch.distributed.launch --nproc_per_node=8 --master_port ${port} run.py --num_workers 32 --sample_num 8'
shopt -s expand_aliases

dataset=coco-voc
task=voc
lr_init=0.00005

path=checkpoints/step/${dataset}-${task}/
dataset_pars="--dataset ${dataset} --task ${task} --batch_size 16 $ov --val_interval 50"
exp --name FTwide --step 0 --bce --lr ${lr_init} ${dataset_pars} --epochs 200 --optim adam --weight_decay 0

# phase1
pretr_FT=${path}FTwide_0.pth
lr=0.001

exp --name OURS --step 1 --weakly ${dataset_pars} --alpha 0.9 --lr ${lr} --step_ckpt $pretr_FT \
 --loss_de 1 --lr_policy warmup --affinity --epochs 30 --optim sgd --phase 1

# phase 2
lr=0.00005
pretr_seg=${path}OURS_1.pth

exp --name OURS --step 1 --weakly ${dataset_pars} --alpha 0.9 --lr ${lr} --step_ckpt $pretr_FT \
 --loss_de 1 --lr_policy warmup --affinity --epochs 50 --optim adam --weight_decay 0 --seg_ckpt $pretr_seg --phase 2

# inference on single GPU