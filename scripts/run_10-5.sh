#!/bin/bash

# We assume to have a parameter indicating whether to use overlap (0 or 1)
port=$(python get_free_port.py)
echo ${port}
alias exp='python -m torch.distributed.launch --nproc_per_node=1 --master_port ${port} run.py --num_workers 4 --sample_num 8'
shopt -s expand_aliases
overlap=1

dataset=voc
task=10-5
lr_init=0.00005

if [ ${overlap} -eq 0 ]; then
  path=checkpoints/step/${dataset}-${task}/
  ov=""
else
  path=checkpoints/step/${dataset}-${task}-ov/
  ov="--overlap"
  echo "Overlap"
fi

########## step0
dataset_pars="--dataset ${dataset} --task ${task} --batch_size 16 $ov --val_interval 10"
exp --name OURS --step 0 --bce --lr ${lr_init} ${dataset_pars} --epochs 100 --optim adam --weight_decay 0


########## step1
# phase 1
lr=0.001

pretr_FT=${path}OURS_0.pth

exp --name OURS --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr_FT \
 --loss_de 1 --lr_policy warmup --affinity --epochs 40 --optim sgd --phase 1

# phase 2
lr=0.00005

pretr_seg=${path}OURS_1.pth

exp --name OURS --step 1 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr_FT \
 --loss_de 1 --lr_policy warmup --affinity --epochs 50 --optim adam --weight_decay 0 --seg_ckpt $pretr_seg --phase 2

########## step2
# phase 1
lr=0.001

pretr_FT=${path}OURS_1.pth

exp --name OURS --step 2 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr_FT \
 --loss_de 1 --lr_policy warmup --affinity --epochs 40 --optim sgd --phase 1

# phase 2
lr=0.00005

pretr_seg=${path}OURS_2.pth

exp --name OURS --step 2 --weakly ${dataset_pars} --alpha 0.5 --lr ${lr} --step_ckpt $pretr_FT \
 --loss_de 1 --lr_policy warmup --affinity --epochs 50 --optim adam --weight_decay 0 --seg_ckpt $pretr_seg --phase 2
