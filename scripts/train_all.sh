#!/bin/bash

CONDA=Path_to_Conda
ENV=Conda_env_name

echo "running"
# export WANDB_MODE=offline

# model structure
N_EMBD=768 N_HEAD=12 N_LAYER=12
BATCH_SIZE=16 SEQ_LEN=512 ACC_STEP=16
ITERATIONS=16000 WARMUP_STEPS=2000

# common options for all optimizers
COMMON=(
  --config_format base
  --results_base_folder ./results
  --n_embd "$N_EMBD"
  --n_head "$N_HEAD"
  --n_layer "$N_LAYER"
  --batch_size "$BATCH_SIZE"
  --sequence_length "$SEQ_LEN"
  --acc_steps "$ACC_STEP"
  --model base
  --distributed_backend nccl
  --iterations "$ITERATIONS"
  --experiment_name 124m
  --seed 114514
  --datasets_dir ./datasets
  --dataset c4
  --wandb
  --wandb_project HTnoise
  --eval_interval 100
  --latest_ckpt_interval 1000
  --save_cnt 5
)

## adamw
$CONDA run -n $ENV torchrun --nproc_per_node=8 ./src/main.py \
   "${COMMON[@]}" --warmup_steps "$WARMUP_STEPS"\
   --opt adamw --lr 1e-3 --weight_decay 0.1 --scheduler cos \
   --beta1 0.9 --beta2 0.95 --dropout 0.0 --grad_clip 0.5

# muon
$CONDA run -n $ENV torchrun --nproc_per_node=8 ./src/main.py \
   "${COMMON[@]}" --warmup_steps "$WARMUP_STEPS"\
   --opt muon --lr 1e-3 --muon_lr_factor 1e-2 --scheduler cos \
   --beta1 0.9 --beta2 0.99 --momentum 0.95 --nesterov False --muon_ns_steps 5 --save_cnt 5

### muonlight
$CONDA run -n $ENV torchrun --nproc_per_node=8 ./src/main.py \
   "${COMMON[@]}" --warmup_steps "$WARMUP_STEPS"\
   --opt muonlight --lr 1e-3 --muon_lr_factor 1e-2 --weight_decay 0.1 --scheduler cos \
   --beta1 0.9 --beta2 0.99 --momentum 0.95 --nesterov True --muon_ns_steps 5 --grad_clip 0.5

## signum
$CONDA run -n $ENV torchrun --nproc_per_node=8 ./src/main.py \
   "${COMMON[@]}" --warmup_steps "$WARMUP_STEPS"\
   --opt signum --lr 1e-4 --weight_decay 1 --scheduler linear\
   --momentum 0.9 --grad_clip 0.5

## Lion
$CONDA run -n $ENV torchrun --nproc_per_node=8 ./src/main.py \
   "${COMMON[@]}" --warmup_steps "$WARMUP_STEPS"\
   --opt lion --lr 1e-4 --weight_decay 1 --scheduler linear\
   --beta1 0.9 --beta2 0.95 --grad_clip 0.5 --save_cnt 5

### NSGD
$CONDA run -n $ENV torchrun --nproc_per_node=8 ./src/main.py \
   "${COMMON[@]}" --warmup_steps "$WARMUP_STEPS"\
   --opt nsgd --lr 1e-3 --weight_decay 0.1 --momentum 0.9 --grad_clip 0.5\


echo "finished!"