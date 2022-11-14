#!/bin/bash
#BSUB -W 02:00
#BSUB -nnodes 10
#BSUB -P BIP215
#BSUB -o finetune.o%J
#BSUB -J finetune

module load gcc/9.1.0
module load cuda
#export RANK=$OMPI_COMM_WORLD_RANK
#export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
#export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
#echo "Setting env_var RANK=${RANK}"
#echo "Setting env_var LOCAL_RANK=${LOCAL_RANK}"
#echo "Setting env_var WORLD_SIZE=${WORLD_SIZE}"

#export RANK=0
#export WORLD_SIZE=1
export OMP_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/bip215/proj-shared/torch_extensions
#export NCCL_DEBUG=INFO

#jsrun -n 1 -c 1 -g 1 -a 1 python test.py
jsrun --smpiargs="-disable_gpu_hooks"  -n 10 -c 6 -g 6 -a 6 -r 1 python3 train_multi_gpu.py --gpus 6 --replace_sampler_ddp=True --deepspeed_config_path deepspeed_config.json --resume_from_ckpt /gpfs/alpine/bip215/proj-shared/checkpoints/epoch\=5-step\=89.ckpt
