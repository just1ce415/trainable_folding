#!/bin/bash
#BSUB -W 02:00
#BSUB -nnodes 72
#BSUB -P BIP215
#BSUB -o antibodies.o%J
#BSUB -J antibodies
#BSUB -N ernest.glukhov@stonybrook.edu

module load gcc/9.1.0
module load open-ce/1.5.2-py39-0
conda activate folding

export OMP_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/bip215/proj-shared/eglukhov/torch_extensions

project_dir="/gpfs/alpine/bip215/proj-shared/eglukhov/antibodies";
data_dir=$project_dir/datasets/v3;
project="antibodies_test";
run_name="memory_check_10_no_recycles";
output_dir=$project_dir/output/$run_name/$LSB_JOBID;
mkdir -p $output_dir;
jsrun -n 72 -c 6 -g 6 -a 6 --smpiargs="-disable_gpu_hooks"  python3 train_multi_gpu.py \
          --gpus 6 \
          --num_nodes 72 \
          --replace_sampler_ddp=True \
          --deepspeed_config_path deepspeed_config.json \
          --wandb_output_dir $output_dir \
          --wandb_project $project \
          --wandb_name $run_name \
          --wandb_id $run_name \
          --model_weights_path /gpfs/alpine/bip215/proj-shared/eglukhov/af_params/params_model_1_multimer_v2.npz \
          --model_checkpoint_path $output_dir/checkpoints \
          --preprocessed_data_dir $data_dir/npz_data \
          --train_json_path $data_dir/train.json \
          --val_json_path $data_dir/val.json \
          --test_mode_name "val_metrics" \
          --n_layers_in_lr_group 10 \
          --crop_size 384

wandb sync $output_dir/wandb/latest-run/