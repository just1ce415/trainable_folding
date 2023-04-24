#!/bin/bash
#BSUB -W 02:00
#BSUB -nnodes 2
#BSUB -P BIP215
#BSUB -o antibodies.o%J
#BSUB -J antibodies
#BSUB -N ernest.glukhov@stonybrook.edu

module load gcc/9.1.0
module load open-ce/1.5.2-py39-0
conda activate folding

export OMP_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/bip215/proj-shared/eglukhov/torch_extensions

# Get version from params
#model_version={{MODEL_VERSION}};
model_version=1;

af_params_dir="/gpfs/alpine/bip215/proj-shared/eglukhov/af_params/";
project_dir="/gpfs/alpine/bip215/proj-shared/eglukhov/compress";
data_dir=$project_dir/datasets/v4;
project="compress_test";
run_name="one_with_checkpoint_${model_version}_olcf";
run_version='train_1';
output_dir=$project_dir/output/$run_name/$run_version;

mkdir -p $output_dir;
jsrun -n 2 -c 6 -g 6 -a 6 --smpiargs="-disable_gpu_hooks" python3 train_multi_gpu.py \
          --gpus 6 \
          --num_nodes 2 \
          --wandb_offline \
          --wandb_output_dir $output_dir \
          --wandb_project $project \
          --wandb_name $run_name \
          --wandb_id $run_name \
          --model_weights_path $af_params_dir/params_model_${model_version}_multimer_v2.npz \
          --model_checkpoint_path $output_dir/checkpoints \
          --preprocessed_data_dir $data_dir/npz_data \
          --output_data_path $output_dir/data \
          --train_json_path $data_dir/train.json \
          --val_json_path $data_dir/val.json \
          --test_mode_name "val_metrics" \
          --n_layers_in_lr_group 10 \
          --crop_size 384 \
          --learning_rate 0.0001 \
          --max_epochs 10 \
          --step 'train'

wandb sync $output_dir/wandb/latest-run/