#!/bin/bash
#BSUB -W 02:00
#BSUB -nnodes 5
#BSUB -P BIP215
#BSUB -o new_residue.o%J
#BSUB -J new_residue
#BSUB -N ernest.glukhov@stonybrook.edu

module load gcc/9.1.0
module load open-ce/1.5.2-py39-0
conda activate folding

export OMP_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/bip215/proj-shared/eglukhov/torch_extensions
export PRODY_CONFIG_DIR=/gpfs/alpine/bip215/proj-shared/eglukhov/.prody

# Get version from params
#model_version={{MODEL_VERSION}}
model_version=1

af_params_dir="/gpfs/alpine/bip215/proj-shared/eglukhov/af_params/";
project_dir="/gpfs/alpine/bip215/proj-shared/eglukhov/new_residue";
data_dir=$project_dir/datasets/v9;
project="new_residue";
run_name="get_data_${model_version}";
run_version='v1'
#restore_version='train_1'
output_dir=$project_dir/output/$run_name/$run_version;

mkdir -p $output_dir;
jsrun -n 5 -c 6 -g 6 -a 6 --smpiargs="-disable_gpu_hooks" python3 -m train_feature_process \
      --gpus 6 \
      --num_nodes 5 \
      --train_json_path $data_dir/debug_train.json \
      --val_json_path $data_dir/train.json \
      --preprocessed_data_dir $data_dir/npz_files \
      --model_weights_path $af_params_dir/params_model_${model_version}_multimer_v2.npz \
      --model_checkpoint_path $output_dir/checkpoints \
      --output_data_path $output_dir \
      --wandb_offline \
      --wandb_output_dir $output_dir \
      --wandb_project $project \
      --wandb_name $run_name \
      --wandb_id $run_name \
      --test_mode_name "val" \
      --max_epochs 50 \
      --step 'predict'



jsrun -n 5 -c 6 -g 6 -a 6 --smpiargs="-disable_gpu_hooks" python3 -m train_feature_process \
      --gpus 6 \
      --num_nodes 5 \
      --train_json_path $data_dir/debug_train.json \
      --val_json_path $data_dir/val.json \
      --preprocessed_data_dir $data_dir/npz_files \
      --model_weights_path $af_params_dir/params_model_${model_version}_multimer_v2.npz \
      --model_checkpoint_path $output_dir/checkpoints \
      --output_data_path $output_dir \
      --wandb_offline \
      --wandb_output_dir $output_dir \
      --wandb_project $project \
      --wandb_name $run_name \
      --wandb_id $run_name \
      --test_mode_name "val" \
      --max_epochs 50 \
      --step 'predict'

#      --preprocessed_data_dir /gpfs/alpine/bip215/proj-shared/eglukhov/new_residue/output/get_data_5/val/structures/npz_files \

#      --resume_from_ckpt $project_dir/output/$run_name/$restore_version/checkpoints/best.ckpt \
#      --resume_from_ckpt $project_dir/output/$run_name/$restore_version/checkpoints/last.ckpt \


#wandb sync --append $output_dir/wandb/latest-run/