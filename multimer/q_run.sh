#!/bin/bash
#$ -l gpus=1
#$ -l gpu_type=A100
#$ -l h_rt=48:00:00
#$ -j y
#$ -N new_residue

module load miniconda/4.12.0 cuda/11.6 gcc
export NCCL_P2P_LEVEL=LOC
conda activate folding
cd /projectnb2/sc3dm/eglukhov/new_residue/trainable_folding/multimer

run_name="test";
python -m train_feature_process \
      --json_path /projectnb2/sc3dm/eglukhov/new_residue/new_res_train_3_20.json \
      --val_json_path /projectnb2/sc3dm/eglukhov/new_residue/new_res_val_3_20.json \
      --preprocessed_data_dir /projectnb2/sc3dm/eglukhov/new_residue/npz_files \
      --model_weights_path /projectnb2/sc3dm/eglukhov/new_residue/params_model_1_multimer_v2.npz \
      --model_checkpoint_path /projectnb2/sc3dm/eglukhov/new_residue/checkpoints/$run_name\
      --wandb_logger_dir /projectnb2/sc3dm/eglukhov/new_residue/wandb_logs \
      --wandb_name $run_name \
      --wandb_id $run_name \
      --accumulate_grad_batches 8