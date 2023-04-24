#python preprocess.py \
#          --pre_alignment_path /storage/erglukhov/antibodies/antibody_MSAs/ \
#          --json_data_path /storage/erglukhov/antibodies/datasets/v3/all.json \
#          --preprocessed_data_dir /storage/erglukhov/antibodies/datasets/v3/ \
#          --mmcif_dir /pool-data/data/thu/mmcif_files/ \
#          --n_jobs 12


module load miniconda/4.12.0
eval "$(conda shell.bash hook)"
module load cuda/11.6 gcc
export NCCL_P2P_LEVEL=LOC
conda activate folding

run=18;
project_dir="/projectnb2/sc3dm/eglukhov/compress/";
data_dir=$project_dir/datasets/v4;
project="compress";
run_name="${run}_continue";
output_dir=$project_dir/output/$run_name/;
mkdir -p $output_dir;
python train_multi_gpu.py \
          --gpus 10 \
          --num_nodes 1 \
          --wandb_output_dir $output_dir \
          --wandb_project $project \
          --wandb_name $run_name \
          --wandb_id $run_name \
          --model_weights_path /projectnb2/sc3dm/eglukhov/new_residue/params_model_1_multimer_v2.npz \
          --model_checkpoint_path $output_dir/checkpoints \
          --resume_from_ckpt /projectnb2/sc3dm/eglukhov/compress/output/18/checkpoints/step=1679-val_structure_loop_loss=1.37.ckpt \
          --preprocessed_data_dir $project_dir/datasets/v4/npz_data \
          --output_data_path $output_dir/data \
          --train_json_path $data_dir/train.json \
          --val_json_path $data_dir/val.json \
          --test_mode_name "val_metrics" \
          --n_layers_in_lr_group 10 \
          --crop_size 384 \
          --max_epochs 50 \
          --hyperparams_seed $run \
          --step 'search'