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

model_version=5;

#for run in {10..20}; do
#  echo "Starting training run: ${run}"
#  project_dir="/projectnb2/sc3dm/eglukhov/phospho/";
#  data_dir=$project_dir/datasets/v2;
#  project="phospho";
#  run_name="search_${model_version}_${run}";
#  output_dir=$project_dir/output/$run_name/;
#  mkdir -p $output_dir;
#  python train_multi_gpu.py \
#            --gpus 10 \
#            --num_nodes 1 \
#            --wandb_output_dir $output_dir \
#            --wandb_project $project \
#            --wandb_name $run_name \
#            --wandb_id $run_name \
#            --model_weights_path /projectnb2/sc3dm/eglukhov/af_params/params_model_${model_version}_multimer_v2.npz \
#            --model_checkpoint_path $output_dir/checkpoints \
#            --preprocessed_data_dir $data_dir/npz_data \
#            --output_data_path $output_dir/data \
#            --train_json_path $data_dir/train.json \
#            --val_json_path $data_dir/val.json \
#            --test_mode_name "val_metrics" \
#            --max_epochs 10 \
#            --hyperparams_seed $run \
#            --step 'search'
#  sleep 1
#done
#
#echo "All jobs finished."


run='baseline';
project_dir="/projectnb2/sc3dm/eglukhov/phospho/";
data_dir=$project_dir/datasets/v2;
project="phospho";
run_name="${model_version}_${run}";
output_dir=$project_dir/output/$run_name/;
mkdir -p $output_dir;
python train_multi_gpu.py \
          --gpus 10 \
          --num_nodes 1 \
          --wandb_output_dir $output_dir \
          --wandb_project $project \
          --wandb_name $run_name \
          --wandb_id $run_name \
          --model_weights_path /projectnb2/sc3dm/eglukhov/af_params/params_model_${model_version}_multimer_v2.npz \
          --model_checkpoint_path $output_dir/checkpoints \
          --preprocessed_data_dir $data_dir/npz_data \
          --output_data_path $output_dir/data \
          --train_json_path $data_dir/train.json \
          --val_json_path $data_dir/val.json \
          --test_mode_name "baseline" \
          --max_epochs 10 \
          --hyperparams_seed 0 \
          --step 'test'