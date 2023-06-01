#python preprocess.py \
#          --pre_alignment_path /home/kikodze/projects/tmp/mhc_msa/ \
#          --json_data_path /home/kikodze/projects/mhc/summary.json \
#          --preprocessed_data_dir /home/kikodze/projects/mhc/datasets/v1/ \
#          --mmcif_dir /home/kikodze/projects/mmcif_files/ \
#          --n_jobs 32

# export CUDA_VISIBLE_DEVICES=0,1

export CUDA_VISIBLE_DEVICES=1

# model_version=5;

NOW=$(date +"%Y%m%d%H%M$S");

project_dir="/home/yimzhu/workspace/mhc/";
data_dir="/home/eglukhov/projects/mhc/datasets/v1";
project="mhc_inference";
# output_dir=$project_dir/output/$run_name/;

############
# Search #
############
# val_ds="val";
# # run=1;
# model_version=1;
# for run in {1..10}; do
# # for model_version in {1}; do
#  echo "Starting run: $run"
#  echo "Starting model version: $model_version"
#  project_dir="/home/yimzhu/workspace/mhc/";
#  data_dir="/home/eglukhov/projects/mhc/datasets/v1";
#  project="mhc_search";
#  run_name="search_${run}_${model_version}";
#  output_dir=$project_dir/output/$run_name/;
#  mkdir -p $output_dir;
#  python train_multi_gpu.py \
#     --gpus 3 \
#     --num_nodes 1 \
#     --wandb_output_dir $output_dir \
#     --wandb_project $project \
#     --wandb_name $run_name \
#     --wandb_id ${run_name}_${NOW} \
#     --model_weights_path /home/eglukhov/projects/af_params/params_model_${model_version}_multimer_v2.npz \
#     --model_checkpoint_path $output_dir/checkpoints \
#     --preprocessed_data_dir $data_dir/npz_data \
#     --output_data_path $output_dir \
#     --train_json_path $data_dir/train.json \
#     --val_json_path $data_dir/${val_ds}.json \
#     --test_mode_name "val_metrics" \
#     --n_layers_in_lr_group 10 \
#     --crop_size 384 \
#     --max_epochs 5 \
#     --hyperparams_seed $run \
#     --learning_rate 0.001 \
#     --accumulate_grad_batches 1 \
#     --step 'search'
#     # --wandb_offline
#  sleep 1
# # done
# done
#

############
# Best #
############
run_name="best_${model_version}";
output_dir=$project_dir/output/${run_name}_${NOW}/;
mkdir -p $output_dir;
python train_multi_gpu.py \
    --gpus 1 \
    --num_nodes 1 \
    --wandb_output_dir $output_dir \
    --wandb_project $project \
    --wandb_name $run_name \
    --wandb_id ${run_name}_${NOW} \
    --model_weights_path /home/eglukhov/projects/af_params/params_model_${model_version}_multimer_v2.npz \
    --model_checkpoint_path $output_dir/checkpoints \
    --preprocessed_data_dir $data_dir/npz_data \
    --output_data_path $output_dir \
    --train_json_path $data_dir/train.json \
    --val_json_path $data_dir/val.json \
    --test_mode_name "val_metrics" \
    --n_layers_in_lr_group 10 \
    --crop_size 384 \
    --max_epochs 10 \
    --step 'test' \
    --resume_from_ckpt /home/yimzhu/workspace/mhc//output/search_9_1//checkpoints/best.ckpt \
    --accumulate_grad_batches 6 \
    --evoformer_num_block 3 \
    --learning_rate 0.00001 \
    --hyperparams_seed 1 
    # --wandb_offline
    sleep 1

############
# Baseline #
############
# for model_version in {1..5}; do
# run_name="init_5_${model_version}";
# output_dir=$project_dir/output/${run_name}_${NOW}/;
# mkdir -p $output_dir;
# python train_multi_gpu.py \
#     --gpus 1 \
#     --num_nodes 1 \
#     --wandb_output_dir $output_dir \
#     --wandb_project $project \
#     --wandb_name $run_name \
#     --wandb_id ${run_name}_${NOW} \
#     --model_weights_path /home/eglukhov/projects/af_params/params_model_${model_version}_multimer_v2.npz \
#     --model_checkpoint_path $output_dir/checkpoints \
#     --preprocessed_data_dir $data_dir/npz_data \
#     --output_data_path $output_dir \
#     --train_json_path $data_dir/train.json \
#     --val_json_path $data_dir/val.json \
#     --test_mode_name "val_metrics" \
#     --n_layers_in_lr_group 10 \
#     --crop_size 384 \
#     --max_epochs 10 \
#     --step 'test' \
#     --resume_from_ckpt /home/yimzhu/workspace/mhc//output/search_9_1//checkpoints/best.ckpt \
#     --accumulate_grad_batches 6 \
#     --evoformer_num_block 3 \
#     --learning_rate 0.00001 \
#     --hyperparams_seed 1 \
#     --wandb_offline
#     sleep 1
# done

echo "All jobs submitted."
