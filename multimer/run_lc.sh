export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

model_version=1;
run=1;
project_dir="/titan/eric/projects/mhc/";
data_dir=$project_dir/datasets/v1;
project="test";
run_name="wo_linear_${model_version}_${run}";
output_dir=$project_dir/output/$run_name/;
python train_multi_gpu.py \
    --gpus 8 \
    --num_nodes 1 \
    --wandb_output_dir $output_dir \
    --wandb_project $project \
    --wandb_name $run_name \
    --wandb_id $run_name \
    --wandb_offline \
    --model_weights_path /titan/eric/projects/af_params/params_model_${model_version}_multimer_v2.npz \
    --model_checkpoint_path $output_dir/checkpoints \
    --preprocessed_data_dir $project_dir/datasets/v1/npz_data \
    --output_data_path $output_dir/data \
    --train_json_path $data_dir/train.json \
    --val_json_path $data_dir/val.json \
    --test_mode_name "val_metrics" \
    --crop_size 384 \
    --max_epochs 5 \
    --hyperparams_seed $run \
    --step 'search'