python preprocess.py \
          --pre_alignment_path /home/kikodze/projects/phospho/datasets/msa/ \
          --json_data_path /home/kikodze/projects/phospho/datasets/v2/all_samples.json \
          --preprocessed_data_dir /home/kikodze/projects/phospho/datasets/v2 \
          --mmcif_dir /home/kikodze/projects/mmcif_files/ \
          --n_jobs 32


#export CUDA_VISIBLE_DEVICES=0
#
#run=1;
#model_version=5;
#
#project_dir="/home/eglukhov/projects/mhc/";
#data_dir=$project_dir/datasets/v1;
#project="mhc_test";
#run_name="init_5_${run}_${model_version}";
#output_dir=$project_dir/output/$run_name/;
#mkdir -p $output_dir;
#python train_multi_gpu.py \
#    --gpus 1 \
#    --num_nodes 1 \
#    --wandb_output_dir $output_dir \
#    --wandb_project $project \
#    --wandb_name $run_name \
#    --wandb_id $run_name \
#    --model_weights_path /home/eglukhov/projects/af_params/params_model_${model_version}_multimer_v2.npz \
#    --model_checkpoint_path $output_dir/checkpoints \
#    --preprocessed_data_dir $data_dir/npz_data \
#    --output_data_path $output_dir \
#    --train_json_path $data_dir/train.json \
#    --val_json_path $data_dir/val.json \
#    --test_mode_name "val_metrics" \
#    --n_layers_in_lr_group 10 \
#    --crop_size 384 \
#    --max_epochs 10 \
#    --hyperparams_seed $run \
#    --learning_rate 0.001 \
#    --accumulate_grad_batches 1 \
#    --step 'train'


#for model_version in {1..5}; do
#  echo "Starting training model version: $model_version"
#  run=18;
#  project_dir="/projectnb2/sc3dm/eglukhov/compress/";
#  data_dir=$project_dir/datasets/v4;
#  project="compress_final";
#  run_name="init_5_${run}_${model_version}";
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
#            --preprocessed_data_dir $project_dir/datasets/v4/npz_data \
#            --output_data_path $output_dir/val_init_5 \
#            --train_json_path $data_dir/train.json \
#            --val_json_path $data_dir/val.json \
#            --test_mode_name "val_metrics" \
#            --n_layers_in_lr_group 10 \
#            --crop_size 384 \
#            --max_epochs 10 \
#            --hyperparams_seed $run \
#            --learning_rate 0.001 \
#            --accumulate_grad_batches 1 \
#            --step 'test'
#  sleep 1
#
#done
#
#echo "All jobs submitted."
