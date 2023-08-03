for model_version in {1..5}; do
  echo "Starting training model version: $model_version"
  run=0;
  project_dir="/home/kikodze/projects/phospho/";
  data_dir=$project_dir/datasets/confidence/v1/model_${model_version};
  project="phospho_train_conf";
  run_name="train_conf_${model_version}_${run}";
  output_dir=$project_dir/output/$run_name/;

  mkdir -p $output_dir;
  python train_confidence.py \
          --gpus 1 \
          --num_nodes 1 \
          --wandb_output_dir $output_dir \
          --wandb_project $project \
          --wandb_name ${run_name} \
          --model_checkpoint_path $output_dir/checkpoints \
          --train_folder_path $data_dir/train \
          --val_folder_path $data_dir/val \
          --max_epochs 20 \
          --learning_rate 0.0001 \
          --num_layers 0 \
          --accumulate_grad_batches 64 \
          --step 'train'
#
  python train_confidence.py \
          --init_model_weights_path /home/kikodze/projects/phospho/output/step_one/model_states_${model_version}_clean.pt \
          --plddt_model_weights_path $output_dir/checkpoints/plddt_weights.pt \
          --new_model_weights_path /home/kikodze/projects/phospho/output/step_one/model_states_${model_version}_plddt.pt \
          --step 'update_model'

done;
