python -m preprocess \
        --json_data_path /storage/erglukhov/new_residue/human/data.json \
        --pre_alignment_path /storage/erglukhov/new_residue/human/a3m \
        --preprocessed_data_dir /storage/erglukhov/new_residue/human/npz_files \
        --mmcif_dir /storage/thu/fragment_cifs \
        --new_res_a3m_path /storage/erglukhov/new_residue/new_residue.a3m \
        --mode "inference" \
        --n_jobs 12

#model_version='1';
#af_params_dir="/data/thu/colabfold_batch_multiseed/colabfold_batch/colabfold/params/";
#project_dir="/storage/erglukhov/new_residue/human/";
#data_dir="/storage/erglukhov/new_residue/human/";
#project="human";
#run_name="check";
#run_version='2'
#restore_version='1'
#output_dir=$project_dir/output/$run_name/$run_version;
#
#python3 -m train_feature_process \
#      --gpus 1 \
#      --num_nodes 1 \
#      --test_json_path /storage/erglukhov/new_residue/human/data.json \
#      --preprocessed_data_dir $data_dir/npz_files \
#      --model_weights_path $af_params_dir/params_model_${model_version}_multimer_v2.npz \
#      --model_checkpoint_path $output_dir/checkpoints \
#      --output_data_path $output_dir \
#      --wandb_offline \
#      --wandb_output_dir $output_dir \
#      --wandb_project $project \
#      --wandb_name $run_name \
#      --wandb_id $run_name \
#      --test_mode_name "test" \
#      --max_epochs 1 \
#      --step 'predict'