#python -m preprocess \
#        --json_data_path /storage/erglukhov/new_residue/test_code/debug_inference.json \
#        --pre_alignment_path /storage/thu/missing_frag_rec_MSAs \
#        --preprocessed_data_dir /storage/erglukhov/new_residue/test_code/new_version \
#        --mmcif_dir /storage/thu/fragment_cifs \
#        --new_res_a3m_path /storage/erglukhov/new_residue/new_residue.a3m \
#        --verification_sdf /storage/erglukhov/new_residue/fragment_database/v4_database/carve_fragments_v3/fragment_db/7epe/7epe_ligand_FMN_A_1201_fragment_0000003_0.sdf \
#        --test_mode \
#        --n_jobs 1

model_version='1';
af_params_dir="/data/thu/colabfold_batch_multiseed/colabfold_batch/colabfold/params/";
project_dir="/storage/erglukhov/new_residue/test_code/";
data_dir=$project_dir/new_version;
project="new_residue_test";
run_name="check";
run_version='2'
restore_version='1'
output_dir=$project_dir/output/$run_name/$run_version;

python3 -m train_feature_process \
      --gpus 1 \
      --num_nodes 1 \
      --train_json_path $data_dir/debug_train.json \
      --val_json_path $data_dir/debug_inference.json \
      --preprocessed_data_dir $data_dir/npz_files \
      --model_weights_path $af_params_dir/params_model_${model_version}_multimer_v2.npz \
      --resume_from_ckpt $project_dir/output/$run_name/$restore_version/checkpoints/last.ckpt \
      --model_checkpoint_path $output_dir/checkpoints \
      --output_data_path $output_dir/structures \
      --wandb_offline \
      --wandb_output_dir $output_dir \
      --wandb_project $project \
      --wandb_name $run_name \
      --wandb_id $run_name \
      --test_mode_name "val" \
      --max_epochs 1 \
      --step 'predict'