python -m preprocess \
        --json_data_path /storage/erglukhov/new_residue/test_code/debug_val.json \
        --pre_alignment_path /storage/thu/missing_frag_rec_MSAs \
        --preprocessed_data_dir /storage/erglukhov/new_residue/test_code/new_code \
        --mmcif_dir /storage/thu/fragment_cifs \
        --new_res_a3m_path /storage/erglukhov/new_residue/new_residue.a3m \
        --verification_sdf /storage/erglukhov/new_residue/fragment_database/v4_database/carve_fragments_v3/fragment_db/7epe/7epe_ligand_FMN_A_1201_fragment_0000003_0.sdf \
        --test_mode \
        --n_jobs 1

#
#run_name="new_metrics";
#python -m train_feature_process \
#      --json_path /projectnb2/sc3dm/eglukhov/new_residue/new_res_train_3_20.json \
#      --val_json_path /projectnb2/sc3dm/eglukhov/new_residue/new_res_val_3_20.json \
#      --preprocessed_data_dir /projectnb2/sc3dm/eglukhov/new_residue/npz_files \
#      --model_weights_path /projectnb2/sc3dm/eglukhov/new_residue/params_model_1_multimer_v2.npz \
#      --model_checkpoint_path /projectnb2/sc3dm/eglukhov/new_residue/checkpoints/$run_name\
#      --wandb_logger_dir /projectnb2/sc3dm/eglukhov/new_residue/wandb_logs \
#      --wandb_name $run_name \
#      --wandb_id $run_name \
#      --accumulate_grad_batches 8