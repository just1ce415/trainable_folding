python -m preprocess \
        --json_data_path /storage/erglukhov/new_residue/new_res_train2.json \
        --pre_alignment_path /storage/thu/missing_frag_rec_MSAs \
        --preprocessed_data_dir /storage/erglukhov/new_residue/npz_files \
        --mmcif_dir /storage/thu/fragment_cifs \
        --new_res_a3m_path /storage/erglukhov/new_residue/new_residue.a3m \


#python -m train_feature_process \
#      --json-path /projectnb2/sc3dm/eglukhov/new_residue/new_res_train_3.json \
#      --preprocessed_data_dir /projectnb2/sc3dm/eglukhov/new_residue/npz_files \
#      --model_weights_path /projectnb2/sc3dm/eglukhov/new_residue//params_model_1_multimer_v2.npz
