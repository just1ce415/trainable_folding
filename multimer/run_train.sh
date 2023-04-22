python -m preprocess \
        --json_data_path /storage/erglukhov/fragments/0000035/renumbered_summary.json \
        --pre_alignment_path /storage/thu/frag_MSAs \
        --preprocessed_data_dir /storage/erglukhov/fragments/0000035/v1 \
        --mmcif_dir /storage/thu/fragment_cifs \
        --new_res_a3m_path /storage/erglukhov/new_residue/new_residue.a3m \
        --verification_sdf /storage/erglukhov/fragments/0000035/renumbered_sdf_files/4k0t_ligand_D9Z_A_303_fragment_0000035_1.sdf \
        --n_jobs 12