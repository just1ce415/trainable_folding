python preprocess.py \
          --pre_alignment_path /storage/erglukhov/antibodies/test_data/antibody_MSAs/ \
          --json_data_path /storage/erglukhov/antibodies/trainable_folding/multimer/ab_loop_multimer_test_2.json \
          --preprocessed_data_dir /storage/erglukhov/antibodies/test_data/npz_data \
          --mmcif_dir /pool-data/data/thu/mmcif_files/ \
          --n_jobs 2
#
python train_multi_gpu.py \
          --output_dir ./output/ \
          --gpus 1 \
          --replace_sampler_ddp=True \
          --deepspeed_config_path deepspeed_config.json \
          --wandb_id test_10 \
          --trainer_dir_path /storage/erglukhov/antibodies/test_data/ \
          --model_checkpoint_path /storage/erglukhov/antibodies/test_data/checkpoints \
          --json_data_path /storage/erglukhov/antibodies/trainable_folding/multimer/ab_loop_multimer_test.json \
          --preprocessed_data_dir /storage/erglukhov/antibodies/test_data/npz_data \
          --model_weights_path /storage/erglukhov/antibodies/test_dataparams/params_model_1_multimer_v2.npz