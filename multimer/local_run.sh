#python preprocess.py \
#          --pre_alignment_path /storage/erglukhov/antibodies/test_data/antibody_MSAs/ \
#          --json_data_path /storage/erglukhov/antibodies/trainable_folding/multimer/ab_loop_multimer_test_2.json \
#          --preprocessed_data_dir /storage/erglukhov/antibodies/test_data/npz_data \
#          --mmcif_dir /pool-data/data/thu/mmcif_files/ \
#          --n_jobs 2

run_name="val";
python train_multi_gpu.py \
          --output_dir ./output/ \
          --gpus 4 \
          --replace_sampler_ddp=True \
          --deepspeed_config_path deepspeed_config.json \
          --wandb_id $run_name \
          --wandb_name $run_name \
          --trainer_dir_path /home/eglukhov/antibodies/test_data/ \
          --model_checkpoint_path /home/eglukhov/antibodies/test_data/checkpoints/$run_name \
          --json_data_path /home/eglukhov/antibodies/trainable_folding/multimer/ab_loop_multimer_test.json \
          --preprocessed_data_dir /home/eglukhov/antibodies/npz_data \
          --model_weights_path /data1/thunguyen/params/params_model_1_multimer_v2.npz \
          --n_layers_in_lr_group 10 \
          --accum_grad_batches 16
