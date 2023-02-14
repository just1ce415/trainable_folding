#python preprocess.py \
#          --pre_alignment_path /storage/erglukhov/antibodies/antibody_MSAs/ \
#          --json_data_path /storage/erglukhov/antibodies/datasets/v3/all.json \
#          --preprocessed_data_dir /storage/erglukhov/antibodies/datasets/v3/ \
#          --mmcif_dir /pool-data/data/thu/mmcif_files/ \
#          --n_jobs 12


run_name="test";
python train_multi_gpu.py \
          --output_dir ./output/ \
          --replace_sampler_ddp=True \
          --deepspeed_config_path deepspeed_config.json \
          --wandb_id $run_name \
          --wandb_name $run_name \
          --project "antibodies_test" \
          --test_table_name "val_metrics" \
          --trainer_dir_path /home/eglukhov/antigen/train_data/ \
          --model_checkpoint_path /home/eglukhov/antigen/train_data/checkpoints/$run_name \
          --train_json_path /storage/erglukhov/antibodies/datasets/v3/train.json \
          --val_json_path /storage/erglukhov/antibodies/datasets/v3/val.json \
          --preprocessed_data_dir /storage/erglukhov/antibodies/datasets/v3/npz_data \
          --model_weights_path /storage/erglukhov/antibodies/test_dataparams/params_model_1_multimer_v2.npz \
          --accum_grad_batches 16 \
          --n_layers_in_lr_group 10 \
          --gpus 1 \
          --crop_size 384
