#!/bin/bash

# Set the environment variable for CUDA devices
export WANDB_PROJECT="..."

# Define base directory
BASE_DIR="..."

# Loop over different cases #e.g. 800 0.67,0.045,0.045,0.045,0.15,0.02,0.025
for case in "..."
do
    IFS=' ' read -r -a array <<< "$case"
    N="${array[0]}"
    MULTIPLIERS="${array[1]}"

    # Define training and validation sizes
    N_TRAIN=$((N * 1000 * 7))
    N_VAL=70000
    BATCH_SIZE=1

    BASE_TRAIN_PATH="$BASE_DIR/train_reweight_${N}_size_..._weight.txt"
    BASE_VAL_PATH="$BASE_DIR/val_reweight_${N}_size_..._weight.txt"

    python $BASE_DIR/prepare_data.py --n_train $N_TRAIN --n_val $N_VAL --train_file_path $BASE_TRAIN_PATH --val_file_path $BASE_VAL_PATH --multipliers $MULTIPLIERS

    # Run training model with base data
    torchrun --nproc_per_node=8 $BASE_DIR/run_clm.py \
        --model_type gpt2 \
        --config_overrides "n_embd=1280,n_layer=36,n_head=20" \
        --tokenizer_name gpt2 \
        --train_file $BASE_TRAIN_PATH \
        --validation_file $BASE_VAL_PATH \
        --output_dir $BASE_DIR/training_gpt2_large/reweight_${N}_size_..._weight \
        --overwrite_output_dir \
        --num_train_epochs 3 \
        --learning_rate 2e-4 \
        --logging_first_step \
        --logging_steps 100 \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps 10 \
        --save_steps 2400 \
        --seed 42 \
        --do_train \
        --do_eval \
        --fp16 \
        --report_to wandb \
        --run_name "reweight_${N}_size_..._weight" \
        --warmup_ratio 0.1 \
        --weight_decay 0.01
done
