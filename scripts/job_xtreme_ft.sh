#! /bin/bash

MODEL_DIR="../models"
MODEL_TYPE="myt5"
MODEL_SIZE="large"
MODEL_NAME="mt-v8.1c"
MODEL_STEPS=250000

CUDA_VISIBLE_DEVICES=3 python3.10 ../src/xtreme_ft.py \
    --model_dir $MODEL_DIR \
    --task "translation" \
    --model_type $MODEL_TYPE \
    --model_size $MODEL_SIZE \
    --model_name $MODEL_NAME \
    --model_steps $MODEL_STEPS