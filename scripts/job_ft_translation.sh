#! /bin/bash

MODEL_TYPE="myt5"
MODEL_SIZE="large"
MODEL_NAME="mt-v8"
MODEL_STEPS=250000

CUDA_VISIBLE_DEVICES=0,1,2,3 python3.10 ../src/ft_generative_task.py \
    --checkpoint_dir "../models" \
    --task "translation" \
    --directory "../flores200_dataset" \
    --model_type $MODEL_TYPE \
    --model_size $MODEL_SIZE \
    --model_name $MODEL_NAME \
    --model_steps $MODEL_STEPS