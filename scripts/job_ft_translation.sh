#! /bin/bash

MODEL_TYPE=$1
MODEL_SIZE=$2

MODEL_STEPS=250000

CUDA_VISIBLE_DEVICES=0 \

python3.10 ../src/ft_generative_task.py --checkpoint_dir "../models" \
                                    --task "translation" \
                                    --directory "../flores200_dataset" \
                                    --model_type $MODEL_TYPE \
                                    --model_size $MODEL_SIZE \
                                    --model_steps $MODEL_STEPS