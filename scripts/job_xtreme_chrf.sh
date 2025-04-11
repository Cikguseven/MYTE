#! /bin/bash

MODEL_TYPE=$1
MODEL_SIZE=$2

MODEL_STEPS=250000

CUDA_VISIBLE_DEVICES=0,1,2,3 \

python3.10 ../src/xtreme_chrf.py --model_dir "../models" \
                                 --task $task \
                                 --model_type $MODEL_TYPE \
                                 --model_size $MODEL_SIZE
