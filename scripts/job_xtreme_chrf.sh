#! /bin/bash

MODEL_DIR="../models"
MODEL_TYPE="myt5"
MODEL_SIZE="large-mt-v8.1c"

CUDA_VISIBLE_DEVICES=0 python3.10 ../src/xtreme_chrf.py --model_dir $MODEL_DIR \
                                 --model_type $MODEL_TYPE \
                                 --model_size $MODEL_SIZE
