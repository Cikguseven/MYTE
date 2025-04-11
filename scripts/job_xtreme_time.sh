#! /bin/bash

MODEL_TYPE=$1
MODEL_SIZE=$2

MODEL_STEPS=250000

CUDA_VISIBLE_DEVICES=0,1,2,3 \

# for MODEL_TYPE in myt5 byt5
for MODEL_TYPE in myt5
do
  # for task in qa_in_lang translation semantic_parsing ner
  for task in translation
  do
    python3.10 ../src/xtreme_time.py --checkpoint_dir "../models" \
                                     --task $task \
                                     --model_type $MODEL_TYPE \
                                     --model_size $MODEL_SIZE

  done
done