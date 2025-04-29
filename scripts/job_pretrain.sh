#!/bin/bash

# Define paths and parameters for BPE pretraining
DECOMPOSE_MAP="../byte_maps/decompose_map.json"
MERGE_MAP="../byte_maps/merge_map.json"
DOWNLOAD_ARTICLES=FALSE # Set to TRUE if you want to download articles
MAX_ARTICLES=10000 # Adjust as needed
VOCAB_SIZE=16000 # Adjust as needed
CORPUS_BYTE_OUTPUT_DIR="../bpe/corpus"  # Output path for the BPE corpus
BPE_TOKENIZER_OUTPUT_DIR="../src/myt5/${VOCAB_SIZE}.json" # Output path for the trained tokenizer

# Define log file
TIMESTAMP="$(date +%b%d_%H:%M)"
LOG_FILE="../logs/BPE_PRETRAIN_${TIMESTAMP}_${VOCAB_SIZE}.log"

# Create directories if they don't exist
mkdir -p ../logs
mkdir -p "$(dirname "$BPE_TOKENIZER_OUTPUT_DIR")"
mkdir -p "$CORPUS_BYTE_OUTPUT_DIR"

# Run the pretrain_BPE.py script using nohup for background execution
nohup python3.10 ../src/pretrain_BPE.py \
    --decompose_map "$DECOMPOSE_MAP" \
    --merge_map "$MERGE_MAP" \
    --download_articles $DOWNLOAD_ARTICLES \
    --max_articles $MAX_ARTICLES \
    --vocab_size $VOCAB_SIZE \
    --corpus_byte_output_dir "$CORPUS_BYTE_OUTPUT_DIR" \
    --bpe_tokenizer_output_dir "$BPE_TOKENIZER_OUTPUT_DIR" \
    > "$LOG_FILE" 2>&1 &

echo "BPE pretraining script started in the background."
echo "Output and errors will be logged to: $LOG_FILE"
echo "Trained tokenizer will be saved to: $BPE_TOKENIZER_OUTPUT_DIR"