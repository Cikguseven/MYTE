from transformers import T5Config, T5ForConditionalGeneration

# Load a pretrained model and resize embeddings
model = T5ForConditionalGeneration.from_pretrained("google/mt5-small") # Or "t5-small"
model.resize_token_embeddings(len(tokenizer)) # Resize to match tokenizer

print(f"Model Embedding Size: {model.get_input_embeddings().weight.shape[0]}")