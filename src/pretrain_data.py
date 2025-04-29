# filepath: /path/to/your/pretraining_script.py
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
import random

# --- T5 Span Corruption (Simplified Example) ---
# See Hugging Face examples for more robust implementations
def corrupt_text(text, noise_density=0.15, mean_noise_span_length=3.0):
    tokens = text.split() # Simple whitespace split for demo
    num_tokens = len(tokens)
    num_to_corrupt = int(round(num_tokens * noise_density))
    num_spans = int(round(num_to_corrupt / mean_noise_span_length))

    if num_spans == 0:
        return text, text # No corruption

    indices_to_corrupt = sorted(random.sample(range(num_tokens), num_to_corrupt))

    corrupted_input = []
    target_output = []
    extra_id_counter = 0
    last_corrupted_idx = -2

    i = 0
    while i < num_tokens:
        if i in indices_to_corrupt:
            start = i
            while i + 1 in indices_to_corrupt:
                i += 1
            end = i

            # Add sentinel to input if not consecutive corruption
            if start > last_corrupted_idx + 1:
                 corrupted_input.append(f"<extra_id_{extra_id_counter}>")

            # Add sentinel and corrupted span to target
            target_output.append(f"<extra_id_{extra_id_counter}>")
            target_output.extend(tokens[start : end + 1])

            extra_id_counter += 1
            last_corrupted_idx = end
        else:
            corrupted_input.append(tokens[i])
        i += 1

    # Add final sentinel to target if needed
    if target_output:
         target_output.append(f"<extra_id_{extra_id_counter}>")

    return " ".join(corrupted_input), " ".join(target_output)
# --- End Span Corruption Example ---


def preprocess_function(examples):
    inputs = []
    targets = []
    for text in examples["text"]: # Assuming your dataset has a "text" column
        input_text, target_text = corrupt_text(text)
        inputs.append(input_text)
        targets.append(target_text)

    # Tokenize
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
    labels = tokenizer(text_target=targets, max_length=512, padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Load dataset (replace with your actual dataset)
# raw_datasets = load_dataset("mc4", "en", split="train[:1%]") # Example: small subset of mc4
# tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets.column_names)

# Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=tokenizer.pad_token_id,
    pad_to_multiple_of=8 # Optional: for efficiency
)