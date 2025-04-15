from collections import defaultdict
import accelerate
import argparse
import csv
import json
import os
import torch
import transformers
from accelerate import Accelerator
from copy import deepcopy
from datasets import Dataset
from functools import partial
from itertools import islice, cycle
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import parse_data_example
from utils_modeling import get_model_tokenizer

N_EVAL_BATCHES = 64

TASK_LANGUAGES = [
    'en2am',
    'en2de',
    'en2el',
    'en2fr',
    'en2hy',
    'en2ja',
    'en2kk',
    'en2ko',
    'en2mt',
    'en2pl',
    'en2ru',
    'en2sn',
    'en2ta',
    'en2te',
    'en2vi'
    ]


def preprocess_function(examples, tokenizer, max_length=300):
    model_inputs = tokenizer(examples["text"], padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    targets = tokenizer(examples["target"], padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")

    model_inputs["labels"] = targets["input_ids"]

    return model_inputs


def get_dataset(directory, tokenizer, task, train_batch_size=4, eval_batch_size=2, map_batch_size=1000):
    train_examples = []
    eval_examples = []

    # Iterate over all languages in TASK_LANGUAGES
    for lang in TASK_LANGUAGES:
        train_file = os.path.join(directory, task, "train", f"{lang}.jsonl")
        eval_file = os.path.join(directory, task, "validation", f"{lang}.jsonl")

        # Load training dataset
        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                train_examples.extend([parse_data_example(json.loads(line)) for line in f.readlines()])

        # Load evaluation dataset
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                eval_examples.extend([parse_data_example(json.loads(line)) for line in f.readlines()])

    # Convert to Hugging Face Dataset format
    train_dataset = Dataset.from_list(train_examples)
    eval_dataset = Dataset.from_list(eval_examples)

    processed_train_dataset = train_dataset.map(
        partial(preprocess_function, tokenizer=tokenizer),
        batched=True,
        batch_size=map_batch_size, # Process in larger chunks for map efficiency
        remove_columns=train_dataset.column_names, # Remove original 'text', 'target' columns
        desc="Running tokenizer on train set"
    )
    processed_eval_dataset = eval_dataset.map(
        partial(preprocess_function, tokenizer=tokenizer),
        batched=True,
        batch_size=map_batch_size, # Process in larger chunks for map efficiency
        remove_columns=eval_dataset.column_names, # Remove original 'text', 'target' columns
        desc="Running tokenizer on eval set"
    )

    # Set format to PyTorch tensors for the DataLoader
    processed_train_dataset.set_format("torch")
    processed_eval_dataset.set_format("torch")

    # Create *single* DataLoaders to batch the processed tensors
    train_loader = DataLoader(
        processed_train_dataset.shuffle(seed=42), # Shuffle the processed dataset
        batch_size=train_batch_size,
        shuffle=True # Shuffle batches each epoch
    )
    eval_loader = DataLoader(
        processed_eval_dataset, # No need to shuffle eval data
        batch_size=eval_batch_size
    )

    return train_loader, eval_loader # Return single loaders


def train_evaluate(model, train_loader, eval_loader, lr=1e-3, n_epochs=30, orig_patience=2):
    patience = orig_patience

    accelerator = Accelerator(gradient_accumulation_steps=16, mixed_precision="bf16")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = transformers.get_inverse_sqrt_schedule(optimizer, num_warmup_steps=100)

    # Prepare everything together
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    prev_loss = float('inf')
    best_model_state = None

    accelerator.print(f"Starting training for {n_epochs} epochs on device: {accelerator.device}")

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        # Use the prepared train_loader directly
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} Training", disable=False)
        for batch in progress_bar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                # Batch should already be a dict of tensors on the correct device
                # No need for isinstance checks or .to(device) if prepare worked correctly
                try:
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs.loss
                except KeyError as e:
                    print(f"Error: Missing key {e} in batch. Batch keys: {batch.keys()}")
                    raise e
                except Exception as e:
                    print(f"Error during model forward pass: {e}")
                    raise e

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.detach().item()
                num_train_batches += 1
                progress_bar.set_postfix(loss=total_train_loss / num_train_batches)

        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0

        # Evaluation
        model.eval()
        total_eval_loss = 0
        num_eval_batches = 0
        with torch.no_grad():
            eval_progress_bar = tqdm(islice(eval_loader, N_EVAL_BATCHES), desc=f"Epoch {epoch+1}/{n_epochs} Evaluation", total=N_EVAL_BATCHES, disable=False)
            for batch in eval_progress_bar:
                try:
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs.loss
                except KeyError as e:
                    print(f"Error: Missing key {e} in eval batch. Batch keys: {batch.keys()}")
                    raise e
                except Exception as e:
                    print(f"Error during eval forward pass: {e}")
                    raise e

                # Gather loss across processes
                total_eval_loss += loss.detach().item()
                num_eval_batches += 1
                eval_progress_bar.set_postfix(loss=total_eval_loss / num_eval_batches)


        avg_eval_loss = total_eval_loss / num_eval_batches if num_eval_batches > 0 else 0.0
        print(f'Epoch {epoch + 1}, Loss train: {avg_train_loss:.4f}, eval: {avg_eval_loss:.4f}')

        # Early stopping logic (on main process)
        if avg_eval_loss < prev_loss:
            print(f"Eval loss decreased ({prev_loss:.4f} --> {avg_eval_loss:.4f}). Saving model state...")
            prev_loss = avg_eval_loss
            patience = orig_patience
            unwrapped_model = accelerator.unwrap_model(model)
            best_model_state = deepcopy(unwrapped_model.state_dict())
        else:
            patience -= 1
            print(f"Eval loss did not decrease. Patience: {patience}/{orig_patience}")

        if patience == 0:
            print('Early stopping triggered.')
            break

    # Load the best model state if early stopping occurred
    if best_model_state is not None:
        print("Loading best model state from early stopping.")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(best_model_state)
        model = unwrapped_model # Use the unwrapped model with the best state

    # Return the final model (best one if early stopping happened)
    # Return the unwrapped model so it can be saved normally
    return accelerator.unwrap_model(model)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_dir", required=True, type=str)
    argparser.add_argument("--task", required=True, type=str)
    argparser.add_argument("--dataset_dir", required=False, default="../xtreme_up_v1.1", type=str)
    argparser.add_argument("--model_type", required=True, type=str)
    argparser.add_argument("--model_size", required=True, type=str)
    argparser.add_argument("--model_name", required=True, type=str)
    argparser.add_argument("--patience", default=2, type=int)
    argparser.add_argument("--lr", default=1e-3, type=float)
    argparser.add_argument("--model_steps", required=False, type=int, default=250000)
    argparser.add_argument("--n_epochs", default=30, type=int)
    argparser.add_argument("--train_batch_size", default=4, type=int, help="Batch size for training")
    argparser.add_argument("--eval_batch_size", default=2, type=int, help="Batch size for evaluation")
    argparser.add_argument("--map_batch_size", default=1000, type=int, help="Batch size for dataset mapping (adjust based on RAM)")

    args = argparser.parse_args()

    accelerator = Accelerator()

    model_save_path = f"{args.model_dir}/{args.model_type}_{args.model_size}-{args.model_name}_{args.model_steps}"

    if os.path.isdir(model_save_path):
         print(f"Fine-tuned model directory exists: {model_save_path}, exiting...")
    else:
        print("Model directory does not exist, starting training...")

        model, tokenizer = get_model_tokenizer(args.model_type, args.model_size, args.model_steps, args.model_dir)

        print("Loading and preprocessing dataset...")
        train_loader, eval_loader = get_dataset(
            args.dataset_dir,
            tokenizer,
            args.task,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            map_batch_size=args.map_batch_size
        )

        print("Starting model training...")
        model = train_evaluate(
            model,
            train_loader,
            eval_loader,
            lr=args.lr,
            n_epochs=args.n_epochs,
            orig_patience=args.patience
        )

        print(f"Saving model to {model_save_path}...")
        model.save_pretrained(model_save_path, use_safetensors=True)
        tokenizer.save_pretrained(model_save_path)
        print(f"Model and tokenizer saved successfully.")
