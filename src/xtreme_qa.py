import os
import json
import torch
import argparse
import re
import string
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from functools import partial
from collections import Counter

from utils import normalize_text
from utils_modeling import get_model_tokenizer

TASK_LANGUAGES = ['ar', 'bn', 'en', 'fi', 'id', 'ko', 'ru', 'sw', 'te']


def parse_data_example(example):
    text = ' '.join([example['context'], example['question']])
    target = example['target']

    return {"text": normalize_text(text), "target": normalize_text(target)}


def preprocess_function(examples, tokenizer, max_length=1024):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(examples["text"], padding="longest", max_length=max_length, truncation=True, return_tensors="pt").to(device)
    targets = tokenizer(examples["target"], padding="longest", max_length=max_length, truncation=True, return_tensors="pt").to(device)

    model_inputs = inputs
    model_inputs["labels"] = targets["input_ids"]

    return model_inputs


def get_dataset(lang, dataset_dir, task, tokenizer, sample_size=100, split='test'):
    data_file = os.path.join(dataset_dir, task, split, f"{lang}.jsonl")
    with open(data_file, 'r') as f:
        examples = [parse_data_example(json.loads(line)) for line in f.readlines()]

    dataset = Dataset.from_list(examples)
    dataset = dataset.shuffle(seed=42).select(range(sample_size))
    dataset = dataset.map(partial(preprocess_function, tokenizer=tokenizer), desc="Running tokenizer", batched=True, batch_size=32)

    return DataLoader(dataset, batch_size=1)


def reconstruct(inp, tokenizer, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = []
    tokenized = tokenizer(inp, padding=True, return_tensors="pt").to(device)
    out = model.generate(**tokenized, max_length=300)
    out = out.cpu().numpy().tolist()
    for seq in out:
        seq = [i for i in seq if i != 0 and i != 1]
        result.append(normalize_text(tokenizer.decode(seq)))

    return result


def _normalize_answer(text, punc_chars, punc_repl):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(s):
    return re.sub(r"\b(a|an|the)\b", " ", s)

  def replace_punctuation(s):
    to_replace = set(punc_chars)
    return "".join(punc_repl if ch in to_replace else ch for ch in s)

  def white_space_fix(s):
    return " ".join(s.split())

  text = text.lower()
  text = replace_punctuation(text)
  text = remove_articles(text)
  text = white_space_fix(text)

  return text


def normalize_squad(answer):
  """Normalization used in official SQuAD evaluation script."""
  return _normalize_answer(answer, punc_chars=string.punctuation, punc_repl="")


def f1_score(target, prediction):
  """Computes token f1 score for a single target and prediction."""
  prediction_tokens = prediction.split()
  target_tokens = target.split()
  common = (Counter(prediction_tokens) &
            Counter(target_tokens))
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(target_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def compute_f1_for_task(model, tokenizer, lang, dataset_dir, task):
    dataset = get_dataset(lang, dataset_dir, task, tokenizer)
    f1_scores = []

    for batch in tqdm(dataset, desc=f"Processing inference for {lang}"):
        input_text = batch["text"][0]  # Extract text from the batch
        target = batch["target"][0]    # Extract target from the batch
        target = normalize_squad(target)

        # Generate predictions
        prediction = reconstruct(input_text, tokenizer, model)
        prediction = normalize_squad(prediction[0])

        # Calculate F1 score for this example
        f1 = f1_score(target, prediction) * 100
        f1_scores.append(f1)

    # Calculate average F1 score across all examples
    avg_f1 = np.mean(f1_scores)
    return avg_f1


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_dir", required=True, type=str)
    argparser.add_argument("--dataset_dir", required=False, default="../xtreme_up_v1.1", type=str)
    argparser.add_argument("--results_dir", required=False, default="../xtreme_up_results", type=str)
    argparser.add_argument("--model_type", required=True, type=str)
    argparser.add_argument("--model_size", required=True, type=str)
    argparser.add_argument("--model_steps", required=False, type=int, default=250000)

    args = argparser.parse_args()

    task = "qa_in_lang"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, tokenizer = get_model_tokenizer(args.model_type, args.model_size, args.model_steps, args.model_dir, device=device)

    f1_scores = {}

    for lang in TASK_LANGUAGES:
        print(f"Processing inference for {lang}")
        score = compute_f1_for_task(model, tokenizer, lang, args.dataset_dir, task=task)
        f1_scores[lang] = score
        print(f"F1 Score for {lang}: {score:.4f}")

    # Save results
    results_file = os.path.join(args.results_dir, f"{args.model_type}_{args.model_size}_{task}_qa.json")
    with open(results_file, 'w') as f:
        json.dump(f1_scores, f, indent=4)

    print(f"F1 scores saved to {results_file}")