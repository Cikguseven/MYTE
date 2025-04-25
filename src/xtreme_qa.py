from itertools import islice
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

from utils import normalize_text, parse_data_example, preprocess_function
from utils_modeling import get_model_tokenizer

QA_LANGUAGES = [
  'ar',
  'bn',
  'en',
  'fi',
  'id',
  'ko',
  'ru',
  'sw',
  'te'
  ]


def get_dataset(lang, dataset_dir, task, tokenizer, read_size=225, sample_size=200, split='test'):
    data_file = os.path.join(dataset_dir, task, split, f"{lang}.jsonl")
    with open(data_file, 'r') as f:
        examples = [parse_data_example(task, json.loads(line)) for line in islice(f, read_size)]

    dataset = Dataset.from_list(examples)
    dataset = dataset.shuffle(seed=42).select(range(sample_size))
    dataset = dataset.map(partial(preprocess_function, tokenizer=tokenizer), desc="Running tokenizer", batched=True, batch_size=64)

    return DataLoader(dataset, batch_size=4)


def reconstruct(inp, tokenizer, model):
    result = []
    tokenized = tokenizer(inp, padding=True, return_tensors="pt")

    # Move tokenized inputs to the same device as the model
    device = model.device
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

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
        input_texts = batch["text"]    # List of input texts in the batch
        targets = batch["target"]      # List of targets in the batch

        # Generate predictions for the batch
        predictions = reconstruct(input_texts, tokenizer, model)

        for target, prediction in zip(targets, predictions):
            target = normalize_squad(target)
            prediction = normalize_squad(prediction)
            f1 = f1_score(target, prediction) * 100
            f1_scores.append(f1)

    # Calculate average F1 score across all examples
    avg_f1 = np.mean(f1_scores)
    return avg_f1


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_dir", required=True, type=str)
    argparser.add_argument("--dataset_dir", required=False, default="../xtreme_up_v1.1", type=str)
    argparser.add_argument("--results_dir", required=False, default="../xtreme_up_results/qa", type=str)
    argparser.add_argument("--model_type", required=True, type=str)
    argparser.add_argument("--model_size", required=True, type=str)
    argparser.add_argument("--model_steps", required=False, type=int, default=250000)

    args = argparser.parse_args()

    task = "qa_in_lang"

    model, tokenizer = get_model_tokenizer(args.model_type, args.model_size, args.model_steps, args.model_dir, trained=True)

    f1_scores = {}

    for lang in QA_LANGUAGES:
        print(f"Processing inference for {lang}")
        score = compute_f1_for_task(model, tokenizer, lang, args.dataset_dir, task=task)
        f1_scores[lang] = score
        print(f"F1 Score for {lang}: {score:.4f}")

    # Save results
    results_file = os.path.join(args.results_dir, f"{args.model_type}_{args.model_size}.json")
    with open(results_file, 'w') as f:
        json.dump(f1_scores, f, indent=4)

    print(f"F1 scores saved to {results_file}")