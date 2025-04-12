import os
import json
import torch
import argparse
import sacrebleu
import unicodedata2
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from functools import partial

from utils import normalize_text
from utils_modeling import get_model_tokenizer

TASK_LANGUAGES = [
    'en2am',
    'en2de',
    'en2el',
    'en2fr',
    'en2hy',
    'en2ja',
    'en2kk',
    'en2ko',
    'en2pl',
    'en2ru',
    'en2sn',
    'en2ta',
    'en2te',
    'en2vi'
    ]


def parse_data_example(example):
    text = example['input']
    target = example['target']

    return {"text": normalize_text(text), "target": normalize_text(target)}


def preprocess_function(examples, tokenizer, max_length=1024):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
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

    return DataLoader(dataset, batch_size=32)

def reconstruct(inp, tokenizer, model):
	device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
	result = []
	tokenized = tokenizer(inp, padding=True, return_tensors="pt").to(device)
	out = model.generate(**tokenized, max_length=300)
	out = out.cpu().numpy().tolist()
	for seq in out:
		seq = [i for i in seq if i != 0 and i != 1]
		result.append(normalize_text(tokenizer.decode(seq)))

	return result


def unicode_normalization(text: str) -> str:
  """Applies Unicode normalization."""
  return unicodedata2.normalize("NFKC", text)


def normalize_targets_predictions(predictions: list[str], targets: list[str]) -> tuple[list[str], list[str]]:
    """Normalize predictions and targets for all tasks."""
    predictions = [unicode_normalization(p) for p in predictions]
    targets = [unicode_normalization(t) for t in targets]
    return predictions, targets


def compute_chrf_for_task(model, tokenizer, lang, dataset_dir, task):
    dataset = get_dataset(lang, dataset_dir, task, tokenizer)
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(dataset, desc=f"Processing inference for {lang}"):
            input_texts = batch["text"]
            target_texts = batch["target"]

            # Generate predictions
            batch_predictions = reconstruct(input_texts, tokenizer, model)

            predictions.extend(batch_predictions)
            targets.extend(target_texts)

    # Compute chrF score
    predictions, targets = normalize_targets_predictions(predictions, targets)
    return sacrebleu.corpus_chrf(predictions, [targets])


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_dir", required=True, type=str)
    argparser.add_argument("--dataset_dir", required=False, default="../xtreme_up_v1.1", type=str)
    argparser.add_argument("--results_dir", required=False, default="../xtreme_up_results", type=str)
    argparser.add_argument("--model_type", required=True, type=str)
    argparser.add_argument("--model_size", required=True, type=str)
    argparser.add_argument("--model_steps", required=False, type=int, default=250000)

    args = argparser.parse_args()

    task = "translation"
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    model, tokenizer = get_model_tokenizer(args.model_type, args.model_size, args.model_steps, args.model_dir, device=device)

    chrf_scores = {}

    for lang in TASK_LANGUAGES:
        print(f"Processing inference for {lang}")
        score = compute_chrf_for_task(model, tokenizer, lang, args.dataset_dir, task=task)
        chrf_scores[lang] = score.score
        print(f"chrF Score for {lang}: {score.score:.4f}")

    # Save results
    results_file = os.path.join(args.results_dir, f"{args.model_type}_{args.model_size}_chrf_scores.json")
    with open(results_file, 'w') as f:
        json.dump(chrf_scores, f, indent=4)

    print(f"chrF scores saved to {results_file}")