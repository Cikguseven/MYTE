import os
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from functools import partial
from sacrebleu.metrics import CHRF

from utils import normalize_text
from utils_modeling import get_model_tokenizer

TASK_LANGUAGES = {
    'translation': ['en2ta', 'en2te', 'en2el', 'en2hy', 'en2ru', 'en2kk', 'en2am', 'en2vi', 'en2ja', 'en2fr',
                    'en2ko', 'en2de', 'en2pl', 'en2sn'],
}


def parse_data_example(example, task):
    text, target = '', ''
    if task == 'translation':
        text = example['input']
        target = example['target']

    return {"text": normalize_text(text), "target": normalize_text(target)}


def preprocess_function(examples, tokenizer, max_length=1024):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(examples["text"], padding="longest", max_length=max_length, truncation=True, return_tensors="pt").to(device)
    examples["inputs"] = inputs
    return examples


def get_dataset(lang, task, dataset_dir, tokenizer, sample_size=100, split='test'):
    data_file = os.path.join(dataset_dir, task, split, f"{lang}.jsonl")
    with open(data_file, 'r') as f:
        examples = [parse_data_example(json.loads(line), task) for line in f.readlines()]

    dataset = Dataset.from_list(examples)
    dataset = dataset.shuffle(seed=42).select(range(sample_size))
    dataset = dataset.map(partial(preprocess_function, tokenizer=tokenizer), desc="Running tokenizer", batched=True, batch_size=32)

    return dataset


def compute_chrf_for_task(model, tokenizer, lang, task, dataset_dir, device):
    dataset = get_dataset(lang, task, dataset_dir, tokenizer)
    predictions = []
    references = []

    for example in tqdm(dataset, desc=f"Processing {task} inference in {lang}"):
        inputs = example["inputs"]
        target = example["target"]

        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=1024)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(prediction)
        references.append(target)

    # Compute chrF score
    chrf = CHRF()
    score = chrf.corpus_score(predictions, [references])
    return score


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_dir", required=True, type=str)
    argparser.add_argument("--dataset_dir", required=False, default="../xtreme_up_v1.1", type=str)
    argparser.add_argument("--results_dir", required=False, default="../xtreme_up_results", type=str)

    argparser.add_argument("--task", required=True, type=str)
    argparser.add_argument("--model_type", required=True, type=str)
    argparser.add_argument("--model_size", required=True, type=str)
    argparser.add_argument("--model_steps", required=False, type=int, default=250000)

    args = argparser.parse_args()

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    model, tokenizer = get_model_tokenizer(args.model_type, args.model_size, args.model_steps, args.model_dir, device=device)

    chrf_scores = {}

    for lang in TASK_LANGUAGES[args.task]:
        print(f"Processing {args.task} inference in {lang}")
        score = compute_chrf_for_task(model, tokenizer, lang, args.task, args.dataset_dir, device)
        chrf_scores[lang] = score.score
        print(f"chrF Score for {lang}: {score.score:.4f}")

    # Save results
    results_file = os.path.join(args.results_dir, f"{args.model_type}_{args.model_size}_{args.task}_chrf_scores.json")
    with open(results_file, 'w') as f:
        json.dump(chrf_scores, f, indent=4)

    print(f"chrF scores saved to {results_file}")