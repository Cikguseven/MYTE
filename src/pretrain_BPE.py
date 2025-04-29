import argparse
import os
from datasets import load_dataset
from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from myt5.myt5_tokenizer import MyT5Tokenizer

# Data from wikipedia dumps dated 20231101
SOUTHEAST_ASIAN_LANGS = [
    "en",  # English: 6.41M
    "zh",  # Chinese: 1.38M
    # "id",  # Indonesian: 666k
    "vi",  # Vietnamese: 1.29M
    "ms",  # Malay: 369k
    "th",  # Thai: 160k
    "my",  # Burmese: 109k
    "lo",  # Lao: 5.01k
    "tl",  # Tagalog: 45.3k
    "ta",  # Tamil: 161k
    "km",  # Khmer: 12k
]

def preprocess_and_write(language, tokenizer, output_dir, max_articles):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{max_articles}_{language}.txt")
    with open(out_path, "w", encoding="utf-8") as fout:
        try:
            dataset = load_dataset('wikipedia', f"20220301.{language}", split='train[:{max_articles}]', streaming=True)
        except Exception as e:
            print(f"Could not load Wikipedia for {language}: {e}")
            return None
        for i, article in enumerate(tqdm(dataset, desc=f"Processing {language}")):
            if i >= max_articles:
                break
            text = article.get("text", "")
            if not text.strip():
                continue
            # Tokenize to bytes, then morphologically encode
            tokens = [f"{b:02x}" for b in text.encode("utf-8")]
            tokens = tokenizer.morphological_encode(tokens)
            fout.write(" ".join(tokens) + "\n")
    return out_path

def train_bpe_tokenizer(corpus_files, max_vocab_size, output_path="bpe_tokenizer.json"):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        max_vocab_size=max_vocab_size,
        special_tokens=["[UNK]", "<pad>", "</s>"]
    )
    tokenizer.train(corpus_files, trainer)
    tokenizer.save(output_path)
    print(f"BPE tokenizer saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=SOUTHEAST_ASIAN_LANGS, help="Languages to process")

    parser.add_argument("--decompose_map", type=str, help="Path to decompose map")
    parser.add_argument("--merge_map", type=str, help="Path to merge map")

    parser.add_argument("--download_articles", type=bool, default=False, help="Flag to download articles")
    parser.add_argument("--max_articles", type=int, help="Max articles per language")
    parser.add_argument("--max_vocab_size", type=int, help="BPE vocab size")

    parser.add_argument("--corpus_byte_output_dir", type=str, help="Where to save processed corpus")
    parser.add_argument("--bpe_tokenizer_output_dir", type=str, help="Where to save BPE tokenizer")

    args = parser.parse_args()

    tokenizer = MyT5Tokenizer(
        decompose_map=args.decompose_map,
        merge_map=args.merge_map
    )

    corpus_files = []
    if args.download_articles:
        for lang in args.languages:
            print(f"Processing {lang} articles...")
            out_file = preprocess_and_write(lang, tokenizer, args.corpus_byte_output_dir, max_articles=args.max_articles)
            if out_file:
                corpus_files.append(out_file)
    else:
        corpus_files = [os.path.join(args.corpus_byte_output_dir, f"{lang}_{args.max_articles}.txt") for lang in args.languages]

    if corpus_files:
        train_bpe_tokenizer(corpus_files, max_vocab_size=args.max_vocab_size, output_path=args.bpe_tokenizer_output_dir)
    else:
        print("No corpus files generated. Exiting.")