import argparse
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Data from wikipedia dumps dated 20231101
SOUTHEAST_ASIAN_LANGS = [
    "en",  # English: 6.41M
    "zh",  # Chinese: 1.38M
    "id",  # Indonesian: 666k
    "vi",  # Vietnamese: 1.29M
    "ms",  # Malay: 369k
    "th",  # Thai: 160k
    "my",  # Burmese: 109k
    "lo",  # Lao: 5.01k
    "tl",  # Tagalog: 45.3k
    "ta",  # Tamil: 161k
    "km",  # Khmer: 12k
]

def train_bpe_tokenizer(corpus_files, max_vocab_size, output_path):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=max_vocab_size,
        special_tokens=["[UNK]", "<pad>", "</s>"]
    )
    tokenizer.train(corpus_files, trainer)
    tokenizer.save(output_path)
    print(f"BPE tokenizer saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=SOUTHEAST_ASIAN_LANGS, help="Languages to process")
    parser.add_argument("--max_articles", type=int, required=True, help="Max articles per language")
    parser.add_argument("--max_vocab_size", type=int, required=True, help="BPE vocab size")
    parser.add_argument("--corpus_byte_output_dir", type=str, required=True, help="Where processed corpus is stored")
    parser.add_argument("--bpe_tokenizer_output_dir", type=str, required=True, help="Where to save BPE tokenizer")
    args = parser.parse_args()

    corpus_files = [os.path.join(args.corpus_byte_output_dir, f"{args.max_articles}_{lang}.txt") for lang in args.languages]

    if corpus_files:
        train_bpe_tokenizer(
            corpus_files,
            args.max_vocab_size,
            args.bpe_tokenizer_output_dir
        )
    else:
        print("No corpus files generated. Exiting.")