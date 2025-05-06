import os
import argparse
from datasets import load_dataset
from tqdm import tqdm
from myt5.myt5_tokenizer import MyT5Tokenizer

# Data from wikipedia dumps dated 20231101
SOUTHEAST_ASIAN_LANGS = [
    # "en",  # English: 6.41M
    "id",  # Indonesian: 666k
    # "km",  # Khmer: 12k
    # "lo",  # Lao: 5.01k
    # "ms",  # Malay: 369k
    # "my",  # Burmese: 109k
    # "ta",  # Tamil: 161k
    # "th",  # Thai: 160k
    # "tl",  # Tagalog: 45.3k
    # "vi",  # Vietnamese: 1.29M
    # "zh",  # Chinese: 1.38M
]

def preprocess_and_write(language, tokenizer, output_dir, max_articles):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{max_articles}_{language}.txt")
    with open(out_path, "w", encoding="utf-8") as fout:
        try:
            dataset = load_dataset('wikimedia/wikipedia', f"20231101.{language}", split='train', streaming=True)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=SOUTHEAST_ASIAN_LANGS, help="Languages to process")
    parser.add_argument("--decompose_map", type=str, help="Path to decompose map")
    parser.add_argument("--merge_map", type=str, help="Path to merge map")
    parser.add_argument("--max_articles", type=int, help="Max articles per language")
    parser.add_argument("--corpus_byte_output_dir", type=str, help="Where to save processed corpus")
    args = parser.parse_args()

    tokenizer = MyT5Tokenizer(
        decompose_map=args.decompose_map,
        merge_map=args.merge_map
    )

    for lang in args.languages:
        print(f"Processing {lang} articles...")
        preprocess_and_write(lang, tokenizer, args.corpus_byte_output_dir, max_articles=args.max_articles)