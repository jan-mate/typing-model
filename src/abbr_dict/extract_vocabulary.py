import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import pandas as pd
from src.abbr_dict.vocabulary import extract_words, extract_suffixes

CORPUS_PATH = "data/raw/combined_corpus.parquet"
TEXT_COLUMN = "body"
OUTPUT_PATH = "data/abbr_dict/vocabulary.json"

TOP_K_WORDS = 3000
TOP_K_SUFFIXES = 500
SUFFIX_MIN_LEN = 2
SUFFIX_MAX_LEN = 5
SUFFIX_MIN_DISTINCT_WORDS = 50

ENABLE_WORDS = True
ENABLE_SUFFIXES = True


def main():
    if not os.path.exists(CORPUS_PATH):
        print(f"Error: {CORPUS_PATH} not found. Run download_expanded_corpus.py first.")
        return

    df = pd.read_parquet(CORPUS_PATH)
    texts = df[TEXT_COLUMN].dropna().tolist()
    # suffixes use Wikipedia text only because Reddit concatenation artifacts like
    # "activelooking" would let noisy suffixes pass the distinct word gate
    wiki_texts = df[df["source"] == "wikitext"][TEXT_COLUMN].dropna().tolist() if "source" in df.columns else texts
    print(f"Loaded {len(texts)} texts ({len(wiki_texts)} Wikipedia) from {CORPUS_PATH}")
    del df

    items = []

    if ENABLE_WORDS:
        words = extract_words(texts, top_k=TOP_K_WORDS)
        items.extend(words)
        print(f"Extracted {len(words)} words")

    if ENABLE_SUFFIXES:
        word_set = {w["text"] for w in words} if ENABLE_WORDS else set()
        suffixes = extract_suffixes(
            wiki_texts,
            top_k=TOP_K_SUFFIXES,
            min_len=SUFFIX_MIN_LEN,
            max_len=SUFFIX_MAX_LEN,
            min_distinct_words=SUFFIX_MIN_DISTINCT_WORDS,
            exclude_words=word_set,
        )
        items.extend(suffixes)
        print(f"Extracted {len(suffixes)} suffixes")

    result = {"items": items}

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {len(items)} items to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()