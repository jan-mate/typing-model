import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from src.config import CORPUS_PATH
from src.oneshot_shift.transformer import OneShotShiftTransformer

class FrequencyBuilder:
    def __init__(self, transformer: OneShotShiftTransformer):
        self.transformer = transformer

    def build_and_save(self, output_dir: str, corpus_path: str = CORPUS_PATH):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        print(f"Loading corpus from {corpus_path}...")
        df = pd.read_parquet(corpus_path)

        print("Transforming corpus text for 1SS...")
        raw_text = " ".join(df["body"].astype(str).tolist())
        transformed_text = self.transformer.transform(raw_text)
        transformed_text = " ".join(transformed_text.split())

        print("Counting n-grams...")
        u_counts = Counter(transformed_text)
        b_counts = Counter(transformed_text[i:i+2] for i in range(len(transformed_text)-1))
        
        self._save_ngram(u_counts, "unigrams_1ss", out)
        self._save_ngram(b_counts, "bigrams_1ss", out)
        print(f"1SS frequencies saved to {out}")

    def _save_ngram(self, counts, name, output_dir: Path):
        filtered = {k: v for k, v in counts.items() if v >= 5}
        if not filtered:
            return

        total = sum(filtered.values())
        # zipf score: log10(prob) + 9 
        zipfs = {k: float(np.clip(np.log10(v/total)+9, 0, 10)) for k, v in filtered.items()}
        zipfs = dict(sorted(zipfs.items(), key=lambda item: item[1], reverse=True))

        output_path = output_dir / f"{name}_zipf.json"
        with open(output_path, "w") as f:
            json.dump(zipfs, f, indent=4)
