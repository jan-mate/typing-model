import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from wordfreq import top_n_list, word_frequency, zipf_frequency

def run(output_dir, corpus_path="data/raw/combined_corpus.parquet"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    words = top_n_list('en', 50000)
    w_zipf, w_rel = {}, {}
    for w in words:
        w_zipf[w] = float(zipf_frequency(w, 'en'))
        w_rel[w] = float(word_frequency(w, 'en'))

    with open(out / "words_zipf.json", "w") as f: json.dump(w_zipf, f, indent=4)
    with open(out / "words.json", "w") as f: json.dump(w_rel, f, indent=4)

    df = pd.read_parquet(corpus_path)
    text = " ".join(df["body"].astype(str).tolist())
    text = " ".join(text.split())
    
    u_counts = Counter(text)
    b_counts = Counter(text[i:i+2] for i in range(len(text)-1))
    t_counts = Counter(text[i:i+3] for i in range(len(text)-2))

    def save_ngram(counts, name):
        filtered = {k: v for k, v in counts.items() if v >= 5}
        if not filtered:
            return
        total = sum(filtered.values())
        zipfs = {k: float(np.clip(np.log10(v/total)+9, 0, 10)) for k, v in filtered.items()}
        probs = {k: float(v/total) for k, v in filtered.items()}
        
        zipfs = dict(sorted(zipfs.items(), key=lambda item: item[1], reverse=True))
        probs = dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))

        with open(out / f"{name}_zipf.json", "w") as f: json.dump(zipfs, f, indent=4)
        with open(out / f"{name}.json", "w") as f: json.dump(probs, f, indent=4)

    save_ngram(u_counts, "unigrams")
    save_ngram(b_counts, "bigrams")
    save_ngram(t_counts, "trigrams")

if __name__ == "__main__":
    run("data/frequencies")