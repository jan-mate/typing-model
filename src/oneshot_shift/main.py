import os
import sys
import json
import random
import pandas as pd
import re
import argparse

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import DATA_ROOT, CORPUS_PATH, model_dir
from src.utils.layout_utils import build_layout_and_engine
from src.models.wrappers import MLP as MLPSpeedModel, LGBM as LGBMSpeedModel, LinReg as LinRegSpeedModel
from src.oneshot_shift.transformer import OneShotShiftTransformer
from src.oneshot_shift.frequency_builder import FrequencyBuilder
from src.oneshot_shift.enricher import OneShotShiftEnricher
from src.oneshot_shift.evaluator import SpeedEvaluator
from src.enrichment.engine import EnrichmentEngine
from src.enrichment.features import frequency

N_SENTENCES = 500
MAX_SENT_LEN = 70
MIN_SENT_LEN = 5
RANDOM_SEED = 42

WORDS_ZIPF = os.path.join(DATA_ROOT, "data/frequencies/words_zipf.json")
FREQ_1SS_DIR = os.path.join(DATA_ROOT, "data/frequencies")

def main():
    parser = argparse.ArgumentParser(description="Evaluate 1SS speedup.")
    parser.add_argument("--model", type=str, default="lgbm", choices=["lgbm", "mlp", "linreg"])
    parser.add_argument("--sentences", type=int, default=N_SENTENCES)
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    
    transformer = OneShotShiftTransformer()
    
    u_1ss_path = os.path.join(FREQ_1SS_DIR, "unigrams_1ss_zipf.json")
    b_1ss_path = os.path.join(FREQ_1SS_DIR, "bigrams_1ss_zipf.json")
    
    actual_corpus_path = CORPUS_PATH
    if not os.path.exists(actual_corpus_path):
        alt_path = CORPUS_PATH.replace("data/", "data/raw/")
        if os.path.exists(alt_path):
            actual_corpus_path = alt_path
    
    if not os.path.exists(u_1ss_path) or not os.path.exists(b_1ss_path):
        print(f"Generating 1SS-aware frequency dictionaries from {actual_corpus_path}...")
        builder = FrequencyBuilder(transformer)
        builder.build_and_save(FREQ_1SS_DIR, corpus_path=actual_corpus_path)
    
    print("Setting up enrichment engines...")
    # rpt_key=True selects the layout with the 1SS slot (see transformer.ONE_SHOT_CHAR)
    base_layout_paths, _, _ = build_layout_and_engine(DATA_ROOT, rpt_key=False)
    oss_layout_paths, _, _ = build_layout_and_engine(DATA_ROOT, rpt_key=True)
    
    move_feats = os.path.join(DATA_ROOT, "data/layouts/movement_features_rpt_trg.json")
    u_reg = os.path.join(FREQ_1SS_DIR, "unigrams_zipf.json")
    b_reg = os.path.join(FREQ_1SS_DIR, "bigrams_zipf.json")

    base_engine = EnrichmentEngine(u_reg, b_reg, WORDS_ZIPF, move_feats)
    oss_engine = EnrichmentEngine(u_reg, b_reg, WORDS_ZIPF, move_feats)
    o = OneShotShiftEnricher.ONE_SHOT_CHAR_ORD
    ss_bi = frequency.load_bigram_map(b_1ss_path)
    oss_engine.bi_matrix[o, :] = ss_bi[o, :]
    oss_engine.bi_matrix[:, o] = ss_bi[:, o]
    oss_engine.uni_map[o] = frequency.load_unigram_map(u_1ss_path)[o]
    
    print(f"Loading {args.model.upper()} model...")
    if args.model == "lgbm":
        model = LGBMSpeedModel(model_dir("lgbm"))
    elif args.model == "mlp":
        model = MLPSpeedModel(model_dir("mlp_main"))
    else:
        model = LinRegSpeedModel(model_dir("linreg"))
    
    print(f"Loading corpus for evaluation: {actual_corpus_path}")
    df_corpus = pd.read_parquet(actual_corpus_path)
    all_sentences = []
    for text in df_corpus["body"].dropna():
        for fragment in re.split(r"[.\n!?]+", str(text)):
            fragment = fragment.strip()
            if MIN_SENT_LEN <= len(fragment) <= MAX_SENT_LEN:
                all_sentences.append(fragment)
    
    random.shuffle(all_sentences)
    sentences = all_sentences[:args.sentences]
    print(f"Sampled {len(sentences)} sentences.")
    
    with open(WORDS_ZIPF) as f:
        word_freq_map = json.load(f)
    
    class PatchedEngineWrapper:
        def __init__(self, engine, word_map):
            self.engine = engine
            self.enricher = OneShotShiftEnricher(engine, word_map)
        def enrich_linguistics(self, df, n_pads):
            return self.enricher.engine.enrich_linguistics(df, n_pads)
        def enrich_layout(self, base_dict, **kwargs):
            return self.enricher.enrich_layout(base_dict, **kwargs)

    patched_engine = PatchedEngineWrapper(oss_engine, word_freq_map)
    evaluator = SpeedEvaluator(model, base_engine, patched_engine, base_layout_paths, oss_layout_paths)
    print("\nRunning Standard 1SS Evaluation...")
    results_df = evaluator.evaluate(sentences, transformer)

    def report_results(df, title, n_cap_events):
        t_base_mean = df['time_base_mean'].sum()
        t_base_median = df['time_base_median'].sum()
        
        speedup_mean = df['savings_mean'].sum() / t_base_mean * 100
        speedup_median = df['savings_median'].sum() / t_base_median * 100
        speedup_motion = df['savings_motion'].sum() / t_base_mean * 100
        
        savings_mean_ms = df['savings_mean'].sum()
        penalty_per_event = -savings_mean_ms / n_cap_events if n_cap_events > 0 else 0
        
        print("\n" + "="*45)
        print(f" {title} ({args.model.upper()}) ")
        print("="*45)
        print(f"Total Sentences:    {len(sentences)}")
        print(f"Capitalizations:    {n_cap_events}")
        print(f"Base Time (Mean):   {t_base_mean/1000:.2f}s")
        print(f"Base Time (Median): {t_base_median/1000:.2f}s")
        print("-" * 45)
        print(f"1. Strict (Mean):   {speedup_mean:>6.2f}% speedup")
        print(f"   Penalty/Shift:   {penalty_per_event:.2f} ms")
        print(f"2. Strict (Median): {speedup_median:>6.2f}% speedup")
        print(f"3. Motion-Only:     {speedup_motion:>6.2f}% speedup")
        print("="*45)

    n_cap_events = sum(transformer.transform(s).count(transformer.one_shot_char) for s in sentences)
    report_results(results_df, "STANDARD 1SS RESULTS", n_cap_events)

if __name__ == "__main__":
    main()
