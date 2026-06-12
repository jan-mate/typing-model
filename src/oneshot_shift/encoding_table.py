import os
import sys
import json

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import DATA_ROOT
from src.utils.layout_utils import build_layout_and_engine
from src.oneshot_shift.transformer import OneShotShiftTransformer
from src.oneshot_shift.enricher import OneShotShiftEnricher
from src.enrichment.engine import EnrichmentEngine
from src.enrichment.features import frequency
from src.utils.inference import build_synthetic_df

WORD = "Cat"
W_BACK, W_AHEAD = 3, 1

FEATURE_ROWS = [
    ("shift",             "Shift",       "{:.0f}"),
    ("bigram_frequency",  "Bigram freq", "{:.2f}"),
    ("word_frequency",    "Word freq",   "{:.2f}"),
    ("is_word_start",     "Word start",  "{:.0f}"),
    ("word_relative_pos", "Rel. pos",    "{:.2f}"),
]

FREQ_DIR = os.path.join(DATA_ROOT, "data/frequencies")


def make_enricher(one_ss_freqs):
    engine = EnrichmentEngine(
        unigrams_path=os.path.join(FREQ_DIR, "unigrams_zipf.json"),
        bigrams_path=os.path.join(FREQ_DIR, "bigrams_zipf.json"),
        words_path=os.path.join(FREQ_DIR, "words_zipf.json"),
        movement_features_path=os.path.join(DATA_ROOT, "data/layouts/movement_features_rpt_trg.json"),
    )
    if one_ss_freqs:
        o = OneShotShiftEnricher.ONE_SHOT_CHAR_ORD
        ss_bi = frequency.load_bigram_map(os.path.join(FREQ_DIR, "bigrams_1ss_zipf.json"))
        engine.bi_matrix[o, :] = ss_bi[o, :]
        engine.bi_matrix[:, o] = ss_bi[:, o]
        engine.uni_map[o] = frequency.load_unigram_map(os.path.join(FREQ_DIR, "unigrams_1ss_zipf.json"))[o]
    with open(os.path.join(FREQ_DIR, "words_zipf.json")) as f:
        word_map = json.load(f)
    return OneShotShiftEnricher(engine, word_map)


def encode(text, enricher, layout_paths):
    df, offsets = build_synthetic_df([text], W_BACK, W_AHEAD)
    base_dict = enricher.engine.enrich_linguistics(df, n_pads=0)
    enriched, encoded = enricher.enrich_layout(base_dict, **layout_paths)
    start, length = offsets[0]
    rows = {}
    for feat, _, _ in FEATURE_ROWS:
        arr = enriched.get(feat)
        rows[feat] = [None if arr is None else float(arr[start + i]) for i in range(length)]
    keys = [chr(c) for c in encoded[start:start + length]]
    return keys, rows


def fmt_val(v, fmt):
    if v is None or (isinstance(v, float) and v != v):  # NaN
        return "NaN"
    return fmt.format(v)


def display_key(k, oss_char):
    return "1SS" if k == oss_char else k


def print_table(title, keys, rows, oss_char):
    disp = [display_key(k, oss_char) for k in keys]
    print(f"\n{title}")
    header = "Key".ljust(12) + "".join(d.center(8) for d in disp)
    print(header)
    print("-" * len(header))
    for feat, label, fmt in FEATURE_ROWS:
        cells = "".join(fmt_val(v, fmt).center(8) for v in rows[feat])
        print(label.ljust(12) + cells)


def one_table(keys, rows, oss_char, subcap):
    ncol = len(keys) + 1
    head = ", ".join(["[*Key*]"] + [f"[{display_key(k, oss_char)}]" for k in keys])
    lines = [
        "  table(",
        f"    columns: {ncol},",
        "    inset: 5pt,",
        "    align: center + horizon,",
        "    stroke: 0.5pt + gray,",
        "    fill: (col, row) => if col == 0 or row == 0 { gray.lighten(90%) },",
        "",
        f"    {head},",
    ]
    for feat, label, fmt in FEATURE_ROWS:
        cs = [fmt_val(rows[feat][i], fmt) for i in range(len(keys))]
        row = ", ".join([f"[*{label}*]"] + [f"[{c}]" for c in cs])
        lines.append(f"    {row},")
    lines.append("  ),")
    return f"  [{subcap}]\n" + "\n".join(lines)


def typst_block(keys_std, rows_std, keys_oss, rows_oss, oss_char):
    print("\n\n===== TYPST FIGURE =====\n")
    block = (
        "#figure(\n"
        "  stack(\n"
        "    dir: ttb,\n"
        "    spacing: 1em,\n"
        + one_table(keys_std, rows_std, oss_char, "(a) Standard")
        + "\n"
        + one_table(keys_oss, rows_oss, oss_char, "(b) One-shot shift")
        + "\n"
        "  ),\n"
        "  caption: [Enriched features for \"Cat\", typed normally (a) and with "
        "one-shot shift (b). NaN marks values that do not apply.],\n"
        ") <oss-encoding>"
    )
    print(block)


def main():
    transformer = OneShotShiftTransformer()
    oss_char = transformer.one_shot_char
    text_std = WORD
    text_oss = transformer.transform(WORD)

    # rpt_key=True selects the layout with the 1SS slot (see transformer.ONE_SHOT_CHAR)
    base_layout_paths, _, _ = build_layout_and_engine(DATA_ROOT, rpt_key=False)
    oss_layout_paths, _, _ = build_layout_and_engine(DATA_ROOT, rpt_key=True)

    base_enricher = make_enricher(one_ss_freqs=False)
    oss_enricher = make_enricher(one_ss_freqs=True)

    keys_std, rows_std = encode(text_std, base_enricher, base_layout_paths)
    keys_oss, rows_oss = encode(text_oss, oss_enricher, oss_layout_paths)

    print_table("STANDARD CHORD: 'Cat'", keys_std, rows_std, oss_char)
    print_table("TRANSPARENT 1SS: '<1SS>cat'", keys_oss, rows_oss, oss_char)
    typst_block(keys_std, rows_std, keys_oss, rows_oss, oss_char)


if __name__ == "__main__":
    main()