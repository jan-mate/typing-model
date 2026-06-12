# write _zipf_rpt_trg.json variants with synthetic TRG/RPT entries
import json
import math
import os
import sys

LOCAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

TRG_CHAR = "\x01"
RPT_CHAR = "\x02"

UNIGRAMS_IN = f"{LOCAL_ROOT}/data/frequencies/unigrams_zipf.json"
BIGRAMS_IN  = f"{LOCAL_ROOT}/data/frequencies/bigrams_zipf.json"
UNIGRAMS_OUT = f"{LOCAL_ROOT}/data/frequencies/unigrams_zipf_rpt_trg.json"
BIGRAMS_OUT  = f"{LOCAL_ROOT}/data/frequencies/bigrams_zipf_rpt_trg.json"


def main():
    with open(UNIGRAMS_IN) as f:
        unigrams = json.load(f)
    with open(BIGRAMS_IN) as f:
        bigrams = json.load(f)

    # TRG: 6.0, a common-key frequency (recalculated per-dict at eval time).
    # RPT: log10 of the summed same-letter bigram frequencies, reflecting total RPT usage.
    same_letter_linear_sum = sum(
        10 ** v for k, v in bigrams.items() if len(k) == 2 and k[0] == k[1]
    )
    rpt_zipf = math.log10(same_letter_linear_sum)

    unigrams_out = dict(unigrams)
    unigrams_out[TRG_CHAR] = 6.0
    unigrams_out[RPT_CHAR] = rpt_zipf

    bigram_mean = sum(bigrams.values()) / len(bigrams)

    bigrams_out = dict(bigrams)
    letters = [c for c in unigrams.keys() if c not in (TRG_CHAR, RPT_CHAR)]
    for c in letters:
        # TRG bigram from the unigram freq, shifted down by 1.0 (heuristic), so the model can tell
        # that exiting a common letter like 'e' is faster than exiting 'z'
        adjusted_zipf = max(0.0, unigrams.get(c, 0.0) - 1.0)
        bigrams_out[c + TRG_CHAR] = adjusted_zipf
        bigrams_out[TRG_CHAR + c] = adjusted_zipf
        bigrams_out[c + RPT_CHAR] = bigrams.get(c + c, bigram_mean)
        bigrams_out[RPT_CHAR + c] = bigram_mean

    for a in (TRG_CHAR, RPT_CHAR):
        for b in (TRG_CHAR, RPT_CHAR):
            bigrams_out[a + b] = 0.0

    with open(UNIGRAMS_OUT, "w") as f:
        json.dump(unigrams_out, f, indent=2, ensure_ascii=True)
    with open(BIGRAMS_OUT, "w") as f:
        json.dump(bigrams_out, f, indent=2, ensure_ascii=True)

    print(f"Wrote {UNIGRAMS_OUT}")
    print(f"  added TRG=6.0000, RPT={rpt_zipf:.4f}")
    print(f"  total entries: {len(unigrams_out)}")
    print(f"Wrote {BIGRAMS_OUT}")
    print(f"  letter-control bigram value: {bigram_mean:.4f}")
    print(f"  added {4 * len(letters) + 4} entries, total: {len(bigrams_out)}")


if __name__ == "__main__":
    main()
