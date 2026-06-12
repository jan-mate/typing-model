import os
import sys
import json
from rapidfuzz.distance import Levenshtein

try:
    import pyphen as _pyphen_module
    _PYPHEN = _pyphen_module.Pyphen(lang='en_US')
    HAS_PYPHEN = True
except ImportError:
    HAS_PYPHEN = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.abbr_dict.config import data_path

VOWELS = frozenset("aeiou")

# common English phonetic digraphs mapped to canonical sounds
_PHONETIC_REPLACEMENTS = [
    ("ph", "f"),
    ("gh", "f"),
    ("ee", "i"),
    ("ie", "i"),
    ("oo", "u"),
    ("ou", "u"),
    ("ea", "e"),
    ("ck", "k"),
    ("qu", "k"),
    ("ch", "C"), # placeholder for soft ch, to avoid c->k conversion
    ("sh", "s"),
    ("th", "t"),
    ("ks", "x"),
    ("c", "k"),  # hard c to k
    ("C", "ch"), # restore ch
]

# similarity between phonetically or visually similar characters
_SIM_MAP = {
    # vowels (very similar)
    ('e', 'i'): 0.85, ('o', 'u'): 0.90, ('i', 'y'): 0.95,
    # vowels (somewhat similar)
    ('a', 'e'): 0.60, ('a', 'o'): 0.50, ('e', 'y'): 0.60,
    # consonants (voiced/unvoiced or similar sound)
    ('k', 'c'): 0.95, ('k', 'q'): 0.90, ('c', 'q'): 0.85,
    ('s', 'z'): 0.90, ('s', 'c'): 0.80, # soft c
    ('f', 'v'): 0.85, ('p', 'b'): 0.85, ('t', 'd'): 0.85,
    ('m', 'n'): 0.80, ('l', 'r'): 0.65, ('j', 'g'): 0.85, # soft g
    ('v', 'w'): 0.60, ('u', 'w'): 0.70,
    ('x', 'k'): 0.60, ('x', 's'): 0.60,
}
# make bidirectional
for (a, b), val in list(_SIM_MAP.items()):
    _SIM_MAP[(b, a)] = val

# single letters whose spoken name sounds like the word
LETTER_SOUNDS_LIKE = {
    "u": {"you"},
    "r": {"are", "our"},
    "y": {"why"},
    "n": {"and", "in"},
    "b": {"be", "been"},
    "c": {"see", "sea"},
    "k": {"ok", "okay"},
    "x": {"ex"},
}


def _phonetic_normalize(text):
    text = text.lower()
    for old, new in _PHONETIC_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def _char_sim(c1, c2):
    if c1 == c2:
        return 1.0
    if (c1, c2) in _SIM_MAP:
        return _SIM_MAP[(c1, c2)]
    if c1 in VOWELS and c2 in VOWELS:
        return 0.35
    if (c1 in VOWELS) != (c2 in VOWELS):
        return 0.05
    return 0.1


def _sequence_sim(abbr, word):
    if not abbr or not word:
        return 0.0
    return sum(max(_char_sim(c, w) for w in word) for c in abbr) / len(abbr)


def _calculate_penalty(abbr, word, normalize=False):
    # penalize chars in abbr that aren't in word
    a_proc = _phonetic_normalize(abbr) if normalize else abbr.lower()
    w_proc = _phonetic_normalize(word) if normalize else word.lower()
    w_chars = list(w_proc)
    
    total_penalty = 0.0
    for a in a_proc:
        if a in w_chars:
            w_chars.remove(a)
            continue
            
        best_sim = 0.0
        best_char = None
        for w in w_chars:
            sim = _char_sim(a, w)
            if sim > best_sim:
                best_sim = sim
                best_char = w
        
        if best_sim >= 0.8: # threshold for a 'valid' substitution
            w_chars.remove(best_char)
            total_penalty += (1.0 - best_sim) * 0.5
        else:
            total_penalty += 1.0
            
    return total_penalty


def _subsequence_contiguity(abbr, word):
    if not abbr or not word:
        return 0.0, 0
    j = 0
    positions = []
    for c in abbr:
        while j < len(word) and word[j] != c:
            j += 1
        if j >= len(word):
            return 0.0, 0
        positions.append(j)
        j += 1
    span = positions[-1] - positions[0] + 1
    return len(abbr) / span, span


def _shared_prefix_len(a, b):
    n = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            n += 1
        else:
            break
    return n


def _syllable_acronym(word):
    # first letter of each pyphen syllable (e.g. 'without' -> 'wo')
    if not HAS_PYPHEN:
        return None
    parts = _PYPHEN.inserted(word).split('-')
    if len(parts) >= 2:
        return ''.join(p[0] for p in parts if p)
    return None


def compute_heuristic_score(word, abbr):
    word, abbr = word.lower(), abbr.lower()
    word_len = max(len(word), 1)
    abbr_len = max(len(abbr), 1)

    if not abbr:
        return 0.0

    # letter-name match, short-circuit (e.g. "u" -> "you")
    if (abbr_len == 1
            and abbr in LETTER_SOUNDS_LIKE
            and word in LETTER_SOUNDS_LIKE[abbr]):
        return 0.95

    # similarity metrics
    literal_sim = _sequence_sim(abbr, word)

    n_abbr = _phonetic_normalize(abbr)
    n_word = _phonetic_normalize(word)
    phonetic_sim = _sequence_sim(n_abbr, n_word)

    edit_sim = Levenshtein.normalized_similarity(abbr, word)

    # structural metrics, literal
    subseq_ratio, span = _subsequence_contiguity(abbr, word)
    gaps = max(0, span - abbr_len) if subseq_ratio > 0 else 0
    contiguity_score = (0.9 ** gaps) if subseq_ratio > 0 else 0.0

    # structural metrics, phonetic
    p_subseq_ratio, p_span = _subsequence_contiguity(n_abbr, n_word)
    p_gaps = max(0, p_span - len(n_abbr)) if p_subseq_ratio > 0 else 0
    p_contiguity_score = (0.9 ** p_gaps) if p_subseq_ratio > 0 else 0.0

    first_match = 1.0 if (abbr and word and abbr[0] == word[0]) else 0.0
    p_first_match = 1.0 if (n_abbr and n_word and n_abbr[0] == n_word[0]) else 0.0
    
    last_match = 1.0 if (abbr and word and abbr[-1] == word[-1]) else 0.0
    
    prefix_len = _shared_prefix_len(abbr, word)
    ratio = prefix_len / word_len
    prefix_utility = max(0.0, 1.0 - ((ratio - 0.4) / 0.5) ** 2) if ratio > 0 else 0.0
    is_prefix = 1.0 if word.startswith(abbr) else 0.0
    
    p_prefix_len = _shared_prefix_len(n_abbr, n_word)
    p_ratio = p_prefix_len / max(len(n_word), 1)
    p_prefix_utility = max(0.0, 1.0 - ((p_ratio - 0.4) / 0.5) ** 2) if p_ratio > 0 else 0.0
    p_is_prefix = 1.0 if n_word.startswith(n_abbr) else 0.0

    # reward chars saved but penalize being too short; ~4 letters is the recall sweet spot
    chars_saved = word_len - abbr_len
    # len factor peaks at 4+: len 1->0.2, 2->0.5, 3->0.8, 4+->1.0
    len_factor = min(1.0, 0.2 + 0.3 * (abbr_len - 1)) if abbr_len >= 1 else 0.0
    compression = min(1.0, (chars_saved / 8.0) * len_factor)

    first_letter_bonus = 1.0 if (abbr_len == 1 and first_match) else 0.0
    # single letters suit short words (and, the, cat) but not long ones (information)
    if abbr_len == 1:
        first_letter_bonus *= max(0.2, 1.0 - (word_len - 3) / 10.0)

    syl_acronym = _syllable_acronym(word)
    is_syl_acronym = 1.0 if (syl_acronym and abbr == syl_acronym) else 0.0

    final_contiguity = max(contiguity_score, p_contiguity_score)
    final_prefix_util = max(prefix_utility, p_prefix_utility)
    final_is_prefix = max(is_prefix, p_is_prefix)
    final_first_match = max(first_match, p_first_match)

    score = (
        0.25 * final_contiguity
        + 0.15 * final_first_match
        + 0.10 * final_is_prefix
        + 0.10 * final_prefix_util
        + 0.10 * is_syl_acronym
        + 0.10 * literal_sim
        + 0.10 * phonetic_sim
        + 0.08 * first_letter_bonus
        + 0.08 * compression
        + 0.03 * edit_sim
        + 0.03 * last_match
    )

    penalty_literal = _calculate_penalty(abbr, word, normalize=False)
    penalty_phonetic = _calculate_penalty(abbr, word, normalize=True)
    penalty = min(penalty_literal, penalty_phonetic)
    
    if penalty > 0:
        score *= max(0.1, 1.0 - 0.8 * penalty)

    return float(min(max(score, 0.0), 1.0))


CANDIDATES_PATH = data_path("candidates.json")
OUTPUT_PATH = data_path("candidates_scored.json")
SAMPLE_WORDS = ["the", "and", "classification", "information", "people", "cat", "see", "manager", "message", "birthday", "without"]
TOP_K_PER_ITEM = 80

def main():
    if not os.path.exists(CANDIDATES_PATH):
        print(f"Error: {CANDIDATES_PATH} not found. Run candidate generation first.")
        return

    with open(CANDIDATES_PATH) as f:
        data = json.load(f)

    for item in data["items"]:
        word = item["text"]
        for cand in item["candidates"]:
            cand["intuitiveness"] = round(
                compute_heuristic_score(word, cand["abbr"]), 4
            )
        cands = item["candidates"]
        cands.sort(key=lambda c: c["intuitiveness"], reverse=True)
        top = cands[:TOP_K_PER_ITEM]
        # always keep repeat-key forms past the top-K.
        rpt_extra = [c for c in cands[TOP_K_PER_ITEM:]
                     if c["trigger_form"] in ("doubletap", "rpt_trg")]
        item["candidates"] = top + rpt_extra

    for sample_word in SAMPLE_WORDS:
        entry = next((i for i in data["items"] if i["text"] == sample_word), None)
        if entry:
            print(f"\n{sample_word}: top 5")
            for c in entry["candidates"][:5]:
                print(f"  {c['abbr']:10s} ({c['trigger_form']:10s})  {c['intuitiveness']:.3f}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved scored candidates to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
