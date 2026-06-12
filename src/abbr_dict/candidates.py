import random
import string
from itertools import combinations, product

import pyphen as _pyphen_module
_PYPHEN = _pyphen_module.Pyphen(lang='en_US')

VOWELS = set("aeiou")
CONSONANTS = set(string.ascii_lowercase) - VOWELS

# reverse lookup: word -> single letters whose name sounds like it ("see" -> "c")
_WORD_SOUNDS_LIKE_LETTER = {}
for _letter, _words in {
    "u": {"you"},
    "r": {"are", "our"},
    "y": {"why"},
    "n": {"and", "in"},
    "b": {"be", "been"},
    "c": {"see", "sea"},
    "k": {"ok", "okay"},
    "x": {"ex"},
}.items():
    for _w in _words:
        _WORD_SOUNDS_LIKE_LETTER.setdefault(_w, set()).add(_letter)

# phonetically similar substitutions per character
_PHONETIC_SUBS = {
    'a': ['e', 'o'],
    'b': ['p', 'v'],
    'c': ['k', 's', 'q'],
    'd': ['t'],
    'e': ['i', 'a', 'y'],
    'f': ['v', 'p'],
    'g': ['j', 'k'],
    'h': [],
    'i': ['e', 'y'],
    'j': ['g', 'y'],
    'k': ['c', 'q', 'x'],
    'l': ['r'],
    'm': ['n'],
    'n': ['m'],
    'o': ['u', 'a'],
    'p': ['b', 'f'],
    'q': ['k', 'c'],
    'r': ['l'],
    's': ['z', 'c', 'x'],
    't': ['d'],
    'u': ['o', 'w'],
    'v': ['f', 'b', 'w'],
    'w': ['v', 'u'],
    'x': ['k', 's'],
    'y': ['i', 'e', 'j'],
    'z': ['s'],
}


def get_first_syllable(word):
    # approximates the first syllable with a vowel/consonant heuristic ('information'
    # -> 'in', 'computer' -> 'com'); ideally this would use pyphen instead
    if len(word) <= 2:
        return None

    found_vowel = False
    for i, ch in enumerate(word):
        if ch in VOWELS:
            found_vowel = True
        elif found_vowel:
            return word[: i + 1]
    return None


def _is_doublechar(abbr):
    return len(abbr) >= 2 and len(set(abbr)) == 1


def _is_valid_candidate(abbr, word):
    # abbr must be >=25% shorter than the word, except a doublechar 'cc' is valid for
    # any word containing 'c'
    if _is_doublechar(abbr) and len(abbr) == 2 and word.count(abbr[0]) >= 1:
        return True
    return len(abbr) <= int(len(word) * 0.75)


def _generate_deletions(word, min_len=1, max_results=5000):
    # single-char deletions via BFS, capped at max_results valid candidates
    results = set()
    queue = [word]
    seen = {word}

    while queue:
        current = queue.pop()
        for i in range(len(current)):
            candidate = current[:i] + current[i + 1 :]
            if len(candidate) >= min_len and candidate not in seen:
                seen.add(candidate)
                if _is_valid_candidate(candidate, word):
                    results.add(candidate)
                    if len(results) >= max_results:
                        return results
                queue.append(candidate)
    return results


def _generate_replacements(word, max_replacements=2, min_len=1):
    # random replacements, kept only if <=3 chars (short combos like 'tx' for 'typewriter')
    results = set()
    letters = string.ascii_lowercase

    for n_replace in range(1, max_replacements + 1):
        for positions in combinations(range(len(word)), n_replace):
            for _ in range(3):
                candidate = list(word)
                for pos in positions:
                    candidate[pos] = random.choice(letters)
                candidate = "".join(candidate)
                if (candidate != word
                        and len(candidate) >= min_len
                        and len(candidate) <= 3
                        and _is_valid_candidate(candidate, word)):
                    results.add(candidate)
    return results


def _generate_phonetic_replacements(word, max_replacements=2, min_len=1):
    # shorten first (deletions), then respell phonetically
    results = set()
    deletions = _generate_deletions(word, min_len=min_len)
    if not deletions:
        return results

    sample = random.sample(list(deletions), min(30, len(deletions)))
    for base in sample:
        for n_replace in range(1, max_replacements + 1):
            for positions in combinations(range(len(base)), n_replace):
                alternatives = [_PHONETIC_SUBS.get(base[pos], []) for pos in positions]
                if not any(alternatives):
                    continue
                for subs in product(*[alts if alts else [base[pos]] for alts, pos in zip(alternatives, positions)]):
                    candidate = list(base)
                    for pos, rep in zip(positions, subs):
                        candidate[pos] = rep
                    candidate = "".join(candidate)
                    if candidate != word and _is_valid_candidate(candidate, word):
                        results.add(candidate)
    return results


def _generate_doublechars(word):
    # 'cc' for every char c in the word ('meow' -> {'mm','ee','oo','ww'})
    results = set()
    for char in set(word):
        if word.count(char) >= 1:
            results.add(char * 2)
    return results


def _generate_letter_name_candidates(word):
    # single letter whose spoken name sounds like the word ('see' -> 'c')
    return set(_WORD_SOUNDS_LIKE_LETTER.get(word, set()))


def _generate_syllable_candidates(word):
    # pyphen syllables: acronym ('without' -> 'wo') and full first syllable ('without' -> 'with')
    results = set()
    if len(word) <= 2:
        return results

    parts = _PYPHEN.inserted(word).split('-')

    if len(parts) >= 2:
        acronym = ''.join(p[0] for p in parts if p)
        if _is_valid_candidate(acronym, word):
            results.add(acronym)

    if parts:
        first_syl = parts[0]
        if first_syl != word and _is_valid_candidate(first_syl, word):
            results.add(first_syl)

    return results


def _generate_deletion_then_replace(word, max_replacements=2, min_len=1):
    results = set()
    deletions = _generate_deletions(word, min_len=min_len)

    if not deletions:
        return results

    sample_deletions = random.sample(list(deletions), min(50, len(deletions)))
    for deleted in sample_deletions:
        replacements = _generate_replacements(
            deleted, max_replacements=max_replacements, min_len=min_len
        )
        for r in replacements:
            if _is_valid_candidate(r, word):
                results.add(r)
    return results


def generate_candidates(word, n=5, min_len=1, max_replacements=2):
    # returns plain abbr strings; expansion into (abbr, trigger_form) is in generate_all_candidates
    candidates = set()

    syllable = get_first_syllable(word)
    if syllable and syllable != word:
        if _is_valid_candidate(syllable, word):
            candidates.add(syllable)

    candidates.update(_generate_deletions(word, min_len=min_len))
    candidates.update(_generate_replacements(word, max_replacements=max_replacements, min_len=min_len))
    candidates.update(_generate_phonetic_replacements(word, max_replacements=max_replacements, min_len=min_len))
    candidates.update(_generate_deletion_then_replace(word, max_replacements=max_replacements, min_len=min_len))
    candidates.update(_generate_doublechars(word))
    candidates.update(_generate_letter_name_candidates(word))
    candidates.update(_generate_syllable_candidates(word))

    candidates.discard(word)

    final_pool = {c for c in candidates if _is_valid_candidate(c, word)}

    if not final_pool:
        return []

    return list(random.sample(list(final_pool), min(n, len(final_pool))))


def _expand_trigger_forms(abbr, word, rpt_key):
    from src.abbr_dict.speed import expand_with_rpt, build_abbr_keystream, TRG_CHAR

    word_keys = expand_with_rpt(word, rpt_key)
    word_cost = len(word_keys) + 1  # +1 for TRG_CHAR

    if not _is_doublechar(abbr) or len(abbr) != 2 or not rpt_key:
        potential = [{"abbr": abbr, "trigger_form": "trg"}]
    else:
        if word.count(abbr[0]) < 1:
            potential = [{"abbr": abbr, "trigger_form": "trg"}]
        else:
            # drop the plain "cc"+TRG form: with rpt on it's keystroke-identical to
            # rpt_trg, so it would be a duplicate trigger; keep only doubletap + rpt_trg
            potential = [
                {"abbr": abbr, "trigger_form": "doubletap"},
                {"abbr": abbr, "trigger_form": "rpt_trg"},
            ]

    # keep only triggers strictly shorter than the word's keystroke count
    valid = []
    for p in potential:
        keys = build_abbr_keystream(p["abbr"], p["trigger_form"], rpt_key=rpt_key)
        if len(keys) < word_cost:
            valid.append(p)

    return valid


def generate_all_candidates(vocabulary, n_per_item=1000, min_len=1, max_replacements=2,
                            enable_words=True, enable_suffixes=True, rpt_key=False):
    items = vocabulary.get("items", [])
    result_items = []

    for item in items:
        text = item["text"]
        item_type = item["type"]

        if item_type == "multiword":
            continue
        if item_type == "singleword" and not enable_words:
            continue
        if item_type == "suffix" and not enable_suffixes:
            continue

        if len(text) <= 1:
            continue

        cands = generate_candidates(
            text, n=n_per_item, min_len=min_len, max_replacements=max_replacements
        )

        if cands:
            expanded = []
            for c in cands:
                expanded.extend(_expand_trigger_forms(c, text, rpt_key))

            result_item = item.copy()
            result_item["candidates"] = expanded
            result_items.append(result_item)

    print(f"Generated candidates for {len(result_items)} items")
    return {"items": result_items}