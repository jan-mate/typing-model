# real-context strip sampling. builds (word, abbr) keystream pairs from corpus
import re
import random
from dataclasses import dataclass

import numpy as np

from src.abbr_dict.speed import expand_with_rpt, build_abbr_keystream

WORD_RE = re.compile(r"[a-zA-Z']+")

_SUFFIX_CORPUS_CAP = 500  # max candidate sentences to collect for a suffix lookup


@dataclass(frozen=True)
class Occurrence:
    sent: str
    start: int   # char index in sent where the matched word starts
    end: int     # char index in sent where the matched word ends
    body: str    # matched text in original case (e.g. "Department")


@dataclass
class ContextStrip:
    word_ks: str
    abbr_ks: str
    host_word: str   # lowercase host word for frequency override
    word_len: int    # len of word_part in word_ks (for z-score slicing)
    abbr_len: int    # len of abbr_part in abbr_ks
    weight: float    # Zipf weight; 1.0 for singlewords


def build_word_index(sentences: list[str], max_per_word: int = 200) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = {}
    for sent in sentences:
        for tok in set(WORD_RE.findall(sent.lower())):
            lst = idx.setdefault(tok, [])
            if len(lst) < max_per_word:
                lst.append(sent)
    return idx


def build_suffix_index(
    word_to_sents: dict[str, list[str]],
    max_suffix_len: int = 8,
) -> dict[str, list[str]]:
    # map each suffix (len 2..max_suffix_len) to words ending with it, for fast lookup
    idx: dict[str, list[str]] = {}
    for word in word_to_sents:
        for n in range(2, min(max_suffix_len + 1, len(word))):
            suf = word[-n:]
            idx.setdefault(suf, []).append(word)
    return idx


def find_occurrences(
    word_lower: str,
    ctype: str,
    word_to_sents: dict[str, list[str]],
    suffix_index: dict[str, list[str]],
    N: int,
) -> list[Occurrence]:
    # up to N occurrences of word_lower (or words ending in it, for suffixes)
    if ctype in ("singleword", "linked"):
        cand_sents = list(word_to_sents.get(word_lower, []))
    else:
        host_words = suffix_index.get(word_lower, [])
        cand_sents = []
        for hw in host_words:
            cand_sents.extend(word_to_sents.get(hw, []))
            if len(cand_sents) >= _SUFFIX_CORPUS_CAP:
                break

    random.shuffle(cand_sents)

    occurrences: list[Occurrence] = []
    for sent in cand_sents:
        for m in WORD_RE.finditer(sent):
            tok = m.group()
            tok_lower = tok.lower()
            if ctype in ("singleword", "linked"):
                match = tok_lower == word_lower
            else:
                match = tok_lower.endswith(word_lower) and len(tok) > len(word_lower)

            if match:
                occurrences.append(Occurrence(sent=sent, start=m.start(), end=m.end(), body=tok))
                if len(occurrences) >= N:
                    return occurrences

    return occurrences


def build_context_strip(
    occ: Occurrence,
    word_lower: str,
    ctype: str,
    abbr: str,
    trigger_form: str,
    left_len: int,
    right_len: int,
    rpt_key: bool,
    word_freq_map: dict[str, float],
) -> ContextStrip:
    # the word portion starts at index left_len in both strips, so slicing uses a constant offset
    sent, start, end, body = occ.sent, occ.start, occ.end, occ.body

    left  = sent[max(0, start - left_len) : start].rjust(left_len)
    right = (sent[end : end + right_len] if end < len(sent) else " ").ljust(right_len)

    if ctype in ("singleword", "linked"):
        word_part = "".join(expand_with_rpt(body, rpt_key))
        abbr_part = "".join(build_abbr_keystream(abbr, trigger_form, rpt_key))
        if body[0].isupper() and abbr_part:
            abbr_part = abbr_part[0].upper() + abbr_part[1:]
        host_word = word_lower
        weight = 1.0
    else:  # suffix
        stem = body[: -len(word_lower)]
        word_part = "".join(expand_with_rpt(body, rpt_key))
        abbr_part = "".join(expand_with_rpt(stem, rpt_key)) + "".join(
            build_abbr_keystream(abbr, trigger_form, rpt_key)
        )
        host_word = body.lower()
        weight = 10.0 ** word_freq_map.get(host_word, 0.0)

    return ContextStrip(
        word_ks=left + word_part + right,
        abbr_ks=left + abbr_part + right,
        host_word=host_word,
        word_len=len(word_part),
        abbr_len=len(abbr_part),
        weight=weight,
    )


def weighted_mean_savings(savings: list[float], weights: list[float]) -> float:
    if not savings:
        return 0.0
    return float(np.average(savings, weights=weights))


def compute_costs_sliced(
    df_feats,
    offsets: list[tuple[int, int]],
    model,
    left_len: int,
    word_lens: list[int],
) -> list[tuple[float, float]]:
    # sum z-scores only over the word portion [left_len : left_len+word_len]; context
    # chars are excluded since they differ between word and abbr strips and add noise
    if not offsets:
        return []

    feat_matrix = df_feats[list(model.features)].values.astype(np.float32)

    predict_indices: list[int] = []
    text_mapping: list[int] = []

    for text_idx, (start, length) in enumerate(offsets):
        wl = word_lens[text_idx]
        # position 0 has no IKI (no preceding keystroke); word starts at left_len
        for i in range(max(1, left_len), left_len + wl):
            if i < length:
                predict_indices.append(start + i)
                text_mapping.append(text_idx)

    if not predict_indices:
        return [(0.0, 0.0)] * len(offsets)

    idx_arr = np.array(predict_indices)
    rel = np.arange(-model.w_back, model.w_ahead + 1)
    X = feat_matrix[idx_arr[:, None] + rel].reshape(len(idx_arr), -1)

    means, stds = model.predict(X)

    tm = np.array(text_mapping)
    sum_means = np.bincount(tm, weights=means, minlength=len(offsets))
    sum_vars  = np.bincount(tm, weights=stds ** 2, minlength=len(offsets))
    return [(float(m), float(np.sqrt(v))) for m, v in zip(sum_means, sum_vars)]