# abbreviation-expansion primitives and word-frequency overrides
TRG_CHAR = "\x01"
RPT_CHAR = "\x02"


def expand_with_rpt(word, rpt_key):
    # with rpt_key on, a char equal to its predecessor becomes RPT_CHAR: the engine
    # claims all double-taps, so literal double letters must use the repeat key
    # ("letter" -> l,e,t,RPT,e,r)
    if not rpt_key:
        return list(word)
    out = []
    prev = None
    for ch in word:
        if ch == prev:
            out.append(RPT_CHAR)
            # prev stays as ch, so a third identical char also maps to RPT_CHAR
        else:
            out.append(ch)
            prev = ch
    return out


def build_abbr_keystream(abbr, trigger_form, rpt_key=False):
    if trigger_form == "trg":
        return expand_with_rpt(abbr, rpt_key) + [TRG_CHAR]
    if trigger_form == "doubletap":
        return [abbr[0], abbr[0]]
    if trigger_form == "rpt_trg":
        return [abbr[0], RPT_CHAR, TRG_CHAR]
    raise ValueError(f"Unknown trigger_form: {trigger_form!r}")


_DELIMITER_CHARS = set(' .,!?') | {TRG_CHAR}


def override_word_frequency(df_feats, offsets, texts, target_words, word_freq_map):
    # overwrite word_frequency with Zipf(target), except trailing delimiters which
    # stay at the pad value they have in training data
    col_idx = df_feats.columns.get_loc("word_frequency")
    for (start, length), text, target in zip(offsets, texts, target_words):
        zipf = float(word_freq_map.get(target.lower(), 0.0))
        end = start + length
        while end > start and text[end - start - 1] in _DELIMITER_CHARS:
            end -= 1
        if end > start:
            df_feats.iloc[start:end, col_idx] = zipf