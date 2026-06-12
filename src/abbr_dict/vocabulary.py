import re
from collections import Counter


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower())


def extract_words(corpus_texts, top_k=4000):
    counter = Counter()
    for text in corpus_texts:
        counter.update(tokenize(text))

    return [
        {"text": word, "type": "singleword", "frequency": count}
        for word, count in counter.most_common(top_k)
    ]


def extract_suffixes(corpus_texts, top_k=500, min_len=2, max_len=6,
                     min_distinct_words=50, exclude_words=None):
    # rank suffixes by total occurrences, but require each to appear across at least
    # min_distinct_words word types
    exclude_words = exclude_words or set()
    total_counts = Counter()
    distinct_words: dict[str, set] = {}

    for text in corpus_texts:
        for word in tokenize(text):
            if len(word) < min_len:
                continue
            for length in range(min_len, min(max_len + 1, len(word) + 1)):
                suffix = word[-length:]
                total_counts[suffix] += 1
                if suffix not in distinct_words:
                    distinct_words[suffix] = set()
                distinct_words[suffix].add(word)

    candidates = [
        (suffix, count)
        for suffix, count in total_counts.items()
        if (len(distinct_words.get(suffix, set())) >= min_distinct_words
            and suffix not in exclude_words)
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    return [
        {"text": suffix, "type": "suffix", "frequency": count}
        for suffix, count in candidates[:top_k]
    ]