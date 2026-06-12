# sources:
#   reddit:   https://huggingface.co/datasets/anhchanghoangsg/reddit_pushshift_dataset_cleaned
#   wikitext: https://huggingface.co/datasets/wikitext  (wikitext-103-v1)
import os
import re
import random
import math
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.fs import PyFileSystem, FSSpecHandler
from datasets import load_dataset
from huggingface_hub import HfApi, HfFileSystem
from tqdm import tqdm

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

REDDIT_REPO = "anhchanghoangsg/reddit_pushshift_dataset_cleaned"
ALLOWED_CHARS_STR = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?'"
ALLOWED_CHARS = set(ALLOWED_CHARS_STR)
SENTENCE_SPLIT = re.compile(r"[.\n!?]+")
MIN_LEN = 3
MAX_LEN = 70
REDDIT_START_DATE = "2023-01-01"  # keep only Pushshift comments on/after this date
FLUSH_EVERY = 10_000              # write to parquet (and free RAM) every N sentences
READ_BATCH = 5_000                # rows read per parquet batch (bounds memory)
MAX_SCAN_PER_FILE = 200_000       # give up on a subreddit after scanning this many rows

# signatures of bot / moderator / ad-spam text, checked on the raw comment body
BOT_SPAM_SIGNATURES = (
    "i am a bot", "performed automatically", "this action was performed",
    "contact the moderators", "beep boop", "bleep bloop", "^^^",
    "chat with", "meet a girl", "recommended subreddits", "click here",
    "18free", "free now", "onlyfans", "subscribe to", "dm me", "promo code",
)


def is_bot_or_spam(raw):
    if not isinstance(raw, str):
        return True
    low = raw.lower()
    return any(sig in low for sig in BOT_SPAM_SIGNATURES)


def is_real_sentence(s):
    toks = s.split()
    if len(toks) < 2:
        return False
    # reject the WikiText rare-word placeholder
    low_toks = [t.lower() for t in toks]
    if "unk" in low_toks or "<unk>" in low_toks:
        return False
    alpha = [t for t in toks if any(c.isalpha() for c in t)]
    if len(alpha) < 2 or len(alpha) / len(toks) < 0.5:
        return False
    return True


def iter_sentences(raw):
    if not isinstance(raw, str):
        return
    raw = re.sub(r"http\S+|www\S+", "", raw)
    for piece in SENTENCE_SPLIT.split(raw):
        # reject rather than strip disallowed chars, since stripping would fabricate text
        if any(c not in ALLOWED_CHARS for c in piece):
            continue
        s = piece
        if re.search(r"(.)\1{3,}", s):  # discard 4+ repeated chars (e.g. "aaaa")
            continue
        s = re.sub(r"\s+", " ", s).strip()
        if MIN_LEN <= len(s) <= MAX_LEN and is_real_sentence(s):
            yield s


def _flush(writer, buf, source):
    if buf:
        writer.write_table(pa.Table.from_pydict(
            {"body": list(buf), "source": [source] * len(buf)}))
        buf.clear()


def _read_reddit_file(fp, cutoff, per_file_cap, pa_fs):
    res, scanned = [], 0
    try:
        with pa_fs.open_input_file(f"datasets/{REDDIT_REPO}/{fp}") as f:
            pf = pq.ParquetFile(f)
            cu_idx = pf.schema.names.index("created_utc")

            # use row-group stats to skip groups entiely older than the cutoff
            valid_rgs = []
            for rg_idx in range(pf.num_row_groups):
                stats = pf.metadata.row_group(rg_idx).column(cu_idx).statistics
                if stats and stats.has_min_max:
                    if str(stats.max) >= cutoff:
                        valid_rgs.append(rg_idx)
                else:
                    valid_rgs.append(rg_idx)  # if no stats: must read to check

            if not valid_rgs:
                return res

            # only the last valid group: finds recent data faster
            valid_rgs = valid_rgs[-1:]

            for batch in pf.iter_batches(batch_size=READ_BATCH, row_groups=valid_rgs, columns=["created_utc", "body"]):
                cu = batch.column("created_utc").to_pylist()
                bd = batch.column("body").to_pylist()
                
                for c, body in zip(cu, bd):
                    scanned += 1
                    if c is None or str(c) < cutoff or is_bot_or_spam(body):
                        continue
                    for s in iter_sentences(body):
                        res.append(s)
                        if len(res) >= per_file_cap:
                            return res
                if scanned >= MAX_SCAN_PER_FILE:
                    return res
    except Exception:
        pass
    return res


def get_reddit(writer, limit, cutoff=REDDIT_START_DATE, per_file_cap=250, seed=42, workers=12):
    print(f"Reddit (Pushshift, >= {cutoff}): collecting {limit:,} sentences "
          f"across subreddits weighted by size (cap {per_file_cap}/subreddit, {workers} workers)...")
    fs = HfFileSystem()
    pa_fs = PyFileSystem(FSSpecHandler(fs))

    print("Fetching subreddit file sizes for weighted sampling...")
    files_info = fs.find(f"datasets/{REDDIT_REPO}", detail=True)
    # skip files > 20MB to not OOM
    valid_files = [info for path, info in files_info.items()
                   if path.endswith(".parquet") and "comments" in path and info["size"] < 20_000_000]

    # weighted sampling without replacement: sort by -log(U) / weight
    rng = random.Random(seed)
    valid_files.sort(key=lambda x: -math.log(max(1e-10, rng.random())) / max(x["size"], 1))

    prefix = f"datasets/{REDDIT_REPO}/"
    files = [info["name"].replace(prefix, "") for info in valid_files]
    file_iter = iter(files)

    buf, total, subreddits_used = [], 0, 0
    pbar = tqdm(total=limit, desc="reddit")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = set()

        def submit_next():
            try:
                fp = next(file_iter)
            except StopIteration:
                return False
            futures.add(ex.submit(_read_reddit_file, fp, cutoff, per_file_cap, pa_fs))
            return True

        for _ in range(workers * 2):
            if not submit_next():
                break

        while futures and total < limit:
            done, pending = wait(futures, return_when=FIRST_COMPLETED)
            futures = pending
            for fut in done:
                take = fut.result()
                if take:
                    subreddits_used += 1
                take = take[: max(0, limit - total)]
                buf.extend(take)
                total += len(take)
                pbar.update(len(take))
                if len(buf) >= FLUSH_EVERY:
                    _flush(writer, buf, "reddit")
                if total < limit:
                    submit_next()
    _flush(writer, buf, "reddit")
    pbar.close()
    print(f"Collected from {subreddits_used} distinct subreddits.")
    return total


def get_wikitext(writer, limit):
    print(f"WikiText-103: collecting {limit:,} sentences...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
    ds = ds.shuffle(seed=43, buffer_size=20000)

    buf, total = [], 0
    pbar = tqdm(total=limit, desc="wikitext")
    for rec in ds:
        text = rec.get("text", "")
        stripped = text.strip()
        if stripped.startswith("=") and stripped.endswith("="):  # skip section headers
            continue
        for s in iter_sentences(text):
            buf.append(s)
            total += 1
            pbar.update(1)
            if len(buf) >= FLUSH_EVERY:
                _flush(writer, buf, "wikitext")
            if total >= limit:
                break
        if total >= limit:
            break
    _flush(writer, buf, "wikitext")
    pbar.close()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-each", type=int, default=50_000)
    ap.add_argument("--per-file-cap", type=int, default=100)
    ap.add_argument("--reddit-start-date", default=REDDIT_START_DATE)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN not set; downloads may be slower.")

    out_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / "combined_corpus.parquet"

    schema = pa.schema([("body", pa.string()), ("source", pa.string())])
    with pq.ParquetWriter(out_path, schema) as writer:
        n_reddit = get_reddit(writer, args.limit_each, cutoff=args.reddit_start_date,
                              per_file_cap=args.per_file_cap, seed=args.seed, workers=args.workers)

        n_wiki = get_wikitext(writer, args.limit_each)

    print(f"Saved {n_reddit + n_wiki:,} sentences "
          f"({n_reddit:,} reddit + {n_wiki:,} wikitext) to {out_path}")


if __name__ == "__main__":
    main()
