from src.abbr_dict.eval_core import (
    display_ks, IKI_MEAN_MS, IKI_STD_MS, MIN_ENTRY_USES,
)


def print_examples(debug, n_show=5):
    all_data        = debug["all_data"]
    normal_costs    = debug["normal_costs"]
    abbr_costs      = debug["abbr_costs"]
    debug_df_feats  = debug["debug_df_feats"]
    debug_offsets   = debug["debug_offsets"]
    debug_batch_end = debug["debug_batch_end"]

    print()
    print("=" * 62)
    print("EXAMPLES")
    print("=" * 62)

    n_shown = 0
    for i, (sent, normal_text, abbr_text, freq_overrides, stats, match_keys) in enumerate(all_data):
        if not freq_overrides or n_shown >= n_show:
            continue
        if i >= debug_batch_end:
            continue

        n_normal = len(normal_text)
        n_abbr   = len(abbr_text)
        t_normal = max(0, n_normal - 1) * IKI_MEAN_MS + normal_costs[i][0] * IKI_STD_MS
        t_abbr   = max(0, n_abbr   - 1) * IKI_MEAN_MS + abbr_costs[i][0]   * IKI_STD_MS
        savings_ms = t_normal - t_abbr

        print(f"  Original: {sent}")
        print(f"  Abbr:     {display_ks(abbr_text)}")

        seq_start = debug_offsets[i][0]
        for char_start, char_end, target_word in freq_overrides:
            token_display = display_ks(abbr_text[char_start:char_end])
            row = seq_start + char_start
            if row < len(debug_df_feats):
                wf = float(debug_df_feats.at[row, "word_frequency"])
                bf = float(debug_df_feats.at[row, "bigram_frequency"])
                print(f"    [{target_word}] -> [{token_display}]  "
                      f"word_freq={wf:.3f}  bigram_freq={bf:.3f}")

        print(f"  Normal: {t_normal/1000:.2f}s  Abbr: {t_abbr/1000:.2f}s  "
              f"Savings: {savings_ms:+.1f} ms ({savings_ms/max(1, t_normal)*100:+.1f}%)")
        print()
        n_shown += 1


def print_summary(result, model_name="mlp"):
    has_suffixes = result["n_suffix_matches"] > 0

    print("=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print(f"Model: {model_name}")
    print(f"Dict: {result['dict_size']} entries  "
          f"mean_intuitiveness={result['dict_mean_intuitiveness']}  "
          f"min_intuitiveness={result['dict_min_intuitiveness']}")
    print(f"Sentences:  {result['n_sentences']}  |  Avg length: {result['avg_sentence_len']:.1f} chars\n")

    total_tokens = result["total_tokens"]
    n_sw  = result["n_singleword_matches"]
    n_suf = result["n_suffix_matches"]
    n_cov = n_sw + n_suf
    print("Token coverage:")
    print(f"  Singleword: {n_sw:,} / {total_tokens:,}  ({100*n_sw/max(total_tokens,1):.1f}%)")
    if has_suffixes:
        print(f"  Suffix:     {n_suf:,} / {total_tokens:,}  ({100*n_suf/max(total_tokens,1):.1f}%)")
    print(f"  Total:      {n_cov:,} / {total_tokens:,}  ({100*n_cov/max(total_tokens,1):.1f}%)\n")

    print(f"Real-time estimated speedup (μ={IKI_MEAN_MS:.1f}ms, σ={IKI_STD_MS:.1f}ms):")
    print(f"  Without dict:  {result['total_t_normal_s']:.2f} s total")
    print(f"  With dict:     {result['total_t_abbr_s']:.2f} s total")
    print(f"  Total savings: {result['total_ms_saved']/1000:.2f} s")
    print(f"  Agg speedup:   {result['agg_speedup_pct']:.2f}%\n")

    print("Sentence-level savings:")
    print(f"  Mean savings:  {result['mean_sentence_savings_ms']:+.1f} ± {result['std_sentence_savings_ms']:.1f} ms")
    print(f"  Z-score Δ:     {result['z_savings_mean']:+.4f} ± {result['z_savings_std']:.4f} z\n")

    if result["singleword_ms_per_match"] is not None:
        print(f"Singleword speedup:  {result['singleword_ms_per_match']:+.1f} ms per match")
    if has_suffixes and result["suffix_ms_per_match"] is not None:
        print(f"Suffix speedup:      {result['suffix_ms_per_match']:+.1f} ms per match")
    print("  (Positive speedup means the abbreviation made typing faster)")

    rows = result["per_entry_stats"]
    if not rows:
        return

    print()
    print("=" * 62)
    print(f"PER-ENTRY SAVINGS  (min {MIN_ENTRY_USES} uses; {len(rows)} entries)")
    print("=" * 62)
    hdr = f"  {'text':<14} {'abbr':<8} {'trig':<10} {'intuit':>6}  {'uses':>5}  {'total_ms':>9}  {'mean_ms':>8}  {'speedup%':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    def _fmt_row(r):
        intuit = f"{r['intuitiveness']:.3f}" if r["intuitiveness"] is not None else "  ?"
        return (f"  {r['text']:<14} {r['abbr']:<8} {r['trigger_form']:<10} {intuit:>6}  {r['uses']:>5}  "
                f"{r['total_ms']:>+9.0f}  {r['mean_ms']:>+8.1f}  {r['speedup_pct']:>+7.1f}%")

    n_top       = 20
    bottom_rows = [r for r in rows if r["total_ms"] < 0]
    for r in rows[:n_top]:
        print(_fmt_row(r))
    n_mid = len(rows) - n_top - len(bottom_rows)
    if n_mid > 0:
        print(f"  ... ({n_mid} more entries) ...")
    for r in bottom_rows:
        print(_fmt_row(r))