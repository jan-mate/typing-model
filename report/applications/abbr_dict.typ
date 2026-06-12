#import "/styles.typ": *

== Abbreviation Dictionary Optimization <sec-abbr-dict>

=== Introduction

Abbreviated typing replaces short strings with the longer text they stand for: whole words or common word-endings.
The typist enters an abbreviation, and an expansion engine writes out the full form.
The hard part is choosing the abbreviations.
A good one is fast to type and easy to remember.

Speed is a difficult criterion.
The typing time of an arbitrary key sequence cannot be measured without collecting new data for every candidate.
The trained IKI models supply it instead.
They predict the typing cost of any sequence, so thousands of candidates can be compared.

Such expansion should raise typing speed, but by how much is unknown.
This section quantifies it.


=== Methodology

Every abbreviation needs an expansion method: a way to type it and trigger the replacement.
This choice comes first, because it sets which abbreviation strings are possible at all.
The rest of the pipeline then has five stages: vocabulary extraction, candidate generation, intuitiveness scoring, speed precomputation, and optimization.
The first three build a pool of scored candidates.
The last two select a dictionary from it.

==== Expansion Methods

Two extra keys are added to a standard QWERTY keyboard.
Each key gives one way to expand an abbreviation.

The first method uses a *trigger key* (`TRG`).
An abbreviation fires when `TRG` follows its string.
So `"th" + TRG` expands to `"the "`, with a trailing space.
Plain `"th"` inside a word is left untouched.
The trigger key removes any ambiguity with normal typing, so every abbreviation can use it. `TRG` is a thumb key beside the spacebar.
It displaces no letter, and the model gives it the spacebar's position and finger.

The second method reuses the *repeat key* (`RPT`) of @repeat-key.
Because `RPT` already types literal double letters, the plain double-press of a key is freed for other uses.
The expansion engine claims it: a double-press fires an expansion directly, with no trigger key. This is the `doubletap` form.
A double-press of `t`, for example, can expand to `"the "`.

The repeat key changes only how a _doublechar_ abbreviation, which is a doubled letter such as `"tt"`, is triggered.
Its available trigger forms depend on the repeat regime:

+ *Repeat key off.* One form exists: the plain `"tt" + TRG`.
+ *Repeat key on.* The plain form is dropped for two alternatives. The `doubletap` form is a double-press of `t`. The `rpt_trg` form is `t` then `RPT` then `TRG`.

With the repeat key on, the two forms can map to two different words.
Claiming double-presses has a cost.
A literal double letter inside a word, such as the `"rr"` in `"squirrel"`, must come from `r` then `RPT`, or it triggers an accidental expansion. `RPT` is assumed to sit at the right-pinky home slot, displacing `;`.

Both placements are fixed assumptions, not optimized.
The trigger and repeat keys are modeled as real keystrokes with their own typing cost, not as free actions.

==== Vocabulary

The vocabulary holds two entry types, drawn from the combined Reddit and Wikipedia corpus of 100 000 texts (@sec-corpus).

*Words* are the 3 000 most frequent tokens, matched with the regex `[a-z]+(?:'[a-z]+)*` (ASCII lowercase; contractions kept whole).

*Suffixes* are word endings of 2 to 5 characters.
A suffix qualifies if it ends at least 50 distinct words.
This keeps real word parts (`-tion`, `-ing`, `-ment`) and drops noise endings.
The count of distinct words uses Wikipedia text only.
Reddit posts run words together (`"activelooking"`), which would inflate the count for false endings.
Ranking by total frequency still uses the full corpus.
The 500 most frequent suffixes are kept.
Overlapping ones such as `-ation` and `-tion` both stay, and the optimizer chooses between them.

A word or a suffix is called an _entry_. The pipeline treats the two alike: each entry is abbreviated, scored, and selected the same way.

==== Candidate Generation

Each entry generates abbreviation candidates by seven strategies:

+ *Deletions* — every single-character deletion, applied recursively to `min_len = 1`, capped at 5 000 results per entry.
+ *Replacements* — one or two letters of the entry swapped for random ones. The result is kept only when it is at most three characters long. This admits short combinations like `"tx"` for `"text"` and blocks long random strings.
+ *Phonetic deletion-then-replace* — the entry is first shortened by deletion, then some remaining letters are swapped for phonetically similar ones, producing candidates like `"kat"` for `"category"`. Deletion alone (strategy 1) cannot change letters, and replacement alone (strategy 2) does not shorten past three characters, so this combination reaches forms that neither produces.
+ *Random deletion-then-replace* — the same, with random letters in place of phonetic ones.
+ *Syllable* — the entry is split into syllables; both the syllable acronym (`"without"` → `"wo"`) and the first full syllable (`"without"` → `"with"`) are emitted.
+ *Doublechar* — for each character in the entry, the doubled string is emitted (`"the"` → `{"tt", "hh", "ee"}`).
+ *Letter-name* — a word that sounds like a letter name yields that letter (`"you"` → `"u"`).

A candidate must be at least 25% shorter than its entry (`len(abbr) ≤ 0.75 × len(entry)`); doublechar candidates are exempt.
Up to 8 000 candidates per entry survive deduplication.

==== Intuitiveness Scoring

Intuitiveness is scored by a hand-built heuristic.
It combines 11 weighted components.
#figure(
  table(
    columns: (auto, auto, 1fr),
    inset: 6pt,
    stroke: none,
    align: (center, center, left),
    fill: zebra,
    [*Weight*], [*Component*], [*Description*],
    [0.25], [contiguity], [Whether the whole abbreviation is an in-order subsequence of the entry, and how few letters it skips. A gapless block scores 1, each skipped letter multiplies the score by 0.9, and a non-subsequence scores 0. The higher of the raw-letter and phonetic-normalized score is taken.],
    [0.15], [first\_char], [Whether `abbr[0] == entry[0]`.],
    [0.10], [is\_prefix], [Whether the abbreviation is a prefix of the entry.],
    [0.10], [prefix\_utility], [Quadratic score peaking when the abbreviation is \~40% the length of the entry. It is short enough to save keystrokes, yet long enough to be recognizable.],
    [0.10], [literal\_seq\_sim], [Each abbreviation letter scored by its closest entry letter, then averaged; position is ignored. `"sqrl"` for `"squirrel"` scores 1, since `s`, `q`, `r`, `l` all occur.],
    [0.10], [phonetic\_seq\_sim], [Same as above after phonetic normalization.],
    [0.10], [syl\_acronym], [Whether the abbreviation matches the syllable acronym, which is the first letter of each syllable (e.g. `"without"` → `"wo"`, `"squirrel"` → `"sr"`).],
    [0.08], [first\_letter\_bonus], [Bonus for single-letter abbreviations, scaled down for longer entries.],
    [0.08], [compression], [Reward proportional to characters saved, capped at 8. Short sequences are more memorable; this is unrelated to speed.],
    [0.03], [edit\_dist\_sim], [$1 - "normalized Levenshtein distance"$ over the whole strings, so position matters. `"sqrl"` for `"squirrel"` scores 0.5.],
    [0.03], [last\_char], [Whether `abbr[-1] == entry[-1]`.],
  ),
  caption: [Intuitiveness score components and their weights.],
)

Phonetically related characters count as partly similar, not different: `k/c/q` are near-identical, `s/z/c` sound alike, and `f/v` and `p/b` are voiced pairs.
This lets `"kat"` score better against `"category"` than `"iat"`. A penalty then counts abbreviation characters that pair with no entry character, even a phonetically similar one.
One unmatched character multiplies the score by 0.2, and two or more by a floor of 0.1.
So `"kat"` for `"category"` keeps its score, while `"clzscficati"` for `"classification"` collapses near zero.
The result is clipped to $[0, 1]$.

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 6pt,
    stroke: none,
    align: (left, center, right),
    fill: (col, row) => if row == 0 { luma(235) },
    [*Word*], [*Abbreviation*], [*Score*],
    [`otter`],        [`ot`],    [0.93],
    [`rhododendron`], [`rho`],   [0.87],
    [`fox`],          [`fx`],    [0.73],
    [`squirrel`],     [`sl`],    [0.64],
    [`mango`],        [`ngo`],   [0.51],
    [`poppy`],        [`pi`],    [0.44],
    [`panda`],        [`pth`],   [0.39],
    [`fennec`],       [`vec`],   [0.24],
    [`fluffy`],       [`fyy`],   [0.10],
    [`meow`],         [`cat`],   [0.01],
  ),
  caption: [Example intuitiveness scores.],
)

==== Speed Precomputation

For every entry and every candidate, the DL and LightGBM ensembles predict the typing cost: the sum of per-keystroke IKI Z-scores over the keystream.
The saving is the difference.

$ "savings" = "entry_cost" - "abbr_cost" $

Each saving is measured in real context, not in isolation.
Up to six occurrences of the entry are sampled from the corpus.
Because they are drawn from real text, common surroundings appear in proportion to how often they occur.
Each occurrence keeps the entry's real neighbors around it, as far back and ahead as the model can see.
The entry keystream and the abbreviation keystream share those neighbors, so the saving compares them in the same surroundings.
Scoring every candidate at every position in the corpus would be too slow, so six samples are the compromise.
The candidate's saving is the mean over its samples.
These samples serve only the optimizer; the final evaluation instead re-scores each abbreviation inside whole corpus sentences.

Each sample also carries an uncertainty.
The 10-fold ensemble returns a standard deviation alongside its mean, and these are pooled into one uncertainty per candidate.
A noisy candidate is later penalized for it.

The model reads two frequency features: the bigram frequency and the word frequency.
An abbreviation has neither.
Its `TRG` and `RPT` keys have never been typed, so no real bigram involves them.
And the abbreviation string itself has no corpus frequency.

The trigger-key frequency is circular.
How often a `TRG` bigram occurs depends on which abbreviations the dictionary holds.
That dictionary is the optimizer's output, so the frequency cannot be known before the optimizer runs.

Optimization therefore uses placeholders.
Every trigger bigram takes the mean of all bigram frequencies.
The approach assumes it biases every abbreviation pattern about equally, so their relative savings stay roughly correct.

Evaluation uses real values instead.
Once the dictionary is fixed, each trigger bigram's true frequency is computed from it: the bigram is as frequent as the entries that fire through it.
The final speedup uses these real, precomputed frequencies, not the placeholder.
A literal double letter, such as the `"tt"` in `"letter"`, is typed as `t` then `RPT`; that bigram takes the frequency of `tt`.
The values are computed once, not iterated, since the dictionary is not re-optimized against them.

The word frequency works the same in both stages.
A word entry takes the frequency of the word it abbreviates.
A suffix entry takes the frequency of the word it appears in.
This treats the abbreviation as just as familiar as that word.

==== Model Choice and Goodhart Avoidance

Two models score speed: the deep MLP ensemble (DL) and LightGBM. Both predict per-keystroke IKI far better than a keystroke-count baseline ($R^2 approx 0.29$ versus 0).

The dictionary is optimized with DL and evaluated with LightGBM. Using one model for both inflates the result: the optimizer exploits that model's errors, and the same model then rewards them (Goodhart's law).
A second model as the judge reduces this.
LightGBM is not fully independent, since it trains on the same IKI data as DL. But it is a different kind of model, using gradient-boosted trees rather than a neural network, so it makes different errors.
The optimizer cannot exploit errors the judge does not share.
The baseline dictionary, the configuration used throughout, scores 13.29% under DL, its own model, and 12.99% under LightGBM, the judge.
The 0.30-point gap is the inflation an in-sample score would report.
The LightGBM score is the one reported throughout.

==== Optimization

Dictionary selection is a binary Integer Linear Program (ILP) solved with Google OR-Tools CP-SAT. A variable $x_(w,a) in {0, 1}$ marks whether entry $w$ takes abbreviation $a$. The objective trades speed against intuitiveness:

$ max sum_(w,a) f_w x_(w,a) [ (1 - lambda)(s_(w,a) - k_sigma sigma_(w,a)) + lambda dot 10 dot "intuit"_(w,a) ] - sum_((b,a)) f_b s_a z_(b,a) $

$f_w$ is corpus frequency, $s_(w,a)$ the mean predicted Z-score saving, $sigma_(w,a)$ the ensemble standard deviation, and $lambda in [0, 1]$ the trade-off weight. At $lambda = 0$ the objective is pure speed. At $lambda = 1$ it is pure intuitiveness. The uncertainty penalty $k_sigma$ (default 1.0) subtracts one standard deviation, so a noisy prediction ranks below a confident one of equal mean. A candidate is admitted only if two things hold: its intuitiveness clears a floor of 0.3, and its uncertainty-penalized saving $s_(w,a) - k_sigma sigma_(w,a)$ stays positive.

Three hard constraints apply:

+ At most one abbreviation per entry.
+ No two entries share the same `(abbreviation, trigger form)` pair.
+ The dictionary holds at most $N$ entries.

The final term prevents double counting when two selected suffixes overlap, such as `"ion"` and `"on"`. A word ending in `"ion"` — like `"dandelion"` — is matched by the longer suffix first, so `"ion"` claims it.
If both suffixes are selected, `"on"`'s saving would also be credited on those words, counting the gain twice.
The term subtracts the shorter suffix's own saving $s_a$ once, weighted by the longer suffix's frequency $f_b$. The indicator $z_(b,a)$ is 1 only when both are selected.
Here $s_a$ is the same quantity as $s_(w,a)$, but for the shorter suffix's entry.

The penalty is only an estimate, and it leans toward over-punishing.
It values the shorter suffix at its best candidate rather than the one actually chosen, so it subtracts a little more than the true overlap.
Valuing each suffix at a single representative candidate also keeps the optimization small enough to solve exactly.

The true objective is harder than this.
It would score each abbreviation in full sentence context, and the dictionary would itself change the trigger-key frequency.
That problem cannot be solved exactly.
The approach simplifies it instead. It uses sampled-context savings, fixed candidate scores, and synthetic trigger frequencies, which makes the objective linear.
CP-SAT then solves this simplified problem to its exact optimum.
That optimum is the best dictionary for the simplified problem, not necessarily for the true one.
Evaluation re-runs the model in full context to measure the real speedup.

==== Top-K Candidates per Word

Each word offers its $K$ best abbreviations to the optimizer.
The shortlist is ranked by the same speed–intuitiveness blend the objective uses at the current $lambda$.
With $K = 1$ each word is locked to its single best abbreviation.
Two words can then collide: `"t"` may be the top pick for both `"the"` and `"that"`, yet only one can use it.
With $K > 1$ the optimizer has fallbacks.
It can keep `"t"` for `"the"` and send `"that"` to its next-best form.
A larger $K$ never hurts the optimizer's own objective, since more options can only help it.
But it multiplies the candidate count, so solver time and memory bound how large it can be.

#figure(
  table(
    columns: (auto, auto),
    inset: 6pt,
    stroke: none,
    align: (center, right),
    fill: (col, row) => if row == 0 { luma(235) },
    [*K*], [*Speedup*],
    [3],  [7.25%],
    [5],  [9.13%],
    [8],  [11.50%],
    [10], [12.23%],
    [12], [12.99%],
    [15], [12.75%],
  ),
  caption: [Speedup versus candidates per word $K$ ($lambda = 0.1$, $N = 160$, LightGBM evaluation).],
) <ksweep-tbl>

@ksweep-tbl shows speedup rising steeply with $K$, peaking at $K = 12$ with 12.99%.
$K = 15$ scores slightly lower, 12.75%, even with strictly more candidates.
The extra candidates only help the optimizer's own objective, scored with DL.
They do not necessarily help the LightGBM evaluation.
They also give the optimizer more room to exploit DL's errors, the same Goodhart effect the two-model setup guards against.
That exploitation does not transfer to the LightGBM judge.
So past $K = 12$ the dictionary can look better to the optimizer yet score worse in evaluation.
$K = 12$ is the compromise used throughout: it gives the peak measured speedup, and a larger $K$ only adds solver cost.

==== Evaluation

The dictionary is applied to 25 000 sentences, typed with and without it.
Each sentence becomes one keystream, so the model sees the full surrounding context for each keystroke.
Each token is matched by greedy longest-match: whole-word entries beat suffixes, and the longest suffix wins.

Matching is case-insensitive, and capitalization carries to the abbreviation's first letter.
With `ca → cat`, the token `"Cat"` is typed as `"Ca"` then the trigger.
The shift cost appears on both sides of the comparison, so capitalization does not bias the saving.

Punctuation attached to the start or end of a token is separated before lookup, so the bare word is matched and the punctuation typed around the abbreviation.
Apostrophes count as word characters, so `"don't"` stays whole.
When punctuation follows an expansion, _smart-space deletion_ drops the trailing space the trigger would otherwise insert: `"squirrel."` is typed as the abbreviation, its trigger, then the period, and the removed space costs no keystroke.

For a matched token, the word-frequency feature is set to the parent word's Zipf value, not the abbreviation's.
The model should see how familiar the real word is.
Keystreams are scored in batches.

=== Results

The optimizer is run across $lambda$ from 0 to 1 at fixed $K = 12$ and $N = 160$. Each run yields a dictionary, evaluated on the same sentences. @sweep_pareto plots the frontier: mean intuitiveness against measured speedup, one point per $lambda$.

#figure(
  image("../abbr_pareto.png", width: 80%),
  caption: [Pareto frontier of the $lambda$ sweep. Each point is a dictionary optimized at a different $lambda$ ($K = 12$, $N = 160$), evaluated with LightGBM. $lambda = 0$ is pure speed, $lambda = 1$ pure intuitiveness.],
) <sweep_pareto>

The frontier is a trade-off.
Speedup peaks at $lambda = 0.1$, 12.99%, and falls steadily to 9.74% at $lambda = 1$, a span of 3.25 percentage points.
Over the same range mean intuitiveness rises from 0.63 to 0.80. $lambda = 0.1$ is the baseline used throughout: it gives the highest measured speedup while already raising intuitiveness from 0.63 ($lambda = 0$) to 0.69.
Past this point, each further gain in intuitiveness costs measurable speed, because the per-word shortlist itself is $lambda$-aware and begins offering more intuitive but slower candidates.

Dictionary size sets a second trade-off. @size_sweep sweeps $N$ at fixed $lambda = 0.1$ and $K = 12$. More entries cover more tokens and raise the aggregate speedup.

#figure(
  image("../abbr_size.png", width: 80%),
  caption: [Aggregate speedup against dictionary size $N$ at $lambda = 0.1$, $K = 12$.],
) <size_sweep>

#figure(
  table(
    columns: (auto, auto),
    inset: 6pt,
    stroke: none,
    align: (center, right),
    fill: (col, row) => if row == 0 { luma(235) },
    [*N*], [*Speedup*],
    [10],   [4.61%],
    [20],   [6.15%],
    [40],   [8.02%],
    [80],   [10.30%],
    [160],  [12.99%],
    [320],  [15.44%],
    [612],  [17.29%],
  ),
  caption: [Speedup versus dictionary size $N$ ($lambda = 0.1$, $K = 12$).
The last row is the largest dictionary reachable, not a requested size.],
) <size-tbl>

Speedup rises steeply, then flattens.
The first 160 entries reach 13.0%.
Adding more raises it to 17.3%, but the dictionary saturates at 612 entries (@size-tbl).
At $lambda = 0.1$ only that many words have an abbreviation that clears every filter, so a request for 640 or 1 280 both return 612.
Beyond this point the limit is the candidate pool, not $N$.

@examples-tbl shows some of the highest-impact entries.
Frequent short words take single-letter abbreviations, and common endings such as `-tion` and `-ing` shorten across many words at once.

#figure(
  table(
    columns: (auto, auto),
    inset: 6pt,
    stroke: none,
    align: (left, center),
    fill: (col, row) => if row == 0 { luma(235) },
    [*Entry*], [*Abbreviation*],
    [`the`],    [`t`],
    [`was`],    [`w`],
    [`you`],    [`u`],
    [`and`],    [`d`],
    [`-ation`], [`an`],
    [`-ted`],   [`e`],
    [`-ing`],   [`in`],
  ),
  caption: [Example entries from the $lambda = 0.1$, $N = 160$ dictionary, among the highest-impact abbreviations. A leading `-` marks a suffix.],
) <examples-tbl>

==== Contribution of the Repeat Key

The two expansion methods can be separated by re-optimizing with some trigger forms switched off (@rpt-tbl).
Every variant uses the same baseline ($lambda = 0.1$, $K = 12$, $N = 160$, LightGBM evaluation).

#figure(
  table(
    columns: (auto, auto),
    inset: 6pt,
    stroke: none,
    align: (left, right),
    fill: (col, row) => if row == 0 { luma(235) },
    [*Configuration*], [*Speedup*],
    [Full dictionary (`TRG` + repeat-key forms)], [12.99%],
    [`TRG` forms only (repeat key unused)],        [12.79%],
    [Double-tap only (no `TRG` key)],              [1.52%],
    [Empty dictionary (repeat regime, no expansions)], [0.26%],
  ),
  caption: [Speedup with the trigger forms restricted, isolating the repeat key's contribution.],
) <rpt-tbl>

The full dictionary reaches 12.99%.
Dropping the two repeat-key forms lowers this to 12.79%, a difference of 0.20 percentage points; the remaining entries are all plain `abbr + TRG` strings.
A dictionary built from double-taps alone reaches 1.52%, as only 23 words have a usable doublechar abbreviation.
An empty dictionary, which still types literal double letters through `RPT`, scores 0.26%.
Most of the speedup comes from the trigger key, not the repeat key.

The `TRG`-only variant still runs on repeat-key hardware, with double letters typed through `RPT`.
A keyboard with no repeat key at all, where `;` was restored and double letters were typed plainly, was built and scored.
It reaches 12.53%, against 12.79% for `TRG`-only and 12.99% for the full dictionary.

==== Contribution of Suffixes

Suffixes are isolated the same way, by re-optimizing with whole words or suffixes dropped (@suffix-tbl).
The baseline is again $lambda = 0.1$, $K = 12$, $N = 160$, LightGBM evaluation.

#figure(
  table(
    columns: (auto, auto),
    inset: 6pt,
    stroke: none,
    align: (left, right),
    fill: (col, row) => if row == 0 { luma(235) },
    [*Configuration*], [*Speedup*],
    [Full dictionary (words + suffixes)], [12.99%],
    [Whole words only],                   [12.33%],
    [Suffixes only],                      [3.18%],
  ),
  caption: [Speedup with whole-word or suffix entries restricted, isolating the suffix contribution.],
) <suffix-tbl>

Suffixes add 0.66 percentage points over a words-only dictionary.
Most of the speedup comes from whole words.

=== Discussion <sec-abbr-discussion>

The method works as a demonstration.
The IKI model picks abbreviations that measurably speed typing, and the optimizer turns thousands of scored candidates into a usable dictionary.
This shows the model can drive abbreviation design.

Speed and intuitiveness trade off measurably, a 3.25-point span across the frontier.
The candidate pool softens the conflict: candidates are generated to look like their words, not at random, so the strategies tend to produce intuitive abbreviations.

The size curve points to a different limit.
The limit on a useful dictionary is not the model, but how many abbreviations a person will learn.
The candidate pool also caps out at 612 entries at this floor: beyond that, more candidates are needed, for example by lowering the intuitiveness floor.

The optimizer is crude. But it limits the dictionary's quality, not the trustworthiness of the speedup number. Its shortcuts are several:

- *Vocabulary cap.* Only the 3 000 most frequent words and 500 suffixes are eligible, so a gain from a rarer entry is never captured.
- *Candidate pool.* Each entry keeps at most 8 000 candidates from seven strategies, a small slice of the possible abbreviations, so a better abbreviation that no strategy proposes is never scored.
- *Shortlist.* Each word offers only its $K = 12$ best candidates to the optimizer. That cutoff is tuned at one setting, not proven optimal, and its best value may shift with dictionary size and $lambda$.
- *Sampled context.* Savings come from a few sampled contexts, not full text.
- *Fixed inputs.* Candidate scores are fixed and trigger frequencies held synthetic to keep the program linear; the real frequencies from evaluation are never fed back to the optimizer.
- *Suffix overlap.* The double-counting penalty over-estimates the overlap, so the optimizer can discard an overlapping suffix pair that the true optimum would keep.
- *Noise.* The uncertainty penalty curbs noisy candidates but does not remove them, and $k_sigma$ is an arbitrary one standard deviation.

A direct search could instead optimize the true objective in full context. Simulated annealing or a genetic algorithm would work. This drops the simplifications, but gives up the optimality guarantee and costs far more compute.

Two assumptions, unlike the optimizer's crude shortcuts, can push the measured number too high.
The largest is the frequency assumption.
When an abbreviation is scored, its word-frequency feature is set to the value of the word it replaces, not of the abbreviation itself.
This assumes the typist recalls the mapping at once, as if it were learned as well as any common word.
In reality a typist pauses to recall a mapping, more so for rare or odd entries.
The measured speedups are therefore likely too high.
Modeling this needs behavioral data and is left for future work.

The second is the trigger key's hand assignment.
It inherits the spacebar's ambidextrous assignment, since the model was trained with the spacebar typed by either thumb.
A dedicated trigger bound to one thumb could cost slightly more than a space, so the reported speedup is optimistic on this count.
A fixed-hand assignment was rejected because it would place the key outside the model's training distribution.

The keys' slots are a further fixed assumption, but a limitation rather than a bias.
They are placed by hand at plausible slots, not optimized.
The repeat key adds only 0.20 percentage points, so its placement barely affects the result.
It could be dropped for a simpler one-key device at almost no cost.
The trigger key is different.
It carries almost the whole speedup, so its placement and cost matter more.
The trigger placement here is plausible but unoptimized, and finding a better one is left open.
The current trigger key placement is not hypothetical: split-spacebar keyboards already put a second thumb key in this spot.

Intuitiveness is the softest part.
There is no objective measure of how memorable an abbreviation is, so the heuristic encodes one person's judgment.
An LLM scorer was tried but did not work: its scores were too inconsistent.

The dictionary is also unsystematic.
Each abbreviation is chosen on its own, so related words need not get related forms. `"panda"` may shorten to `"pa"`, yet `"pandas"` need not become `"pas"`. The typist has no rule to generalize from and must memorize each entry separately.
Memorizing many unrelated mappings is itself costly. The intuitiveness heuristic scores each abbreviation alone, so it misses this set-wide cost. The burden grows with dictionary size.
A systematic scheme, with fixed rules from words to abbreviations, may be easier to learn, but would give up per-word speed tuning.
A speed model would still help there: it could rank which rules save the most time, or reserve unsystematic, speed-optimized forms for the few highest-frequency words, where the gain is largest.

Whole words outperform suffixes by far, 12.33% against 3.18%.
Which points past single words to multi-word phrases.
A phrase like "red panda" could collapse to `rp`, or `btw` to "by the way".
Extending the candidate pool to phrases is left for future work.

=== Conclusion

Trained IKI models can drive abbreviation-dictionary design end to end.
Judged by a separate model, the speedup runs from about 5% at 10 entries to 17% at 612 entries, an upper bound that assumes instant recall of each mapping.
Whole words carry most of it; suffixes add under a point.
Size and required intuitiveness set where a dictionary lands in that range.