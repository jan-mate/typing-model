#import "/styles.typ": *

== Data enrichment <data-enrichment>
Each keystroke is enriched with features computed from the typed text.
The features split into two groups: motor features (key position, movement, finger and hand use) and linguistic features (frequency, word and syllable structure).

Most motor features assume standard touch typing: each key is always pressed by a fixed finger, with fingers resting on the home row `asdf hjk;`.
@finger-map shows the assumed finger assignment.
Space is treated as `hand=2` (ambidextrous) since typists press it with either thumb.

#figure(
  image("../finger.png"),
  caption: [Assumed touch-typing finger map. Colors indicate finger type; the dashed line marks the left/right hand boundary.],
) <finger-map>

@features-sample shows a sample of enriched features for one word.

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    inset: 5pt,
    align: center + horizon,
    stroke: none,
    fill: zebra,

    [*Key*], [S], [q], [u], [i], [r], [r], [e], [l],
    [*x*], [1.00], [-0.25], [3.25], [2.25], [2.75], [2.75], [1.75], [1.00],
    [*y*], [0.00], [1.00], [1.00], [1.00], [1.00], [1.00], [1.00], [0.00],
    [*Hand*], [0], [0], [1], [1], [0], [0], [0], [1],
    [*Repetition*], [--], [0], [0], [0], [0], [1], [0], [0],
    [*Out roll*], [--], [1], [0], [1], [0], [0], [1], [0],
  ),
  caption: [Selected enriched features for "Squirrel" on QWERTY.],
) <features-sample>

=== Spatial position (x, y)
`x` and `y` encode a key's position on the layout.
@position-map shows the values assigned to each key on QWERTY.

#figure(
  image("../position.png"),
  caption: [Key positions on QWERTY. `x` is symmetric across hands; `y=0` is the home row.],
) <position-map>

`y=0` is the home row, `y=1` the top row, `y=-1` the bottom row, and `y=2` the number row.
Both `x` and `y` are encoded as continuous features.
The space bar sits at `y=-2`; this could arguably be treated as a distinct category rather than simply two rows below home.

`x` uses a symmetric encoding, mirrored across the two hands: the pinky home key on each hand receives `x=0`, increasing toward the index finger and turning negative for keys beyond the pinky.
"a" (left pinky) and ";" (right pinky) therefore share the coordinate `x=0`.
Cross-layout generalisation is the main motivation for symmetric encoding: ";" does not appear in training data, so without it the model would see no examples of the right-pinky home row position.
On Dvorak, "s" sits on the ";"-slot and receives `x=0`, so the model applies what it learned from "a" directly.

=== Movement (move_dist, move_sin, move_cos)
Three features encode how far and in what direction the finger moves to reach each key.
Movement is a bigram feature: each key's movement is measured relative to the previous key, so the first key of a sequence has no movement value.

The starting position depends on whether the same finger typed the previous key.
For a same-finger bigram — one finger types two keys in a row, e.g. "rt" (both left index) — the finger is still at the previous key, so `move_dist` is the Euclidean distance from "r" to "t" in key-grid coordinates.
For a different-finger bigram — e.g. "er" (left middle then left index) — the current finger starts from its home key: `move_dist` for "r" is the distance from "f" to "r", not from "e" to "r".

`move_sin` and `move_cos` are the sine and cosine of the movement angle.
The angle is split into two components so that similar directions have close values: a raw angle would treat 1° and 359° as opposite extremes despite being almost the same movement.

=== Hand, finger, and finger type
`hand` is 0 for the left hand, 1 for the right hand, and 2 for space.

`finger` encodes which of the ten fingers presses a key (0 = left pinky, 1 = left ring, 4 = left thumb, 5 = right thumb, 9 = right pinky).
`finger_type` collapses this to the finger category: pinky, ring, middle, index, or thumb.
Both are derived from @finger-map.

All three are categorical, not ordinal.
The linear and MLP models receive them one-hot encoded.
LightGBM uses them as native categorical features.

=== Shift
`shift` flags a character that requires the Shift modifier (e.g. "A", "!").
"A" and "a" share all position and finger features but differ on `shift`.
A shift flag is used rather than a dedicated Shift-key column because the data does not record which hand pressed Shift.

=== Bigram and trigram indicators
These binary features flag motor patterns across two or three consecutive keys.

- *same_hand* — the bigram uses the same hand (e.g. "hi").
- *same_finger* — the bigram uses the same finger (e.g. "nu").
- *same_finger_trigram* — all three keys of the trigram use the same finger (e.g. "num").
- *same_finger_skipgram* — the first and last keys share a finger and are different keys, while the middle key uses a different finger (e.g. "fur": "f" and "r" are both left index; "u" is right index).
- *repetition* — the same key is pressed twice in a row (e.g. "rr").
- *skipgram_repetition* — the same key appears at the first and last position of a trigram with a different key in between (e.g. "UwU").
- *double_row_jump* — a same-hand bigram with one key on the bottom row (`y=-1`) and one on the top or number row (`y≥1`) (e.g. "cr").

=== Rolls
A roll is a same-hand sequence of keys pressed by fingers moving in one consistent direction.
An inward roll moves toward the index finger (pinky → index); an outward roll moves toward the pinky (index → pinky).

- *in_roll* — a same-hand bigram rolling inward (e.g. "at", "ok").
- *out_roll* — a same-hand bigram rolling outward (e.g. "ca", "ts").
- *in_triroll* — a same-hand trigram rolling inward continuously (e.g. "asd").
- *out_triroll* — a same-hand trigram rolling outward continuously (e.g. "hil").
- *redirects* — a same-hand trigram where the roll direction reverses (e.g. "cat": "ca" rolls outward, "at" rolls inward).

=== Scissors
`scissors` flags a same-hand bigram where adjacent fingers — pinky, ring, or middle (not index or thumb) — reach across two or more rows (e.g. "ex", ",o").

=== Frequency features
Three features encode how often a character or sequence appears in text.
Unigram and bigram frequencies are computed from a text corpus; word frequencies use the `wordfreq` library.
Character and word frequencies follow a power law: a small number of items are very common and most are rare.
Raw counts would give a highly skewed distribution; the Zipf scale, $log_(10)("count" / "total") + 9$, compresses this into a bounded, more uniform range that is easier for a model to learn from.

- *unigram_frequency* — Zipf frequency of the single character.
- *bigram_frequency* — Zipf frequency of the two-character sequence, recorded at the second character.
- *word_frequency* — Zipf frequency of the whole word, with word boundaries at punctuation and whitespace.

=== Word and syllable position
These features encode where a character falls within its word and syllable.
Word boundaries come from punctuation and whitespace; syllable boundaries are estimated by the `pyphen` library (e.g. "squirrel" → "squir-rel").

- *word_index*, *word_length* — the character's position within the word and the word's total length.
- *word_relative_pos* — progress through the word, normalized from 0.0 to 1.0.
- *is_word_start*, *is_word_end* — binary flags marking the first and last character of a word.
- *is_syllable_start*, *is_syllable_end* — binary flags marking the first and last character of each estimated syllable.

=== Sequence position
These features encode where a character falls within the whole typed sequence.

- *sequence_pos* — the character's position in the sequence.
- *sequence_length* — the total number of characters in the sequence.
- *sequence_relative_pos* — progress through the sequence, normalized from 0.0 to 1.0.

