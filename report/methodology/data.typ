#import "/styles.typ": *

== Data
This project uses two datasets.
The 136M Keystrokes dataset provides the keystroke timings that train and evaluate the IKI model.
A combined Reddit and Wikipedia corpus aims to provide realistic text for frequency features and for evaluating typing systems.

=== 136M Keystrokes dataset
Training uses the Aalto 136M Keystrokes dataset @dhakal2018: over 136 million keystrokes from $168 thin 594$ participants performing transcription tasks.
Each participant transcribed 15 sentences while the system logged key-press and key-release timestamps.
From the original corpora, the dataset's creators chose random sentences containing at least 3 words and at most 70 characters, fewer than five numerical symbols, and only simple punctuation marks (`,`, `.`, `!`, `?`, `'`).
The dataset's scale and coverage of both QWERTY and Dvorak typing motivate its choice for this project.
QWERTY data trains and validates the models; the pipeline holds out the smaller pool of Dvorak data to test cross-layout generalisation.
The two layouts are heavily imbalanced: of the $168 thin 594$ participants, $165 thin 324$ type on QWERTY and only $209$ on Dvorak.
Visual representations of QWERTY, Dvorak, and the other layouts used for evaluation are provided in @appendix-layouts.
@log-sample shows the format of a processed sequence.

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    stroke: none,
    fill: flat,
    [*Key*], [p], [o], [p], [p], [y],
    [*IKI (ms)*], [NaN], [170], [150], [180], [110]
  ),
  caption: [A processed keystroke sequence: each key with the interval since the previous key.],
) <log-sample>

Cross-layout evaluation requires distinguishing a typed character (keycode) from its physical position on the board (keyslot).
The model covers 42 keyslots.
These positions cover the 26 letters, 10 digits, space, and simple punctuation marks (`,`, `.`, `!`, `?`, `'`).
The set includes any keyslot required to type these characters on either QWERTY or Dvorak.
The QWERTY training sentences lack the `;` keycode.
However, Dvorak maps the letter `s` to the `;` keyslot.
The model includes this keyslot to evaluate Dvorak.
Conversely, the model excludes the QWERTY `-`, `[`, and `]` keyslots.
Neither layout places a modeled character on these positions (Dvorak maps them to the unmodeled `[`, `/`, and `=`).
The model does not treat Shift as a separate keystroke.
It encodes Shift as a binary feature on the shifted keyslot, discussed in @data-enrichment.
@keyboard-layout shows the 42 modeled keyslots on a US QWERTY layout.

#figure(
  image("../modeled_keys.png", width: 95%),
  caption: [The 42 keyslots corresponding to the modeled keycodes, shown on a US QWERTY layout.],
) <keyboard-layout>

=== Corpus <sec-corpus>
The IKI model needs realistic text to estimate the character n-gram frequencies used as features.
It also needs realistic text to evaluate typing systems such as abbreviation dictionaries on text people actually write.
The corpus aims to represent what is typed on a keyboard.

It combines two sources in equal parts.
Reddit comments from the Pushshift dataset @baumgartner2020 supply informal, modern writing.
WikiText-103 @merity2016, drawn from Wikipedia articles, supplies formal writing.
Mixing the two spans a range of formality rather than a single register.

The pipeline keeps only Reddit comments from 2023 onward, to reflect modern language.
To avoid any single community dominating, the pipeline samples Reddit sentences across randomly chosen subreddits, with at most $250$ sentences taken from each.

A limitation is that Reddit text is often typed on a phone rather than a physical keyboard, which makes it less representative of keyboard typing.
Including WikiText partly offsets this, since it is formal text likely written on a keyboard.

The pipeline applies cleaning: it strips URLs and characters outside a standard typing set, and a simple filter removes bot, moderator, and spam comments.
The filter is imperfect, so some such text remains.
The pipeline drops single-word fragments, since a sentence of one word carries almost no key-to-key transition signal.

The pipeline splits each text into sentences; each sentence becomes one corpus entry.
Splitting by sentence approximates the larger chunks people type in, since a sentence is a rough unit of fluent typing.
This is not fully realistic: typists sometimes stop mid-sentence, and sometimes type fluently straight across a full stop, so a sentence boundary does not always mark a real pause.

The pipeline filters sentences to 3–70 characters.
It samples $50 thin 000$ sentences from each source, for $100 thin 000$ in total.
