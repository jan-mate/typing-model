= Introduction
Faster typing can increase productivity.
This project builds a machine learning model to predict keyboard typing speed, providing a way to evaluate text input systems.

Many hypotheses have been proposed about what affects typing speed.
Some are about typing system design: alternative layouts such as Dvorak and Colemak, abbreviation expansion, chording, dead keys, one-shot modifiers, layer keys, and repeat keys.
Others are about specific motor patterns: same-finger bigrams, finger travel distance, hand alternation, and rolling movements.

Testing these hypotheses empirically is difficult because it is hard to control for familiarity.
Almost everyone is already familiar with standard QWERTY without advanced features such as repeat keys or one-shot modifiers.
Any trial involving a novel layout or input mechanism confounds the system's intrinsic effect with the practice needed to learn it.
Learning times vary considerably by system, but can take hundreds of hours @anderson2009 @strong1956.

A predictive model bypasses this learning confound.
Trained on real keystroke data, the model evaluates typing speed on novel sequences without requiring any participant to type them.
The model targets skilled, fast typists, because the systems it evaluates, such as abbreviation expansion, are designed for people who want to type fast.
Such a model has several uses:

*Testing the above hypotheses.* The model estimates the speed benefit of typing system features, without requiring participants to learn them.

*Layout optimization.* A model of what determines typing speed could enable optimizing keyboard layouts for speed, without recruiting participants.

*Abbreviation expansion.* Text expansion systems let users type a short sequence that automatically expands to a full word or phrase.

*Research normalization.* Experiments often compare typing speed across different text.
Some text is simply harder to type than other text.
The model measures this difficulty, so researchers can control for it @williams2026.

The project has two parts.
The first builds and validates the predictive model.
The second applies it in four studies.
Each study tests one way to type faster.
These come from the designs listed above: two new keys, an abbreviation dictionary, and a different keyboard layout.
Each is meant to speed up typing, but its real effect has been hard to measure.
The model can now estimate it:

+ Does a one-shot shift key --- tapped once to shift only the next keystroke, instead of being held down --- reduce typing time?
+ Does a repeat key --- a key that re-emits the previous keystroke --- reduce typing time?
+ Does an abbreviation dictionary reduce typing time, and by how much?
+ Is Dvorak faster than QWERTY?

The project also examines whether the model can guide layout optimization.

