== Validation set and data splitting
Data splitting determines whether a model learns layout-independent motor costs or memorizes QWERTY timings.
Because the training data is overwhelmingly QWERTY, a naive split causes the model to overfit to specific letter sequences.
This split is an important design choice for cross-layout generalization.

=== Choosing the validation unit
The split must prevent data leakage at the word and bigram levels.
Leakage occurs if the model sees a sequence during training and encounters it again during validation.

*Word-level leakage versus sentences.*
Typing is chunk-based @salthouse1986, and whole words approximate these chunks well.
Sentence-level splitting is less effective because sentences share common words.
Training on "The arctic fox" and validating on "The fennec fox" leaks the timings for the words "The" and "fox".
Holding out whole words yields finer-grained validation units than sentences, and guarantees no validation word appeared in training.

*Bigram-level leakage.*
Even with unseen words, models can leak information because bigrams are memorized across different words.
If the model sees the bigram "th" in thousands of QWERTY instances of "the", it memorizes that specific motor rhythm.
It can then apply this rhythm when predicting "th" in an unseen validation word like "that", bypassing biomechanical features.

*Context leakage.*
Conversely, validating only at the bigram level introduces context leakage.
Because typing is chunk-based, models can use the surrounding keystrokes of a word to predict the timing of a held-out bigram.
Because of this context leakage, words serve as a better validation unit than bigrams.
While word-level validation still allows some context leakage from the surrounding keys of unseen words, it leaks less data than validating only at the bigram level.

*Alternative split approaches.*
Creating a validation set that holds out both specific words and specific bigrams simultaneously is infeasible because the validation sets would overlap heavily.
Language is a densely connected graph, and this overlap makes strict cross-validation impossible.
A diverse ensemble needs distinct, non-overlapping validation sets. 10 is a pragmatic choice. It gives enough models for a useful ensemble while leaving each fold ample training data.
Training one model per fold gives 10 models, and the spread of their predictions estimates how uncertain the model is about a given input.

Alternatively, a sentence-level split combined with both word and bigram regularization might have worked better.
However, it would have added complexity to the data pipeline, so it was avoided.

=== The splitting algorithm
The split uses two components: word-based validation folds and bigram-based training dropout.

*Word-level validation folds.*
A greedy bin-packing algorithm divides the dataset into 10 non-overlapping validation folds.
The algorithm sorts 50 000 unique words by descending total frequency.
Processing them in that order, it assigns each word to the fold with the lowest running frequency total.
This process distributes both highly common and rare words evenly across all folds, making each fold representative of the overall frequency distribution.
The algorithm balances the 10 folds to within 0.001% of each other in total dataset rows.
Any words outside the top 50 000 are assigned to a fold randomly.

*Bigram-level dropout.*
The same greedy algorithm assigns 1 865 bigrams to 10 buckets based on descending frequency.
During training on a given fold, every row whose bigram belongs to that fold's bucket is withheld.
Each bucket is one tenth of all bigram occurrences.
The model therefore never sees 10% of them during training.
Combining word-level folds with bigram dropout leaves 81% ($0.9 times 0.9$) of the data for training, enough to learn the biomechanical patterns.

