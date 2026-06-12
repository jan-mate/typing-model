# How to Run

## Setup
```bash
pip install -r requirements.txt
```

## Data
- **Quick (eval only):** download the enriched parquet files from the [Releases page](https://github.com/jan-mate/typing-model/releases) into `data/enriched/`. With the committed `trained_models/`, this runs evaluation with no preprocessing or training.
- **Full rebuild:**
  - Keystrokes: download the [Aalto dataset](https://userinterfaces.aalto.fi/136Mkeystrokes/); put `metadata_participants.txt` in `data/raw/` and the `*_keystrokes.txt` files in `data/raw/files/`.
  - Text corpus: `python src/preprocess_data/download_expanded_corpus.py`

## Pipeline (build training data)
```bash
python src/preprocess_data/calculate_frequencies.py
python src/preprocess_data/pipeline.py   # edit CONFIG inside to switch QWERTY/Dvorak
```
Outputs `data/enriched/enriched_with_folds.parquet`.

## Training (Colab, needs a GPU)
Open `colab.ipynb`: it mounts Drive, clones the repo, and runs each training script in `src/train_models/{linear_regression,lgbm,mlp}/`. Upload `data/enriched/` to `MyDrive/typing-model/enriched_data/` first. Every script is standalone: `python src/train_models/<model>/<script>.py`. Models save to `trained_models/`.

## Testing
```bash
pytest tests/ --ignore=tests/preprocess_data/   # preprocess tests need the raw 18GB keystrokes
```

## Evaluation
Each `eval/*.py` is standalone and prints results, run locally or via the matching cell in `colab.ipynb`:
```bash
python eval/<script>.py
```
Uses the committed `trained_models/`; needs only the enriched data (see Data).

## Applications
- **Abbreviation dictionary** (`src/abbr_dict/`), in order: `extract_vocabulary.py` → `generate_candidates.py` → `intuitiveness.py` → `precompute_speed.py` (Colab) → `optimize.py`.
- **One-shot shift:** `python src/oneshot_shift/main.py`

## Figures & report
```bash
python src/visualization/generate_all_html.py    # writes figures to report/
typst compile --root report report/main.typ report/machine-learning-model-of-iki-for-eval-typing-systems-anon.pdf
```