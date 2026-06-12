import os

LOCAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DRIVE_ROOT = os.environ.get("TYPING_MODEL_DRIVE_ROOT", "/content/drive/MyDrive/typing-model")
STORAGE_ROOT = DRIVE_ROOT if os.path.exists(DRIVE_ROOT) else LOCAL_ROOT
DATA_ROOT = LOCAL_ROOT

def _enriched(name: str) -> str:
    flat = os.path.join(STORAGE_ROOT, "data", name)
    nested = os.path.join(STORAGE_ROOT, "data", "enriched", name)
    return flat if os.path.exists(flat) else nested


ENRICHED_DATA_PATH = _enriched("enriched_with_folds.parquet")
DVORAK_DATA_PATH = _enriched("enriched_with_folds_dvorak.parquet")

CORPUS_PATH = os.path.join(STORAGE_ROOT, "data/combined_corpus.parquet")


def model_dir(model_name: str, subdir: str = "ensemble") -> str:
    # models live under models/ on Drive, trained_models/ locally
    artifacts_dir = "models" if STORAGE_ROOT == DRIVE_ROOT else "trained_models"
    return os.path.join(STORAGE_ROOT, artifacts_dir, model_name, subdir)
