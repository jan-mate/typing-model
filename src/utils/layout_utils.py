import os
from src.enrichment.engine import EnrichmentEngine


def build_layout_and_engine(data_root: str, rpt_key: bool, rpt_variant: str = "trg"):
    # rpt_key picks the layout/frequencies with the RPT/TRG keys; rpt_variant ("trg"
    # semicolon or "j") only matters when rpt_key=True
    if rpt_key:
        layout_map_name = (
            "layout_map_rpt_j" if rpt_variant == "j" else "layout_map_rpt_trg"
        )
        layout_paths = {
            "layout_path":     os.path.join(data_root, "data/layouts/qwerty_us_rpt_trg.json"),
            "layout_map_path": os.path.join(data_root, f"data/layouts/{layout_map_name}.json"),
            "shifts_path":     os.path.join(data_root, "data/layouts/shifts_us.json"),
        }
        engine_config = {
            "unigrams_path":          os.path.join(data_root, "data/frequencies/unigrams_zipf_rpt_trg.json"),
            "bigrams_path":           os.path.join(data_root, "data/frequencies/bigrams_zipf_rpt_trg.json"),
            "words_path":             os.path.join(data_root, "data/frequencies/words_zipf.json"),
            "movement_features_path": os.path.join(data_root, "data/layouts/movement_features_rpt_trg.json"),
        }
        bigrams_file = os.path.join(data_root, "data/frequencies/bigrams_zipf_rpt_trg.json")
    else:
        layout_paths = {
            "layout_path":     os.path.join(data_root, "data/layouts/qwerty_us.json"),
            "layout_map_path": os.path.join(data_root, "data/layouts/layout_map.json"),
            "shifts_path":     os.path.join(data_root, "data/layouts/shifts_us.json"),
        }
        engine_config = {
            "unigrams_path":          os.path.join(data_root, "data/frequencies/unigrams_zipf.json"),
            "bigrams_path":           os.path.join(data_root, "data/frequencies/bigrams_zipf.json"),
            "words_path":             os.path.join(data_root, "data/frequencies/words_zipf.json"),
            "movement_features_path": os.path.join(data_root, "data/layouts/movement_features.json"),
        }
        bigrams_file = os.path.join(data_root, "data/frequencies/bigrams_zipf.json")

    engine = EnrichmentEngine(**engine_config)
    return layout_paths, engine, bigrams_file
