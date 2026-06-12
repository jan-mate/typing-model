# streaming filter for large speed_savings.json files
import json
import os
import sys
from decimal import Decimal

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.abbr_dict.config import data_path

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

INPUT_PATH = data_path("speed_savings.json")
OUTPUT_PATH = data_path("speed_savings_filtered.json")

THRESHOLD = 0.25

# model output is a sum of per-keystroke z-scores. ALPHA converts one saved keystroke
# into those same z-units (savings_ms = length_diff*IKI_MEAN + savings_z*IKI_STD), so the
# length term z-standardization removed can be added back
IKI_MEAN_MS = 110.5
IKI_STD_MS  = 50.4
ALPHA       = IKI_MEAN_MS / IKI_STD_MS  # ≈ 2.19

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        return

    print(f"Loading (Streaming): {INPUT_PATH}")
    print(f"Outputting to:       {OUTPUT_PATH}")
    
    total_before = 0
    total_after = 0
    
    try:
        import ijson
    except ImportError:
        print("Installing ijson...")
        import subprocess
        subprocess.check_call([os.sys.executable, "-m", "pip", "install", "ijson"])
        import ijson

    filtered_flat_items = []
    
    with open(INPUT_PATH, "r") as f:
        word_items = ijson.items(f, 'items.item')

        for word_item in word_items:
            word_text = word_item["text"]
            word_type = word_item["type"]
            freq = word_item["frequency"]

            # 
            for cand in word_item.get("candidates", []):
                total_before += 1

                s = cand.get("savings", {})
                dl_z    = float(s.get("dl",   {}).get("mean", 0))
                lgbm_z  = float(s.get("lgbm", {}).get("mean", 0))
                # ks_save is deterministic, so combined-savings std equals the model std
                dl_std   = float(s.get("dl",   {}).get("std", 0))
                lgbm_std = float(s.get("lgbm", {}).get("std", 0))
                # +1 because the trigger replaces the trailing word-boundary space
                ks_save = float(s.get("keystrokes", {}).get("mean", 0)) + 1

                # combine the timing saving (model z-sum) with the keystroke-count saving,
                # converted to the same z-units by ALPHA
                dl_combined   = ks_save * ALPHA + dl_z
                lgbm_combined = ks_save * ALPHA + lgbm_z

                if max(dl_combined, lgbm_combined) >= THRESHOLD:
                    flat_item = {
                        "text": word_text,
                        "type": word_type,
                        "frequency": freq,
                        "abbr": cand["abbr"],
                        "trigger_form": cand["trigger_form"],
                        "intuitiveness": cand.get("intuitiveness", 0),
                        "savings_z_dl":       dl_combined,
                        "savings_z_lgbm":     lgbm_combined,
                        "savings_z_dl_std":   dl_std,
                        "savings_z_lgbm_std": lgbm_std,
                        "savings_keystrokes": ks_save,
                        "savings_z_mean":     (dl_combined + lgbm_combined) / 2,
                    }
                    filtered_flat_items.append(flat_item)
                    total_after += 1
            
            if total_before % 10000 == 0 and total_before > 0:
                print(f"Processed {total_before} candidates... (kept {total_after})")

    print(f"Filtering complete. Kept {total_after}/{total_before} candidates.")
    
    output_data = {
        "metadata": {
            "threshold": THRESHOLD,
            "total_before": total_before,
            "total_after": total_after,
            "note": "Flattened and filtered using filter_streaming.py"
        },
        "items": filtered_flat_items
    }
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2, cls=DecimalEncoder)
    
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
