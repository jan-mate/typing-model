import numpy as np
import pandas as pd
from src.enrichment.engine import EnrichmentEngine

class OneShotShiftEnricher:
    ONE_SHOT_CHAR_ORD = 2  # \x02

    def __init__(self, engine: EnrichmentEngine, word_map: dict):
        self.engine = engine
        self.word_map = word_map

    def enrich_layout(self, base_dict, layout_path, layout_map_path, shifts_path, features=None):
        enriched, encoded = self.engine.enrich_layout(base_dict, layout_path, layout_map_path, shifts_path, features)
        enriched['word_frequency'] = self._get_1ss_word_frequencies(encoded)
        return enriched, encoded

    def enrich(self, df_raw: pd.DataFrame, layout_paths: dict, n_pads: int = 0, features: list = None):
        enriched, encoded = self.engine.enrich(df_raw, **layout_paths, n_pads=n_pads, features=features)
        enriched['word_frequency'] = self._get_1ss_word_frequencies(encoded)
        return enriched, encoded

    def _get_1ss_word_frequencies(self, ids):
        # word frequency lookup that ignores the 1SS character
        freqs = np.full(len(ids), np.nan, dtype=np.float32)
        delimiters = {0, 32, 46, 44, 33, 63} # PAD, space, . , ! ?

        i = 0
        while i < len(ids):
            if ids[i] in delimiters:
                i += 1
                continue

            start = i
            while i < len(ids) and ids[i] not in delimiters:
                i += 1
            end = i

            # strip the 1SS character before the lookup
            word_chars = []
            for j in range(start, end):
                if ids[j] != self.ONE_SHOT_CHAR_ORD:
                    word_chars.append(chr(ids[j]).lower())
            
            word = "".join(word_chars)
            val = self.word_map.get(word, 0.0)
            freqs[start:end] = val

        return freqs
