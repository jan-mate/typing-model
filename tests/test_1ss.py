import numpy as np
import pandas as pd
from src.oneshot_shift.transformer import OneShotShiftTransformer
from src.oneshot_shift.enricher import OneShotShiftEnricher
from src.enrichment.engine import EnrichmentEngine

def test_transformer_isolated_caps():
    transformer = OneShotShiftTransformer()
    assert transformer.transform("Cat") == "\x02cat"
    assert transformer.transform("The cat") == "\x02the cat"
    assert transformer.transform("A dog") == "\x02a dog"
    assert transformer.transform("I am.") == "\x02i am."

def test_transformer_consecutive_caps():
    transformer = OneShotShiftTransformer()
    assert transformer.transform("NASA") == "NASA"
    assert transformer.transform("USA is big") == "USA is big"
    assert transformer.transform("HTML") == "HTML"

def test_transformer_mixed():
    transformer = OneShotShiftTransformer()
    # T is isolated, NASA is not, C is isolated.
    assert transformer.transform("The NASA Cat") == "\x02the NASA \x02cat"

def test_transformer_punctuation():
    transformer = OneShotShiftTransformer()
    # At least for now, only [A-Z] are handled as 1SS.
    # If the user wants punctuation shifted via 1SS, the pattern needs expansion.
    assert transformer.transform("Wow!") == "\x02wow!"

def test_enricher_word_frequency_override():
    # Mock engine and word map
    engine = EnrichmentEngine()
    word_map = {"cat": 5.5, "the": 4.2}
    enricher = OneShotShiftEnricher(engine, word_map)
    
    # " \x02cat" (Space, 1SS, c, a, t)
    # Key IDs: 32, 2, 99, 97, 116
    ids = np.array([32, 2, 99, 97, 116])
    
    freqs = enricher._get_1ss_word_frequencies(ids)
    
    # Index 0 is a space delimiter -> left as NaN (skipped by the word scan)
    # Indices 1, 2, 3, 4 are the \x02cat word -> all 5.5
    assert np.isnan(freqs[0])
    assert np.all(freqs[1:] == 5.5)

def test_enricher_shift_is_zero():
    from src.utils.layout_utils import build_layout_and_engine
    # We need real layout paths for this
    layout_paths, engine, _ = build_layout_and_engine(".", rpt_key=True)
    word_map = {"cat": 5.5}
    enricher = OneShotShiftEnricher(engine, word_map)
    
    text = "\x02cat"
    keys = [ord(c) for c in text]
    df_raw = pd.DataFrame({
        'PARTICIPANT_ID': ['S']*len(keys), 'SEQUENCE_ID': ['T']*len(keys), 'ORIGINAL_SEQUENCE_ID': ['T']*len(keys),
        'KEY_ID': keys, 'iki': [0.0]*len(keys), 'iki_z': [0.0]*len(keys), 'iki_log_z': [0.0]*len(keys)
    })
    
    enriched, _ = enricher.enrich(df_raw, layout_paths)
    
    # shift feature should be 0 for all characters in \x02cat
    # In Baseline "Cat", 'C' would have shift=1.
    assert np.all(enriched['shift'] == 0.0)
    # word_frequency should be 5.5 for all
    assert np.all(enriched['word_frequency'] == 5.5)
