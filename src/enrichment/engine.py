import json
import numpy as np
import pandas as pd
from src.enrichment.features import (
    finger, hand, space_coords, shift, frequency, 
    finger_type, movement, same_finger_bigram,
    same_hand_bigram, same_finger_trigram, word_frequency, 
    in_roll, out_roll, repetition, skipgram_repetition, 
    same_finger_skipgram, in_triroll, out_triroll, redirects, 
    double_row_jump, scissors, sequence_position, word_position, 
    syllable_position
)

FEATURE_REGISTRY = {
    "finger": finger.get_map,
    "finger_type": finger_type.get_map,
    "hand": hand.get_map,
    "coords": space_coords.get_map,
    "shift": shift.get_map
}

PAD_VALUES = {
    "move_dist": -1.0, "move_sin": -2.0, "move_cos": -2.0,
    "coords": -5.0,    "x": -5.0, "y": -5.0, "finger": -1.0, "finger_type": -1.0, "hand": -1.0, "shift": -1.0,
    "unigram_frequency": -1.0, "bigram_frequency": -1.0, "word_frequency": -1.0,
    "same_hand": -1.0, "same_finger": -1.0, "same_finger_trigram": -1.0,
    "repetition": -1.0, "skipgram_repetition": -1.0, "same_finger_skipgram": -1.0,
    "in_roll": -1.0, "out_roll": -1.0, "in_triroll": -1.0, "out_triroll": -1.0,
    "redirects": -1.0, "double_row_jump": -1.0, "scissors": -1.0,
    "sequence_pos": -1.0, "sequence_length": -1.0, "sequence_relative_pos": -1.0,
    "word_index": -1.0, "word_length": -1.0, "word_relative_pos": -1.0,
    "is_word_start": -1.0, "is_word_end": -1.0,
    "is_syllable_start": -1.0, "is_syllable_end": -1.0
}

class EnrichmentEngine:
    def __init__(self, unigrams_path=None, bigrams_path=None, words_path=None, movement_features_path=None):
        self.uni_map = frequency.load_unigram_map(unigrams_path) if unigrams_path else None
        self.bi_matrix = frequency.load_bigram_map(bigrams_path) if bigrams_path else None
        self.word_map = word_frequency.load_map(words_path) if words_path else None
        self.move_data = None
        if movement_features_path:
            with open(movement_features_path, 'r') as f:
                self.move_data = json.load(f)

    def _build_lookups(self, layout_data, layout_map, shifts_data, selected_features):
        keycode_to_slot = {item['keycode']: item['slot'] for item in layout_data}
        shift_map = {item['shift']: item['base'] for item in shifts_data}
        tables = {name: FEATURE_REGISTRY[name](keycode_to_slot, layout_map, shift_map) for name in selected_features if name in FEATURE_REGISTRY}
        
        if self.uni_map is not None:
            tables["unigram_frequency"] = self.uni_map
        
        finger_1d = finger._get_base_map(keycode_to_slot, layout_map, shift_map)
        hand_1d = hand._get_base_map(keycode_to_slot, layout_map, shift_map)
        
        sf_matrix = same_finger_bigram.get_matrix(finger_1d) if "finger" in tables else None
        sh_matrix = same_hand_bigram.get_matrix(hand_1d) if "hand" in tables else None
        sft_matrix = same_finger_trigram.get_matrix(finger_1d) if "finger" in tables else None
        
        sf_skip_matrix = same_finger_skipgram.get_matrix(finger_1d) if "finger" in tables else None
        
        in_roll_matrix, out_roll_matrix, in_triroll_matrix, out_triroll_matrix = None, None, None, None
        redir_matrix, drj_matrix, scissor_matrix = None, None, None
        
        if "finger" in tables and "hand" in tables:
            in_roll_matrix = in_roll.get_matrix(finger_1d, hand_1d)
            out_roll_matrix = out_roll.get_matrix(finger_1d, hand_1d)
            in_triroll_matrix = in_triroll.get_matrix(finger_1d, hand_1d)
            out_triroll_matrix = out_triroll.get_matrix(finger_1d, hand_1d)
            redir_matrix = redirects.get_matrix(finger_1d, hand_1d)
            
        if "coords" in tables and "hand" in tables:
            drj_matrix = double_row_jump.get_matrix(tables["coords"], hand_1d)
            if "finger" in tables:
                scissor_matrix = scissors.get_matrix(finger_1d, hand_1d, tables["coords"])

        move_matrix = None
        if self.move_data:
            move_matrix = movement.get_matrix(keycode_to_slot, self.move_data["bigram"])
            
        return tables, move_matrix, sf_matrix, sh_matrix, sft_matrix, in_roll_matrix, out_roll_matrix, sf_skip_matrix, in_triroll_matrix, out_triroll_matrix, redir_matrix, drj_matrix, scissor_matrix

    def enrich_linguistics(self, df, n_pads=8, target_cols=['iki_z', 'iki_log_z'], show_progress=False):
        res = {
            "participant_id": [], "sequence_id": [], "original_sequence_id": [], "key_ids": [],
            "sequence_pos": [], "sequence_length": [], "sequence_relative_pos": [],
            "word_frequency": [], "word_idx": [], "word_length": [], "word_relative_pos": [],
            "is_word_start": [], "is_word_end": [], "is_syllable_start": [], "is_syllable_end": [],
            "repetition": [], "skipgram_repetition": [], "bigram_frequency": [],
            "targets": {t: [] for t in target_cols}
        }
        
        groups = df.groupby('SEQUENCE_ID', sort=False)
        if show_progress:
            from tqdm import tqdm
            groups = tqdm(groups, desc="Linguistic Setup")
            
        rep_matrix = repetition.get_matrix()
        skip_rep_matrix = skipgram_repetition.get_matrix()

        for _, group in groups:
            ids = group['KEY_ID'].values
            seq_len = len(ids)
            p_id = str(group['PARTICIPANT_ID'].iloc[0])
            s_id = str(group['SEQUENCE_ID'].iloc[0])
            os_id = str(group['ORIGINAL_SEQUENCE_ID'].iloc[0])
            
            res["participant_id"].extend(["[PAD]"] * n_pads + [p_id] * seq_len)
            res["sequence_id"].extend(["[PAD]"] * n_pads + [s_id] * seq_len)
            res["original_sequence_id"].extend(["[PAD]"] * n_pads + [os_id] * seq_len)
            
            res["key_ids"].append(np.concatenate([np.zeros(n_pads, dtype=np.uint8), ids]))

            seq_pos, seq_len_arr, seq_rel_pos = sequence_position.get_features(seq_len)
            res["sequence_pos"].append(np.concatenate([np.full(n_pads, PAD_VALUES["sequence_pos"], dtype=np.float32), seq_pos]))
            res["sequence_length"].append(np.concatenate([np.full(n_pads, PAD_VALUES["sequence_length"], dtype=np.float32), seq_len_arr]))
            res["sequence_relative_pos"].append(np.concatenate([np.full(n_pads, PAD_VALUES["sequence_relative_pos"], dtype=np.float32), seq_rel_pos]))
            
            wf = word_frequency.get_word_frequencies(ids, self.word_map) if self.word_map else np.full(seq_len, PAD_VALUES["word_frequency"], dtype=np.float32)
            res["word_frequency"].append(np.concatenate([np.full(n_pads, PAD_VALUES["word_frequency"], dtype=np.float32), wf]))
            
            w_idx, w_len, w_rel_pos, w_start, w_end = word_position.get_features(ids)
            s_start, s_end = syllable_position.get_features(ids)
            res["word_idx"].append(np.concatenate([np.full(n_pads, PAD_VALUES["word_index"], dtype=np.float32), w_idx]))
            res["word_length"].append(np.concatenate([np.full(n_pads, PAD_VALUES["word_length"], dtype=np.float32), w_len]))
            res["word_relative_pos"].append(np.concatenate([np.full(n_pads, PAD_VALUES["word_relative_pos"], dtype=np.float32), w_rel_pos]))
            res["is_word_start"].append(np.concatenate([np.full(n_pads, PAD_VALUES["is_word_start"], dtype=np.float32), w_start]))
            res["is_word_end"].append(np.concatenate([np.full(n_pads, PAD_VALUES["is_word_end"], dtype=np.float32), w_end]))
            res["is_syllable_start"].append(np.concatenate([np.full(n_pads, PAD_VALUES["is_syllable_start"], dtype=np.float32), s_start]))
            res["is_syllable_end"].append(np.concatenate([np.full(n_pads, PAD_VALUES["is_syllable_end"], dtype=np.float32), s_end]))

            b_freqs = np.full(seq_len, PAD_VALUES["bigram_frequency"], dtype=np.float32)
            rep_vals = np.full(seq_len, PAD_VALUES["repetition"], dtype=np.float32)
            skip_rep_vals = np.full(seq_len, PAD_VALUES["skipgram_repetition"], dtype=np.float32)

            for i in range(1, seq_len):
                p, c = ids[i-1], ids[i]
                if p < 128 and c < 128:
                    if self.bi_matrix is not None: b_freqs[i] = self.bi_matrix[p, c]
                    rep_vals[i] = rep_matrix[p, c]
                    if i >= 2:
                        pp = ids[i-2]
                        if pp < 128:
                            skip_rep_vals[i] = skip_rep_matrix[pp, p, c]
                            
            res["bigram_frequency"].append(np.concatenate([np.full(n_pads, PAD_VALUES["bigram_frequency"], dtype=np.float32), b_freqs]))
            res["repetition"].append(np.concatenate([np.full(n_pads, PAD_VALUES["repetition"], dtype=np.float32), rep_vals]))
            res["skipgram_repetition"].append(np.concatenate([np.full(n_pads, PAD_VALUES["skipgram_repetition"], dtype=np.float32), skip_rep_vals]))

            for t in target_cols:
                res["targets"][t].append(np.concatenate([np.full(n_pads, np.nan, dtype=np.float32), group[t].values.astype(np.float32)]))

        base_dict = {
            "participant_id": np.array(res["participant_id"]),
            "sequence_id": np.array(res["sequence_id"]),
            "original_sequence_id": np.array(res["original_sequence_id"]),
            "key_id": np.concatenate(res["key_ids"]),
            "sequence_pos": np.concatenate(res["sequence_pos"]),
            "sequence_length": np.concatenate(res["sequence_length"]),
            "sequence_relative_pos": np.concatenate(res["sequence_relative_pos"]),
            "word_frequency": np.concatenate(res["word_frequency"]),
            "word_index": np.concatenate(res["word_idx"]),
            "word_length": np.concatenate(res["word_length"]),
            "word_relative_pos": np.concatenate(res["word_relative_pos"]),
            "is_word_start": np.concatenate(res["is_word_start"]),
            "is_word_end": np.concatenate(res["is_word_end"]),
            "is_syllable_start": np.concatenate(res["is_syllable_start"]),
            "is_syllable_end": np.concatenate(res["is_syllable_end"]),
            "bigram_frequency": np.concatenate(res["bigram_frequency"]),
            "repetition": np.concatenate(res["repetition"]),
            "skipgram_repetition": np.concatenate(res["skipgram_repetition"]),
        }
        for t in target_cols:
            base_dict[t] = np.concatenate(res["targets"][t])
            
        return base_dict

    def enrich_layout(self, base_dict, layout_path, layout_map_path, shifts_path, features=None):
        with open(layout_path, 'r') as f: l_data = json.load(f)
        with open(layout_map_path, 'r') as f: l_map = json.load(f)
        with open(shifts_path, 'r') as f: s_data = json.load(f)
        
        selected = features or list(FEATURE_REGISTRY.keys())
        tables, move_matrix, sf_matrix, sh_matrix, sft_matrix, ir_matrix, or_matrix, sf_skip_matrix, in_triroll_matrix, out_triroll_matrix, redir_matrix, drj_matrix, scissor_matrix = self._build_lookups(l_data, l_map, s_data, selected)
        
        encoded = base_dict["key_id"]
        mask = (encoded == 0)
        
        enriched = base_dict.copy()
        
        # direct table lookups via fancy indexing (x, y, finger, hand, shift)
        for feat, table in tables.items():
            data = table[encoded].astype(np.float32)
            pad = PAD_VALUES.get(feat, -1.0)
            if data.ndim == 1: 
                data[mask] = pad
            else: 
                data[mask, :] = pad
            enriched[feat] = data
            if feat == "coords":
                enriched["x"] = data[:, 0]
                enriched["y"] = data[:, 1]
                
        # bigram/trigram features need prev (p) and prev-prev (pp) keys, computed via pandas shift
        p = pd.Series(encoded).shift(1).fillna(0).astype(int).values
        pp = pd.Series(encoded).shift(2).fillna(0).astype(int).values
        valid_bi = (p != 0) & (encoded != 0) & (p < 128) & (encoded < 128)
        valid_tri = valid_bi & (pp != 0) & (pp < 128)

        def apply_matrix(matrix, mask, p_idx, c_idx, pad_val, pp_idx=None):
            if matrix is None:
                return np.full(len(encoded), pad_val, dtype=np.float32)
            res = np.full(len(encoded), pad_val, dtype=np.float32)
            if pp_idx is not None:
                res[mask] = matrix[pp_idx[mask], p_idx[mask], c_idx[mask]]
            else:
                res[mask] = matrix[p_idx[mask], c_idx[mask]]
            return res

        enriched['same_finger'] = apply_matrix(sf_matrix, valid_bi, p, encoded, PAD_VALUES["same_finger"])
        enriched['same_hand'] = apply_matrix(sh_matrix, valid_bi, p, encoded, PAD_VALUES["same_hand"])
        enriched['in_roll'] = apply_matrix(ir_matrix, valid_bi, p, encoded, PAD_VALUES["in_roll"])
        enriched['out_roll'] = apply_matrix(or_matrix, valid_bi, p, encoded, PAD_VALUES["out_roll"])
        enriched['double_row_jump'] = apply_matrix(drj_matrix, valid_bi, p, encoded, PAD_VALUES["double_row_jump"])
        enriched['scissors'] = apply_matrix(scissor_matrix, valid_bi, p, encoded, PAD_VALUES["scissors"])
        
        enriched['same_finger_trigram'] = apply_matrix(sft_matrix, valid_tri, p, encoded, PAD_VALUES["same_finger_trigram"], pp)
        enriched['same_finger_skipgram'] = apply_matrix(sf_skip_matrix, valid_tri, p, encoded, PAD_VALUES["same_finger_skipgram"], pp)
        enriched['in_triroll'] = apply_matrix(in_triroll_matrix, valid_tri, p, encoded, PAD_VALUES["in_triroll"], pp)
        enriched['out_triroll'] = apply_matrix(out_triroll_matrix, valid_tri, p, encoded, PAD_VALUES["out_triroll"], pp)
        enriched['redirects'] = apply_matrix(redir_matrix, valid_tri, p, encoded, PAD_VALUES["redirects"], pp)

        if move_matrix is not None:
            m_data = np.full((len(encoded), 3), [PAD_VALUES["move_dist"], PAD_VALUES["move_sin"], PAD_VALUES["move_cos"]], dtype=np.float32)
            m_data[valid_bi] = move_matrix[p[valid_bi], encoded[valid_bi]]
            enriched['movement'] = m_data
        else:
            enriched['movement'] = np.full((len(encoded), 3), [PAD_VALUES["move_dist"], PAD_VALUES["move_sin"], PAD_VALUES["move_cos"]], dtype=np.float32)

        return enriched, encoded

    def enrich(self, df, layout_path, layout_map_path, shifts_path, n_pads=8, target_cols=['iki_z', 'iki_log_z'], features=None, show_progress=False):
        base_dict = self.enrich_linguistics(df, n_pads, target_cols, show_progress)
        return self.enrich_layout(base_dict, layout_path, layout_map_path, shifts_path, features)