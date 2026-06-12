import os
from abc import ABC, abstractmethod
import numpy as np

MLP_MAIN_FEATURES = [
    "move_dist", "move_cos", "move_sin", "x", "y", "shift",
    "bigram_frequency", "word_frequency",
    "repetition", "skipgram_repetition", "same_finger_skipgram",
    "same_finger", "same_hand",
    "in_roll", "out_roll", "redirects", "double_row_jump",
    "sequence_pos", "word_length", "word_relative_pos",
    "is_syllable_start", "is_syllable_end",
    "finger_0", "finger_1", "finger_2", "finger_3", "finger_4",
    "finger_5", "finger_6", "finger_7", "finger_8", "finger_9",
    "hand_0", "hand_1", "hand_2",
]

MLP_DL_FEATURES = [
    "x", "y", "shift", "hand",
    "bigram_frequency", "word_frequency",
    "sequence_pos", "word_length", "word_relative_pos",
    "is_syllable_start", "is_syllable_end",
]

LGBM_FEATURES = [
    'move_dist', 'move_cos', 'move_sin', 'x', 'y', 'shift', 
    'bigram_frequency', 'word_frequency', 'same_hand', 'same_finger', 
    'repetition', 'skipgram_repetition', 'same_finger_skipgram', 
    'in_roll', 'out_roll', 'redirects', 'double_row_jump', 
    'sequence_pos', 'word_index', 'word_length', 'word_relative_pos', 
    'finger', 'hand'
]

LINREG_FEATURES = [
    "is_pad", "move_dist", "move_sin", "move_cos", "x", "y", "shift",
    "unigram_frequency", "bigram_frequency", "word_frequency",
    "same_hand", "same_finger", "same_finger_trigram",
    "repetition", "skipgram_repetition", "same_finger_skipgram",
    "in_roll", "out_roll", "in_triroll", "out_triroll",
    "redirects", "double_row_jump", "scissors",
    "word_index", "word_length", "word_relative_pos",
    "is_word_start", "is_word_end", "is_syllable_start", "is_syllable_end",
    "finger_0", "finger_1", "finger_3", "finger_6", "finger_8", "finger_9",
    "finger_type_0", "finger_type_1", "finger_type_2", "finger_type_3",
    "hand_0",
]

MLP_LINGUISTIC_FEATURES = [
    "bigram_frequency", "word_frequency",
    "sequence_pos", "word_length", "word_relative_pos",
    "is_syllable_start", "is_syllable_end",
]

# union of MLP_MAIN + LGBM, for single-pass enrichment in inference.py
INFERENCE_FEATURES = list(dict.fromkeys(MLP_MAIN_FEATURES + LGBM_FEATURES))


class SpeedModel(ABC):
    @abstractmethod
    def predict(self, X):
        # returns (means, stds) over the ensemble folds
        ...


class LGBM(SpeedModel):
    features = LGBM_FEATURES

    def __init__(self, model_dir, n_folds=10):
        import json
        import lightgbm as lgb

        self.models = []
        self.w_back = 3
        self.w_ahead = 1

        params_path = os.path.join(os.path.dirname(model_dir), "best_optuna_lgbm_params.json")
        if os.path.exists(params_path):
            with open(params_path) as f:
                params = json.load(f)
            self.w_back = params.get("w_back", self.w_back)
            self.w_ahead = params.get("w_ahead", self.w_ahead)
        else:
            print(f"Warning: {params_path} not found — using default w_back={self.w_back}, w_ahead={self.w_ahead}.")

        for fold in range(n_folds):
            path = os.path.join(model_dir, f"final_lgbm_fold_{fold}.txt")
            if os.path.exists(path):
                self.models.append(lgb.Booster(model_file=path))
        if not self.models:
            raise FileNotFoundError(f"No LGBM models found in {model_dir}")

    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return preds.mean(axis=0), preds.std(axis=0)


class LinReg(SpeedModel):
    features = LINREG_FEATURES

    def __init__(self, model_dir, n_folds=10):
        import json
        import joblib

        self.models = []
        self.scalers = []
        self.w_back = 2
        self.w_ahead = 1

        params_path = os.path.join(os.path.dirname(model_dir), "best_linreg_params.json")
        if os.path.exists(params_path):
            with open(params_path) as f:
                params = json.load(f)
            self.w_back = params.get("w_back", self.w_back)
            self.w_ahead = params.get("w_ahead", self.w_ahead)
        else:
            print(f"Warning: {params_path} not found — using default w_back={self.w_back}, w_ahead={self.w_ahead}.")

        for fold in range(n_folds):
            model_path = os.path.join(model_dir, f"final_linreg_fold_{fold}.pkl")
            scaler_path = os.path.join(model_dir, f"final_linreg_scaler_fold_{fold}.pkl")
            if os.path.exists(model_path):
                self.models.append(joblib.load(model_path))
                self.scalers.append(joblib.load(scaler_path))
        if not self.models:
            raise FileNotFoundError(f"No LinReg models found in {model_dir}")

    def predict(self, X):
        preds = []
        for model, scaler in zip(self.models, self.scalers):
            X_scaled = scaler.transform(X)
            preds.append(model.predict(X_scaled))
        preds = np.array(preds)
        return preds.mean(axis=0), preds.std(axis=0)


class MLP(SpeedModel):
    features = MLP_MAIN_FEATURES

    def __init__(self, model_dir, n_folds=10,
                 _prefix="final_mlp_main",
                 _params_file="best_optuna_mlp_params.json"):
        import json
        import torch
        import joblib
        from src.models.mlp_arch import DynamicMLP

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        self.scalers = []
        self.w_back = 2
        self.w_ahead = 1

        params_path = os.path.join(os.path.dirname(model_dir), _params_file)
        if os.path.exists(params_path):
            with open(params_path) as f:
                params = json.load(f)
            self.w_back = params.get("w_back", self.w_back)
            self.w_ahead = params.get("w_ahead", self.w_ahead)
        else:
            params = {}
            print(f"Warning: {params_path} not found — using default w_back={self.w_back}, w_ahead={self.w_ahead}.")

        for fold in range(n_folds):
            model_path = os.path.join(model_dir, f"{_prefix}_fold_{fold}.pth")
            scaler_path = os.path.join(model_dir, f"{_prefix}_scaler_fold_{fold}.pkl")
            if not os.path.exists(model_path):
                continue

            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            linear_keys = [k for k in state_dict if k.endswith(".weight") and "net" in k]
            in_dim = state_dict[linear_keys[0]].shape[1]
            hidden_dim = state_dict[linear_keys[0]].shape[0]
            n_layers = len(linear_keys) - 1

            model = DynamicMLP(
                in_dim=in_dim,
                n_layers=params.get("n_layers", n_layers),
                hidden_dim=params.get("hidden_dim", hidden_dim),
                dropout=0.0,
                activation_name=params.get("activation", "SiLU"),
            )
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            self.models.append(model)
            self.scalers.append(joblib.load(scaler_path))

        if not self.models:
            raise FileNotFoundError(f"No MLP models found in {model_dir}")

    def predict(self, X, batch_size=16384):
        import torch
        preds = []
        for model, scaler in zip(self.models, self.scalers):
            X_scaled = scaler.transform(X)
            fold_preds = []
            for i in range(0, len(X_scaled), batch_size):
                batch = X_scaled[i : i + batch_size]
                X_tensor = torch.FloatTensor(batch).to(self.device)
                with torch.no_grad():
                    p = model(X_tensor).cpu().numpy().flatten()
                fold_preds.append(p)
            preds.append(np.concatenate(fold_preds))
        preds = np.array(preds)
        return preds.mean(axis=0), preds.std(axis=0)


class MLPDL(MLP):
    features = MLP_DL_FEATURES

    def __init__(self, model_dir, n_folds=10):
        super().__init__(model_dir, n_folds,
                         _prefix="final_mlp_dl",
                         _params_file="best_dl_params.json")

def load_model(name: str):
    from src.config import model_dir
    if name == "lgbm":
        return LGBM(model_dir("lgbm"))
    elif name == "mlp_main":
        return MLP(model_dir("mlp_main"))
    elif name == "mlp_dl":
        return MLPDL(model_dir("mlp_dl"))
    elif name == "linreg":
        return LinReg(model_dir("linreg"))
    else:
        raise ValueError(f"Unknown model name: {name}")
