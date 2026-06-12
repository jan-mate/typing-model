import sys
sys.path.append('/content/typing-model')

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib
import shap
import warnings

from src.config import ENRICHED_DATA_PATH, model_dir
from src.models.mlp_arch import FixedMLP
from src.utils.data_loader import prepare_sequential_data, get_train_val_masks, get_all_features
from src.utils.training_utils import SEED, mae_with_ci, seed_everything, seed_worker

warnings.filterwarnings('ignore')

CFG = {
    "data_path": ENRICHED_DATA_PATH,
    "model_save_dir": model_dir("mlp_feature_importance"),
    "target": "iki_z",
    "w_back": 3,
    "w_ahead": 1,
    "fold_id": 0,
    "add_noise": True,
    "use_manual_features": False,
    "lr": 2e-3,
    "epochs": 15,
    "batch_size": 32768,
    "patience": 2,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

MANUAL_FEATURES = [
    'move_dist', 'move_cos', 'move_sin', 'x', 'y', 'shift', 
    'bigram_frequency', 'word_frequency', 'same_hand', 'same_finger', 
    'repetition', 'skipgram_repetition', 'same_finger_skipgram', 
    'in_roll', 'out_roll', 'redirects', 'double_row_jump', 
    'sequence_pos', 'word_index', 'word_length', 'word_relative_pos', 
    'finger', 'hand'
]

if CFG["use_manual_features"]:
    ACTIVE_FEATURES = list(MANUAL_FEATURES)
    if CFG["add_noise"] and "random_noise" not in ACTIVE_FEATURES:
        ACTIVE_FEATURES.append("random_noise")
else:
    ACTIVE_FEATURES = get_all_features(add_noise=CFG["add_noise"])


class SHAPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x): return self.model(x).unsqueeze(-1)


def main():
    seed_everything(use_torch=True)
    os.makedirs(CFG["model_save_dir"], exist_ok=True)
    df_raw = pd.read_parquet(CFG["data_path"])
    
    X, y, f_b, f_w = prepare_sequential_data(df_raw, CFG["target"], CFG["w_back"], CFG["w_ahead"], ACTIVE_FEATURES, add_noise=CFG["add_noise"])
    del df_raw
    
    train_mask, val_mask = get_train_val_masks(f_b, f_w, CFG["fold_id"])
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    y_train = y[train_mask]
    X_val = scaler.transform(X[val_mask])
    y_val = y[val_mask]
    
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=CFG["batch_size"], shuffle=True, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=CFG["batch_size"], pin_memory=True)
    
    model = FixedMLP(X_train.shape[1]).to(CFG["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    criterion = nn.L1Loss()
    
    print(f"\nTraining MLP on {sum(train_mask)} samples with {len(ACTIVE_FEATURES)} features...")
    best_val_mae = float('inf')
    epochs_no_improve = 0
    model_path = os.path.join(CFG["model_save_dir"], "mlp_shap_model.pth")
    
    for epoch in range(CFG["epochs"]):
        model.train()
        for bx, by in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            bx, by = bx.to(CFG["device"], non_blocking=True), by.to(CFG["device"], non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_maes = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(CFG["device"], non_blocking=True), by.to(CFG["device"], non_blocking=True)
                val_maes.append(criterion(model(bx), by).item())
                
        avg_val_mae = np.mean(val_maes)
        print(f"End Epoch {epoch+1} | Val MAE: {avg_val_mae:.4f} | LR: {optimizer.param_groups[0]['lr']}")
        
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            joblib.dump(scaler, os.path.join(CFG["model_save_dir"], "shap_scaler.pkl"))
        else:
            epochs_no_improve += 1
            
        scheduler.step(avg_val_mae)
        if epochs_no_improve >= CFG["patience"]: break

    # --- SHAP ANALYSIS ---
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("\nCalculating Validation MAE with CI...")
    with torch.no_grad():
        preds = model(torch.from_numpy(X_val).to(CFG["device"])).cpu().numpy()
        mae, ci_low, ci_high = mae_with_ci(y_val, preds)
        print(f"Word Fold Validation MAE: {mae:.4f} [95% CI: {ci_low:.4f}, {ci_high:.4f}]")
    
    print("\nCalculating SHAP values...")
    if len(X_val) > 5000:
        np.random.seed(42)
        indices = np.random.permutation(len(X_val))
        X_val_sample = X_val[indices[:5000]]
    else:
        X_val_sample = X_val
        
    shap_model = SHAPWrapper(model).to(CFG["device"])

    background_tensor = torch.from_numpy(X_val_sample[:300]).to(CFG["device"])
    test_tensor = torch.from_numpy(X_val_sample[300:1000]).to(CFG["device"])
    
    explainer = shap.DeepExplainer(shap_model, background_tensor)
    shap_values = explainer.shap_values(test_tensor)
    
    if isinstance(shap_values, list): shap_values = shap_values[0]
    if shap_values.ndim == 3: shap_values = shap_values.squeeze(-1)
    
    num_features = len(ACTIVE_FEATURES)
    num_steps = CFG["w_back"] + CFG["w_ahead"] + 1
    
    feature_importances = {}
    for i, feat in enumerate(ACTIVE_FEATURES):
        cols =[i + (step * num_features) for step in range(num_steps)]
        importance = np.mean(np.sum(np.abs(shap_values[:, cols]), axis=1))
        feature_importances[feat] = importance
        
    print("\n--- SHAP Feature Importances ---")
    for i, (feat, imp) in enumerate(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)):
        print(f"{i+1:2d}. {feat:<25}: {imp:.5f}")

if __name__ == "__main__":
    main()