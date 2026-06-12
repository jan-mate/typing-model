import sys
sys.path.append('/content/typing-model')

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

from src.config import ENRICHED_DATA_PATH, model_dir
from src.models.mlp_arch import FixedMLP
from src.utils.data_loader import prepare_sequential_data, get_train_val_masks
from src.utils.training_utils import SEED, mae_with_ci, seed_everything, seed_worker

CFG = {
    "data_path": ENRICHED_DATA_PATH,
    "model_save_dir": model_dir("mlp_main"),
    "target": "iki_z",
    "w_back": 3,
    "w_ahead": 1,
    "fold_id": 0,
    "subset": None,
    "lr": 2e-3,
    "epochs": 15,
    "batch_size": 32768,
    "patience": 2,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

ACTIVE_FEATURES = [
    'move_dist', 'move_cos', 'move_sin', 'x', 'y', 'shift', 
    'bigram_frequency', 'word_frequency', 'repetition', 
    'skipgram_repetition', 'same_finger_skipgram', 'same_finger', 'same_hand',
    'in_roll', 'out_roll', 'redirects', 'double_row_jump', 
    'sequence_pos', 'word_index', 'word_length', 'word_relative_pos'
]
for i in range(10): ACTIVE_FEATURES.append(f'finger_{i}')
for i in range(3): ACTIVE_FEATURES.append(f'hand_{i}')

def main():
    seed_everything(use_torch=True)
    os.makedirs(CFG["model_save_dir"], exist_ok=True)
    df_raw = pd.read_parquet(CFG["data_path"])
    if CFG["subset"]: df_raw = df_raw.head(CFG["subset"]).copy()
        
    X, y, f_b, f_w = prepare_sequential_data(df_raw, CFG["target"], CFG["w_back"], CFG["w_ahead"], ACTIVE_FEATURES)
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
    
    best_val_mae = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(CFG["epochs"]):
        model.train()
        train_losses =[]
        for bx, by in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            bx, by = bx.to(CFG["device"], non_blocking=True), by.to(CFG["device"], non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        model.eval()
        val_maes =[]
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(CFG["device"], non_blocking=True), by.to(CFG["device"], non_blocking=True)
                val_maes.append(criterion(model(bx), by).item())
                
        avg_val_mae = np.mean(val_maes)
        print(f"End Epoch {epoch+1} | Train Loss: {np.mean(train_losses):.4f} | Val MAE: {avg_val_mae:.4f} | LR: {optimizer.param_groups[0]['lr']}")
        
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(CFG["model_save_dir"], "mlp_model.pth"))
            joblib.dump(scaler, os.path.join(CFG["model_save_dir"], "scaler.pkl"))
        else:
            epochs_no_improve += 1
            
        scheduler.step(avg_val_mae)
        if epochs_no_improve >= CFG["patience"]: break

    model.load_state_dict(torch.load(os.path.join(CFG["model_save_dir"], "mlp_model.pth")))
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_val).to(CFG["device"])).cpu().numpy()
        mae, ci_low, ci_high = mae_with_ci(y_val, preds)
        print("\n=== Subset Result ===")
        print(f"ACTIVE_FEATURES ({len(ACTIVE_FEATURES)}): {ACTIVE_FEATURES}")
        print(f"Val MAE: {mae:.4f}  [95% CI: {ci_low:.4f}, {ci_high:.4f}]")
        print(f"Val R2:  {r2_score(y_val, preds):.4f}")

if __name__ == "__main__":
    main()