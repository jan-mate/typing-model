import sys
sys.path.append('/content/typing-model')
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import ENRICHED_DATA_PATH, model_dir
from src.models.mlp_arch import FixedMLP
from src.utils.data_loader import get_all_features, prepare_sequential_data, get_train_val_masks
from src.utils.training_utils import SEED, mae_with_ci, seed_everything, seed_worker

CFG = {
    "data_path": ENRICHED_DATA_PATH,
    "model_save_dir": model_dir("mlp_main"),
    "target": "iki_z",
    "w_back": 3,
    "w_ahead": 1,
    "fold_id": 0,
    "add_noise": True,
    "subset": None, 
    "lr": 1e-3,
    "epochs": 30,
    "batch_size": 8192,
    "patience": 3,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def train_model(X, y, f_b, f_w, cfg):
    train_mask, val_mask = get_train_val_masks(f_b, f_w, cfg["fold_id"])
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    y_train = y[train_mask]
    X_val = scaler.transform(X[val_mask])
    y_val = y[val_mask]
    
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=cfg["batch_size"], shuffle=True, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=cfg["batch_size"], pin_memory=True)
    
    model = FixedMLP(X_train.shape[1]).to(cfg["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.L1Loss()
    
    best_val_mae = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(cfg["epochs"]):
        model.train()
        train_losses =[]
        for bx, by in train_loader:
            bx, by = bx.to(cfg["device"], non_blocking=True), by.to(cfg["device"], non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        model.eval()
        val_maes =[]
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(cfg["device"], non_blocking=True), by.to(cfg["device"], non_blocking=True)
                val_maes.append(criterion(model(bx), by).item())
                
        avg_val_mae = np.mean(val_maes)
        print(f"Epoch {epoch+1} | Train Loss: {np.mean(train_losses):.4f} | Val MAE: {avg_val_mae:.4f}")
        
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(cfg["model_save_dir"], "mlp_model.pth"))
            joblib.dump(scaler, os.path.join(cfg["model_save_dir"], "scaler.pkl"))
        else:
            epochs_no_improve += 1
            
        scheduler.step(avg_val_mae)
        if epochs_no_improve >= cfg["patience"]: break
        
    model.load_state_dict(torch.load(os.path.join(cfg["model_save_dir"], "mlp_model.pth")))
    return model, scaler

def main():
    seed_everything(use_torch=True)
    os.makedirs(CFG["model_save_dir"], exist_ok=True)
    df_raw = pd.read_parquet(CFG["data_path"])

    if CFG["subset"]:
        df_raw = df_raw.head(CFG["subset"]).copy()

    features = get_all_features(add_noise=CFG["add_noise"])
    X, y, f_b, f_w = prepare_sequential_data(df_raw, CFG["target"], CFG["w_back"], CFG["w_ahead"], features, add_noise=CFG["add_noise"])
    del df_raw

    model, scaler = train_model(X, y, f_b, f_w, CFG)
    train_mask, val_mask = get_train_val_masks(f_b, f_w, CFG["fold_id"])

    model.eval()
    with torch.no_grad():
        X_v = scaler.transform(X[val_mask])
        preds = model(torch.from_numpy(X_v).to(CFG["device"])).cpu().numpy()
        mae, ci_low, ci_high = mae_with_ci(y[val_mask], preds)
        print("\n=== All-Features Baseline ===")
        print(f"ACTIVE_FEATURES ({len(features)}): {features}")
        print(f"Val MAE: {mae:.4f}  [95% CI: {ci_low:.4f}, {ci_high:.4f}]")

if __name__ == "__main__":
    main()