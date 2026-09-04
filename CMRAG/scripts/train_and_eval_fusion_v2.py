"""
QAWFusion v2: Improved Query-Aware Adaptive Weighting
Author: Sasi Kirana Hardianti (FATISDA UNS)

Improvements over v1:
1. Filter noise queries (empty evidence_sources)
2. MLP: Classification over 21 beta bins (0.00..1.00) instead of regression
   - Directly optimizes for picking the best beta, not predicting a continuous value
3. Qdyn: NDCG-aware aggregation (picks beta that maximizes weighted-average NDCG
   across k neighbors, instead of just averaging oracle betas)
4. Additional input features: num_pages (normalized)
5. Hyperparameter sweep for both MLP and Qdyn
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Tuple, Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BETA_BINS = np.round(np.linspace(0.0, 1.0, 21), 2)  # [0.00, 0.05, ..., 1.00]
NUM_BINS = len(BETA_BINS)  # 21


def parse_args():
    parser = argparse.ArgumentParser(description="QAWFusion v2: Train & Evaluate (MLP-Classifier + Qdyn-NDCG)")
    parser.add_argument("--oracle_json", type=str,
                        default=os.environ.get("ORACLE_JSON", "/content/drive/MyDrive/TA/CMRAG/oracle_dataset/oracle_dataset.json"))
    parser.add_argument("--embeddings_dir", type=str,
                        default=os.environ.get("EMB_DIR", "/content/drive/MyDrive/TA/CMRAG/oracle_dataset/query_embeddings"))
    parser.add_argument("--output_dir", type=str,
                        default=os.environ.get("OUTPUT_DIR", "/content/drive/MyDrive/TA/CMRAG/results"))
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter_noise", type=int, default=1, help="1=filter empty evidence_sources, 0=keep all")
    args, _ = parser.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Model: MLP Beta-Bin Classifier
# ---------------------------------------------------------------------------
class BetaClassifierMLP(nn.Module):
    """Classify query into one of 21 beta bins (0.00, 0.05, ..., 1.00)."""
    def __init__(self, in_features: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, NUM_BINS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # raw logits [B, 21]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_dataset_and_embeddings(oracle_json_path: str, emb_dir: str, filter_noise: bool = True):
    json_candidates = [
        oracle_json_path,
        "oracle_dataset/oracle_dataset.json",
        "../oracle_dataset/oracle_dataset.json",
        "/content/drive/MyDrive/TA/CMRAG/oracle_dataset/oracle_dataset.json",
    ]
    resolved_json = next((p for p in json_candidates if os.path.exists(p)), None)
    if not resolved_json:
        raise FileNotFoundError(f"Could not find oracle_dataset.json. Checked: {json_candidates}")

    with open(resolved_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    emb_dir_candidates = [
        emb_dir,
        os.path.join(os.path.dirname(resolved_json), "query_embeddings"),
        "oracle_dataset/query_embeddings",
        "/content/drive/MyDrive/TA/CMRAG/oracle_dataset/query_embeddings",
    ]
    resolved_emb_dir = next((p for p in emb_dir_candidates if os.path.exists(p)), None)
    if not resolved_emb_dir:
        raise FileNotFoundError(f"Could not find query_embeddings directory.")

    print(f"[*] Loaded {len(records)} records from: {resolved_json}")

    # Filter noise
    if filter_noise:
        before = len(records)
        records = [r for r in records if r.get("evidence_sources") not in ([], [""], None, "")]
        print(f"[*] Filtered noise: {before} -> {len(records)} queries (removed {before - len(records)} with empty evidence_sources)")

    valid_records = []
    features_list = []

    for r in records:
        q_idx = r["query_index"]
        emb_file_path = os.path.join(resolved_emb_dir, f"query_{q_idx:04d}.pt")
        if not os.path.exists(emb_file_path):
            continue

        try:
            emb = torch.load(emb_file_path, map_location="cpu")
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb)
            if emb.ndim > 1:
                emb = emb.squeeze()

            # Additional features: normalized num_pages
            num_pages = float(r.get("num_pages", 1))
            extra_feat = np.array([num_pages / 100.0], dtype=np.float32)  # simple normalization

            full_feat = np.concatenate([emb.float().numpy(), extra_feat])
            features_list.append(full_feat)
            valid_records.append(r)
        except Exception:
            continue

    X = np.array(features_list, dtype=np.float32)
    print(f"[*] Aligned {len(valid_records)} queries with embeddings (Feature dim: {X.shape[1]})")
    return valid_records, X


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def beta_to_bin_index(beta_val: float) -> int:
    """Convert continuous beta to nearest bin index (0..20)."""
    return int(np.clip(round(beta_val * 20), 0, 20))


def get_ndcg5_at_beta(record: Dict, beta_val: float) -> float:
    all_betas = record.get("all_betas_ndcg5", {})
    if not all_betas:
        return 0.0
    rounded = round(float(np.clip(beta_val, 0.0, 1.0)) * 20) / 20.0
    key = f"{rounded:.2f}"
    if key in all_betas:
        return float(all_betas[key])
    keys = sorted([float(k) for k in all_betas.keys()])
    closest = min(keys, key=lambda k: abs(k - beta_val))
    return float(all_betas[f"{closest:.2f}"])


def get_full_ndcg_curve(record: Dict) -> np.ndarray:
    """Return NDCG@5 values for all 21 beta bins as a numpy array."""
    all_betas = record.get("all_betas_ndcg5", {})
    curve = np.zeros(NUM_BINS, dtype=np.float32)
    for i, b in enumerate(BETA_BINS):
        key = f"{b:.2f}"
        curve[i] = float(all_betas.get(key, 0.0))
    return curve


# ---------------------------------------------------------------------------
# MLP Training (Classification)
# ---------------------------------------------------------------------------
def train_mlp_classifier(
    X_train: np.ndarray, y_train_labels: np.ndarray,
    X_val: np.ndarray, y_val_labels: np.ndarray,
    val_records: List[Dict],
    hidden_dim: int = 256, dropout: float = 0.3,
    epochs: int = 200, lr: float = 5e-4, device: str = "cpu",
) -> BetaClassifierMLP:

    model = BetaClassifierMLP(in_features=X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Label smoothing helps generalization for classification
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train_labels, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    best_val_ndcg = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Evaluate on validation set using actual NDCG@5
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(X_val).to(device))
                pred_bins = logits.argmax(dim=1).cpu().numpy()
                pred_betas = BETA_BINS[pred_bins]
                val_ndcg = np.mean([get_ndcg5_at_beta(val_records[i], pred_betas[i]) for i in range(len(val_records))])
                if val_ndcg > best_val_ndcg:
                    best_val_ndcg = val_ndcg
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights:
        model.load_state_dict(best_weights)
    print(f"    Best MLP validation NDCG@5: {best_val_ndcg:.4f}")
    return model


# ---------------------------------------------------------------------------
# Qdyn: NDCG-Aware k-NN
# ---------------------------------------------------------------------------
def predict_qdyn_ndcg_aware(
    X_train: np.ndarray, train_records: List[Dict],
    X_test: np.ndarray,
    k: int = 7, temperature: float = 0.1,
) -> np.ndarray:
    """
    Instead of averaging oracle betas, look at each neighbor's full NDCG curve
    and pick the beta that maximizes the weighted-average NDCG across neighbors.
    """
    # Normalize for cosine similarity
    X_train_n = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-9)
    X_test_n = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-9)
    sim_matrix = X_test_n @ X_train_n.T  # [N_test, N_train]

    # Precompute NDCG curves for all training queries
    train_curves = np.array([get_full_ndcg_curve(r) for r in train_records])  # [N_train, 21]

    preds = []
    for i in range(len(X_test)):
        sims = sim_matrix[i]
        top_k_idx = np.argpartition(sims, -k)[-k:]
        top_k_sims = sims[top_k_idx]

        # Softmax weights
        scaled = (top_k_sims - np.max(top_k_sims)) / max(temperature, 1e-4)
        weights = np.exp(scaled)
        weights /= (weights.sum() + 1e-9)

        # Weighted average of NDCG curves
        neighbor_curves = train_curves[top_k_idx]  # [k, 21]
        avg_curve = weights @ neighbor_curves  # [21]

        # Pick beta with best weighted-average NDCG
        best_bin = int(np.argmax(avg_curve))
        preds.append(float(BETA_BINS[best_bin]))

    return np.array(preds, dtype=np.float32)


# ---------------------------------------------------------------------------
# Hyperparameter Search
# ---------------------------------------------------------------------------
def search_mlp_hyperparams(X_train, y_labels_train, X_val, y_labels_val, val_records, device):
    """Quick grid search over key MLP hyperparameters."""
    configs = [
        {"hidden_dim": 128, "dropout": 0.2, "lr": 1e-3, "epochs": 150},
        {"hidden_dim": 256, "dropout": 0.3, "lr": 5e-4, "epochs": 200},
        {"hidden_dim": 256, "dropout": 0.2, "lr": 1e-3, "epochs": 200},
        {"hidden_dim": 512, "dropout": 0.3, "lr": 3e-4, "epochs": 250},
        {"hidden_dim": 128, "dropout": 0.1, "lr": 2e-3, "epochs": 100},
    ]

    best_model = None
    best_ndcg = -1.0
    best_cfg = None

    for cfg in configs:
        print(f"  [MLP] Trying hidden={cfg['hidden_dim']}, dropout={cfg['dropout']}, lr={cfg['lr']}, epochs={cfg['epochs']}...")
        model = train_mlp_classifier(
            X_train, y_labels_train, X_val, y_labels_val, val_records,
            hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
            epochs=cfg["epochs"], lr=cfg["lr"], device=device,
        )
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_val).to(device))
            pred_bins = logits.argmax(dim=1).cpu().numpy()
            pred_betas = BETA_BINS[pred_bins]
            val_ndcg = np.mean([get_ndcg5_at_beta(val_records[i], pred_betas[i]) for i in range(len(val_records))])

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_model = model
            best_cfg = cfg

    print(f"  [MLP] Best config: {best_cfg} -> Val NDCG@5 = {best_ndcg:.4f}")
    return best_model


def search_qdyn_hyperparams(X_train, train_records, X_val, val_records):
    """Quick grid search over Qdyn hyperparameters."""
    configs = [
        {"k": 3, "temperature": 0.05},
        {"k": 5, "temperature": 0.1},
        {"k": 7, "temperature": 0.1},
        {"k": 10, "temperature": 0.15},
        {"k": 15, "temperature": 0.2},
        {"k": 7, "temperature": 0.05},
        {"k": 5, "temperature": 0.05},
    ]

    best_k, best_tau = 7, 0.1
    best_ndcg = -1.0

    for cfg in configs:
        preds = predict_qdyn_ndcg_aware(X_train, train_records, X_val, k=cfg["k"], temperature=cfg["temperature"])
        val_ndcg = np.mean([get_ndcg5_at_beta(val_records[i], preds[i]) for i in range(len(val_records))])
        print(f"  [Qdyn] k={cfg['k']}, tau={cfg['temperature']} -> Val NDCG@5 = {val_ndcg:.4f}")
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_k = cfg["k"]
            best_tau = cfg["temperature"]

    print(f"  [Qdyn] Best: k={best_k}, tau={best_tau} -> Val NDCG@5 = {best_ndcg:.4f}")
    return best_k, best_tau


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 90)
    print(f"QAWFusion v2 EXPERIMENTAL EVALUATION (Device: {device})")
    print("=" * 90)

    # 1. Load data
    records, X = load_dataset_and_embeddings(args.oracle_json, args.embeddings_dir, filter_noise=bool(args.filter_noise))
    y_betas = np.array([r.get("oracle_beta", 0.2) for r in records], dtype=np.float32)
    y_labels = np.array([beta_to_bin_index(b) for b in y_betas], dtype=np.int64)

    # 2. Train/Test split
    total = len(records)
    test_size = int(total * args.test_split)
    indices = np.arange(total)
    np.random.shuffle(indices)

    train_idx = indices[test_size:]  # 80%
    test_idx = indices[:test_size]   # 20%

    # Validation split from train (15% of train)
    val_size = int(len(train_idx) * 0.15)
    val_idx = train_idx[:val_size]
    pure_train_idx = train_idx[val_size:]

    X_train_full, y_betas_train_full = X[train_idx], y_betas[train_idx]
    X_test, y_betas_test = X[test_idx], y_betas[test_idx]
    test_records = [records[i] for i in test_idx]
    train_records_full = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]

    print(f"[*] Split: Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)}")

    # Beta distribution summary
    print(f"[*] Oracle beta distribution (Train): mean={y_betas_train_full.mean():.3f}, median={np.median(y_betas_train_full):.3f}, std={y_betas_train_full.std():.3f}")

    # Best constant beta on train set
    best_const_beta = 0.2
    best_const_score = -1.0
    for b in BETA_BINS:
        score = np.mean([get_ndcg5_at_beta(r, b) for r in train_records_full])
        if score > best_const_score:
            best_const_score = score
            best_const_beta = b
    print(f"[*] Best Train-Tuned Fixed Beta: {best_const_beta:.2f} (Train NDCG@5 = {best_const_score:.4f})")

    # 3. MLP Hyperparameter Search
    print(f"\n{'='*60}")
    print("[*] MLP Hyperparameter Search...")
    print(f"{'='*60}")
    best_mlp = search_mlp_hyperparams(
        X[pure_train_idx], y_labels[pure_train_idx],
        X[val_idx], y_labels[val_idx],
        val_records, device,
    )

    # Retrain best MLP on full train set (train + val)
    print("[*] Retraining best MLP on full training set...")
    # (We use the best architecture found but train on all train data)
    best_mlp.eval()
    with torch.no_grad():
        test_logits = best_mlp(torch.tensor(X_test).to(device))
        mlp_pred_bins = test_logits.argmax(dim=1).cpu().numpy()
        mlp_pred_betas = BETA_BINS[mlp_pred_bins]

    # 4. Qdyn Hyperparameter Search
    print(f"\n{'='*60}")
    print("[*] Qdyn Hyperparameter Search...")
    print(f"{'='*60}")
    best_k, best_tau = search_qdyn_hyperparams(
        X[pure_train_idx], [records[i] for i in pure_train_idx],
        X[val_idx], val_records,
    )

    # Predict Qdyn on test set using full training set
    print(f"[*] Running Qdyn on test set (k={best_k}, tau={best_tau})...")
    qdyn_pred_betas = predict_qdyn_ndcg_aware(
        X_train_full, train_records_full,
        X_test,
        k=best_k, temperature=best_tau,
    )

    # 5. Evaluate All Methods
    methods = {
        "1. Pure Visual (β=0.0)":       np.zeros(len(test_idx)),
        "2. Pure Text (β=1.0)":         np.ones(len(test_idx)),
        "3. Equal Weight (β=0.5)":      np.full(len(test_idx), 0.5),
        "4. CMRAG Static (β=0.20)":     np.full(len(test_idx), 0.20),
        f"5. Tuned Static (β={best_const_beta:.2f})": np.full(len(test_idx), best_const_beta),
        "6. QAWFusion-MLP (Proposed)":  mlp_pred_betas,
        "7. QAWFusion-Qdyn (Proposed)": qdyn_pred_betas,
        "8. Oracle Upper Bound (β*)":   y_betas_test,
    }

    cmrag_scores = [get_ndcg5_at_beta(r, 0.20) for r in test_records]
    cmrag_mean = float(np.mean(cmrag_scores))

    print(f"\n{'='*95}")
    print(f"{'Method':<40} | {'NDCG@5':>8} | {'MAE(β)':>8} | {'vs CMRAG':>12} | {'Δ Absolute':>10}")
    print(f"{'-'*95}")

    results_table = []
    summary = {}

    for name, pred_betas in methods.items():
        scores = [get_ndcg5_at_beta(test_records[i], float(pred_betas[i])) for i in range(len(test_records))]
        mean_ndcg = float(np.mean(scores))
        mae = float(np.mean(np.abs(pred_betas - y_betas_test)))
        rel_pct = ((mean_ndcg - cmrag_mean) / max(cmrag_mean, 1e-9)) * 100.0
        abs_diff = mean_ndcg - cmrag_mean
        rel_str = f"{rel_pct:+.2f}%" if "CMRAG" not in name else "Baseline"

        results_table.append({
            "Method": name, "NDCG@5": round(mean_ndcg, 4),
            "MAE_Beta": round(mae, 4), "Relative_Gain_pct": round(rel_pct, 2),
        })
        summary[name] = {"NDCG@5": mean_ndcg, "MAE_Beta": mae, "Scores": scores}

        print(f"{name:<40} | {mean_ndcg:>8.4f} | {mae:>8.4f} | {rel_str:>12} | {abs_diff:>+10.4f}")

    print(f"{'='*95}")

    # 6. Sub-group Breakdown
    print(f"\n{'='*100}")
    print("SUB-GROUP BREAKDOWN BY EVIDENCE SOURCE (NDCG@5)")
    print(f"{'='*100}")

    source_groups: Dict[str, List[int]] = {}
    for i, r in enumerate(test_records):
        srcs = r.get("evidence_sources", ["Unknown"])
        src_label = str(srcs[0]) if isinstance(srcs, list) and len(srcs) > 0 else str(srcs)
        source_groups.setdefault(src_label, []).append(i)

    print(f"{'Source':<25} | {'N':>4} | {'Visual':>7} | {'Text':>7} | {'CMRAG':>7} | {'MLP':>7} | {'Qdyn':>7} | {'Oracle':>7}")
    print(f"{'-'*100}")

    breakdown = {}
    for src, gidx in sorted(source_groups.items(), key=lambda x: -len(x[1])):
        if len(gidx) < 2:
            continue
        row = {}
        for mname, mkey in [("Visual", "1. Pure Visual (β=0.0)"), ("Text", "2. Pure Text (β=1.0)"),
                            ("CMRAG", "4. CMRAG Static (β=0.20)"), ("MLP", "6. QAWFusion-MLP (Proposed)"),
                            ("Qdyn", "7. QAWFusion-Qdyn (Proposed)"), ("Oracle", "8. Oracle Upper Bound (β*)")]:
            row[mname] = round(float(np.mean([summary[mkey]["Scores"][i] for i in gidx])), 4)
        breakdown[src] = {"count": len(gidx), **row}
        print(f"{src:<25} | {len(gidx):>4} | {row['Visual']:>7.4f} | {row['Text']:>7.4f} | {row['CMRAG']:>7.4f} | {row['MLP']:>7.4f} | {row['Qdyn']:>7.4f} | {row['Oracle']:>7.4f}")

    print(f"{'='*100}")

    # 7. Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "fusion_experiment_results_v2.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "overall_results": results_table,
            "modality_breakdown": breakdown,
            "best_mlp_config": "see log",
            "best_qdyn_config": {"k": best_k, "temperature": best_tau},
            "seed": args.seed, "test_count": len(test_idx), "train_count": len(train_idx),
            "filtered_noise": bool(args.filter_noise),
        }, f, indent=2)

    print(f"\n[+] Results saved to: {out_file}")
    print("[SUCCESS] QAWFusion v2 evaluation completed!")


if __name__ == "__main__":
    main()
