"""
QAWFusion: Query-Aware Adaptive Weighting for Multimodal RAG Scoring Fusion
Author: Sasi Kirana Hardianti (FATISDA UNS)

This script trains and evaluates:
1. Baseline: Pure Visual Retrieval (beta = 0.0)
2. Baseline: Pure Text Retrieval (beta = 1.0)
3. Baseline: CMRAG Static Weighting (beta = 0.2)
4. Baseline: Equal Static Weighting (beta = 0.5)
5. Baseline: Train-set Best Constant Beta
6. Proposed Model 1: MLP (Multi-Layer Perceptron Regressor)
7. Proposed Model 2: Qdyn (Query Dynamic Classes via k-NN Cosine Similarity)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train & Evaluate QAWFusion (MLP vs Qdyn)")
    parser.add_argument(
        "--oracle_json",
        type=str,
        default="/content/drive/MyDrive/TA/CMRAG/oracle_dataset/oracle_dataset.json",
        help="Path to oracle_dataset.json",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="/content/drive/MyDrive/TA/CMRAG/oracle_dataset/query_embeddings",
        help="Path to query embeddings directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/TA/CMRAG/results",
        help="Directory to save evaluation results and comparison table",
    )
    parser.add_argument("--test_split", type=float, default=0.2, help="Test set fraction (e.g. 0.2 = 20%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--k_neighbors", type=int, default=7, help="k nearest neighbors for Qdyn")
    parser.add_argument("--qdyn_temp", type=float, default=0.1, help="Softmax temperature for Qdyn weighting")
    parser.add_argument("--mlp_epochs", type=int, default=150, help="Training epochs for MLP")
    parser.add_argument("--mlp_lr", type=float, default=1e-3, help="Learning rate for MLP")
    parser.add_argument("--mlp_hidden", type=int, default=256, help="Hidden dimension for MLP")
    args, _ = parser.parse_known_args()
    return args


class QueryMLPRegressor(nn.Module):
    """Multi-Layer Perceptron to predict optimal fusion weight beta in [0, 1]."""
    def __init__(self, in_features: int = 1152, hidden_dim: int = 256, dropout: float = 0.2):
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
            nn.Linear(hidden_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Bound output strictly to [0.0, 1.0]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_dataset_and_embeddings(oracle_json_path: str, emb_dir: str):
    # Resolve local or Google Drive paths
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
        raise FileNotFoundError(f"Could not find query_embeddings directory. Checked: {emb_dir_candidates}")

    print(f"[*] Loaded {len(records)} records from: {resolved_json}")
    print(f"[*] Using query embeddings from: {resolved_emb_dir}")

    valid_records = []
    features_list = []

    for r in records:
        q_idx = r["query_index"]
        emb_filename = f"query_{q_idx:04d}.pt"
        emb_file_path = os.path.join(resolved_emb_dir, emb_filename)

        if not os.path.exists(emb_file_path):
            continue

        try:
            emb_tensor = torch.load(emb_file_path, map_location="cpu")
            if isinstance(emb_tensor, np.ndarray):
                emb_tensor = torch.from_numpy(emb_tensor)
            if emb_tensor.ndim > 1:
                emb_tensor = emb_tensor.squeeze()
            features_list.append(emb_tensor.float().numpy())
            valid_records.append(r)
        except Exception as e:
            continue

    X = np.array(features_list, dtype=np.float32)
    print(f"[*] Successfully aligned {len(valid_records)} queries with embeddings (Feature dim: {X.shape[1]}).")
    return valid_records, X


def get_ndcg5_at_beta(record: Dict[str, Any], beta_val: float) -> float:
    """Lookup the NDCG@5 performance for a specific beta from precomputed grid search curve."""
    all_betas = record.get("all_betas_ndcg5", {})
    if not all_betas:
        return 0.0

    # Find closest discrete beta in grid (step 0.05)
    rounded_beta = round(float(np.clip(beta_val, 0.0, 1.0)) * 20) / 20.0
    key_str = f"{rounded_beta:.2f}"
    if key_str in all_betas:
        return float(all_betas[key_str])

    # Fallback to nearest key
    keys = sorted([float(k) for k in all_betas.keys()])
    closest_key = min(keys, key=lambda k: abs(k - beta_val))
    return float(all_betas[f"{closest_key:.2f}"])


def train_mlp_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 150,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    device: str = "cpu",
) -> QueryMLPRegressor:
    model = QueryMLPRegressor(in_features=X_train.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            vx = torch.tensor(X_val).to(device)
            vy = torch.tensor(y_val).to(device)
            val_loss = criterion(model(vx), vy).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights:
        model.load_state_dict(best_weights)
    return model


def predict_qdyn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int = 7,
    temperature: float = 0.1,
) -> np.ndarray:
    """
    Qdyn (Dynamic Query Classes via k-NN with Cosine Similarity Weighting)
    """
    # Normalize query embeddings for cosine similarity
    X_train_norm = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-9)
    X_test_norm = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-9)

    # Cosine Similarity Matrix: [N_test, N_train]
    sim_matrix = np.dot(X_test_norm, X_train_norm.T)

    preds = []
    for i in range(len(X_test)):
        sims = sim_matrix[i]
        # Top-k neighbors
        top_k_indices = np.argpartition(sims, -k)[-k:]
        top_k_sims = sims[top_k_indices]
        top_k_betas = y_train[top_k_indices]

        # Softmax weighting over cosine similarities
        scaled_sims = (top_k_sims - np.max(top_k_sims)) / max(temperature, 1e-4)
        weights = np.exp(scaled_sims)
        weights = weights / (np.sum(weights) + 1e-9)

        pred_beta = np.sum(weights * top_k_betas)
        preds.append(float(np.clip(pred_beta, 0.0, 1.0)))

    return np.array(preds, dtype=np.float32)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 85)
    print(f"QAWFusion EXPERIMENTAL EVALUATION FRAMEWORK (Device: {device})")
    print("=" * 85)

    # 1. Load records and embeddings
    records, X = load_dataset_and_embeddings(args.oracle_json, args.embeddings_dir)
    y = np.array([r.get("oracle_beta", 0.2) for r in records], dtype=np.float32)

    total_samples = len(records)
    test_size = int(total_samples * args.test_split)
    train_size = total_samples - test_size

    # 2. Train / Test Split
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    # Extract train/val splits for MLP (85% train, 15% val inside train set)
    mlp_val_size = int(len(train_idx) * 0.15)
    mlp_train_idx = train_idx[mlp_val_size:]
    mlp_val_idx = train_idx[:mlp_val_size]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    test_records = [records[i] for i in test_idx]

    print(f"[*] Dataset split: Train={len(train_idx)} queries ({100*(1-args.test_split):.0f}%) | Test={len(test_idx)} queries ({100*args.test_split:.0f}%)")

    # Find best fixed constant beta on Train set (Grid baseline)
    candidate_constants = np.linspace(0.0, 1.0, 21)
    train_records = [records[i] for i in train_idx]
    best_const_beta = 0.2
    best_const_score = -1.0
    for c_beta in candidate_constants:
        score = np.mean([get_ndcg5_at_beta(r, c_beta) for r in train_records])
        if score > best_const_score:
            best_const_score = score
            best_const_beta = c_beta

    print(f"[*] Best Train-Tuned Fixed Beta: {best_const_beta:.2f} (Train NDCG@5 = {best_const_score:.4f})")

    # 3. Train MLP Model
    print(f"\n[*] Training Query-MLP Regressor on {len(mlp_train_idx)} samples ({args.mlp_epochs} epochs)...")
    mlp_model = train_mlp_model(
        X_train=X[mlp_train_idx],
        y_train=y[mlp_train_idx],
        X_val=X[mlp_val_idx],
        y_val=y[mlp_val_idx],
        epochs=args.mlp_epochs,
        lr=args.mlp_lr,
        hidden_dim=args.mlp_hidden,
        device=device,
    )

    # Predict MLP on test set
    mlp_model.eval()
    with torch.no_grad():
        mlp_pred_betas = mlp_model(torch.tensor(X_test).to(device)).cpu().numpy()

    # 4. Predict Qdyn on test set
    print(f"[*] Running Qdyn (k-NN Cosine Similarity, k={args.k_neighbors}, tau={args.qdyn_temp})...")
    qdyn_pred_betas = predict_qdyn(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        k=args.k_neighbors,
        temperature=args.qdyn_temp,
    )

    # 5. Evaluate All Methods on Test Set
    methods = {
        "1. Pure Visual (beta=0.0)": np.zeros(len(test_idx)),
        "2. Pure Text (beta=1.0)": np.ones(len(test_idx)),
        "3. Equal Weight (beta=0.5)": np.full(len(test_idx), 0.5),
        "4. CMRAG Static (beta=0.20)": np.full(len(test_idx), 0.20),
        f"5. Tuned Static (beta={best_const_beta:.2f})": np.full(len(test_idx), best_const_beta),
        "6. QAWFusion - MLP (Proposed)": mlp_pred_betas,
        "7. QAWFusion - Qdyn (Proposed)": qdyn_pred_betas,
        "8. Oracle Upper Bound (beta*)": y_test,
    }

    results_table = []
    print("\n" + "=" * 90)
    print(f"{'Method / Model':<35} | {'NDCG@5':<10} | {'MAE (Beta)':<12} | {'vs CMRAG (Rel %)':<16}")
    print("-" * 90)

    cmrag_ndcg_scores = [get_ndcg5_at_beta(r, 0.20) for r in test_records]
    cmrag_mean_ndcg = float(np.mean(cmrag_ndcg_scores))

    summary_dict = {}

    for name, pred_betas in methods.items():
        ndcg_scores = [get_ndcg5_at_beta(test_records[i], float(pred_betas[i])) for i in range(len(test_records))]
        mean_ndcg = float(np.mean(ndcg_scores))
        mae = float(np.mean(np.abs(pred_betas - y_test)))

        diff_pct = ((mean_ndcg - cmrag_mean_ndcg) / max(cmrag_mean_ndcg, 1e-9)) * 100.0
        diff_str = f"{diff_pct:+.2f}%" if "CMRAG" not in name else "Baseline"

        results_table.append({
            "Method": name,
            "NDCG@5": round(mean_ndcg, 4),
            "MAE_Beta": round(mae, 4),
            "Relative_Gain_pct": round(diff_pct, 2),
        })

        summary_dict[name] = {
            "NDCG@5": mean_ndcg,
            "MAE_Beta": mae,
            "Relative_Gain": diff_pct,
            "Scores": ndcg_scores,
        }

        print(f"{name:<35} | {mean_ndcg:<10.4f} | {mae:<12.4f} | {diff_str:<16}")

    print("=" * 90)

    # 6. Breakdown by Evidence Modality Type (Chart vs Table vs Text)
    print("\n" + "=" * 90)
    print("SUB-GROUP BREAKDOWN BY EVIDENCE SOURCE (NDCG@5)")
    print("=" * 90)

    source_groups: Dict[str, List[int]] = {}
    for i, r in enumerate(test_records):
        srcs = r.get("evidence_sources", ["Unknown"])
        if isinstance(srcs, list) and len(srcs) > 0:
            src_label = str(srcs[0])
        else:
            src_label = str(srcs)
        source_groups.setdefault(src_label, []).append(i)

    print(f"{'Evidence Source':<20} | {'Count':<6} | {'Visual':<8} | {'Text':<8} | {'CMRAG (0.2)':<12} | {'MLP':<8} | {'Qdyn':<8} | {'Oracle':<8}")
    print("-" * 90)

    breakdown_results = {}
    for src_name, group_indices in sorted(source_groups.items(), key=lambda x: -len(x[1])):
        if len(group_indices) < 2:
            continue
        g_vis = np.mean([summary_dict["1. Pure Visual (beta=0.0)"]["Scores"][i] for i in group_indices])
        g_txt = np.mean([summary_dict["2. Pure Text (beta=1.0)"]["Scores"][i] for i in group_indices])
        g_cmr = np.mean([summary_dict["4. CMRAG Static (beta=0.20)"]["Scores"][i] for i in group_indices])
        g_mlp = np.mean([summary_dict["6. QAWFusion - MLP (Proposed)"]["Scores"][i] for i in group_indices])
        g_qdy = np.mean([summary_dict["7. QAWFusion - Qdyn (Proposed)"]["Scores"][i] for i in group_indices])
        g_orc = np.mean([summary_dict["8. Oracle Upper Bound (beta*)"]["Scores"][i] for i in group_indices])

        breakdown_results[src_name] = {
            "count": len(group_indices),
            "Visual": round(float(g_vis), 4),
            "Text": round(float(g_txt), 4),
            "CMRAG": round(float(g_cmr), 4),
            "MLP": round(float(g_mlp), 4),
            "Qdyn": round(float(g_qdy), 4),
            "Oracle": round(float(g_orc), 4),
        }

        print(f"{src_name:<20} | {len(group_indices):<6} | {g_vis:<8.4f} | {g_txt:<8.4f} | {g_cmr:<12.4f} | {g_mlp:<8.4f} | {g_qdy:<8.4f} | {g_orc:<8.4f}")

    print("=" * 90)

    # 7. Save outputs to Google Drive
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "fusion_experiment_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "overall_results": results_table,
            "modality_breakdown": breakdown_results,
            "seed": args.seed,
            "test_count": len(test_idx),
            "train_count": len(train_idx),
        }, f, indent=2)

    print(f"\n[+] Full experimental results saved to: {out_file}")
    print("[SUCCESS] Evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()
