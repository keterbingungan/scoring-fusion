"""
Adaptive Multimodal Scoring Fusion for Visual Document RAG
Penulis: Sasi Kirana Hardianti (FATISDA UNS)
Judul: Optimasi Penggabungan Skor Multimodal pada Visual Document
       Retrieval-Augmented Generation Menggunakan Pembobotan Dinamis Berbasis Kueri

Deskripsi:
Skrip ini melatih dan mengevaluasi model pembobotan fusi dinamis berbasis kueri (MLP)
dan membandingkannya dengan berbagai baseline:
  1. Pure Visual Retrieval (beta = 0.0)
  2. Pure Text Retrieval (beta = 1.0)
  3. Equal Weighting (beta = 0.5)
  4. CMRAG Static Baseline (beta = 0.20)
  5. Tuned Static Baseline (beta konstan terbaik dari set training)
  6. k-NN Baseline / Qdyn (Similarity-weighted NDCG curve)
  7. Dynamic MLP (Metode yang diajukan: Residual / Standard MLP + Expected NDCG Loss)
  8. Oracle Upper Bound (beta optimal per kueri)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Tuple, Any


# ---------------------------------------------------------------------------
# 1. Konstanta & Hyperparameter Bins
# ---------------------------------------------------------------------------
# Rentang bobot beta visual vs teks dibagi menjadi 21 bin diskret [0.00, 0.05, ..., 1.00]
BETA_BINS = np.round(np.linspace(0.0, 1.0, 21), 2)  # [0.00, 0.05, ..., 1.00]
NUM_BINS = len(BETA_BINS)  # 21
BETA_TENSOR = torch.tensor(BETA_BINS, dtype=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pelatihan dan Evaluasi Dynamic Multimodal Scoring Fusion untuk Visual Doc RAG"
    )
    parser.add_argument(
        "--oracle_json",
        type=str,
        default=os.environ.get("ORACLE_JSON", "oracle_dataset/oracle_dataset.json"),
        help="Path ke file oracle_dataset.json",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=os.environ.get("EMB_DIR", "oracle_dataset/query_embeddings"),
        help="Direktori berisi embedding kueri (.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", "results"),
        help="Direktori untuk menyimpan hasil evaluasi JSON",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Proporsi data pengujian (default: 0.2 = 20%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed untuk reprodusibilitas eksperimen",
    )
    parser.add_argument(
        "--filter_noise",
        type=int,
        default=1,
        help="1 = filter kueri tanpa evidence_sources yang valid, 0 = gunakan semua",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default=os.environ.get("LOSS_TYPE", "expected_ndcg"),
        choices=["expected_ndcg", "soft_kl", "hybrid", "ce_smooth"],
        help="Fungsi loss untuk optimasi MLP (default: expected_ndcg)",
    )
    parser.add_argument(
        "--soft_target_tau",
        type=float,
        default=float(os.environ.get("SOFT_TARGET_TAU", 0.08)),
        help="Temperatur untuk distribusi target NDCG pada Soft Target KL Loss",
    )
    parser.add_argument(
        "--entropy_reg",
        type=float,
        default=float(os.environ.get("ENTROPY_REG", 0.01)),
        help="Koefisien regularisasi entropi untuk Expected NDCG Loss",
    )
    args, _ = parser.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# 2. Arsitektur Model: Standard MLP & Residual MLP
# ---------------------------------------------------------------------------
class BetaClassifierMLP(nn.Module):
    """
    Standard Feed-Forward MLP:
    Menerima embedding kueri + fitur metadata, memprediksi logits untuk 21 bin beta.
    """
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
        return self.net(x)


class ResBlock(nn.Module):
    """Blok Residual dengan LayerNorm, GELU, dan Dropout untuk propagasi gradien yang stabil."""
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ResidualBetaMLP(nn.Module):
    """
    Residual MLP:
    Menggunakan skip connections dan LayerNorm untuk pelatihan yang lebih stabil
    pada kapasitas model yang lebih dalam.
    """
    def __init__(self, in_features: int, hidden_dim: int = 512, dropout: float = 0.25):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res1 = ResBlock(hidden_dim, dropout=dropout)
        self.down1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res2 = ResBlock(hidden_dim // 2, dropout=dropout * 0.8)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, NUM_BINS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.res1(h)
        h = self.down1(h)
        h = self.res2(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# 3. Fungsi Loss: Ranking-Aware & Utility-Driven
# ---------------------------------------------------------------------------
class ExpectedNDCGLoss(nn.Module):
    """
    Expected NDCG Loss (Utility-Driven):
    Memaksimalkan secara langsung ekspektasi reward NDCG@5 pada distribusi probabilitas output.
    L = - E_{b ~ p}[NDCG(b)] - alpha * Entropy(p)
    """
    def __init__(self, entropy_coef: float = 0.01):
        super().__init__()
        self.entropy_coef = entropy_coef

    def forward(self, logits: torch.Tensor, ndcg_curves: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        expected_ndcg = torch.sum(probs * ndcg_curves, dim=-1)
        utility_loss = -torch.mean(expected_ndcg)

        if self.entropy_coef > 0.0:
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1).mean()
            return utility_loss - self.entropy_coef * entropy

        return utility_loss


class SoftTargetKLLoss(nn.Module):
    """
    Soft-Target Distribution Matching via KL-Divergence:
    Menyelaraskan distribusi prediksi model dengan kurva kehalusan NDCG empiris.
    Target Q_i = softmax(NDCG_curve / tau)
    L = KL(Q || P)
    """
    def __init__(self, tau: float = 0.08):
        super().__init__()
        self.tau = tau

    def forward(self, logits: torch.Tensor, ndcg_curves: torch.Tensor) -> torch.Tensor:
        target_q = F.softmax(ndcg_curves / self.tau, dim=-1)
        log_pred_p = F.log_softmax(logits, dim=-1)
        return F.kl_div(log_pred_p, target_q, reduction="batchmean")


class HybridRankingLoss(nn.Module):
    """Gabungan antara Expected NDCG Loss dan Soft-Target KL Loss."""
    def __init__(self, tau: float = 0.08, kl_weight: float = 0.5, entropy_coef: float = 0.005):
        super().__init__()
        self.expected_loss = ExpectedNDCGLoss(entropy_coef=entropy_coef)
        self.kl_loss = SoftTargetKLLoss(tau=tau)
        self.kl_weight = kl_weight

    def forward(self, logits: torch.Tensor, ndcg_curves: torch.Tensor) -> torch.Tensor:
        l_exp = self.expected_loss(logits, ndcg_curves)
        l_kl = self.kl_loss(logits, ndcg_curves)
        return l_exp + self.kl_weight * l_kl


# ---------------------------------------------------------------------------
# 4. Pemuatan Data & Ekstraksi Fitur
# ---------------------------------------------------------------------------
def get_full_ndcg_curve(record: Dict) -> np.ndarray:
    """Mengambil nilai NDCG@5 untuk seluruh 21 bin beta sebagai array 1D float32."""
    all_betas = record.get("all_betas_ndcg5", {})
    curve = np.zeros(NUM_BINS, dtype=np.float32)
    for i, b in enumerate(BETA_BINS):
        key = f"{b:.2f}"
        curve[i] = float(all_betas.get(key, 0.0))
    return curve


def get_ndcg5_at_beta(record: Dict, beta_val: float) -> float:
    """Evaluasi NDCG@5 pada nilai beta kontinu dengan mencocokkan ke bin terdekat."""
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


def load_dataset_and_embeddings(oracle_json_path: str, emb_dir: str, filter_noise: bool = True):
    json_candidates = [
        oracle_json_path,
        "oracle_dataset/oracle_dataset.json",
        "../oracle_dataset/oracle_dataset.json",
        "/content/drive/MyDrive/TA/CMRAG/oracle_dataset/oracle_dataset.json",
    ]
    resolved_json = next((p for p in json_candidates if os.path.exists(p)), None)
    if not resolved_json:
        raise FileNotFoundError(f"File oracle_dataset.json tidak ditemukan. Dicari di: {json_candidates}")

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
        raise FileNotFoundError("Direktori query_embeddings tidak ditemukan.")

    print(f"[*] Berhasil memuat {len(records)} data kueri dari: {resolved_json}")

    # Filter data kueri tanpa evidence sources yang valid
    if filter_noise:
        before = len(records)
        records = [r for r in records if r.get("evidence_sources") not in ([], [""], None, "")]
        print(f"[*] Filter noise: {before} -> {len(records)} kueri (dihapus {before - len(records)} kueri kosong)")

    valid_records = []
    features_list = []
    ndcg_curves_list = []

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

            # Normalisasi fitur tambahan jumlah halaman dokumen
            num_pages = float(r.get("num_pages", 1))
            extra_feat = np.array([num_pages / 100.0], dtype=np.float32)

            full_feat = np.concatenate([emb.float().numpy(), extra_feat])
            features_list.append(full_feat)

            # Kurva NDCG empiris untuk 21 bin
            curve = get_full_ndcg_curve(r)
            ndcg_curves_list.append(curve)

            valid_records.append(r)
        except Exception:
            continue

    X = np.array(features_list, dtype=np.float32)
    R = np.array(ndcg_curves_list, dtype=np.float32)
    print(f"[*] Terpasangkan {len(valid_records)} kueri dengan embedding (Dimensi Fitur: {X.shape[1]}, Dimensi Kurva NDCG: {R.shape[1]})")
    return valid_records, X, R


# ---------------------------------------------------------------------------
# 5. Pelatihan Model & Evaluasi Validasi
# ---------------------------------------------------------------------------
def train_mlp_model(
    X_train: np.ndarray, R_train: np.ndarray, y_labels_train: np.ndarray,
    X_val: np.ndarray, R_val: np.ndarray, val_records: List[Dict],
    arch: str = "standard", hidden_dim: int = 512, dropout: float = 0.3,
    epochs: int = 250, lr: float = 5e-4, device: str = "cpu",
    loss_type: str = "expected_ndcg", tau: float = 0.08, entropy_reg: float = 0.01,
) -> Tuple[nn.Module, float, str]:
    """Melatih satu konfigurasi model MLP dan mengevaluasi performa pada set validasi."""
    if arch == "residual":
        model = ResidualBetaMLP(in_features=X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    else:
        model = BetaClassifierMLP(in_features=X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Inisialisasi fungsi loss
    if loss_type == "expected_ndcg":
        criterion = ExpectedNDCGLoss(entropy_coef=entropy_reg)
    elif loss_type == "soft_kl":
        criterion = SoftTargetKLLoss(tau=tau)
    elif loss_type == "hybrid":
        criterion = HybridRankingLoss(tau=tau, kl_weight=0.5, entropy_coef=entropy_reg)
    elif loss_type == "ce_smooth":
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        raise ValueError(f"Fungsi loss tidak dikenali: {loss_type}")

    t_X_train = torch.tensor(X_train, dtype=torch.float32)
    t_R_train = torch.tensor(R_train, dtype=torch.float32)
    t_Y_train = torch.tensor(y_labels_train, dtype=torch.long)

    train_ds = TensorDataset(t_X_train, t_R_train, t_Y_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    beta_bins_dev = torch.tensor(BETA_BINS, dtype=torch.float32).to(device)
    best_val_ndcg = -1.0
    best_weights = None
    best_mode = "soft"

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, br, by in train_loader:
            bx, br, by = bx.to(device), br.to(device), by.to(device)
            optimizer.zero_grad()

            logits = model(bx)
            if loss_type == "ce_smooth":
                loss = criterion(logits, by)
            else:
                loss = criterion(logits, br)

            loss.backward()
            optimizer.step()
        scheduler.step()

        # Evaluasi pada set validasi secara berkala (setiap 10 epoch atau epoch terakhir)
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.tensor(X_val, dtype=torch.float32).to(device))
                val_probs = F.softmax(val_logits, dim=-1)

                # Mode Soft-Beta: ekspektasi kontinyu beta = sum(p * beta)
                pred_betas_soft = torch.sum(val_probs * beta_bins_dev, dim=-1).cpu().numpy()
                val_ndcg_soft = np.mean([get_ndcg5_at_beta(val_records[i], float(pred_betas_soft[i])) for i in range(len(val_records))])

                # Mode Argmax-Beta: pemilihan bin tertinggi beta[argmax(p)]
                pred_argmax_bins = val_logits.argmax(dim=1).cpu().numpy()
                pred_betas_argmax = BETA_BINS[pred_argmax_bins]
                val_ndcg_argmax = np.mean([get_ndcg5_at_beta(val_records[i], float(pred_betas_argmax[i])) for i in range(len(val_records))])

                curr_best = max(val_ndcg_soft, val_ndcg_argmax)
                curr_mode = "soft" if val_ndcg_soft >= val_ndcg_argmax else "argmax"

                if curr_best > best_val_ndcg:
                    best_val_ndcg = curr_best
                    best_mode = curr_mode
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights:
        model.load_state_dict(best_weights)
    print(f"    Val NDCG@5 ({loss_type}) [{best_mode}]: {best_val_ndcg:.4f}")
    return model, best_val_ndcg, best_mode


# ---------------------------------------------------------------------------
# 6. Baseline k-NN (Qdyn NDCG-Aware)
# ---------------------------------------------------------------------------
def predict_qdyn_ndcg_aware(
    X_train: np.ndarray, train_records: List[Dict],
    X_test: np.ndarray,
    k: int = 10, temperature: float = 0.15,
) -> np.ndarray:
    """
    Baseline k-NN non-parametrik:
    Mencari k kueri pelatihan paling mirip berdasarkan cosine similarity,
    lalu memilih beta yang memaksimalkan rata-rata berbobot kurva NDCG tetangga.
    """
    X_train_n = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-9)
    X_test_n = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-9)
    sim_matrix = X_test_n @ X_train_n.T  # [N_test, N_train]

    train_curves = np.array([get_full_ndcg_curve(r) for r in train_records])  # [N_train, 21]

    preds = []
    for i in range(len(X_test)):
        sims = sim_matrix[i]
        top_k_idx = np.argpartition(sims, -k)[-k:]
        top_k_sims = sims[top_k_idx]

        scaled = (top_k_sims - np.max(top_k_sims)) / max(temperature, 1e-4)
        weights = np.exp(scaled)
        weights /= (weights.sum() + 1e-9)

        neighbor_curves = train_curves[top_k_idx]
        avg_curve = weights @ neighbor_curves
        best_bin = int(np.argmax(avg_curve))
        preds.append(float(BETA_BINS[best_bin]))

    return np.array(preds, dtype=np.float32)


# ---------------------------------------------------------------------------
# 7. Hyperparameter Tuning untuk Model MLP
# ---------------------------------------------------------------------------
def search_mlp_hyperparams(
    X_train, R_train, y_labels_train,
    X_val, R_val, val_records,
    device, loss_type, tau, entropy_reg
):
    """
    Grid search terstruktur pada set validasi:
    Membandingkan arsitektur (Standard vs Residual), dimensi tersembunyi, dropout,
    dan learning rate untuk mendapatkan konfigurasi terbaik.
    """
    configs = [
        # 1. Standard MLP (512 dimensi)
        {"arch": "standard", "hidden_dim": 512, "dropout": 0.30, "lr": 5e-4, "epochs": 250, "entropy": entropy_reg},
        # 2. Residual MLP (512 dimensi, Skip Connection + LayerNorm)
        {"arch": "residual", "hidden_dim": 512, "dropout": 0.25, "lr": 5e-4, "epochs": 250, "entropy": entropy_reg},
        # 3. Residual MLP (512 dimensi, entropi lebih tajam 0.005, 300 epoch)
        {"arch": "residual", "hidden_dim": 512, "dropout": 0.20, "lr": 4e-4, "epochs": 300, "entropy": 0.005},
        # 4. High-Capacity Residual MLP (768 dimensi)
        {"arch": "residual", "hidden_dim": 768, "dropout": 0.30, "lr": 3e-4, "epochs": 250, "entropy": entropy_reg},
        # 5. High-Capacity Standard MLP (768 dimensi)
        {"arch": "standard", "hidden_dim": 768, "dropout": 0.30, "lr": 3e-4, "epochs": 250, "entropy": 0.005},
        # 6. Compact Residual MLP (384 dimensi)
        {"arch": "residual", "hidden_dim": 384, "dropout": 0.20, "lr": 6e-4, "epochs": 200, "entropy": 0.01},
    ]

    best_model = None
    best_ndcg = -1.0
    best_cfg = None

    for cfg in configs:
        cfg_ent = cfg.get("entropy", entropy_reg)
        print(f"  [Tuning] arch={cfg['arch']}, dim={cfg['hidden_dim']}, drop={cfg['dropout']}, lr={cfg['lr']}, ep={cfg['epochs']}, ent={cfg_ent}...")
        model, val_ndcg, mode = train_mlp_model(
            X_train, R_train, y_labels_train,
            X_val, R_val, val_records,
            arch=cfg["arch"], hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
            epochs=cfg["epochs"], lr=cfg["lr"], device=device,
            loss_type=loss_type, tau=tau, entropy_reg=cfg_ent,
        )

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_model = model
            best_cfg = {**cfg, "val_ndcg": round(float(val_ndcg), 4), "preferred_mode": mode}

    print(f"\n  [Tuning Selesai] Konfigurasi Terbaik: {best_cfg} -> Val NDCG@5 = {best_ndcg:.4f}")
    return best_model, best_cfg


# ---------------------------------------------------------------------------
# 8. Eksekusi Utama (Main Pipeline)
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 95)
    print(f"PELATIHAN & EVALUASI DYNAMIC MULTIMODAL FUSION (Device: {device})")
    print(f"Loss Function: {args.loss_type.upper()}")
    print("=" * 95)

    # 1. Muat Dataset dan Embedding
    records, X, R = load_dataset_and_embeddings(
        args.oracle_json, args.embeddings_dir, filter_noise=bool(args.filter_noise)
    )
    y_betas = np.array([r.get("oracle_beta", 0.2) for r in records], dtype=np.float32)
    y_labels = np.clip(np.round(y_betas * 20), 0, 20).astype(np.int64)

    # 2. Pembagian Data (Train, Validation, Test)
    total = len(records)
    test_size = int(total * args.test_split)
    indices = np.arange(total)
    np.random.shuffle(indices)

    train_idx = indices[test_size:]
    test_idx = indices[:test_size]

    val_size = int(len(train_idx) * 0.15)
    val_idx = train_idx[:val_size]
    pure_train_idx = train_idx[val_size:]

    X_train_full = X[train_idx]
    y_betas_test = y_betas[test_idx]
    test_records = [records[i] for i in test_idx]
    train_records_full = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]

    print(f"[*] Pembagian Data: Train={len(pure_train_idx)} | Val={len(val_idx)} | Test={len(test_idx)}")

    # Cari konstanta beta terbaik pada set training
    best_const_beta = 0.2
    best_const_score = -1.0
    for b in BETA_BINS:
        score = np.mean([get_ndcg5_at_beta(r, b) for r in train_records_full])
        if score > best_const_score:
            best_const_score = score
            best_const_beta = b
    print(f"[*] Tuned Static Beta Terbaik (Train): {best_const_beta:.2f} (Train NDCG@5 = {best_const_score:.4f})")

    # 3. Hyperparameter Tuning & Pelatihan Model MLP
    print(f"\n{'='*60}")
    print(f"[*] Menjalankan Hyperparameter Tuning MLP...")
    print(f"{'='*60}")
    best_mlp, best_cfg = search_mlp_hyperparams(
        X[pure_train_idx], R[pure_train_idx], y_labels[pure_train_idx],
        X[val_idx], R[val_idx], val_records,
        device=device, loss_type=args.loss_type,
        tau=args.soft_target_tau, entropy_reg=args.entropy_reg,
    )

    # Prediksi Pengujian dengan MLP Terbaik: Mode Soft-Beta dan Argmax-Beta
    best_mlp.eval()
    beta_bins_dev = torch.tensor(BETA_BINS, dtype=torch.float32).to(device)
    with torch.no_grad():
        test_logits = best_mlp(torch.tensor(X[test_idx], dtype=torch.float32).to(device))
        test_probs = F.softmax(test_logits, dim=-1)

        # Mode A: Soft-Beta (Continuous Expectation: sum(p * beta))
        mlp_soft_betas = torch.sum(test_probs * beta_bins_dev, dim=-1).cpu().numpy()

        # Mode B: Argmax-Beta (Discrete Selection: beta[argmax(p)])
        mlp_argmax_bins = test_logits.argmax(dim=1).cpu().numpy()
        mlp_argmax_betas = BETA_BINS[mlp_argmax_bins]

    # 4. Prediksi Baseline k-NN (Qdyn)
    print(f"\n{'='*60}")
    print("[*] Menjalankan Baseline k-NN / Qdyn (k=10, tau=0.15)...")
    print(f"{'='*60}")
    qdyn_pred_betas = predict_qdyn_ndcg_aware(
        X_train_full, train_records_full,
        X[test_idx],
        k=10, temperature=0.15,
    )

    # 5. Tabel Perbandingan Semua Metode
    methods = {
        "1. Pure Visual (β=0.0)":               np.zeros(len(test_idx)),
        "2. Pure Text (β=1.0)":                 np.ones(len(test_idx)),
        "3. Equal Weight (β=0.5)":              np.full(len(test_idx), 0.5),
        "4. CMRAG Static (β=0.20)":             np.full(len(test_idx), 0.20),
        f"5. Tuned Static (β={best_const_beta:.2f})": np.full(len(test_idx), best_const_beta),
        "6. Dynamic MLP (Soft-Beta)":           mlp_soft_betas,
        "7. Dynamic MLP (Argmax-Beta)":         mlp_argmax_betas,
        "8. k-NN Baseline / Qdyn (k=10)":       qdyn_pred_betas,
        "9. Oracle Upper Bound (β*)":           y_betas_test,
    }

    cmrag_scores = [get_ndcg5_at_beta(r, 0.20) for r in test_records]
    cmrag_mean = float(np.mean(cmrag_scores))

    print(f"\n{'='*100}")
    print(f"{'Metode':<42} | {'NDCG@5':>8} | {'MAE(β)':>8} | {'vs CMRAG':>12} | {'Δ Absolut':>10}")
    print(f"{'-'*100}")

    results_table = []
    summary = {}

    for name, pred_betas in methods.items():
        scores = [get_ndcg5_at_beta(test_records[i], float(pred_betas[i])) for i in range(len(test_records))]
        mean_ndcg = float(np.mean(scores))
        mae = float(np.mean(np.abs(pred_betas - y_betas_test)))
        rel_pct = ((mean_ndcg - cmrag_mean) / max(cmrag_mean, 1e-9)) * 100.0
        abs_diff = mean_ndcg - cmrag_mean
        rel_str = f"{rel_pct:+.2f}%" if "CMRAG Static" not in name else "Baseline"

        results_table.append({
            "Method": name,
            "NDCG@5": round(mean_ndcg, 4),
            "MAE_Beta": round(mae, 4),
            "Relative_Gain_pct": round(rel_pct, 2),
        })
        summary[name] = {"NDCG@5": mean_ndcg, "MAE_Beta": mae, "Scores": scores}

        print(f"{name:<42} | {mean_ndcg:>8.4f} | {mae:>8.4f} | {rel_str:>12} | {abs_diff:>+10.4f}")

    print(f"{'='*100}")

    # 6. Analisis Sub-Group Berdasarkan Jenis Bukti Modalitas (Evidence Source)
    print(f"\n{'='*105}")
    print("ANALISIS PERFORMA SUB-GROUP BERDASARKAN EVIDENCE SOURCE (NDCG@5)")
    print(f"{'='*105}")

    source_groups: Dict[str, List[int]] = {}
    for i, r in enumerate(test_records):
        srcs = r.get("evidence_sources", ["Unknown"])
        src_label = str(srcs[0]) if isinstance(srcs, list) and len(srcs) > 0 else str(srcs)
        source_groups.setdefault(src_label, []).append(i)

    print(f"{'Evidence Source':<25} | {'N':>4} | {'Visual':>7} | {'Text':>7} | {'CMRAG':>7} | {'MLP-Soft':>8} | {'k-NN':>7} | {'Oracle':>7}")
    print(f"{'-'*105}")

    breakdown = {}
    for src, gidx in sorted(source_groups.items(), key=lambda x: -len(x[1])):
        if len(gidx) < 2:
            continue
        row = {}
        for mname, mkey in [
            ("Visual", "1. Pure Visual (β=0.0)"),
            ("Text", "2. Pure Text (β=1.0)"),
            ("CMRAG", "4. CMRAG Static (β=0.20)"),
            ("MLP-Soft", "6. Dynamic MLP (Soft-Beta)"),
            ("k-NN", "8. k-NN Baseline / Qdyn (k=10)"),
            ("Oracle", "9. Oracle Upper Bound (β*)"),
        ]:
            row[mname] = round(float(np.mean([summary[mkey]["Scores"][i] for i in gidx])), 4)
        breakdown[src] = {"count": len(gidx), **row}
        print(f"{src:<25} | {len(gidx):>4} | {row['Visual']:>7.4f} | {row['Text']:>7.4f} | {row['CMRAG']:>7.4f} | {row['MLP-Soft']:>8.4f} | {row['k-NN']:>7.4f} | {row['Oracle']:>7.4f}")

    print(f"{'='*105}")

    # 7. Simpan Hasil Evaluasi
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"fusion_experiment_results_{args.loss_type}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "loss_type": args.loss_type,
            "overall_results": results_table,
            "modality_breakdown": breakdown,
            "best_mlp_config": best_cfg,
            "seed": args.seed,
            "test_count": len(test_idx),
            "train_count": len(train_idx),
            "filtered_noise": bool(args.filter_noise),
        }, f, indent=2)

    print(f"\n[+] Hasil tersimpan di: {out_file}")
    print(f"[SELESAI] Pelatihan dan evaluasi selesai secara sukses!")


if __name__ == "__main__":
    main()
