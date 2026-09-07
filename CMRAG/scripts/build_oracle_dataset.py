import os
import sys
import json
import ast
import re
import tarfile
import argparse
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Setup sys.path for IPython / Jupyter / Colab execution compatibility
# ---------------------------------------------------------------------------
try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()

candidate_roots = [
    CURRENT_DIR,
    os.path.dirname(CURRENT_DIR),
    os.getcwd(),
    os.path.dirname(os.getcwd()),
    "/content/drive/MyDrive/TA/CMRAG",
    "/content/drive/MyDrive/TA/code/CMRAG",
    "/content/drive/MyDrive/TA/code",
    "/content/CMRAG",
    "/content/scoring-fusion",
    "/content",
]

found_root = None
for root in candidate_roots:
    if os.path.exists(os.path.join(root, "retriever", "retriever.py")):
        found_root = root
        if root not in sys.path:
            sys.path.insert(0, root)
        break

if not found_root:
    for p in candidate_roots:
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)

from bs4 import BeautifulSoup
import torch.nn as nn
from transformers import SiglipModel, SiglipProcessor

def _extract_tensor(feat):
    if isinstance(feat, torch.Tensor):
        return feat
    if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
        return feat.pooler_output
    return feat[0]

def interpolate_pos_embed(model, new_len: int):
    old = model.embeddings.position_embedding.weight
    new = F.interpolate(
        old.unsqueeze(0).permute(0, 2, 1),
        size=new_len,
        mode="linear",
        align_corners=False,
    ).permute(0, 2, 1).squeeze(0)
    model.embeddings.position_embedding = nn.Embedding.from_pretrained(new, freeze=False)
    model.embeddings.position_ids = torch.arange(new_len).unsqueeze(0)

try:
    from retriever.retriever import Retriever
    from retriever.model.encoder import LongSigLIP
    from retriever.model.processor import LongSigLIPProcessor
except ImportError:
    print("[*] Module 'retriever' not found in path. Using inline CMRAG LongSigLIP architecture...")

    class LongSigLIP(nn.Module):
        def __init__(self, model_id: str = "google/siglip-so400m-patch14-384", max_text_len: int = 768, train: bool = False):
            super().__init__()
            self.siglip = SiglipModel.from_pretrained(model_id)
            for p in self.siglip.parameters():
                p.requires_grad_(False)
            txt_cfg = self.siglip.text_model.config
            self.long_text_enc = type(self.siglip.text_model)(txt_cfg)
            self.long_text_enc.load_state_dict(self.siglip.text_model.state_dict(), strict=False)
            txt_cfg.max_position_embeddings = max_text_len
            interpolate_pos_embed(self.long_text_enc, max_text_len)
            for p in self.long_text_enc.parameters():
                if not p.is_contiguous():
                    p.data = p.data.contiguous()
            self.text_weight = nn.Parameter(torch.tensor(0.2))
            self.scale = nn.Parameter(torch.tensor(10.0))
            self.bias = nn.Parameter(torch.tensor(-10.0))

        def encode_img_text(
            self, pixel_values: torch.Tensor, input_ids_t: torch.Tensor, attention_mask_t: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                z_i = F.normalize(_extract_tensor(self.siglip.get_image_features(pixel_values)), dim=-1)
                t_out = self.long_text_enc(input_ids=input_ids_t, attention_mask=attention_mask_t)
                z_t = F.normalize(_extract_tensor(t_out), dim=-1)
            return z_i, z_t

        def encode_query(self, input_ids_q: torch.Tensor, attention_mask_q: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                z_q = F.normalize(
                    _extract_tensor(self.siglip.get_text_features(input_ids=input_ids_q, attention_mask=attention_mask_q)),
                    dim=-1,
                )
            return z_q

    class LongSigLIPProcessor:
        def __init__(self, model_id: str, max_query: int = 64, max_text: int = 768):
            self.core = SiglipProcessor.from_pretrained(model_id)
            self.max_q = max_query
            self.max_t = max_text

        def process_queries(self, queries: List[str]):
            tok = self.core.tokenizer
            q = tok(queries, max_length=self.max_q, truncation=True, padding="max_length", return_tensors="pt")
            q_mask = (q.input_ids != self.core.tokenizer.pad_token_id).int()
            return {"input_ids_q": q["input_ids"], "attention_mask_q": q_mask}

        def process_img_text(self, images, extracted_texts: List[str]):
            processed_imgs = []
            for img in images:
                if isinstance(img, Image.Image):
                    processed_imgs.append(img.convert("RGB"))
                else:
                    processed_imgs.append(Image.open(img).convert("RGB"))
            pixel_values = self.core(images=processed_imgs, return_tensors="pt").pixel_values
            tok = self.core.tokenizer
            t = tok(extracted_texts, max_length=self.max_t, truncation=True, padding="max_length", return_tensors="pt")
            t_mask = (t.input_ids != self.core.tokenizer.pad_token_id).int()
            return {
                "pixel_values": pixel_values,
                "input_ids_t": t["input_ids"],
                "attention_mask_t": t_mask,
            }

    class Retriever:
        def __init__(self, ckpt_path: str, max_text_len: int, model_id: str, device: str):
            self.ckpt_path = ckpt_path
            self.max_text_len = max_text_len
            self.model_id = model_id
            self.device = device
            self.load_model()

        def load_model(self):
            if self.ckpt_path and os.path.exists(self.ckpt_path):
                print(f"Loading model from checkpoint: {self.ckpt_path} ...")
                try:
                    from retriever.trainer import LitLongSigLIP
                    pl_model = LitLongSigLIP.load_from_checkpoint(self.ckpt_path).to(self.device)
                    model = pl_model.model
                except Exception as e:
                    print(f"Failed loading checkpoint: {e}. Falling back to LongSigLIP base architecture.")
                    model = LongSigLIP(model_id=self.model_id, max_text_len=self.max_text_len, train=False)
            else:
                print(f"Loading LongSigLIP base architecture (max_text_len={self.max_text_len})...")
                model = LongSigLIP(model_id=self.model_id, max_text_len=self.max_text_len, train=False)

            self.processor = LongSigLIPProcessor(model_id=self.model_id, max_text=self.max_text_len)
            self.model = model.to(self.device)

        def encode_query(self, query: str) -> torch.Tensor:
            with torch.no_grad():
                inputs = self.processor.process_queries([query])
                q_ids = inputs["input_ids_q"].to(self.device)
                q_mask = inputs["attention_mask_q"].to(self.device)
                z_q = F.normalize(
                    _extract_tensor(self.model.siglip.get_text_features(input_ids=q_ids, attention_mask=q_mask)),
                    dim=-1,
                )
            return z_q


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def load_html_file(file_path: str) -> str:
    """Load HTML content from file path safely."""
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def html_to_plain_text(html_content: str) -> str:
    """Convert HTML string to clean plain text."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    for element in soup(["script", "style", "img", "div", "header", "footer", "nav"]):
        element.decompose()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"\bhtml\b", "", text, flags=re.IGNORECASE)
    return text


def parse_list_field(val: Any) -> List:
    """Safely parse string representation of list e.g. '[5]' -> [5] or ['5'] -> ['5']."""
    if isinstance(val, list):
        return val
    if not val or not isinstance(val, str):
        return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except (ValueError, SyntaxError):
        return [val]


def extract_page_num(filename: str) -> int:
    """Extract numerical page number from filename e.g., '5.png' -> 5."""
    stem = os.path.splitext(filename)[0]
    match = re.search(r"(\d+)$", stem)
    if match:
        return int(match.group(1))
    return 999999


# ---------------------------------------------------------------------------
# Retrieval Evaluation Metric Functions
# ---------------------------------------------------------------------------
def compute_ndcg_at_k(ranked_page_ids: List[int], gt_page_ids: List[int], k: int = 5) -> float:
    """Compute NDCG@k for a single query's ranked pages against ground truth page IDs."""
    if not gt_page_ids:
        return 0.0

    ranked_k = ranked_page_ids[:k]
    rel = [1.0 if page in gt_page_ids else 0.0 for page in ranked_k]

    if sum(rel) == 0:
        return 0.0

    dcg = sum(r / np.log2(idx + 2) for idx, r in enumerate(rel))
    ideal_rel = sorted([1.0] * len(gt_page_ids), reverse=True)[:k]
    idcg = sum(r / np.log2(idx + 2) for idx, r in enumerate(ideal_rel))

    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_hit_at_k(ranked_page_ids: List[int], gt_page_ids: List[int], k: int = 5) -> float:
    """Compute HIT@k (Recall/Accuracy) for a single query."""
    ranked_k = set(ranked_page_ids[:k])
    gt_set = set(gt_page_ids)
    return 1.0 if len(ranked_k.intersection(gt_set)) > 0 else 0.0


def compute_mrr_at_k(ranked_page_ids: List[int], gt_page_ids: List[int], k: int = 5) -> float:
    """Compute MRR@k for a single query."""
    gt_set = set(gt_page_ids)
    for rank_idx, page in enumerate(ranked_page_ids[:k]):
        if page in gt_set:
            return 1.0 / (rank_idx + 1.0)
    return 0.0


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Oracle Fusion Beta Dataset via Grid Search on CMRAG-Bench using CMRAG LongSigLIP architecture."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.environ.get("DATASET", "MMLongBench"),
        help="CMRAG-Bench dataset split name (e.g. MMLongBench, finslides, techslides, techreport, finreport)",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=os.environ.get("JSON_PATH", ""),
        help="Optional override path to dataset JSON (will auto-resolve/download if empty)",
    )
    parser.add_argument(
        "--tar_path",
        type=str,
        default=os.environ.get("TAR_PATH", ""),
        help="Optional override path to images.tar (will auto-resolve/download if empty)",
    )
    parser.add_argument(
        "--extract_all_dir",
        type=str,
        default=os.environ.get("EXTRACT_DIR", "/content/extracted_all"),
        help="Local directory to pre-extract all documents from images.tar",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.environ.get("CACHE_DIR", "/content/embeddings_cache"),
        help="Local directory to store pre-computed image and text embeddings",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", "/content/drive/MyDrive/TA/CMRAG/oracle_dataset"),
        help="Directory on Google Drive to save oracle_dataset.json and embeddings",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.environ.get("CKPT_PATH", ""),
        help="Path to fine-tuned CMRAG .ckpt file if available (e.g. /content/drive/MyDrive/TA/CMRAG/long_siglip.ckpt)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=os.environ.get("MODEL_NAME", "google/siglip-so400m-patch14-384"),
        help="HuggingFace base model name for SigLIP",
    )
    parser.add_argument(
        "--max_text_len",
        type=int,
        default=int(os.environ.get("MAX_TEXT_LEN", 768)),
        help="Max text length token limit (768 tokens for LongSigLIP)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", 8)),
        help="Batch size for LongSigLIP image/text encoding",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=int(os.environ.get("MAX_QUERIES", -1)),
        help="Limit total queries to process (-1 for all queries)",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=int(os.environ.get("START_INDEX", 0)),
        help="Starting query index for batch processing",
    )
    parser.add_argument(
        "--grid_step",
        type=float,
        default=float(os.environ.get("GRID_STEP", 0.05)),
        help="Step size for beta grid search in range [0.0, 1.0]",
    )
    parser.add_argument(
        "--eval_k",
        type=int,
        default=int(os.environ.get("EVAL_K", 5)),
        help="Cutoff k for NDCG@k evaluation during grid search",
    )
    args, _ = parser.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Pipeline Phase 1: Pre-extract tar archive to fast local disk
# ---------------------------------------------------------------------------
def pre_extract_tar_if_needed(tar_path: str, extract_dir: str):
    """Extract all document folders from images.tar to local disk if not already done."""
    marker_file = os.path.join(extract_dir, ".extraction_complete")
    if os.path.exists(marker_file):
        print(f"[*] Pre-extracted dataset already exists at: {extract_dir}")
        return

    print(f"[*] Pre-extracting ALL document folders from '{tar_path}' to '{extract_dir}' ...")
    print("    (This will take ~3-5 minutes on Colab local disk, but makes encoding fast)")
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_dir)

    with open(marker_file, "w") as f:
        f.write("complete")
    print("[*] Bulk extraction completed successfully!")


# ---------------------------------------------------------------------------
# Pipeline Phase 2: Document Encoding & Caching using CMRAG LongSigLIP
# ---------------------------------------------------------------------------
def find_extracted_doc_dir(extract_base_dir: str, doc_id: str) -> Optional[str]:
    """Locate the extracted folder for a given doc_id on local disk."""
    base_doc_name = os.path.splitext(doc_id)[0]
    candidates = [
        os.path.join(extract_base_dir, "images", doc_id),
        os.path.join(extract_base_dir, "images", base_doc_name),
        os.path.join(extract_base_dir, doc_id),
        os.path.join(extract_base_dir, base_doc_name),
    ]

    for cand in candidates:
        if os.path.exists(cand) and os.path.isdir(cand):
            return cand

    images_dir = os.path.join(extract_base_dir, "images")
    search_dir = images_dir if os.path.exists(images_dir) else extract_base_dir

    if os.path.exists(search_dir):
        for folder in os.listdir(search_dir):
            if folder.lower() in {doc_id.lower(), base_doc_name.lower()}:
                return os.path.join(search_dir, folder)

    return None


def get_or_compute_doc_embeddings(
    doc_dir: str,
    doc_name: str,
    cache_dir: str,
    retriever_obj: Retriever,
    batch_size: int,
) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    """
    Load document embeddings from local disk cache, or compute and save them using CMRAG LongSigLIP.
    Returns: (page_numbers, img_embeds [N, D], txt_embeds [N, D])
    """
    doc_cache_path = os.path.join(cache_dir, doc_name)
    img_cache = os.path.join(doc_cache_path, "images.pt")
    txt_cache = os.path.join(doc_cache_path, "texts.pt")
    page_cache = os.path.join(doc_cache_path, "pages.json")

    # If cached, load directly
    if os.path.exists(img_cache) and os.path.exists(txt_cache) and os.path.exists(page_cache):
        img_embeds = torch.load(img_cache, map_location="cpu")
        txt_embeds = torch.load(txt_cache, map_location="cpu")
        with open(page_cache, "r", encoding="utf-8") as f:
            page_numbers = json.load(f)
        return page_numbers, img_embeds, txt_embeds

    # Otherwise compute embeddings using CMRAG Retriever
    os.makedirs(doc_cache_path, exist_ok=True)
    files = os.listdir(doc_dir)
    png_files = sorted(
        [f for f in files if f.endswith(".png") and not f.endswith("_subimg.png")],
        key=extract_page_num,
    )

    page_numbers = []
    images = []
    parsed_texts = []

    for png_file in png_files:
        p_num = extract_page_num(png_file)
        stem = os.path.splitext(png_file)[0]
        img_path = os.path.join(doc_dir, png_file)
        html_path = os.path.join(doc_dir, f"{stem}_parser.html")

        img = Image.open(img_path).convert("RGB")
        raw_html = load_html_file(html_path)
        plain_txt = html_to_plain_text(raw_html)

        if not plain_txt.strip():
            plain_txt = f"Document page {p_num}"

        page_numbers.append(p_num)
        images.append(img)
        parsed_texts.append(plain_txt)

    # Encode using CMRAG LongSigLIP processor and model
    img_embeds_list = []
    txt_embeds_list = []

    for i in range(0, len(images), batch_size):
        batch_images = images[i : i + batch_size]
        batch_texts = parsed_texts[i : i + batch_size]

        with torch.no_grad():
            processed = retriever_obj.processor.process_img_text(
                images=batch_images, extracted_texts=batch_texts
            )
            z_i, z_t = retriever_obj.model.encode_img_text(
                pixel_values=processed["pixel_values"].to(retriever_obj.device),
                input_ids_t=processed["input_ids_t"].to(retriever_obj.device),
                attention_mask_t=processed["attention_mask_t"].to(retriever_obj.device),
            )
            img_embeds_list.append(z_i.cpu())
            txt_embeds_list.append(z_t.cpu())

    img_embeds = torch.cat(img_embeds_list, dim=0)
    txt_embeds = torch.cat(txt_embeds_list, dim=0)

    # Cache to disk
    torch.save(img_embeds, img_cache)
    torch.save(txt_embeds, txt_cache)
    with open(page_cache, "w", encoding="utf-8") as f:
        json.dump(page_numbers, f)

    return page_numbers, img_embeds, txt_embeds


# ---------------------------------------------------------------------------
# Pipeline Phase 3: Grid Search for Oracle Beta
# ---------------------------------------------------------------------------
def z_score_normalize(scores: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply Z-Score normalization (zero mean, unit variance) across candidate page scores."""
    mean = torch.mean(scores)
    std = torch.std(scores)
    if std < eps:
        return scores - mean
    return (scores - mean) / std


def perform_beta_grid_search(
    s_iq: torch.Tensor,
    s_tq: torch.Tensor,
    page_numbers: List[int],
    gt_page_ids: List[int],
    grid_step: float = 0.05,
    k: int = 5,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Perform grid search over beta in range [0.0, 1.0].
    Applies UCMR-style Z-score normalization to text and image similarity scores.
    Tie-breaking: Uses MEDIAN beta if multiple betas yield the same max NDCG@k score.
    """
    z_iq = z_score_normalize(s_iq)
    z_tq = z_score_normalize(s_tq)

    beta_values = np.round(np.arange(0.0, 1.0 + grid_step / 2.0, grid_step), 4).tolist()
    all_beta_scores: Dict[str, float] = {}

    best_ndcg = -1.0
    best_betas: List[float] = []

    for beta in beta_values:
        fused_score = beta * z_tq + (1.0 - beta) * z_iq
        top_indices = torch.argsort(fused_score, descending=True).tolist()
        ranked_page_ids = [page_numbers[i] for i in top_indices]

        ndcg_val = compute_ndcg_at_k(ranked_page_ids, gt_page_ids, k=k)
        beta_str = f"{beta:.2f}"
        all_beta_scores[beta_str] = round(ndcg_val, 4)

        if ndcg_val > best_ndcg:
            best_ndcg = ndcg_val
            best_betas = [beta]
        elif abs(ndcg_val - best_ndcg) < 1e-6:
            best_betas.append(beta)

    if best_betas:
        oracle_beta = float(np.median(best_betas))
    else:
        oracle_beta = 0.5

    return oracle_beta, best_ndcg, all_beta_scores


# ---------------------------------------------------------------------------
# Main Pipeline
def download_file_with_progress(url: str, dest_path: str):
    import urllib.request
    import shutil
    from tqdm import tqdm

    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    tmp_path = dest_path + ".tmp"
    print(f"[*] Downloading dataset component from HuggingFace...")
    print(f"    URL   : {url}")
    print(f"    Target: {dest_path}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        total_size = int(resp.headers.get("content-length", 0))
        block_size = 1024 * 1024
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
    shutil.move(tmp_path, dest_path)
    print(f"[+] Download complete: {dest_path}")


def ensure_dataset(data_name: str, json_path: str = "", tar_path: str = ""):
    hf_base = f"https://huggingface.co/datasets/Yuwh07/CMRAG-Bench/resolve/main/{data_name}"
    local_dir = f"/content/data/CMRAG-Bench/{data_name}"

    json_candidates = [
        json_path if json_path else None,
        os.path.join(local_dir, f"{data_name}.json"),
        os.path.join(local_dir, f"{data_name.lower()}.json"),
        os.path.join(local_dir, "MMLongbench.json"),
        f"/content/drive/MyDrive/TA/CMRAG/data/CMRAG-Bench/{data_name}/{data_name}.json",
        f"/content/drive/MyDrive/TA/CMRAG/data/CMRAG-Bench/{data_name}/MMLongbench.json",
        f"/content/drive/MyDrive/TA/CMRAG/data/{data_name}/{data_name}.json",
        f"/content/drive/MyDrive/TA/CMRAG/data/{data_name}/MMLongbench.json",
        f"data/CMRAG-Bench/{data_name}/{data_name}.json",
        f"data/CMRAG-Bench/{data_name}/MMLongbench.json",
        f"data/{data_name}/{data_name}.json",
    ]
    resolved_json = next((p for p in json_candidates if p and os.path.exists(p)), None)
    if not resolved_json:
        target_json = os.path.join(local_dir, "MMLongbench.json" if data_name == "MMLongBench" else f"{data_name}.json")
        json_url = f"{hf_base}/{os.path.basename(target_json)}"
        download_file_with_progress(json_url, target_json)
        resolved_json = target_json

    tar_candidates = [
        tar_path if tar_path else None,
        os.path.join(local_dir, "images.tar"),
        f"/content/drive/MyDrive/TA/CMRAG/data/CMRAG-Bench/{data_name}/images.tar",
        f"/content/drive/MyDrive/TA/CMRAG/data/{data_name}/images.tar",
        f"data/CMRAG-Bench/{data_name}/images.tar",
        f"data/{data_name}/images.tar",
    ]
    resolved_tar = next((p for p in tar_candidates if p and os.path.exists(p)), None)
    if not resolved_tar:
        target_tar = os.path.join(local_dir, "images.tar")
        tar_url = f"{hf_base}/images.tar"
        download_file_with_progress(tar_url, target_tar)
        resolved_tar = target_tar

    return resolved_json, resolved_tar


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(f"ORACLE BETA DATASET BUILDER (CMRAG LongSigLIP) | Device: {device}")
    print(f"Dataset Split: {args.dataset}")
    print("=" * 80)

    # 0. Auto-mount Google Drive in Colab if not already mounted (for saving outputs)
    if not os.path.exists("/content/drive/MyDrive"):
        try:
            from google.colab import drive
            print("[*] Mounting Google Drive at /content/drive ...")
            drive.mount("/content/drive", force_remount=False)
        except Exception as e:
            print(f"[WARNING] Could not auto-mount Google Drive: {e}")

    # 1. Validate / auto-download dataset paths
    args.json_path, args.tar_path = ensure_dataset(args.dataset, args.json_path, args.tar_path)
    print(f"[*] Using JSON: {args.json_path}")
    print(f"[*] Using TAR : {args.tar_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # 2. Phase 1: Pre-extract all document folders to fast local disk
    pre_extract_tar_if_needed(args.tar_path, args.extract_all_dir)

    # 3. Read dataset
    with open(args.json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_queries = len(dataset)
    print(f"[*] Total dataset size: {total_queries} queries.")

    start_idx = args.start_index
    end_idx = total_queries if args.max_queries < 0 else min(start_idx + args.max_queries, total_queries)
    print(f"[*] Processing query slice [{start_idx} .. {end_idx - 1}] ...")

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Load CMRAG Retriever (LongSigLIP with max_text_len=768)
    print(f"[*] Loading CMRAG Retriever (LongSigLIP, max_text_len={args.max_text_len})...")
    retriever_obj = Retriever(
        ckpt_path=args.ckpt_path,
        max_text_len=args.max_text_len,
        model_id=args.model_name,
        device=device,
    )

    # 5. Incremental dataset output setup
    output_json_path = os.path.join(args.output_dir, "oracle_dataset.json")
    query_embeddings_dir = os.path.join(args.output_dir, "query_embeddings")
    os.makedirs(query_embeddings_dir, exist_ok=True)

    oracle_records = []
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                oracle_records = json.load(f)
            print(f"[*] Resuming from existing dataset file with {len(oracle_records)} existing records.")
        except Exception as e:
            print(f"[WARNING] Could not read existing dataset JSON ({e}). Starting fresh.")
            oracle_records = []

    processed_indices = {r["query_index"] for r in oracle_records}

    # 6. Process each query
    success_count = 0
    skipped_count = 0

    for idx in range(start_idx, end_idx):
        if idx in processed_indices:
            skipped_count += 1
            continue

        row = dataset[idx]
        question = row.get("question", "")
        doc_id = row.get("doc_id", "")
        ev_pages_raw = parse_list_field(row.get("evidence_pages", ""))
        ev_sources_raw = parse_list_field(row.get("evidence_sources", ""))

        gt_page_ids = []
        for p in ev_pages_raw:
            try:
                gt_page_ids.append(int(p))
            except (ValueError, TypeError):
                pass

        doc_dir = find_extracted_doc_dir(args.extract_all_dir, doc_id)
        if not doc_dir:
            print(f"[WARNING] [{idx}] Document folder for '{doc_id}' not found. Skipping.")
            continue

        doc_name = os.path.basename(doc_dir)

        # Compute or load doc embeddings using LongSigLIP
        page_numbers, img_embeds, txt_embeds = get_or_compute_doc_embeddings(
            doc_dir=doc_dir,
            doc_name=doc_name,
            cache_dir=args.cache_dir,
            retriever_obj=retriever_obj,
            batch_size=args.batch_size,
        )

        if len(page_numbers) == 0:
            print(f"[WARNING] [{idx}] Document '{doc_name}' has 0 pages. Skipping.")
            continue

        # Encode query using CMRAG Retriever
        q_emb = retriever_obj.encode_query(question).cpu()  # [1, D]

        # Compute raw similarity scores (inner product)
        s_iq = (q_emb @ img_embeds.T).squeeze(0)  # [N]
        s_tq = (q_emb @ txt_embeds.T).squeeze(0)  # [N]

        # Perform beta grid search
        oracle_beta, best_ndcg, all_beta_scores = perform_beta_grid_search(
            s_iq=s_iq,
            s_tq=s_tq,
            page_numbers=page_numbers,
            gt_page_ids=gt_page_ids,
            grid_step=args.grid_step,
            k=args.eval_k,
        )

        # Save query embedding
        q_emb_file = f"query_{idx:04d}.pt"
        torch.save(q_emb.squeeze(0), os.path.join(query_embeddings_dir, q_emb_file))

        record = {
            "query_index": idx,
            "question": question,
            "doc_id": doc_id,
            "doc_name": doc_name,
            "evidence_pages": ev_pages_raw,
            "evidence_sources": ev_sources_raw,
            "gt_page_ids": gt_page_ids,
            "oracle_beta": round(oracle_beta, 4),
            "best_ndcg5": round(best_ndcg, 4),
            "num_pages": len(page_numbers),
            "query_emb_file": os.path.join("query_embeddings", q_emb_file),
            "all_betas_ndcg5": all_beta_scores,
        }
        oracle_records.append(record)
        success_count += 1

        if success_count % 20 == 0 or idx == end_idx - 1:
            oracle_records_sorted = sorted(oracle_records, key=lambda x: x["query_index"])
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(oracle_records_sorted, f, indent=2)
            print(
                f"[*] Progress: [{idx + 1}/{end_idx}] Processed {success_count} queries | "
                f"Saved checkpoint to {output_json_path}"
            )

    print("\n" + "=" * 80)
    print("ORACLE BETA DATASET GENERATION COMPLETED!")
    print(f"Total processed in run : {success_count}")
    print(f"Total skipped (cached): {skipped_count}")
    print(f"Final dataset path    : {output_json_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
