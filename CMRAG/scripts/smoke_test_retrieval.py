import os
import sys
import json
import ast
import re
import tarfile
import argparse
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModel

from bs4 import BeautifulSoup

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single query retrieval smoke test using SigLIP on MMLongBench dataset."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="/content/drive/MyDrive/TA/CMRAG/data/CMRAG-Bench/MMLongBench/MMLongbench.json",
        help="Path to MMLongbench.json on Google Drive",
    )
    parser.add_argument(
        "--tar_path",
        type=str,
        default="/content/drive/MyDrive/TA/CMRAG/data/CMRAG-Bench/MMLongBench/images.tar",
        help="Path to images.tar on Google Drive",
    )
    parser.add_argument(
        "--extract_dir",
        type=str,
        default="/content/extracted_docs",
        help="Temporary directory on /content to extract document images",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.environ.get("DATASET", "MMLongBench"),
        help="Dataset name (e.g. MMLongBench, finslides, techslides)",
    )
    parser.add_argument(
        "--query_index",
        type=int,
        default=int(os.environ.get("QUERY_INDEX", 0)),
        help="Index of query to test (0 to 1081)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=os.environ.get("MODEL_NAME", "google/siglip-so400m-patch14-384"),
        help="HuggingFace model name for SigLIP",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", 8)),
        help="Batch size for SigLIP encoding to prevent CUDA OOM",
    )
    args, _ = parser.parse_known_args()
    return args


def parse_list_field(val) -> List:
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


def find_matching_folder_fast(tar_path: str, doc_id: str) -> Optional[str]:
    """Scan tar archive lazily using tar.next() to find document folder name matching doc_id."""
    base_doc_name = os.path.splitext(doc_id)[0]
    target_names = {doc_id.lower(), base_doc_name.lower()}

    with tarfile.open(tar_path, "r") as tar:
        while True:
            member = tar.next()
            if member is None:
                break
            parts = member.name.strip("/").split("/")
            folder_name = None
            if len(parts) >= 2 and parts[0] == "images":
                folder_name = parts[1]
            elif len(parts) >= 1 and parts[0] != "images":
                folder_name = parts[0]

            if folder_name and folder_name.lower() in target_names:
                return folder_name

    return None


def extract_single_doc_folder(tar_path: str, doc_folder_name: str, extract_base_dir: str) -> str:
    """Extract ONLY the target document folder from images.tar to local temporary disk."""
    print(f"[*] Extracting document folder '{doc_folder_name}' from tar...")
    os.makedirs(extract_base_dir, exist_ok=True)

    extracted_members = []
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            parts = member.name.strip("/").split("/")
            if (len(parts) >= 2 and parts[0] == "images" and parts[1] == doc_folder_name) or (
                len(parts) >= 1 and parts[0] == doc_folder_name
            ):
                extracted_members.append(member)

        tar.extractall(path=extract_base_dir, members=extracted_members)

    # Locate extracted path
    candidate_1 = os.path.join(extract_base_dir, "images", doc_folder_name)
    candidate_2 = os.path.join(extract_base_dir, doc_folder_name)

    if os.path.exists(candidate_1):
        target_dir = candidate_1
    elif os.path.exists(candidate_2):
        target_dir = candidate_2
    else:
        target_dir = extract_base_dir

    print(f"[*] Extracted {len(extracted_members)} files to: {target_dir}")
    return target_dir


def extract_page_num(filename: str) -> int:
    """Extract numerical page number from filename e.g., '5.png' or 'page_5.png' -> 5."""
    stem = os.path.splitext(filename)[0]
    match = re.search(r"(\d+)$", stem)
    if match:
        return int(match.group(1))
    return 999999


def load_doc_pages_and_texts(doc_dir: str) -> Tuple[List[int], List[Image.Image], List[str]]:
    """Scan extracted document directory for page PNGs and corresponding _parser.html files."""
    files = os.listdir(doc_dir)
    png_files = [
        f for f in files if f.endswith(".png") and not f.endswith("_subimg.png")
    ]
    png_files = sorted(png_files, key=extract_page_num)

    page_numbers = []
    images = []
    parsed_texts = []

    for png_file in png_files:
        page_num = extract_page_num(png_file)
        page_stem = os.path.splitext(png_file)[0]
        img_path = os.path.join(doc_dir, png_file)

        # 1. Load image
        img = Image.open(img_path).convert("RGB")

        # 2. Load corresponding parsed text from HTML
        html_file = os.path.join(doc_dir, f"{page_stem}_parser.html")
        raw_html = load_html_file(html_file)
        plain_text = html_to_plain_text(raw_html)

        # If html text is empty, fallback to placeholder
        if not plain_text.strip():
            plain_text = f"Document page {page_num}"

        page_numbers.append(page_num)
        images.append(img)
        parsed_texts.append(plain_text)

    return page_numbers, images, parsed_texts


def _extract_tensor(feat) -> torch.Tensor:
    """Extract raw PyTorch Tensor from HuggingFace ModelOutput object if necessary."""
    if isinstance(feat, torch.Tensor):
        return feat
    if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
        return feat.pooler_output
    return feat[0]


def encode_images_batched(
    model, processor, images: List[Image.Image], device: str, batch_size: int
) -> torch.Tensor:
    """Encode page images in mini-batches using SigLIP model."""
    all_embeds = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            embeds = model.get_image_features(**inputs)
            embeds = _extract_tensor(embeds)
            embeds = F.normalize(embeds, p=2, dim=-1)
            all_embeds.append(embeds.cpu())
    return torch.cat(all_embeds, dim=0)


def encode_texts_batched(
    model, processor, texts: List[str], device: str, batch_size: int
) -> torch.Tensor:
    """Encode page texts or queries in mini-batches using SigLIP model."""
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = processor(
            text=batch, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            embeds = model.get_text_features(**inputs)
            embeds = _extract_tensor(embeds)
            embeds = F.normalize(embeds, p=2, dim=-1)
            all_embeds.append(embeds.cpu())
    return torch.cat(all_embeds, dim=0)


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


def ensure_dataset(data_name: str, json_path: str, tar_path: str):
    hf_base = f"https://huggingface.co/datasets/Yuwh07/CMRAG-Bench/resolve/main/{data_name}"
    local_dir = f"/content/data/CMRAG-Bench/{data_name}"

    json_candidates = [
        json_path,
        os.path.join(local_dir, f"{data_name}.json"),
        os.path.join(local_dir, "MMLongbench.json"),
        f"/content/drive/MyDrive/TA/CMRAG/data/CMRAG-Bench/{data_name}/{data_name}.json",
        f"/content/drive/MyDrive/TA/CMRAG/data/CMRAG-Bench/{data_name}/MMLongbench.json",
        f"data/CMRAG-Bench/{data_name}/{data_name}.json",
        f"data/CMRAG-Bench/{data_name}/MMLongbench.json",
    ]
    resolved_json = next((p for p in json_candidates if p and os.path.exists(p)), None)
    if not resolved_json:
        target_json = os.path.join(local_dir, "MMLongbench.json" if data_name == "MMLongBench" else f"{data_name}.json")
        json_url = f"{hf_base}/{os.path.basename(target_json)}"
        download_file_with_progress(json_url, target_json)
        resolved_json = target_json

    tar_candidates = [
        tar_path,
        os.path.join(local_dir, "images.tar"),
        f"/content/drive/MyDrive/TA/CMRAG/data/CMRAG-Bench/{data_name}/images.tar",
        f"data/CMRAG-Bench/{data_name}/images.tar",
    ]
    resolved_tar = next((p for p in tar_candidates if p and os.path.exists(p)), None)
    if not resolved_tar:
        target_tar = os.path.join(local_dir, "images.tar")
        tar_url = f"{hf_base}/images.tar"
        download_file_with_progress(tar_url, target_tar)
        resolved_tar = target_tar

    return resolved_json, resolved_tar


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Execution device: {device}")

    # 1. Validate / auto-download dataset paths
    data_name = "MMLongBench"
    args.json_path, args.tar_path = ensure_dataset(data_name, args.json_path, args.tar_path)
    print(f"[*] Using JSON: {args.json_path}")
    print(f"[*] Using TAR : {args.tar_path}")

    # 2. Read query from dataset JSON
    with open(args.json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.query_index < 0 or args.query_index >= len(dataset):
        print(f"[ERROR] query_index {args.query_index} out of range (0..{len(dataset)-1})")
        return

    row = dataset[args.query_index]
    question = row.get("question", "")
    doc_id = row.get("doc_id", "")
    evidence_pages_raw = parse_list_field(row.get("evidence_pages", ""))
    evidence_sources_raw = parse_list_field(row.get("evidence_sources", ""))

    # Convert evidence_pages to set of integers / strings for flexible matching
    evidence_pages_set = set()
    for item in evidence_pages_raw:
        try:
            evidence_pages_set.add(int(item))
        except (ValueError, TypeError):
            evidence_pages_set.add(str(item))

    print("\n" + "=" * 70)
    print(f"RETRIEVAL SMOKE TEST (Query Index: {args.query_index})")
    print("=" * 70)
    print(f"Question       : {question}")
    print(f"Document ID    : {doc_id}")
    print(f"Evidence Pages : {evidence_pages_raw}")
    print(f"Evidence Srcs  : {evidence_sources_raw}")
    print("-" * 70)

    # 3. Match document folder in tar
    doc_folder = find_matching_folder_fast(args.tar_path, doc_id)
    if not doc_folder:
        print(f"[ERROR] Document folder for '{doc_id}' not found in tar archive.")
        return

    print(f"[*] Matched Document Folder: {doc_folder}")

    # 4. Extract ONLY target document folder to local temp storage
    doc_dir = extract_single_doc_folder(args.tar_path, doc_folder, args.extract_dir)

    # 5. Load page images & parsed text
    page_numbers, images, parsed_texts = load_doc_pages_and_texts(doc_dir)
    total_pages = len(images)
    print(f"[*] Total document pages loaded: {total_pages}")

    if total_pages == 0:
        print("[ERROR] No page images found in extracted folder.")
        return

    # 6. Load SigLIP model & processor
    print(f"[*] Loading SigLIP model: {args.model_name} on {device}...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device).eval()

    # 7. Encode query, images, and text
    print("[*] Encoding query...")
    q_embed = encode_texts_batched(model, processor, [question], device, batch_size=1)

    print(f"[*] Encoding {total_pages} page images (batch size = {args.batch_size})...")
    img_embeds = encode_images_batched(model, processor, images, device, batch_size=args.batch_size)

    print(f"[*] Encoding {total_pages} page texts (batch size = {args.batch_size})...")
    txt_embeds = encode_texts_batched(model, processor, parsed_texts, device, batch_size=args.batch_size)

    # 8. Compute similarity scores (cosine similarity = inner product of L2 normalized vectors)
    # Shapes: q_embed [1, D], img_embeds [N, D], txt_embeds [N, D]
    img_scores = (q_embed @ img_embeds.T).squeeze(0).tolist()
    txt_scores = (q_embed @ txt_embeds.T).squeeze(0).tolist()

    # 9. Format & display retrieval table
    print("\n" + "=" * 70)
    print(f"{'page_id':^10} | {'image_score':^15} | {'text_score':^15} | {'relevant':^10}")
    print("-" * 70)

    for i in range(total_pages):
        p_num = page_numbers[i]
        img_s = img_scores[i]
        txt_s = txt_scores[i]
        is_relevant = (p_num in evidence_pages_set) or (str(p_num) in evidence_pages_set) or (i + 1 in evidence_pages_set)

        rel_str = "TRUE" if is_relevant else "False"
        print(f"{p_num:^10} | {img_s:^15.4f} | {txt_s:^15.4f} | {rel_str:^10}")

    print("=" * 70 + "\n")
    print("[SUCCESS] Retrieval pipeline smoke test completed.")


if __name__ == "__main__":
    main()
