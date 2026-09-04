import os
import json
import ast
import tarfile
import argparse
from typing import List, Tuple, Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect a query from MMLongBench and match document folder in images.tar without full extraction."
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
        "--query_index",
        type=int,
        default=0,
        help="Index of the query to inspect (0 to 1081)",
    )
    args, _ = parser.parse_known_args()
    return args


def parse_list_string(val: str):
    """Safely parse string representations of Python lists e.g., '[5]' or '[\'Chart\']'."""
    if isinstance(val, list):
        return val
    if not val or not isinstance(val, str):
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return [val]


def find_matching_folder_fast(tar_path: str, doc_id: str) -> Tuple[Optional[str], List[str]]:
    """
    Search for doc_id in images.tar lazily using tar.next() to avoid reading the entire
    4.6GB archive headers sequentially over Google Drive FUSE mount.
    """
    base_doc_name = os.path.splitext(doc_id)[0]
    target_names = {doc_id.lower(), base_doc_name.lower()}

    print(f"[*] Fast scanning tar metadata for document matching '{doc_id}' ...")

    matched_folder = None
    candidate_folders = set()

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

            if folder_name:
                folder_lower = folder_name.lower()
                # Check exact match
                if folder_lower in target_names:
                    matched_folder = folder_name
                    print(f"[*] Match found early: '{matched_folder}'")
                    return matched_folder, []

                # Substring match candidate
                if base_doc_name.lower() in folder_lower or folder_lower in base_doc_name.lower():
                    candidate_folders.add(folder_name)

    return matched_folder, sorted(list(candidate_folders))


def main():
    args = parse_args()

    # 1. Validate dataset paths
    if not os.path.exists(args.json_path):
        print(f"[ERROR] JSON file not found at: {args.json_path}")
        print("Please ensure Google Drive is mounted at /content/drive in Colab.")
        return

    if not os.path.exists(args.tar_path):
        print(f"[ERROR] TAR file not found at: {args.tar_path}")
        print("Please ensure Google Drive is mounted at /content/drive in Colab.")
        return

    # 2. Read MMLongbench.json
    print(f"[*] Loading dataset from: {args.json_path}")
    with open(args.json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_queries = len(dataset)
    print(f"[*] Total queries in dataset: {total_queries}")

    if args.query_index < 0 or args.query_index >= total_queries:
        print(f"[ERROR] query_index {args.query_index} out of bounds (0..{total_queries - 1})")
        return

    # 3. Select target query row
    row = dataset[args.query_index]
    question = row.get("question", "")
    doc_id = row.get("doc_id", "")
    evidence_pages_raw = row.get("evidence_pages", "")
    evidence_sources_raw = row.get("evidence_sources", "")

    # Parsing string list fields
    evidence_pages = parse_list_string(evidence_pages_raw)
    evidence_sources = parse_list_string(evidence_sources_raw)

    # 4. Search for document folder in images.tar lazily
    matched_folder, candidate_folders = find_matching_folder_fast(args.tar_path, doc_id)

    # 5. Display results
    print("\n" + "=" * 60)
    print(f"MMLONGBENCH QUERY INSPECTION (Index: {args.query_index})")
    print("=" * 60)
    print(f"Question         : {question}")
    print(f"Document ID      : {doc_id}")
    print(f"Evidence Pages   : {evidence_pages}  (raw: {evidence_pages_raw})")
    print(f"Evidence Sources : {evidence_sources}  (raw: {evidence_sources_raw})")
    print("-" * 60)

    if matched_folder:
        print(f"[SUCCESS] Matched Document Folder : {matched_folder}")
    else:
        print(f"[WARNING] Exact document folder match NOT found for doc_id: {doc_id}")
        if candidate_folders:
            print(f"Top candidate folders ({len(candidate_folders)} found):")
            for cand in candidate_folders[:5]:
                print(f"  - {cand}")
        else:
            print("No similar candidate folders found in archive.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
