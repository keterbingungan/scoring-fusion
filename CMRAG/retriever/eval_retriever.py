import argparse
import os
import numpy as np
import json
from typing import List, Dict
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F

# Limit PyTorch CPU threads to 1 to prevent CPU 100% overload & IDE freezing on laptops
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from utils.utils import load_json, save_json, load_html_file, html_to_plain_text
from .eval.eval import eval_retrieval
from .retriever import Retriever


# Define parameters and hyperparameters
def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune SigLIP with long OCR text")
    # data
    parser.add_argument("--ckpt", default="../output_longSigLIP/long_siglip-epochepoch=0.ckpt", help="pretrained checkpoint")
    parser.add_argument("--test_data", default="MMLongBench", help="which evaluation dataset to use")
    parser.add_argument("--results_root", default="../results/retrieval", help="where to save retrieval results")
    
    # model
    parser.add_argument("--model_id", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--max_query", type=int, default=64)
    parser.add_argument("--max_text",  type=int, default=768)
    
    # evaluation
    parser.add_argument("--batch_size", type=int, default=1, help="batch size (set to 1 for CPU/laptop)")
    parser.add_argument("--step", type=int, default=11)
    parser.add_argument("--text_weight", type=float, default=0.2, help="weight for text similarity")
    
    # Need embeddings from this step
    parser.add_argument("--need_emb", type=bool, default=False, help="whether to encode and save embeddings")
    parser.add_argument("--max_docs", type=int, default=-1, help="limit number of document folders to process (for fast laptop testing)")
    parser.add_argument("--dummy_emb", type=bool, default=False, help="generate lightweight dummy embeddings instantly for testing without CPU load")
    
    return parser.parse_args()




def load_eval_data(data_name):
    """ Load evaluation data and QA pairs according to the dataset name """
    possible_roots = [
        f"data/CMRAG-Bench/{data_name}",
        f"../data/CMRAG-Bench/{data_name}",
        f"data/{data_name}",
        f"../data/{data_name}",
    ]
    data_root = None
    for root in possible_roots:
        if os.path.exists(root):
            data_root = root
            break
    if not data_root:
        data_root = f"data/CMRAG-Bench/{data_name}"

    possible_json_names = [
        f"{data_name}.json",
        f"{data_name.lower()}.json",
        "MMLongbench.json",
        "mmlongbench.json",
        "MMLongBench.json",
        "samples.json"
    ]
    qa_file = None
    for jname in possible_json_names:
        candidate = os.path.join(data_root, jname)
        if os.path.exists(candidate):
            qa_file = candidate
            break

    if qa_file and os.path.exists(qa_file):
        qa_pairs = load_json(qa_file)
        return data_root, qa_pairs
    else:
        raise FileNotFoundError(f"Could not find QA pairs JSON file for '{data_name}' in {data_root}. Checked files: {possible_json_names}")




def main():
    args = get_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    R = Retriever(
        ckpt_path=args.ckpt,
        max_text_len=args.max_text,
        model_id=args.model_id,
        device=DEVICE)
    
    
    # Load evaluation data
    data_root, qa_pairs = load_eval_data(args.test_data)
    print(f"Loaded {len(qa_pairs)} QA pairs from {args.test_data} for evaluation.")
    

    # Encoding and saving embeddings if needed
    if args.need_emb:
        print("="*100)
        print(f"Encoding and saving embeddings at step {args.step}...")
        print("="*100)
        
        folder_names = os.listdir(os.path.join(data_root, "images"))
        if args.max_docs > 0:
            folder_names = folder_names[:args.max_docs]
            print(f"--> [Laptop Friendly Mode] Limiting indexing to first {args.max_docs} document folders.")

        for folder_name in folder_names:
            print(f"Encoding images and text in folder: {folder_name}")
            res_path = os.path.join(data_root, f"embeddings_step{args.step}", folder_name)
            os.makedirs(res_path, exist_ok=True)
            
            if args.dummy_emb:
                # Fast dummy embedding generation (0 CPU load, 100% instant for laptop testing)
                num_pages = len([f for f in os.listdir(os.path.join(data_root, "images", folder_name)) if f.endswith(".png")])
                if num_pages == 0:
                    num_pages = 5
                img_emb = F.normalize(torch.randn(num_pages, 1152), dim=-1)
                text_emb = F.normalize(torch.randn(num_pages, 1152), dim=-1)
                torch.save(img_emb, os.path.join(res_path, "images.pt"))
                torch.save(text_emb, os.path.join(res_path, "texts.pt"))
                print(f"--> [Dummy Emb Mode] Saved {num_pages} dummy embeddings instantly to {res_path}")
            else:
                # Load images and texts
                images = []
                texts = []
                i = 1
                while True:
                    img_path = os.path.join(data_root, "images", folder_name, f"{i}.png")
                    text_path = os.path.join(data_root, "images", folder_name, f"{i}_parser.html")
                    if os.path.exists(img_path):
                        images.append(Image.open(img_path).convert("RGB"))
                        if os.path.exists(text_path):
                            html_content = load_html_file(text_path)
                            plain_text = html_to_plain_text(html_content)
                            texts.append(plain_text)
                        else:
                            texts.append("")
                    else:
                        break
                    i += 1
                # Encode and save embeddings
                R.encode_doc(
                    images=images,
                    texts=texts,
                    res_path=res_path,
                    batch_size=args.batch_size,
                )
            
    
    # Now evaluate the retriever
    print("Evaluating retriever...")
    sims = []
    page_ids = []
    gt_page_ids = []
    
    for qa in tqdm(qa_pairs, desc="Evaluating"):
        query = qa["question"]
        doc_name = qa["doc_id"].replace(".pdf", "")
        gt_page_id = qa["evidence_pages"]
        
        emb_dir = os.path.join(data_root, f"embeddings_step{args.step}", doc_name)
        if not os.path.exists(emb_dir):
            continue

        # Retrieve topk pages
        topk_page_ids, topk_scores = R.retrieve(
            doc_emb_dir=emb_dir,
            query=query,
            topk=10,
            text_weight=args.text_weight,
        )
        sims.append(topk_scores)
        page_ids.append(topk_page_ids)
        gt_page_ids.append(gt_page_id)
        
        if args.max_docs > 0 and len(sims) >= args.max_docs:
            break
    
    # Evaluate
    results = eval_retrieval(
        similarities=sims,
        doc_page_ids=page_ids,
        gt_page_ids=gt_page_ids,
    )
    
    print("=" * 100)
    print("Evaluation results at step {args.step}:")
    print(results)

    # Save results
    root = os.path.join(args.results_root, args.test_data)
    os.makedirs(root, exist_ok=True)
    save_json(results, os.path.join(root, f'retrieval_results_step{args.step}.json'))
    
    print(f"Saved retrieval results to {os.path.join(root, f'retrieval_results_step{args.step}.json')}")



if __name__ == "__main__":
    main()