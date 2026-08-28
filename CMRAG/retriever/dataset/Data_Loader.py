import os
import json
from typing import Dict, List
import torch
from torch.utils.data import Dataset
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import io

from utils.utils import load_json, load_html_file, html_to_plain_text



data_paths = {
    "domain": "openbmb/VisRAG-Ret-Train-In-domain-data",
    "synthetic": "openbmb/VisRAG-Ret-Train-Synthetic-data"
}

query_2_text_id_path = {
    "domain": "../data/domain_data/query_2_img_newid.json",
    "synthetic": "../data/synthetic_data/query_2_img_newid.json"
}


class load_data(Dataset):
    """Dataset for text queries vs image+text documents"""

    def __init__(
            self,
            data_type: str = "domain",
            use_text: bool = True,
            batch_size: int = 16,
        ):
        
        assert data_type in ["domain", "synthetic", "combined"], "data_type must be 'domain' or 'synthetic' or 'combined'"
        
        self.data_type = data_type
        self.batch_size = batch_size

        # Loading huggingface dataset
        self.data = load_dataset(data_paths[data_type], split="train")
        
        # # Not decoding images here, will load them on-the-fly in __getitem__
        # self.data = data.cast_column("image", Image(decode=False))
        
        # Use extracted text or not
        self.use_text = use_text
        if self.use_text:
            self.query_2_text_id = load_json(query_2_text_id_path[data_type])


    def __len__(self):
        return len(self.data)
    

    # Load extracted text (1.png --> 1_parser.html)
    def _load_extracted_text(self, img_id: str) -> str:
        text_path = f"../data/{self.data_type}_data/images/{img_id}_parser.html"
        text = load_html_file(text_path)
        text = html_to_plain_text(text)

        return text



    def __getitem__(self, idx):
        item = self.data[idx]
        
        # decode image on-the-fly
        img = item['image']
        
        # If it's a PIL image already
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
        # If it's a dict (decode=False)
        elif isinstance(img, dict):
            if img.get("bytes") is not None:
                img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            else:
                img = Image.open(img["path"]).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
        
        # Load extracted text if needed
        if self.use_text:
            extracted_text = self._load_extracted_text(self.query_2_text_id[idx]['img_id'])
        
        
        return {
            'query': item['query'],
            'image': img,
            'extracted_text': extracted_text if self.use_text else "",
            'idx': idx
        }




class load_val_data(Dataset):
    """Validation Dataset for text queries vs image+text documents"""
    def __init__(
            self,
            data_type: str = "MMLongBench",
            use_text: bool = True,
            batch_size: int = 16,
        ):
        
        assert data_type in ["MMLongBench"], "data_type must be 'MMLongBench'"

        self.use_text = use_text
        self.batch_size = batch_size
        
        self.root = "../data/MMLongBench-Doc/MMLongBench_eval"
        # Load query json file
        self.samples = load_json(os.path.join(self.root, "samples_subset.json"))

        # Mapping each query to all page ids in the corresponding document
        self.pairs = []
        
        for sample in self.samples:
            query = sample['question']
            doc_name = sample["doc_id"].replace(".pdf", "")
            gt_page_ids = sample["evidence_pages"]
            # Get all page ids for this document
            i = 1
            while i < 1000:
                img_path = os.path.join(self.root, "images", doc_name,f"{i}.png")
                text_path = os.path.join(self.root, "images", doc_name,f"{i}_parser.html")
                if os.path.exists(img_path):
                    self.pairs.append({
                        "query": query,
                        "img_path": img_path,
                        "text_path": text_path,
                        "doc_name": doc_name,
                        "gt_page_ids": gt_page_ids,
                        "page_num": i
                    })
                    i += 1
                else:
                    break
    

    def __len__(self):
        return len(self.pairs)



    def __getitem__(self, idx):
        pair = self.pairs[idx]
        query = pair["query"]
        img_path = pair["img_path"]
        doc_name = pair["doc_name"]
        page_num = pair["page_num"]
        gt_page_ids = pair.get("gt_page_ids", [])

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Load extracted text if available
        extracted_text = ""
        if self.use_text:
            text_path = pair.get("text_path", "")
            if text_path and os.path.exists(text_path):
                html_content = load_html_file(text_path)
                extracted_text = html_to_plain_text(html_content)

        return {
            "query": query,
            "image": img,
            "extracted_text": extracted_text,
            "doc_name": doc_name,
            "page_num": page_num,
            "gt_page_ids": gt_page_ids,
            "idx": idx
        }