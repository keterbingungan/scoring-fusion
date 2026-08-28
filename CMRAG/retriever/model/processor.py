# long_siglip/processor.py
from transformers import SiglipProcessor
import torch
from typing import List

from utils.utils import ensure_pil

class LongSigLIPProcessor:
    def __init__(self, model_id: str, max_query: int = 64, max_text: int = 768):
        self.core = SiglipProcessor.from_pretrained(model_id)
        self.max_q = max_query
        self.max_t = max_text


    def __call__(self, *, images, queries: List[str], extracted_texts: List[str]):
        images = [ensure_pil(img) for img in images]
        # image  (keep batch dim)
        pixel_values = self.core(images=images, return_tensors="pt").pixel_values

        tok = self.core.tokenizer
        q = tok(queries, max_length=self.max_q, truncation=True, padding="max_length", return_tensors="pt")
        t = tok(extracted_texts, max_length=self.max_t, truncation=True, padding="max_length", return_tensors="pt")
        
        # 2.  build the mask yourself (1 = real token, 0 = pad)
        q_mask = (q.input_ids != self.core.tokenizer.pad_token_id).int()
        t_mask = (t.input_ids != self.core.tokenizer.pad_token_id).int()
        
        return {
            "pixel_values"      : pixel_values,        # [B, 3, H, W]
            "input_ids_q"       : q["input_ids"],      # [B, Q]
            "attention_mask_q"  : q_mask,              # [B, Q]
            "input_ids_t"       : t["input_ids"],      # [B, S]
            "attention_mask_t"  : t_mask,              # [B, S]
        }
       
    

    def process_queries(self, queries: List[str]):
        """ Process only queries (for inference) """
        tok = self.core.tokenizer
        q = tok(queries, max_length=self.max_q, truncation=True, padding="max_length", return_tensors="pt")
        q_mask = (q.input_ids != self.core.tokenizer.pad_token_id).int()
        return {
            "input_ids_q"       : q["input_ids"],      # [B, Q]
            "attention_mask_q"  : q_mask,              # [B, Q]
        }
    

    def process_img_text(self, images, extracted_texts: List[str]):
        """ Process only images + texts (for indexing) """
        images = [ensure_pil(img) for img in images]
        pixel_values = self.core(images=images, return_tensors="pt").pixel_values
        
        tok = self.core.tokenizer
        t = tok(extracted_texts, max_length=self.max_t, truncation=True, padding="max_length", return_tensors="pt")
        t_mask = (t.input_ids != self.core.tokenizer.pad_token_id).int()
        
        return {
            "pixel_values"      : pixel_values,        # [B, 3, H, W]
            "input_ids_t"       : t["input_ids"],      # [B, S]
            "attention_mask_t"  : t_mask,              # [B, S]
        }