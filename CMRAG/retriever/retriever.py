import torch
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import os
from pathlib import Path

from utils.utils import ensure_pil
from .trainer import LitLongSigLIP
from .model.encoder import LongSigLIP
from .model.processor import LongSigLIPProcessor


class Retriever:
    def __init__(self, ckpt_path: str, max_text_len, model_id, device: str):
        self.ckpt_path = ckpt_path
        self.max_text_len = max_text_len
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        
        self.load_model()


    # ---------- public API --------------------------------------------------
    def load_model(self):
        # Load model and processor
        if self.ckpt_path and os.path.exists(self.ckpt_path):
            print(f"Loading model from checkpoint: {self.ckpt_path} ...")
            pl_model = LitLongSigLIP.load_from_checkpoint(self.ckpt_path).to(self.device)
            model = pl_model.model # plain nn.Module
        else:
            print(f"Checkpoint file '{self.ckpt_path}' not found. Loading base pretrained model '{self.model_id}' directly...")
            model = LongSigLIP(model_id=self.model_id, max_text_len=self.max_text_len, train=False)
        
        processor = LongSigLIPProcessor(
            model_id=self.model_id,
            max_text=self.max_text_len,
        )
        self.model = model.to(self.device)
        self.processor = processor
        print("Model loaded.")


    def encode_doc(
        self,
        images: List[Image.Image],
        texts: Optional[List[str]] = None,
        res_path: str = "sdk/retriever/embeddings",
        batch_size: int = 8,
    ):
        """
        Encode pages (image ± text) in batches and store:
            embeddings.npy   -> np.ndarray [N, D]
            index.json       -> List[int]  page-ids in same order
        """
        os.makedirs(res_path, exist_ok=True)

        # Encode and save embeddings
        img_emb = []
        text_emb = []
        for i in tqdm(range(0, len(images), batch_size), desc=f"Encoding"):
            batch_images = images[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            with torch.no_grad():
                batch_results = self.processor.process_img_text(
                    images=batch_images,
                    extracted_texts=batch_texts,
                )
                # Encode images and texts
                z_i, z_t = self.model.encode_img_text(
                    pixel_values     = batch_results["pixel_values"].to(self.device),
                    input_ids_t      = batch_results["input_ids_t"].to(self.device),
                    attention_mask_t = batch_results["attention_mask_t"].to(self.device),
                )

                img_emb.append(z_i.cpu())
                text_emb.append(z_t.cpu())
        
        # Concatenate all batches
        img_emb = torch.cat(img_emb, dim=0) # [N, D]
        text_emb = torch.cat(text_emb, dim=0)
        
        # Save embeddings
        torch.save(img_emb, os.path.join(res_path, "images.pt"))
        torch.save(text_emb, os.path.join(res_path, "texts.pt"))
        print(f"Saved embeddings to {res_path}")



    def encode_query(self, query: str) -> np.ndarray:
        """Return pooled + L2-norm query vector shape [D]."""
        with torch.no_grad():
            inputs = self.processor.process_queries([query])
            z_q = self.model.encode_query(
                     input_ids_q=inputs["input_ids_q"].to(self.device),
                     attention_mask_q=inputs["attention_mask_q"].to(self.device)
                 ) # [1, D]
        return z_q



    def retrieve(
        self,
        doc_emb_dir: str,
        query: str,
        topk: int = 10,
        text_weight: float = 0.2,
        use_ucmr_norm: bool = True,
    ) -> Tuple[List[int], List[float]]:
        """Load pre-computed document embeddings and return **page-ids** of top-k pages.
        """
        with torch.no_grad():
            img_emb = torch.load(os.path.join(doc_emb_dir, "images.pt")).to(self.device)        # [N, D]
            text_emb = torch.load(os.path.join(doc_emb_dir, "texts.pt")).to(self.device)        # [N, D]

            q_vec = self.encode_query(query)  # [1, D]
            # Repeat query vector for batch computation
            q_vec = q_vec.repeat(img_emb.size(0), 1).to(self.device)  # [N, D]
            

            # Compute raw similarity
            s_iq = torch.einsum('bd,bd->b', img_emb, q_vec) # [N,]
            s_tq = torch.einsum('bd,bd->b', text_emb, q_vec) # [N,]

            if use_ucmr_norm:
                # UCMR Normalization: Sigmoid Normalization -> Z-score Normalization -> Weighted Fusion
                # 1. Sigmoid Normalization
                scale = getattr(self.model, 'scale', 10.0)
                bias = getattr(self.model, 'bias', -10.0)
                sig_iq = torch.sigmoid(s_iq * scale + bias)
                sig_tq = torch.sigmoid(s_tq * scale + bias)

                # 2. Z-score Normalization across document candidates
                std_iq = torch.std(sig_iq)
                std_tq = torch.std(sig_tq)
                z_iq = (sig_iq - torch.mean(sig_iq)) / (std_iq if std_iq > 1e-6 else 1.0)
                z_tq = (sig_tq - torch.mean(sig_tq)) / (std_tq if std_tq > 1e-6 else 1.0)

                # 3. Weighted Scoring Fusion
                sim = (1 - text_weight) * z_iq + text_weight * z_tq  # [N,]
            else:
                # Raw similarity weighted sum (direct combination)
                sim = (1 - text_weight) * s_iq + text_weight * s_tq  # [N,]

            # Get top-k indices according to similarity
            k_val = min(topk, sim.size(0))
            topk_res = torch.topk(sim, k=k_val)
            top_idx = topk_res.indices.detach().cpu().numpy().tolist()
            top_scores = topk_res.values.detach().cpu().numpy().tolist()

            page_ids = [i + 1 for i in range(img_emb.size(0))]  # assuming page ids are 1, 2, 3, ...
            
            return [page_ids[i] for i in top_idx], top_scores        