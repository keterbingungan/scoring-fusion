# long_siglip/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipModel
from typing import Tuple



def _extract_tensor(feat):
    if isinstance(feat, torch.Tensor):
        return feat
    if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
        return feat.pooler_output
    return feat[0]


def interpolate_pos_embed(model, new_len: int):
    """Resize absolute position embeddings to new_len."""
    old = model.embeddings.position_embedding.weight           # (old_len, D)
    new = F.interpolate(old.unsqueeze(0).permute(0, 2, 1),
                       size=new_len, mode='linear', align_corners=False)\
           .permute(0, 2, 1).squeeze(0)
    model.embeddings.position_embedding = nn.Embedding.from_pretrained(new, freeze=False)
    model.embeddings.position_ids = torch.arange(new_len).unsqueeze(0)



class LongSigLIP(nn.Module):
    """
    Frozen SigLIP image + frozen SigLIP short-text towers.
    Trainable long-text encoder (weight-init from SigLIP-text) + learnable λ, τ, bias.
    Returns  similarity logits:  (s_img_query , s_text_query)
    """
    def __init__(self,
                 model_id: str = "google/siglip-so400m-patch14-384",
                 max_text_len: int = 512,
                 train: bool = True):
        super().__init__()
        
        # 1. freeze base model
        self.siglip = SiglipModel.from_pretrained(model_id)
        for p in self.siglip.parameters():
            p.requires_grad_(False)

        # 2. build long encoder with original config
        txt_cfg = self.siglip.text_model.config          # still 64
        self.long_text_enc = type(self.siglip.text_model)(txt_cfg)

        # 3.  copy weights (shapes match)
        self.long_text_enc.load_state_dict(
            self.siglip.text_model.state_dict(), strict=False
        )

        # 4.  Resize position embeddings
        txt_cfg.max_position_embeddings = max_text_len   # 768
        interpolate_pos_embed(self.long_text_enc, max_text_len)
        # ensure contiguity
        for p in self.long_text_enc.parameters():
            if not p.is_contiguous():
                p.data = p.data.contiguous()

        # 5. learnable scalars
        self.text_weight = nn.Parameter(torch.tensor(0.2))
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(-10.0))
        
        self.train = train



    def forward(self,
                pixel_values: torch.Tensor,
                input_ids_q: torch.Tensor,
                attention_mask_q: torch.Tensor,
                input_ids_t: torch.Tensor,
                attention_mask_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        with torch.no_grad():
            # Encoding image
            z_i = F.normalize(
                _extract_tensor(self.siglip.get_image_features(pixel_values)),
                dim=-1
            )
            # Encoding short text (query)
            z_q = F.normalize(
                _extract_tensor(self.siglip.get_text_features(
                    input_ids=input_ids_q,
                    attention_mask=attention_mask_q
                )),
                dim=-1
            )
        
        # Encoding long text
        z_t = F.normalize(
            self.long_text_enc(input_ids=input_ids_t,
                              attention_mask=attention_mask_t).pooler_output,
            dim=-1
        )
        if self.train:
            s_iq = (z_i @ z_q.T) * self.scale + self.bias   # (B, B)
            s_tq = (z_t @ z_q.T) * self.scale + self.bias   # (B, B)
        else:
            s_iq = torch.einsum('bd,bd->b', z_i, z_q) * self.scale + self.bias # (B,)
            s_tq = torch.einsum('bd,bd->b', z_t, z_q) * self.scale + self.bias # (B,)

        return s_iq, s_tq



    # Encoding images and text
    def encode_img_text(self,
                        pixel_values: torch.Tensor,
                        input_ids_t: torch.Tensor,
                        attention_mask_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ For indexing: encode images and long texts """
        with torch.no_grad():
            # Encoding image
            z_i = F.normalize(
                _extract_tensor(self.siglip.get_image_features(pixel_values)),
                dim=-1
            )
        
            # Encoding long text
            z_t = F.normalize(
                self.long_text_enc(input_ids=input_ids_t,
                                attention_mask=attention_mask_t).pooler_output,
                dim=-1
            )
        
        return z_i, z_t # [B, D], [B, D]


    # Encoding queries
    def encode_query(self,
                     input_ids_q: torch.Tensor,
                     attention_mask_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ For retrieval: encode queries """
        with torch.no_grad():
            # Encoding short text (query)
            z_q = F.normalize(
                _extract_tensor(self.siglip.get_text_features(
                    input_ids=input_ids_q,
                    attention_mask=attention_mask_q
                )),
                dim=-1
            )
        
        return z_q # [B, D]