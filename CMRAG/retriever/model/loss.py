# long_siglip/loss.py
import torch
import torch.nn.functional as F
import torch.nn as nn


class SigLIPBCELoss(nn.Module):
    """
    Pair-wise sigmoid BCE (no full-batch softmax) exactly like original SigLIP.
    Expects logits of shape (B, B) or (B,) when diagonal is positive pairs.
    """
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        B = scores.size(0)
        labels = torch.eye(B, dtype=torch.float, device=scores.device)
        
        return F.binary_cross_entropy_with_logits(scores, labels, reduction='mean')