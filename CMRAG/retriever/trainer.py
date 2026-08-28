# long_siglip/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .model.loss import SigLIPBCELoss
from .model.encoder import LongSigLIP



class LitLongSigLIP(pl.LightningModule):
    def __init__(self, model_id: str, max_text_len: int,
                lr_encoder: float, lr_scalar: float, weight_decay: float,
                train: bool = True):
        super().__init__()
        self.save_hyperparameters()
        
        # model
        self.model = LongSigLIP(model_id=model_id, max_text_len=max_text_len, train=train)
        
        # loss
        self.crit = SigLIPBCELoss()

        # Hyperparameters
        self.lr_encoder = lr_encoder
        self.lr_scalar  = lr_scalar
        self.weight_decay = weight_decay


    # ====== forward ======
    def forward(self, batch):
        s_iq, s_tq = self.model(
            pixel_values     = batch["pixel_values"],
            input_ids_q      = batch["input_ids_q"],
            attention_mask_q = batch["attention_mask_q"],
            input_ids_t      = batch["input_ids_t"],
            attention_mask_t = batch["attention_mask_t"],
        )
        return s_iq, s_tq


    # ====== training step ======
    def training_step(self, batch, batch_idx):
        s_iq, s_tq = self(batch)
        loss_i = self.crit(s_iq)
        loss_t = self.crit(s_tq)
        # loss   = (1-self.model.text_weight)*loss_i + self.model.text_weight*loss_t
        loss   = loss_i + loss_t

        self.log("train_loss", loss, prog_bar=True, batch_size=batch["pixel_values"].shape[0])
        # self.log("text_weight", self.model.text_weight, prog_bar=True)
        self.log("scale", self.model.scale, prog_bar=True)
        self.log("bias", self.model.bias, prog_bar=True)
        return loss


    # ====== optimizer ======
    def configure_optimizers(self):
        opt = torch.optim.AdamW([
            {"params": self.model.long_text_enc.parameters(), "lr": self.lr_encoder},
            {"params": [self.model.text_weight, self.model.scale, self.model.bias], "lr": self.lr_scalar},
        ], weight_decay=self.weight_decay)

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.estimated_stepping_batches, eta_min=1e-6)
        return [opt], [{"scheduler": sched, "interval": "step"}]
