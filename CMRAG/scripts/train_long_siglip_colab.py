import os
import sys
import io
import argparse
import subprocess
import torch

# 1. Auto-install missing packages in Colab VM if needed
required_packages = ["pytorch-lightning", "transformers", "datasets", "beautifulsoup4"]
for pkg in required_packages:
    try:
        if pkg == "pytorch-lightning":
            import pytorch_lightning
        elif pkg == "beautifulsoup4":
            import bs4
        else:
            __import__(pkg)
    except ImportError:
        print(f"[*] Package '{pkg}' not found in Colab VM. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
from transformers import SiglipModel, SiglipProcessor
from bs4 import BeautifulSoup
from datasets import load_dataset


# 2. Auto-mount Google Drive
if not os.path.exists("/content/drive/MyDrive"):
    try:
        from google.colab import drive
        print("[*] Mounting Google Drive at /content/drive ...")
        drive.mount("/content/drive", force_remount=False)
    except Exception as e:
        print(f"[WARNING] Could not auto-mount Google Drive: {e}")


# 3. Setup sys.path candidate roots
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

for root in candidate_roots:
    if os.path.exists(os.path.join(root, "retriever", "trainer.py")):
        if root not in sys.path:
            sys.path.insert(0, root)
        break

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
    from retriever.trainer import LitLongSigLIP
    from retriever.dataset.Data_Loader import load_data
    from retriever.model.processor import LongSigLIPProcessor
except ImportError:
    print("[*] Module 'retriever' not found in path. Using inline LitLongSigLIP trainer & dataset loader...")

    class SigLIPBCELoss(nn.Module):
        def forward(self, scores: torch.Tensor) -> torch.Tensor:
            B = scores.size(0)
            labels = torch.eye(B, dtype=torch.float, device=scores.device)
            return F.binary_cross_entropy_with_logits(scores, labels, reduction="mean")

    class LongSigLIP(nn.Module):
        def __init__(self, model_id: str = "google/siglip-so400m-patch14-384", max_text_len: int = 768, train: bool = True):
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
            self.train_mode = train

        def forward(self, pixel_values, input_ids_q, attention_mask_q, input_ids_t, attention_mask_t):
            with torch.no_grad():
                z_i = F.normalize(_extract_tensor(self.siglip.get_image_features(pixel_values)), dim=-1)
                z_q = F.normalize(_extract_tensor(self.siglip.get_text_features(input_ids=input_ids_q, attention_mask=attention_mask_q)), dim=-1)

            z_t = F.normalize(_extract_tensor(self.long_text_enc(input_ids=input_ids_t, attention_mask=attention_mask_t)), dim=-1)

            if self.train_mode:
                s_iq = (z_i @ z_q.T) * self.scale + self.bias
                s_tq = (z_t @ z_q.T) * self.scale + self.bias
            else:
                s_iq = torch.einsum("bd,bd->b", z_i, z_q) * self.scale + self.bias
                s_tq = torch.einsum("bd,bd->b", z_t, z_q) * self.scale + self.bias

            return s_iq, s_tq

    class LongSigLIPProcessor:
        def __init__(self, model_id: str, max_query: int = 64, max_text: int = 768):
            self.core = SiglipProcessor.from_pretrained(model_id)
            self.max_q = max_query
            self.max_t = max_text

        def __call__(self, *, images, queries, extracted_texts):
            processed_imgs = []
            for img in images:
                if isinstance(img, Image.Image):
                    processed_imgs.append(img.convert("RGB"))
                else:
                    processed_imgs.append(Image.open(img).convert("RGB"))
            pixel_values = self.core(images=processed_imgs, return_tensors="pt").pixel_values
            tok = self.core.tokenizer
            q = tok(queries, max_length=self.max_q, truncation=True, padding="max_length", return_tensors="pt")
            t = tok(extracted_texts, max_length=self.max_t, truncation=True, padding="max_length", return_tensors="pt")
            q_mask = (q.input_ids != tok.pad_token_id).int()
            t_mask = (t.input_ids != tok.pad_token_id).int()
            return {
                "pixel_values": pixel_values,
                "input_ids_q": q["input_ids"],
                "attention_mask_q": q_mask,
                "input_ids_t": t["input_ids"],
                "attention_mask_t": t_mask,
            }

    class LitLongSigLIP(pl.LightningModule):
        def __init__(self, model_id: str, max_text_len: int, lr_encoder: float, lr_scalar: float, weight_decay: float, train: bool = True):
            super().__init__()
            self.save_hyperparameters()
            self.model = LongSigLIP(model_id=model_id, max_text_len=max_text_len, train=train)
            self.crit = SigLIPBCELoss()
            self.lr_encoder = lr_encoder
            self.lr_scalar = lr_scalar
            self.weight_decay = weight_decay

        def forward(self, batch):
            return self.model(
                pixel_values=batch["pixel_values"],
                input_ids_q=batch["input_ids_q"],
                attention_mask_q=batch["attention_mask_q"],
                input_ids_t=batch["input_ids_t"],
                attention_mask_t=batch["attention_mask_t"],
            )

        def training_step(self, batch, batch_idx):
            s_iq, s_tq = self(batch)
            loss = self.crit(s_iq) + self.crit(s_tq)
            self.log("train_loss", loss, prog_bar=True, batch_size=batch["pixel_values"].shape[0])
            self.log("scale", self.model.scale, prog_bar=True)
            self.log("bias", self.model.bias, prog_bar=True)
            return loss

        def configure_optimizers(self):
            opt = torch.optim.AdamW(
                [
                    {"params": self.model.long_text_enc.parameters(), "lr": self.lr_encoder},
                    {"params": [self.model.text_weight, self.model.scale, self.model.bias], "lr": self.lr_scalar},
                ],
                weight_decay=self.weight_decay,
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.trainer.estimated_stepping_batches, eta_min=1e-6
            )
            return [opt], [{"scheduler": sched, "interval": "step"}]

    class load_data(torch.utils.data.IterableDataset):
        def __init__(self, data_type: str = "synthetic", use_text: bool = True, batch_size: int = 4, max_samples: int = -1):
            data_paths = {
                "domain": "openbmb/VisRAG-Ret-Train-In-domain-data",
                "synthetic": "openbmb/VisRAG-Ret-Train-Synthetic-data",
            }
            target_ds = data_paths.get(data_type, data_paths["synthetic"])
            print(f"[*] Streaming dataset '{target_ds}' from HuggingFace (0 local disk space required)...")
            self.dataset = load_dataset(target_ds, split="train", streaming=True)
            self.use_text = use_text
            self.max_samples = max_samples

        def __iter__(self):
            count = 0
            iterator = iter(self.dataset)
            while True:
                if self.max_samples > 0 and count >= self.max_samples:
                    break
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                except Exception as e:
                    continue

                try:
                    img = item["image"]
                    if isinstance(img, Image.Image):
                        img = img.convert("RGB")
                    elif isinstance(img, dict):
                        if img.get("bytes") is not None:
                            img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
                        else:
                            img = Image.open(img["path"]).convert("RGB")
                    else:
                        continue

                    query_str = item.get("query", item.get("question", ""))
                    txt_str = item.get("extracted_text", item.get("text", query_str))
                    if not txt_str or not txt_str.strip():
                        txt_str = query_str

                    count += 1
                    yield {
                        "query": query_str,
                        "image": img,
                        "extracted_text": txt_str if self.use_text else "",
                        "idx": count,
                    }
                except Exception:
                    continue


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune LongSigLIP (768 tokens) on Colab T4 GPU."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/TA/CMRAG/output_longSigLIP",
        help="Google Drive output directory to save trained model checkpoints",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/siglip-so400m-patch14-384",
        help="Base SigLIP model ID",
    )
    parser.add_argument(
        "--max_query",
        type=int,
        default=64,
        help="Max query token length",
    )
    parser.add_argument(
        "--max_text",
        type=int,
        default=768,
        help="Max long text token length",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=4,
        help="Micro-batch size per GPU step (reduced to 4 to fit in 15GB T4 VRAM)",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=2,
        help="Gradient accumulation steps (4 * 2 = 8 effective batch size for fast updates)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Total training epochs",
    )
    parser.add_argument(
        "--lr_encoder",
        type=float,
        default=3e-5,
        help="Learning rate for text encoder",
    )
    parser.add_argument(
        "--lr_scalar",
        type=float,
        default=1e-2,
        help="Learning rate for scale and bias scalars",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay factor",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker count for streaming dataset",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="synthetic",
        choices=["domain", "synthetic", "combined"],
        help="Which dataset split to train on from openbmb/VisRAG-Ret-Train",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Limit total training samples (-1 for full dataset)",
    )
    args, _ = parser.parse_known_args()
    return args


processor = None


def collate_fn(batch):
    global processor
    return processor(
        images=[b["image"] for b in batch],
        queries=[b["query"] for b in batch],
        extracted_texts=[b["extracted_text"] for b in batch],
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("CMRAG LongSigLIP Fine-Tuning | Device: CUDA (Colab T4 Optimized)")
    print("=" * 80)

    global processor
    processor = LongSigLIPProcessor(
        model_id=args.model_id,
        max_query=args.max_query,
        max_text=args.max_text,
    )

    print(f"[*] Initializing LitLongSigLIP with max_text={args.max_text}...")
    model = LitLongSigLIP(
        model_id=args.model_id,
        max_text_len=args.max_text,
        lr_encoder=args.lr_encoder,
        lr_scalar=args.lr_scalar,
        weight_decay=args.weight_decay,
        train=True,
    )

    print(f"[*] Loading dataset '{args.data_type}' from HuggingFace (Streaming Mode)...")
    dataset_obj = load_data(
        data_type=args.data_type,
        use_text=True,
        batch_size=args.micro_batch_size,
        max_samples=args.max_samples,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_obj,
        batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    local_output_dir = "/content/output_longSigLIP"
    os.makedirs(local_output_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=local_output_dir,
        filename="long_siglip-epoch{epoch:02d}",
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )

    class DriveSyncCallback(pl.Callback):
        def __init__(self, local_dir, drive_dir):
            self.local_dir = local_dir
            self.drive_dir = drive_dir

        def on_train_epoch_end(self, trainer, pl_module):
            import shutil
            os.makedirs(self.drive_dir, exist_ok=True)
            if os.path.exists(self.local_dir):
                for f in os.listdir(self.local_dir):
                    if f.endswith(".ckpt"):
                        src = os.path.join(self.local_dir, f)
                        dst = os.path.join(self.drive_dir, f)
                        try:
                            shutil.copy2(src, dst)
                            print(f"\n[+] Synced checkpoint to Drive: {f}")
                        except Exception:
                            pass

    drive_sync_cb = DriveSyncCallback(local_output_dir, args.output_dir)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=args.num_epochs,
        gradient_clip_val=1.0,
        default_root_dir=local_output_dir,
        callbacks=[checkpoint_cb, drive_sync_cb],
        logger=False,  # Disable heavy TensorBoard log events to prevent FUSE disk errors
        enable_checkpointing=True,
    )

    print(f"[*] Starting training for {args.num_epochs} epochs...")
    print(f"    Local checkpoint dir : {local_output_dir}")
    print(f"    Google Drive dir     : {args.output_dir}")
    trainer.fit(model, train_loader)

    # Copy all checkpoints to Google Drive
    import shutil
    print("[*] Copying trained checkpoints to Google Drive...")
    for f in os.listdir(local_output_dir):
        if f.endswith(".ckpt"):
            src = os.path.join(local_output_dir, f)
            dst = os.path.join(args.output_dir, f)
            shutil.copy2(src, dst)
            print(f"    [+] Saved to Drive: {dst}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Checkpoints saved in: {args.output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
