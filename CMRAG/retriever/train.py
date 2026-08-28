#!/usr/bin/env python3
# long_siglip/train.py
import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from .trainer import LitLongSigLIP
from .dataset.Data_Loader import load_data
from .model.processor import LongSigLIPProcessor


def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune SigLIP with long OCR text")
    # data
    parser.add_argument("--output_dir", default="../output_longSigLIP", help="where to save")
    # model
    parser.add_argument("--model_id", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--max_query", type=int, default=64)
    parser.add_argument("--max_text",  type=int, default=768)
    # training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=12)
    parser.add_argument("--lr_encoder", type=float, default=3e-5)
    parser.add_argument("--lr_scalar",  type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()



processor = None   # global in train.py

def collate_fn(batch):
    global processor
    return processor(
        images=[b["image"] for b in batch],
        queries=[b["query"] for b in batch],
        extracted_texts=[b["extracted_text"] for b in batch],
    )



def build_dataloader(data_type: str, batch_size: int, num_workers: int):
    """ build dataloader """
    train_loader = load_data(data_type=data_type, use_text=True)
    train_loader = torch.utils.data.DataLoader(
        train_loader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader




def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    global processor
    processor = LongSigLIPProcessor(
        model_id=args.model_id,
        max_query=args.max_query,
        max_text=args.max_text,
    )

    # model
    model = LitLongSigLIP(model_id=args.model_id,
                         max_text_len=args.max_text,
                         lr_encoder=args.lr_encoder,
                         lr_scalar=args.lr_scalar,
                         weight_decay=args.weight_decay)
    # dataloader
    train_loader = build_dataloader(data_type="synthetic",
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)

    # checkpoint_cb = ModelCheckpoint(
    #     dirpath=args.output_dir,
    #     filename="long_siglip-{epoch:02d}-{train_loss:.3f}",
    #     save_top_k=1,
    #     every_n_train_steps=1000,   # or every_n_epochs=1
    # )
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="long_siglip-epoch{epoch:02d}",
        save_top_k=-1,          # keep **all** epochs
        every_n_epochs=1,       # once per epoch
        save_on_train_epoch_end=True,
    )


    # trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=8,
        # strategy=DeepSpeedStrategy(stage=2,
        #                           offload_optimizer=False,
        #                           offload_parameters=False),
        # strategy = DeepSpeedStrategy(stage=1),
        strategy = "ddp_find_unused_parameters_True",
        precision="bf16",
        max_epochs=args.num_epochs,
        gradient_clip_val=1.0,
        default_root_dir=args.output_dir,
        callbacks=[checkpoint_cb],
        enable_checkpointing=True,
    )
    # train
    trainer.fit(model, train_loader)



if __name__ == "__main__":
    main()