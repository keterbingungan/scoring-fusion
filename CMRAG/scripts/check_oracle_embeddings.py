import os
import torch

q_dir = "/content/drive/MyDrive/TA/CMRAG/oracle_dataset/query_embeddings"
if os.path.exists(q_dir):
    files = [f for f in os.listdir(q_dir) if f.endswith(".pt")]
    print(f"[*] query_embeddings folder exists with {len(files)} .pt files.")
    if files:
        sample_path = os.path.join(q_dir, files[0])
        emb = torch.load(sample_path)
        print(f"Sample file '{files[0]}' loaded. Shape: {emb.shape}, Dtype: {emb.dtype}")
else:
    print(f"query_embeddings folder not found at {q_dir}")
