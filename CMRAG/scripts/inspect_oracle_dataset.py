import os
import json

oracle_path = "/content/drive/MyDrive/TA/CMRAG/oracle_dataset/oracle_dataset.json"
if not os.path.exists(oracle_path):
    print(f"File not found: {oracle_path}")
else:
    with open(oracle_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[*] Total records found in oracle_dataset.json: {len(data)}")
    if len(data) > 0:
        print("\n--- SAMPLE RECORD 0 ---")
        print(json.dumps(data[0], indent=2))
        if len(data) > 1:
            print("\n--- SAMPLE RECORD 1 ---")
            print(json.dumps(data[1], indent=2))
        
        # Summary statistics on beta
        betas = [r.get("optimal_beta", r.get("beta", None)) for r in data if "optimal_beta" in r or "beta" in r]
        print(f"\n[*] Total records with optimal_beta: {len(betas)}")
        if betas:
            import numpy as np
            print(f"    Mean beta : {np.mean(betas):.4f}")
            print(f"    Min beta  : {np.min(betas):.4f}")
            print(f"    Max beta  : {np.max(betas):.4f}")
