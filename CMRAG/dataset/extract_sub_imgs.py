import os
from tqdm import tqdm
from typing import List, Dict
import json
from PIL import Image
import argparse

from dataset.load_data import load_img_root



def parse_args():
    parser = argparse.ArgumentParser(description="Extract sub-images from original images based on bounding boxes.")
    parser.add_argument("--data_name", type=str, default="MMLongBench", help="Name of the dataset to extract sub-images from")
    parser.add_argument("--width_ratio", type=float, default=1.4, help="Width ratio for extracting sub-images")
    parser.add_argument("--height_ratio", type=float, default=1.6, help="Height ratio for extracting sub-images")
    parser.add_argument("--idx", type=int, default=0, help="Index of the data split to use.")
    return parser.parse_args()



def extract_sub_images(img_root: str, width_ratio=1.4, height_ratio=1.6, idx = 0):
    """
    Extract sub-images from the original images based on the provided bounding boxes.
    
    Args:
        img_root (str): Root directory containing images.
        
    Returns:
        None
    """
    num_sub_imgs = 0

    folders = os.listdir(img_root)
    folders.sort()
    for folder in tqdm(folders[idx:], desc="Extracting sub-images"):
        folder_path = os.path.join(img_root, folder)
        if not os.path.isdir(folder_path):
            continue
        
        names = os.listdir(folder_path)
        names.sort()
        for name in names:
            # boxes name format: 1_subimg_boxes.json
            if not name.endswith('.json'):
                continue
            
            coms = name.split('_')
            if len(coms) != 3 or coms[1] != 'subimg' or coms[2] != 'boxes.json':
                continue

            # Load bounding boxes
            boxes_path = os.path.join(folder_path, name)
            with open(boxes_path, 'r') as f:
                boxes = json.load(f)
            
            # if boxes is empty, skip
            if not boxes:
                print(f"No boxes found in {boxes_path}, skipping.")
                continue
            
            # Original image path
            img_path = os.path.join(folder_path, f"{name[:-18]}.png")
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}, skipping.")
                continue

            # Load original image
            img = Image.open(img_path)
            W, H = img.size

            # Extract sub-images based on bounding boxes [x1, y1, x2, y2]
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                # Enlarge the bounding box by 10% (1.1 times) if it is too small
                x1 = max(0, int(x1 - (width_ratio - 1)/2 * w))
                y1 = max(0, int(y1 - (height_ratio - 1)/2 * h))
                x2 = min(W-1, int(x2 + (width_ratio - 1)/2 * w))
                y2 = min(H-1, int(y2 + (height_ratio - 1)/2 * h))
                
                # Ensure the aspect ratio is larger than 50:
                if (x2 - x1) / (y2 - y1) < 0.02 or (y2 - y1) / (x2 - x1) < 0.02:
                    print(f"Aspect ratio too small for {name[:-18]}_subimg_{i}.png, skipping.")
                    continue
                # Ensure the sub-image is not too small
                if (x2 - x1) * (y2 - y1) < 0.01 * W * H:
                    print(f"Sub-image {name[:-18]}_subimg_{i}.png is too small, skipping.")
                    continue
                
                try:
                    sub_img = img.crop((
                        x1,  # x1
                        y1,  # y1
                        x2,  # x2
                        y2   # y2
                    ))
                
                    sub_img.save(os.path.join(folder_path, f"{name[:-18]}_subimg_{i}.png"))
                    num_sub_imgs += 1
                except:
                    print(f"Failed to save sub-image: {name[:-18]}_subimg_{i}.png, skipping")
                    continue
                
                if num_sub_imgs % 100 == 0:
                    print(f"Processed {num_sub_imgs} sub-images...")

    print(f"Total sub-images extracted: {num_sub_imgs}")




if __name__ == "__main__":
    args = parse_args()
    img_root = load_img_root(data_name=args.data_name)
    
    extract_sub_images(img_root, width_ratio=args.width_ratio, height_ratio=args.height_ratio, idx=args.idx)
    
    print("Sub-images extracted successfully.")