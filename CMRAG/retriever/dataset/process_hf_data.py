import os
from datasets import load_dataset
import json
from tqdm import tqdm
import argparse
import hashlib



# Define parameters
def parse_args():
    parser = argparse.ArgumentParser(description="Image Parser using QwenVL")
    parser.add_argument("--data_name", type=str, default="domain", choices=["domain", "synthetic"], help="Dataset to process")
    parser.add_argument("--idx", type=int, default=0, help="Index of the first image to process")
    parser.add_argument("--num_imgs", type=int, default=20000, help="Number of images to process")
    args = parser.parse_args()
    return args


def preprocess_dataset(dataset, output_dir):
    args = parse_args()

    """Main preprocessing function"""
    os.makedirs(output_dir, exist_ok=True)

    img_path = os.path.join(output_dir, "images")
    os.makedirs(img_path, exist_ok=True)
    
    processed_images = {}  # To track duplicates
    image_counter = 1 + args.idx
    data = []
    img_ids_unique = []
    img_id_unique = 0
    #dataset = [dataset[i] for i in range(args.idx, min(len(dataset), args.idx + args.num_imgs))]
    for item in tqdm(dataset, desc="Processing dataset"):
        try:
            image = item['image']
            query = item['query']
            source = item['source']
            
            # Convert image to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Check for duplicate images using hash
            image_hash = ImageHash(image)
            
            if image_hash in processed_images:
                # Skip duplicate images
                img_idx = processed_images[image_hash]
                # image_path = os.path.join(img_path, f"{img_idx}.png")
            else:
                img_idx = image_counter
                processed_images[image_hash] = image_counter
                # # Save original image
                # image_path = os.path.join(img_path, f"{img_idx}.png")
                # image.save(image_path)
                image_counter += 1
                img_ids_unique.append(img_id_unique)

            img_id_unique += 1

            data.append({
                'query': query,
                'img_id': img_idx,
                'source': source
                })
            
            if len(processed_images) % 1000 == 0:
                print(f"Processed image {len(processed_images)} unique images so far.")
            
        except Exception as e:
            print(f"Error processing item {e}")
            continue

    # Save metadata to JSON
    # end = min(args.idx + args.num_imgs, len(dataset))
    with open(os.path.join(output_dir, f"query_2_img_newid.json"), 'w') as f:
        json.dump(data, f, indent=4)
    
    # Save unique img ids
    with open(os.path.join(output_dir, f"unique_img_ori_ids.json"), 'w') as f:
        json.dump(img_ids_unique, f, indent=4)



# Function to compute image hash
def ImageHash(img):
    return hashlib.md5(img.tobytes()).hexdigest()





# Combine all data
def combine_data(data_dir):
    combined_data = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.json'):
            with open(os.path.join(data_dir, file_name), 'r') as f:
                data = json.load(f)
                combined_data.extend(data)

    with open(os.path.join(data_dir, "train_pair_all.json"), 'w') as f:
        json.dump(combined_data, f, indent=4)
    print(f"Combined data saved to {os.path.join(data_dir, 'train_pair_all.json')}")




# Usage
if __name__ == "__main__":
    args = parse_args()
    # print(f"Processing {args.data_name} dataset starting from index {args.idx} for {args.num_imgs} images.")
    if args.data_name == "domain":
        dataset_name = "openbmb/VisRAG-Ret-Train-In-domain-data"
    else:
        dataset_name = "openbmb/VisRAG-Ret-Train-Synthetic-data"
    
    # Load dataset
    ds = load_dataset(dataset_name, split="train")
    # Preprocess and save
    output_directory = f"../data/{args.data_name}_data"
    preprocess_dataset(ds, output_directory)
    
    print(f"Preprocessing complete. Files saved in '{output_directory}' directory.")

