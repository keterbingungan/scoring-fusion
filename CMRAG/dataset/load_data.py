"""  Load data according to the given data name """

import os
from pathlib import Path
import json
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import hashlib


# Function to load data paths based on the provided data name for image parsing
def load_img_path(data_name="MMLongBench", start_idx=0, num_imgs=-1):
    """
    Load image data for the specified dataset.
    
    Args:
        data_name (str): Name of the dataset to load.
        
    Returns:
        str: Path to the directory containing images.
    """
    
    name2root = {
        "MMLongBench": "../data/MMLongBench-Doc/images",
        "ViDoSeek": "../data/ViDoSeek/images",
        "SlideVQA": "../data/SlideVQA-refined/images",
    }

    name2hf = {
        "domain": "openbmb/VisRAG-Ret-Train-In-domain-data",
        "synthetic": "openbmb/VisRAG-Ret-Train-Synthetic-data",
    }

    # Root directory containing datasets
    if data_name in name2root:
        root = name2root[data_name]
    
        # All sub-directories in the root
        img_dirs = os.listdir(root)
        img_dirs.sort()
        
        # Load all image paths into a list
        img_paths = []
        for img_dir in img_dirs:
            # Skip non-directory files
            if not os.path.isdir(os.path.join(root, img_dir)):
                continue
            # all image names
            img_names = os.listdir(os.path.join(root, img_dir))
            img_names.sort()
            for img_name in img_names:
                if img_name.endswith('.png'):
                    img_paths.append(os.path.join(root, img_dir, img_name))
    
        return img_paths


    # Load from Huggingface datasets
    elif data_name in name2hf:
        ds = load_dataset(name2hf[data_name], split="train")
        
        # Load unique image ids
        with open(f"../data/{data_name}_data/unique_img_ori_ids.json", 'r') as f:
            img_ids_unique = json.load(f)
        
        # Select image according to the unique ids
        ds_unique = ds.select(img_ids_unique)

        # # Not decode images here, will decode on-the-fly later
        # ds_unique = ds_unique.cast_column("image", Image(decode=False))

        # End idx
        end_idx = len(ds_unique) if num_imgs < 0 else min(start_idx + num_imgs, len(ds_unique))
        # Select images
        imgs = ds_unique.select(range(start_idx, end_idx)).select_columns(['image'])
        imgs = imgs[:]["image"]
        
        # release memory of ds
        del ds
        del ds_unique

        return imgs


    else:
        raise ValueError(f"Dataset {data_name} not supported. Choose from ['MMLongBench', 'ViDoSeek', 'SlideVQA', 'domain', 'synthetic']")









# Load testqa data according to the given data name
def load_testqa(data_name="MMLongBench", num_samples=-1):
    """
    Load testqa data for the specified dataset.
    
    Args:
        data_name (str): Name of the dataset to load.
        
    Returns:
        str: testqa data.
    """
    
    # Root directory containing datasets
    """
    We uniform the data format for all datasets.
        query: the question to be answered
        answer: the answer to the question
        doc_name: the document id
        doc_page: the page number of the document where the answer is found
        source_type: the type of the source (e.g., text, image)
        query_type: the type of the query (e.g., single_hop, multi_hop)
    """

    # "doc_id": "PH_2016.06.08_Economy-Final.pdf",
    # "doc_type": "Research report / Introduction",
    # "question": "According to the report, how do 5% of the Latinos see economic upward mobility for their children?",
    # "answer": "Less well-off",
    # "evidence_pages": "[5]",
    # "evidence_sources": "['Chart']",
    # "answer_format": "Str"
    if data_name == "MMLongBench":
        file_path = "../data/MMLongBench-Doc/samples.json"
        with open(file_path, 'r') as f:
            testqa_data = json.load(f)
        for item in testqa_data:
            item["query"] = item.pop("question")
            item["answer"] = item.pop("answer")
            item["doc_name"] = item.pop("doc_id")
            item["doc_page"] = item.pop("evidence_pages")
            item["source_type"] = item.pop("evidence_sources")
            item["query_type"] = None

    # "uid": "03c7d62afbcc088cad1c810c09a71df29a29c968_1",
    # "query": "Apply for Nordic Swan Ecolabel license, what is recommended as a web browser according to the Nordic Ecolabelling Portal instructions?",
    # "reference_answer": "Microsoft Edge or Google Chrome.",
    # "meta_info": {
    #       "file_name": "03c7d62afbcc088cad1c810c09a71df29a29c968.pdf",
    #       "reference_page": [
    #          4
    #       ],
    #       "source_type": "text",
    #       "query_type": "single_hop"
    #   }
    elif data_name == "ViDoSeek":
        file_path = "../data/ViDoSeek/vidoseek.json"
        with open(file_path, 'r') as f:
            testqa_data = json.load(f)
            testqa_data = testqa_data["examples"]
        for item in testqa_data:
            item["query"] = item.pop("query")
            item["answer"] = item.pop("reference_answer")
            item["doc_name"] = item["meta_info"].pop("file_name")
            item["doc_page"] = item["meta_info"].pop("reference_page")
            item["source_type"] = item["meta_info"].pop("source_type")
            item["query_type"] = item["meta_info"].pop("query_type")
    

    # "uid": "17",
    # "query": "What is the title of the presentation that ALCATEL-LUCENT made for the OVERVIEW of 2012 shown in the image?",
    # "reference_answer": "AT THE SPEED OF IDEAS",
    # "meta_info": {
    #   "file_name": "alcatel-lucenteesoverview-120403090342-phpapp01_95.pdf",
    #   "reference_page": [
    #     1
    #   ],
    # "origin_query": "What is the title of the presentation that ALCATEL-LUCENT made for the OVERVIEW of 2012?",
    # "query_type": "single_hop",
    # "source_type": "single_span"
    elif data_name == "SlideVQA":
        file_path = "../data/SlideVQA-refined/slidevqa_refined.json"
        with open(file_path, 'r') as f:
            testqa_data = json.load(f)
            testqa_data = testqa_data["examples"]
        for item in testqa_data:
            item["query"] = item.pop("query")
            item["answer"] = item.pop("reference_answer")
            item["doc_name"] = item["meta_info"].pop("file_name")
            item["doc_page"] = item["meta_info"].pop("reference_page")
            item["source_type"] = item["meta_info"].pop("source_type")
            item["query_type"] = item["meta_info"].pop("query_type")

    else:
        raise ValueError(f"Dataset {data_name} not supported. Choose from ['MMLongBench', 'ViDoSeek', 'SlideVQA']")

    if num_samples > 0:
        testqa_data = testqa_data[:num_samples]

    return testqa_data






# Function to load image root based on the provided data name for loading images
def load_img_root(data_name="MMLongBench"):
    """
    Load image root for the specified dataset.
    
    Args:
        data_name (str): Name of the dataset to load.
        
    Returns:
        str: Path to the root directory containing images.
    """
    
    # Root directory containing datasets
    if data_name == "MMLongBench":
        root = "../data/MMLongBench-Doc/images"
    elif data_name == "ViDoSeek":
        root = "../data/ViDoSeek/images"
    elif data_name == "SlideVQA":
        root = "../data/SlideVQA-refined/images"
    else:
        raise ValueError(f"Dataset {data_name} not supported. Choose from ['MMLongBench', 'ViDoSeek', 'SlideVQA']")

    return root