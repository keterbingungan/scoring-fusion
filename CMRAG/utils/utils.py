import json
from typing import List, Dict
from bs4 import BeautifulSoup
import re
import os
import numpy as np
from PIL import Image
import io
import torch.cuda as cuda



def load_json(path, type="json"):
    assert type in ["json", "jsonl"] # only support json or jsonl format
    if type == "json":
        outputs = json.loads(open(path, "r", encoding="utf-8").read())
    elif type == "jsonl":
        outputs = []
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                outputs.append(json.loads(line))
    else:
        outputs = []
        
    return outputs


def save_json(data, path, type="json", use_indent=True):

    assert type in ["json", "jsonl"] # only support json or jsonl format
    if type == "json":
        with open(path, "w", encoding="utf-8") as fout:
            if use_indent:
                fout.write(json.dumps(data, indent=4))
            else:
                fout.write(json.dumps(data))

    elif type == "jsonl":
        with open(path, "w", encoding="utf-8") as fout:
            for item in data:
                fout.write("{}\n".format(json.dumps(item)))

    return path



# Function to load HTML content from a file
def load_html_file(file_path):
    """Load HTML content from a file"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}, skipping.")
        return ""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    


# Function to convert HTML content to plain text
def html_to_plain_text(html_content):
    if not html_content:
        return ""
    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove unwanted tags (images, scripts, styles, etc.)
    for element in soup(['script', 'style', 'img', 'div', 'header', 'footer', 'nav']):
        element.decompose()
    
    # Get text content
    text = soup.get_text()
    
    # Clean up the text
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    # Remove multiple empty lines
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # Remove "html"
    text = re.sub(r'\bhtml\b', '', text, flags=re.IGNORECASE)
    
    return text




# Function to ensure the image is in PIL format
def ensure_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    # If it's a dict (decode=False)
    elif isinstance(img, dict):
        if img.get("bytes") is not None:
            img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        else:
            img = Image.open(img["path"]).convert("RGB")
    elif isinstance(img, (str, os.PathLike)):
        return Image.open(img).convert("RGB")
    elif isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")




def print_memory_usage(step_name="Current"):
    allocated = cuda.memory_allocated() / 1024**3
    reserved = cuda.memory_reserved() / 1024**3
    print(f"{step_name} Memory Usage: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")