# Reference: https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/document_parsing.ipynb

import os
from PIL import Image, ImageDraw, ImageFont
from bs4 import BeautifulSoup, Tag
from pathlib import Path
import numpy as np
from typing import List, Dict, Union
import re
import argparse
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import json
import random


from dataset.load_data import load_img_path
from utils.utils import ensure_pil


# Define parameters
def parse_args():
    parser = argparse.ArgumentParser(description="Image Parser using QwenVL")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-72B-Instruct", help="Path to the pre-trained model")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum number of new tokens to generate")
    parser.add_argument("--data_name", type=str, default="MMLongBench", help="Name of the dataset to load images from")
    parser.add_argument("--idx", type=int, default=0, help="Index of the first image to process")
    parser.add_argument("--num_imgs", type=int, default=5000, help="Number of images to process")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing images")
    args = parser.parse_args()
    return args




""" Class for parsing images and generating HTML with bounding boxes """
class ImageParser:
    def __init__(
            self,
            model_path="Qwen/Qwen2.5-VL-72B-Instruct",
            max_new_tokens=1024
    ):
        self.max_new_tokens = max_new_tokens
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map={"": f"cuda:0"},
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True, trust_remote_code=True, padding_side="left")



    # Function to perform inference on the image
    def inference(self, imgs, prompt=None, system_prompt=None, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        # Get input for the model
        inputs = self._get_input(imgs, prompt, system_prompt)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        input_heights = inputs['image_grid_thw'][:,1]*14
        input_widths = inputs['image_grid_thw'][:,2]*14

        return output_text, input_heights, input_widths



    # Function to get input for the model
    def _get_input(self, imgs: List[str], prompt=None, system_prompt=None):
        # Define default system prompt and prompt if not provided
        if system_prompt is None:
            system_prompt="You are an AI specialized in recognizing and extracting text from images. Your mission is to analyze the image document and generate the result in QwenVL Document Parser HTML format using specified tags while maintaining user privacy and data integrity."
        if prompt is None:
            prompt =  "QwenVL HTML "
        
        # We use batch inference
        if not isinstance(imgs, list):
            imgs = [imgs]

        texts = []
        PIL_imgs = []

        for img in imgs:
            img = ensure_pil(img)
            PIL_imgs.append(img)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"}
                ]}
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

        inputs = self.processor(text=texts, images=PIL_imgs, padding=True, return_tensors="pt").to('cuda')

        return inputs



    # Function to draw bounding boxes and text on images based on HTML content
    def _draw_bbox(self, image_path, resized_width, resized_height, full_predict, color="blue", draw_text=False, draw_box=True):
        image = ensure_pil(image_path)
        original_width = image.width
        original_height = image.height
        
        # Parse the provided HTML content
        soup = BeautifulSoup(full_predict, 'html.parser')
        # Extract all elements that have a 'data-bbox' attribute
        elements_with_bbox = soup.find_all(attrs={'data-bbox': True})

        filtered_elements = []
        for el in elements_with_bbox:
            if el.name == 'ol':
                continue  # Skip <ol> tags
            elif el.name == 'li' and el.parent.name == 'ol':
                filtered_elements.append(el)  # Include <li> tags within <ol>
            else:
                filtered_elements.append(el)  # Include all other elements

        draw = ImageDraw.Draw(image)
        img_boxes = []
        
        # Draw bounding boxes and text for each element
        for element in filtered_elements:
            bbox_str = element['data-bbox']
            text = element.get_text(strip=True)
            # Replace commas with spaces, then split into numbers
            bbox_numbers = bbox_str.replace(',', ' ').split()
            try:
                x1, y1, x2, y2 = map(int, bbox_numbers)
            except:
                continue
            
            # Calculate scaling factors
            scale_x = resized_width / original_width
            scale_y = resized_height / original_height
            
            # Scale coordinates accordingly
            x1_resized = int(x1 / scale_x)
            y1_resized = int(y1 / scale_y)
            x2_resized = int(x2 / scale_x)
            y2_resized = int(y2 / scale_y)
            
            if x1_resized > x2_resized:
                x1_resized, x2_resized = x2_resized, x1_resized
            if y1_resized > y2_resized:
                y1_resized, y2_resized = y2_resized, y1_resized
            
            # Check if this is an image element (sub-image)
            # is_image = (
            #     element.name == "img"  # <img> tag
            #     or (element.name == "div" and "image" in element.get("class", []))  # <div class="image">
            # )
            is_image = (
                element.name == "img"  # <img> tag
                #or (element.name == "div" and "image" in element.get("class", []))  # <div class="image">
            ) 
            if is_image:
                # If it's an image, draw a rectangle with a different color
                color = "red"
                img_boxes.append([x1_resized, y1_resized, x2_resized, y2_resized])
            
            if draw_box:
                # Draw bounding box with the specified color
                draw.rectangle([x1_resized, y1_resized, x2_resized, y2_resized], outline=color, width=2)

            if draw_text:
                font = ImageFont.truetype("NotoSansCJK-Regular.ttc", 20)
                # Draw associated text
                draw.text((x1_resized, y2_resized), text, fill='black', font=font)

        return image, img_boxes



    # Function to clean and format HTML content
    def _clean_and_format_html(self, full_predict):
        soup = BeautifulSoup(full_predict, 'html.parser')
        
        # Regular expression pattern to match 'color' styles in style attributes
        color_pattern = re.compile(r'\bcolor:[^;]+;?')

        # Find all tags with style attributes and remove 'color' styles
        for tag in soup.find_all(style=True):
            original_style = tag.get('style', '')
            new_style = color_pattern.sub('', original_style)
            if not new_style.strip():
                del tag['style']
            else:
                new_style = new_style.rstrip(';')
                tag['style'] = new_style
                
        # Remove 'data-bbox' and 'data-polygon' attributes from all tags
        for attr in ["data-bbox", "data-polygon"]:
            for tag in soup.find_all(attrs={attr: True}):
                del tag[attr]

        classes_to_update = ['formula.machine_printed', 'formula.handwritten']
        # Update specific class names in div tags
        for tag in soup.find_all(class_=True):
            if isinstance(tag, Tag) and 'class' in tag.attrs:
                new_classes = [cls if cls not in classes_to_update else 'formula' for cls in tag.get('class', [])]
                tag['class'] = list(dict.fromkeys(new_classes))  # Deduplicate and update class names

        # Clear contents of divs with specific class names and rename their classes
        for div in soup.find_all('div', class_='image caption'):
            div.clear()
            div['class'] = ['image']

        classes_to_clean = ['music sheet', 'chemical formula', 'chart']
        # Clear contents and remove 'format' attributes of tags with specific class names
        for class_name in classes_to_clean:
            for tag in soup.find_all(class_=class_name):
                if isinstance(tag, Tag):
                    tag.clear()
                    if 'format' in tag.attrs:
                        del tag['format']

        # Manually build the output string
        output = []
        if soup.body:  # Check if body exists
            for child in soup.body.children:
                if isinstance(child, Tag):
                    output.append(str(child))
                    output.append('\n')  # Add newline after each top-level element
                elif isinstance(child, str) and not child.strip():
                    continue  # Ignore whitespace text nodes
        else:  # If no body tag exists, use the whole soup
            for child in soup.children:
                if isinstance(child, Tag):
                    output.append(str(child))
                    output.append('\n')
                elif isinstance(child, str) and not child.strip():
                    continue
        complete_html = f"""```html\n<html><body>\n{" ".join(output)}</body></html>\n```"""
        return complete_html






def main():
    args = parse_args()

    # Load image paths based on the specified dataset
    img_paths = load_img_path(data_name=args.data_name, start_idx=args.idx, num_imgs=args.num_imgs)
    # img_paths.sort()
    print(f"Total number of images: {len(img_paths)}")

    # Define class
    parser = ImageParser(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens
    )
    
    # Define prompt
    prompt = "QwenVL HTML "
    system_prompt = "You are an AI specialized in recognizing and extracting text from images. Your mission is to analyze the image document and generate the result in QwenVL Document Parser HTML format using specified tags while maintaining user privacy and data integrity."
    
    # idx_max = min(((args.idx + args.num_imgs) // args.num_imgs) * args.num_imgs, len(img_paths))
    # img_paths = img_paths[args.idx:idx_max]


    for i in tqdm(range(0, len(img_paths), args.batch_size), desc="Processing images"):
        batch_paths = img_paths[i:i+args.batch_size]

        # Get the output from the model
        outputs, input_heights, input_widths = parser.inference(batch_paths, prompt, system_prompt)

        for ii, (output, img_path, input_height, input_width) in enumerate(zip(outputs, batch_paths, input_heights, input_widths)):
            # Define root and image idx
            if isinstance(img_path, (str, os.PathLike)):
                root = Path(img_path).parent
                img_idx = Path(img_path).stem
            else:
                root = f"../data/{args.data_name}_data/images"
                img_idx = i + args.idx + ii + 1
            
            """ We only draw boxes for a few images for demo purposes. """
            draw_box = True if random.random() < 0.0 else False
            # Draw bounding boxes on the image
            image, img_boxes = parser._draw_bbox(img_path, input_width, input_height, output, draw_text=False, draw_box=draw_box)
            if draw_box:
                # Save the annotated image
                output_image_path = os.path.join(root, f"{img_idx}_box.png")
                image.save(output_image_path)

            # Save the bounding boxes to a json file
            output_json_path = os.path.join(root, f"{img_idx}_subimg_boxes.json")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(img_boxes, f, indent=4)
            
            # Clean and format the HTML output
            ordinary_html = parser._clean_and_format_html(output)
            # Save the cleaned HTML output
            output_html_path = os.path.join(root, f"{img_idx}_parser.html")
            with open(output_html_path, 'w', encoding='utf-8') as f:
                f.write(ordinary_html)



if __name__ == "__main__":
    main()