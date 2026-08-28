from typing import List, Dict
import os
import ast


from utils.utils import load_json, load_html_file, html_to_plain_text



# Prompt for input with images and a query
def get_prompt(img_root: str, img_folder: str, img_ids: List[str], query: str) -> List[Dict]:
    # Multi-image input
    content = []
    if isinstance(img_ids, str):
        try:
            img_ids = ast.literal_eval(img_ids)
        except:
            img_ids = [img_ids]

    for img_id in img_ids:
        img_path = os.path.join(img_root, img_folder, f"{img_id}.png")
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}, skipping.")
            continue
        content.append({"type": "image", "image": img_path})
    
    # Prompt text
    content.append({
        "type": "text",
        "text": f"""Your task is to answer the question based on the provided images. Please directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.
                    Question: {query}"""
    })
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    return messages





# Prompt for input with images and parsed text from the images
def get_prompt_with_parsed_text(img_root: str, img_folder: str, img_ids: List[str], query: str) -> List[Dict]:
    # Multi-image input with parsed text
    content = []
    sub_text = []
    if isinstance(img_ids, str):
        try:
            img_ids = ast.literal_eval(img_ids)
        except:
            img_ids = [img_ids]

    for img_id in img_ids:
        # Image path
        img_path = os.path.join(img_root, img_folder, f"{img_id}.png")
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}, skipping.")
            continue
        
        # Parsed text (.html format)
        parsed_text_path = os.path.join(img_root, img_folder, f"{img_id}_parser.html")
        parsed_text = load_html_file(parsed_text_path)
        parsed_text = html_to_plain_text(parsed_text)
        sub_text.append(parsed_text)

        content.append({"type": "image", "image": img_path})
        
    # Concatenate parsed text
    if sub_text:
        parsed_text_combined = "\n".join(sub_text)


    # Prompt text
    content.append({
        "type": "text",
        "text": f"""Your task is to answer the question based on the provided images and their parsed text. Please directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.\n
                    Parsed Text: {parsed_text_combined}\n
                    Question: {query}"""
    })
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    return messages




# Prompt for input with extracted sub-images and parsed text from the images
def get_prompt_with_sub_images_and_parsed_text(img_root: str, img_folder: str, img_ids: List[str], query: str, max_num_imgs: int=10) -> List[Dict]:
    # Multi-image input with sub-images
    content = []
    sub_text = []
    if isinstance(img_ids, str):
        try:
            img_ids = ast.literal_eval(img_ids)
        except:
            img_ids = [img_ids]

    num_imgs = 0

    for img_id in img_ids:
        # Determine how many sub-images are available according to to the boxes
        boxes_path = os.path.join(img_root, img_folder, f"{img_id}_subimg_boxes.json")
        if not os.path.exists(boxes_path):
            print(f"Boxes file not found: {boxes_path}, skipping.")
            continue
        
        boxes = load_json(os.path.join(img_root, img_folder, f"{img_id}_subimg_boxes.json"))
        if not boxes:
            print(f"No sub-image boxes found in {boxes_path}, skipping.")
            continue
        
        # Sub-image path
        for i in range(len(boxes)):
            sub_img_path = os.path.join(img_root, img_folder, f"{img_id}_subimg_{i}.png")
            if not os.path.exists(sub_img_path):
                print(f"Sub-image not found: {sub_img_path}, skipping.")
                continue
            content.append({"type": "image", "image": sub_img_path})
            num_imgs += 1
            if num_imgs >= max_num_imgs:
                break
        
        if num_imgs >= max_num_imgs:
                break
        
        # Parsed text (.html format)
        parsed_text_path = os.path.join(img_root, img_folder, f"{img_id}_parser.html")
        parsed_text = load_html_file(parsed_text_path)
        parsed_text = html_to_plain_text(parsed_text)
        if parsed_text:
            sub_text.append(parsed_text)

    # Concatenate parsed text
    if sub_text:
        parsed_text_combined = "\n".join(sub_text)
    else:
        parsed_text_combined = "No parsed text available for the provided sub-images."
    
    # Prompt text
    content.append({
        "type": "text",
        "text": f"""Your task is to answer the question based on the provided sub-images and their parsed text. Please directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.\n
                    Parsed Text: {parsed_text_combined}\n
                    Question: {query}
    """
    })
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    return messages