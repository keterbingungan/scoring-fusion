from pdf2image import convert_from_path
import os
from pathlib import Path
from tqdm import tqdm
import argparse




def parse_args():
    parser = argparse.ArgumentParser(description="PDF to Image Converter")
    parser.add_argument("--pdf_root", type=str, required=True, help="Path to the PDF root directory")
    parser.add_argument("--dpi", type=int, default=400, help="DPI for image conversion")
    return parser.parse_args()




def PDF2IMG(path, dpi=400):
    """
    Convert a PDF file to images.

    Args:
        path (str): Path to the PDF file.

    Returns:
        list: List of images converted from the PDF pages.
    """

    # Convert PDF to images
    images = convert_from_path(path, dpi=dpi)
    
    return images




if __name__ == "__main__":
    args = parse_args()
    root = args.pdf_root
    pdf_paths = os.listdir(root)
    pdf_paths.sort()

    # image path
    img_root = Path(root).parent / "images"
    os.makedirs(img_root, exist_ok=True)

    for pdf_path in tqdm(pdf_paths, desc="Converting PDFs to images"):
        pdf_path = os.path.join(root, pdf_path)
        if not pdf_path.endswith('.pdf'):
            continue
        images = PDF2IMG(pdf_path, dpi=args.dpi)
        
        # Save all images into a folder
        folder_name = Path(pdf_path).stem
        img_path = img_root / folder_name
        os.makedirs(img_path, exist_ok=True)
        
        # Save images
        for i, img in enumerate(images):
            img.save(os.path.join(img_path, f"{i+1}.png"), "PNG")
    
    print(f"Converted {pdf_path} to {len(images)} images in {img_path}")