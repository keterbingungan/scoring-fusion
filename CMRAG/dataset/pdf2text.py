import pdfplumber
from tqdm import tqdm
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np



path = "../multimodal_RAG/MMLongBench-Doc/documents/2210.02442v1.pdf"
path = "../multimodal_RAG/MMLongBench-Doc/documents/0b85477387a9d0cc33fca0f4becaa0e5.pdf"

# with pdfplumber.open(path) as pdf:
#     text = ""
#     for page in tqdm(pdf.pages[:1], desc="Extracting text from pages"):
#         text += page.extract_text()
# print(text)



# Convert PDF to images
images = convert_from_path(path, dpi=400)

text = ""
for img in images[:1]:  # Only process first page for demo
    # Convert PIL Image to numpy array for OpenCV
    img_np = np.array(img)
    
    # Convert RGB to BGR (OpenCV uses BGR by default)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Now proceed with your processing
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # # Optional: Show the processed image (for debugging)
    # cv2.imshow('Processed', thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    text += pytesseract.image_to_string(thresh, config='--psm 6')

print(text)