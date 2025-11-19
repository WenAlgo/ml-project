from PIL import Image
import numpy as np
import cv2

def enhance_for_detection(pil_img):
    # Convert to RGB then to OpenCV format
    img = pil_img.convert('RGB')
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # Adaptive threshold to increase contrast for text on colored backgrounds
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 8)
    # Use slight median blur to remove speckle noise
    th = cv2.medianBlur(th, 3)
    return Image.fromarray(th)
