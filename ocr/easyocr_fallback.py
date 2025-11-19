# EasyOCR fallback: returns list of boxes and texts
import easyocr
from PIL import Image
reader = easyocr.Reader(['en'], gpu=False)

def easyocr_detect_and_recognize(pil_img):
    arr = pil_img.convert('RGB')
    arr = __import__('numpy').array(arr)
    results = reader.readtext(arr)
    # results: list of (bbox, text, confidence)
    boxes = []
    texts = []
    for bbox, text, conf in results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        boxes.append([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))])
        texts.append(text)
    return boxes, texts
