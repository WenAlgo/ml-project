from PIL import Image
import numpy as np

def sort_boxes(boxes, threshold=10):
    boxes = [list(map(int,b)) for b in boxes]
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    rows = []
    for b in boxes:
        if not rows:
            rows.append([b])
        else:
            last = rows[-1]
            ys = [r[1] for r in last]
            if abs(b[1] - np.mean(ys)) <= threshold:
                last.append(b)
            else:
                rows.append([b])
    sorted_boxes = []
    for row in rows:
        row_sorted = sorted(row, key=lambda r: r[0])
        sorted_boxes.extend(row_sorted)
    return sorted_boxes

def crop_from_boxes(image_pil, boxes, padding=4):
    crops = []
    w,h = image_pil.size
    for (x1,y1,x2,y2) in boxes:
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)
        crop = image_pil.crop((x1,y1,x2,y2))
        crops.append(crop)
    return crops
