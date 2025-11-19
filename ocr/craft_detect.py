# Robust CRAFT wrapper that accepts PIL.Image and filters broken polygons
from craft_text_detector import Craft
from PIL import Image
import numpy as np
import os, tempfile
craft = Craft(output_dir=None, crop_type='box', cuda=False)

def detect_boxes_from_pil(image_pil):
    # save to temp file because craft expects a path
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    try:
        image_pil.save(tmp.name)
        try:
            result = craft.detect_text(tmp.name)
        except Exception as e:
            # Catch CRAFT internals failing (avoid crashing Flask)
            return []
        polys = result.get('boxes', []) or []
        clean_boxes = []
        for poly in polys:
            try:
                arr = np.array(poly)
                if arr.ndim != 2 or arr.shape[0] < 2:
                    continue
                xs = arr[:,0]
                ys = arr[:,1]
                x1,x2 = int(xs.min()), int(xs.max())
                y1,y2 = int(ys.min()), int(ys.max())
                # skip tiny boxes
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    continue
                clean_boxes.append([x1,y1,x2,y2])
            except Exception:
                continue
        return clean_boxes
    finally:
        try:
            tmp.close()
            os.unlink(tmp.name)
        except:
            pass

if __name__ == '__main__':
    img = Image.open('example.jpg')
    print(detect_boxes_from_pil(img))
