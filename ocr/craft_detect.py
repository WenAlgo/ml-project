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
            print("CRAFT error:", e)
            return []

        polys = result.get('boxes', [])
        if polys is None:
            polys = []

        clean_boxes = []
        for poly in polys:
            try:
                arr = np.array(poly)
                # Filter out broken polygons
                if arr.ndim != 2 or arr.shape[0] < 2:
                    continue
                xs = arr[:, 0]
                ys = arr[:, 1]
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                # Skip tiny boxes as they're mostly noise
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    continue
                clean_boxes.append([x1, y1, x2, y2])
            except Exception as e:
                print("poly parse error:", e)
                continue

        return clean_boxes

    finally:
        try:
            tmp.close()
            os.unlink(tmp.name)
        except:
            pass
