from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
from config import TROCR_MODEL, OCR_DEVICE

class TrocrRecognizer:
    def __init__(self, model_name=TROCR_MODEL, device=OCR_DEVICE):
        self.device = device
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.gen_kwargs = dict(max_length=256, num_beams=4)

    def recognize(self, image_pil):
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        pixel_values = self.processor(images=image_pil, return_tensors='pt').pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values, **self.gen_kwargs)
        preds = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return preds[0]

