import io, os, traceback
from flask import Flask, request, jsonify
from PIL import Image
from ocr.craft_detect import detect_boxes_from_pil
from ocr.trocr_recognize import TrocrRecognizer
from ocr.easyocr_fallback import easyocr_detect_and_recognize
from utils.sort_and_crop import sort_boxes, crop_from_boxes
from utils.image_preprocess import enhance_for_detection
from corrector.edit_distance import SimpleCandidateGenerator
from corrector.bert_mlm_corrector import BertMLMCorrector
from evaluation.metrics import compute_wer, cer
from config import OCR_DEVICE, CROP_PADDING
from wordfreq import top_n_list

app = Flask(__name__)

# OCR models
recognizer = TrocrRecognizer()
mlm = BertMLMCorrector(device=OCR_DEVICE)

# The vocabulary list uses the top 50,000 most frequent English words from wordfreq
vocab = top_n_list("en", 50000)

# Candidate generator
candidate_gen = SimpleCandidateGenerator(vocab)

@app.route('/api/ocr', methods=['POST'])
def api_ocr():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'no image uploaded'}), 400

        file = request.files['image']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')

        # Enhance image for detection
        img_for_detect = enhance_for_detection(img)

        # Run CRAFT text detection
        boxes = detect_boxes_from_pil(img_for_detect)
        raw_texts = []

        # If CRAFT found something
        if boxes:
            boxes_sorted = sort_boxes(boxes)
            crops = crop_from_boxes(img, boxes_sorted, padding=CROP_PADDING)

            for crop in crops:
                try:
                    raw = recognizer.recognize(crop)
                except Exception:
                    raw = ''
                raw_texts.append(raw)

        # Fallback to EasyOCR if CRAFT fails
        if not boxes or all(not t for t in raw_texts):
            try:
                fb_boxes, fb_texts = easyocr_detect_and_recognize(img)
                if fb_texts:
                    raw_texts = fb_texts
            except Exception:
                pass

        # Merge OCR results
        raw_sentence = ' '.join([t for t in raw_texts if t]).strip()

        # Correction
        tokens = raw_sentence.split()
        corrected_tokens = mlm.correct_sentence(tokens, candidate_gen) if tokens else []
        corrected_sentence = ' '.join(corrected_tokens).strip()

        # Metrics 
        ground = request.form.get('ground_truth', None)
        metrics = {}

        if ground:
            metrics['wer_before'] = compute_wer(ground, raw_sentence)
            metrics['wer_after'] = compute_wer(ground, corrected_sentence)
            metrics['cer_before'] = cer(ground, raw_sentence)
            metrics['cer_after'] = cer(ground, corrected_sentence)

        return jsonify({
            'raw': raw_sentence,
            'corrected': corrected_sentence,
            'metrics': metrics
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'internal error', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
