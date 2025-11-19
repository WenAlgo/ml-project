import torch

OCR_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TROCR_MODEL = 'microsoft/trocr-base-handwritten'
BERT_MLM = 'bert-base-uncased'
CRAFT_OUTPUT_DIR = './ocr_craft_output'
CROP_PADDING = 4
MAX_CANDIDATES = 10
# Correction thresholds
EDIT_DISTANCE_THRESHOLD = 2   # only consider tokens with edit distance > this to any vocab word
VOCAB_MIN_WORD_LEN = 2
