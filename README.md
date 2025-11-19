# OCR + Spelling Correction

This project implements:

-   CRAFT text detection (using craft-text-detector)
-   TrOCR recognition (HuggingFace transformers)
-   Edit-distance candidate generation + BERT-MLM scoring for correction
-   WER / CER evaluation (jiwer + editdistance)
-   Flask API for image upload and returning raw & corrected text
