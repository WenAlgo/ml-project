from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from config import EDIT_DISTANCE_THRESHOLD, VOCAB_MIN_WORD_LEN

class BertMLMCorrector:
    def __init__(self, model_name='bert-base-uncased', device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)

    def score_candidate(self, sentence_tokens, target_index, candidate):
        tokens = sentence_tokens.copy()
        mask_tok = self.tokenizer.mask_token
        tokens[target_index] = mask_tok
        text = ' '.join(tokens)
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        mask_indices = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        if mask_indices[0].numel() == 0:
            return float('-inf')
        mask_pos = mask_indices[1].item()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        cand_tokens = self.tokenizer.tokenize(candidate)
        if not cand_tokens:
            return float('-inf')
        # heuristic: use first subtoken logits
        cand_id = self.tokenizer.convert_tokens_to_ids(cand_tokens[0])
        score = logits[0, mask_pos, cand_id].item()
        score = score - 0.01 * len(cand_tokens)
        return score

    def correct_sentence(self, sentence_tokens, candidate_generator, max_candidates=10):
        corrected = sentence_tokens.copy()
        for i, tok in enumerate(sentence_tokens):
            if not tok.isalpha() or len(tok) < VOCAB_MIN_WORD_LEN:
                continue
            cands = candidate_generator.candidates(tok, max_cand=max_candidates)
            if cands and cands[0].lower() == tok.lower():
                continue
            import editdistance
            if cands:
                best_dist = editdistance.eval(tok.lower(), cands[0].lower())
                if best_dist <= EDIT_DISTANCE_THRESHOLD:
                    continue
            best = None
            best_score = float('-inf')
            for cand in cands:
                s = self.score_candidate(sentence_tokens, i, cand)
                if s > best_score:
                    best_score = s
                    best = cand
            if best and best_score > -1e9:
                if best.isalpha():
                    corrected[i] = best
        return corrected
