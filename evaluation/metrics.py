from jiwer import wer
import editdistance

def compute_wer(ref, hyp):
    return wer(ref, hyp)

def cer(ref, hyp):
    return editdistance.eval(ref, hyp) / max(1, len(ref))
