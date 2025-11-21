from datasets import load_dataset

# https://huggingface.co/datasets/staghado/Bentham
def load_bentham(split='train'):
    ds = load_dataset('staghado/Bentham')
    return ds[split]
    
if __name__ == '__main__':
    d = load_bentham('train')
    print(next(iter(d)))
