# data_augmentation.py
import numpy as np

class SeqAugmentor:
    @staticmethod
    def mask(seq, rate=0.3, mask_id=0):
        seq = seq.copy()
        if len(seq) <= 2:
            return seq
        k = max(1, int((len(seq) - 2) * rate))
        idx = np.random.choice(range(1, len(seq) - 1), k, replace=False)
        for i in idx:
            seq[i] = mask_id
        return seq

    @staticmethod
    def shuffle(seq):
        if len(seq) <= 3:
            return seq.copy()
        body = seq[1:-1]
        np.random.shuffle(body)
        return [seq[0], *body, seq[-1]]

    @staticmethod
    def cutoff(seq, rate=0.3):
        if len(seq) <= 3:
            return seq.copy()
        k = max(1, int((len(seq) - 2) * rate))
        keep = sorted(np.random.choice(len(seq) - 2, len(seq) - 2 - k, replace=False))
        body = [seq[1:-1][i] for i in keep]
        return [seq[0], *body, seq[-1]]
