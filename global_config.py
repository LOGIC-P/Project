import os
import numpy as np
import pandas as pd
import tensorflow as tf

_EMBED_DIR = "./self-embedding"


def load_global_poi_dictionary(embedding_dir=_EMBED_DIR):
    files = sorted(
        [f for f in os.listdir(embedding_dir) if f.endswith("_poi_weight.csv")]
    )
    offsets, tot = {}, 0
    for f in files:
        city = f.replace("_poi_weight.csv", "")
        n = pd.read_csv(os.path.join(embedding_dir, f)).shape[0]
        offsets[city] = (tot, n)
        tot += n
    return offsets, tot


def load_initial_embedding(embedding_dir=_EMBED_DIR):
    """Return numpy array (total, dim) & offsets dict."""
    offsets, total = load_global_poi_dictionary(embedding_dir)
    d0 = pd.read_csv(os.path.join(embedding_dir, next(iter(offsets)) + "_poi_weight.csv"))
    dim = d0.shape[1]
    arr = np.zeros((total, dim), dtype=np.float32)

    for city, (start, n) in offsets.items():
        f = os.path.join(embedding_dir, f"{city}_poi_weight.csv")
        arr[start : start + n] = pd.read_csv(f).values
    return arr, offsets


def create_trainable_embedding():
    init, offsets = load_initial_embedding()
    return tf.Variable(init, name="global_poi_emb", dtype=tf.float32), offsets
