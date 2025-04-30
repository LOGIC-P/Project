# input_data.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_augmentation import SeqAugmentor
from global_config import create_trainable_embedding


def _normalize_geo(geo):
    geo = geo.astype(np.float32)
    np.fill_diagonal(geo, 1e-6)
    row_sum = geo.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return geo / row_sum


class CityData:
    def __init__(self, city):
        self.city = city
        self.emb_var, offsets = create_trainable_embedding()
        self.offset, self.local_n = offsets[city]
        self.local_mask = np.zeros(self.emb_var.shape[0], dtype=np.float32)
        self.local_mask[self.offset : self.offset + self.local_n] = 1.0

        # load csv
        qf = f"./train_data/{city}-query.csv"
        tf_path = f"./train_data/{city}-trajs.dat"
        self.queries = pd.read_csv(qf).values.tolist()
        with open(tf_path) as f:
            self.trajs = [[int(x) for x in l.split()] for l in f]

        geo_raw = pd.read_csv(f"./dis_matric/{city}_dis_matric.csv").values
        self.geo_prob = _normalize_geo(geo_raw)

    # ---------- supervised ----------
    def _map_q(self, q):
        return np.array(
            [q[0] + self.offset, q[1], q[2] + self.offset, q[3]], dtype=np.int32
        )

    def _map_t(self, t):
        return [p + self.offset for p in t]

    def load_supervised(self, batch=32):
        queries = []
        trajs = []

        for traj in self.trajs:
            if len(traj) < 3:
                continue
            start_poi, end_poi = traj[0], traj[-1]
            n_poi = len(traj)
            start_time = np.random.randint(0, 24)
            end_time = (start_time + np.random.randint(1, 5)) % 24
            queries.append([start_poi + self.offset, start_time, end_poi + self.offset, end_time])
            trajs.append([p + self.offset for p in traj])

            # 加 trip 增强
            for _ in range(2):
                aug1 = SeqAugmentor.mask(traj)
                aug2 = SeqAugmentor.shuffle(traj)
                queries.append([start_poi + self.offset, start_time, end_poi + self.offset, end_time])
                trajs.append([p + self.offset for p in aug1])
                queries.append([start_poi + self.offset, start_time, end_poi + self.offset, end_time])
                trajs.append([p + self.offset for p in aug2])

        Q = np.array(queries, dtype=np.int32)
        T = tf.keras.preprocessing.sequence.pad_sequences(trajs, padding="post")
        q_tr, q_val, t_tr, t_val = train_test_split(Q, T, test_size=0.2, random_state=42)

        tr = tf.data.Dataset.from_tensor_slices((q_tr, t_tr)).shuffle(len(q_tr)).batch(batch).prefetch(tf.data.AUTOTUNE)
        va = tf.data.Dataset.from_tensor_slices((q_val, t_val)).batch(batch)
        return tr, va


    def gen_contrastive_sequences(self):
        from contrastive_poi import build_aug_graph, gen_sequences

        mat = build_aug_graph(
            self.trajs,
            list(range(self.local_n)),
            self.geo_prob,
        )
        seqs = gen_sequences(mat, walk_num=5, walk_len=6)
        return [[x + self.offset for x in s] for s in seqs]

    # ---------- trip-level aug ----------
    def aug_two_views(self, traj):
        a, b = np.random.choice(
            [SeqAugmentor.mask, SeqAugmentor.shuffle, SeqAugmentor.cutoff], 2
        )
        return a(traj), b(traj)
