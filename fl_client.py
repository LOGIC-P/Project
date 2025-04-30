# fl_client.py
import os, argparse, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# 如果你想彻底屏蔽掉 server 端的 loss 输出，可以启用下面这一行：
import logging
logging.getLogger("flwr.server").setLevel(logging.WARNING)

import flwr as fl
import numpy as np
import tensorflow as tf
from input_data import CityData
from model import QueryEncoder, Decoder
from train import step_supervised
from contrastive_poi import POIContrastiveTrainer

LOCAL_EPOCHS = 3


class Client(fl.client.NumPyClient):
    def __init__(self, city, cid):
        self.city = city
        self.cid = cid
        self.data = CityData(city)

        # <<< self-supervised POI >>
        print(f"[{city}] contrastive pretrain ...", flush=True)
        trainer = POIContrastiveTrainer(self.data.emb_var)
        trainer.train_one_epoch(
            self.data.gen_contrastive_sequences(), batch=512, neg_k=64
        )

        self.query_enc = QueryEncoder(self.data.emb_var, 256)
        self.decoder = Decoder(
            self.data.emb_var,
            self.data.emb_var.shape[0],
            self.data.local_mask,
            256,
        )
        self.h_init = tf.keras.layers.Dense(256, activation="relu")
        self.opt = tf.keras.optimizers.Adam(0.001)
        self.train_ds, self.val_ds = self.data.load_supervised(32)

    # FL API ---------------------------------------------------
    def _all_vars(self):
        return (
            self.query_enc.trainable_variables
            + self.decoder.trainable_variables
            + self.h_init.trainable_variables
            + [self.data.emb_var]
        )

    def get_parameters(self, _):
        return [v.numpy() for v in self._all_vars()]

    def set_parameters(self, params):
        for v, p in zip(self._all_vars(), params):
            v.assign(p)

    # ----------------------------------------------------------
    def fit(self, params, _):
        if params:
            self.set_parameters(params)
        for _ in range(LOCAL_EPOCHS):
            for Q, T in self.train_ds:
                step_supervised(self.query_enc, self.decoder, self.h_init, Q, T, self.opt)
        return self.get_parameters(None), len(self.train_ds), {}

    def evaluate(self, params, _):
        self.set_parameters(params)
        f1s, pf1s = [], []
        for Q, T in self.val_ds:
            qv = self.query_enc(Q)
            for i in range(Q.shape[0]):
                real = [x for x in T[i].numpy().tolist() if x != 0]
                if len(real) < 2:
                    continue
                hidden = self.h_init(qv[i : i + 1])
                x = tf.constant([[real[0]]], tf.int32)
                pred, mask = [real[0]], self.data.local_mask.copy()
                mask[pred[0]] = mask[real[-1]] = 0
                for _ in range(len(real) - 2):
                    seq_log, _, hidden = self.decoder(x, qv[i : i + 1], hidden)
                    nxt = int(tf.argmax(seq_log * mask, 1))
                    pred.append(nxt)
                    mask[nxt] = 0
                    x = tf.constant([[nxt]], tf.int32)
                pred.append(real[-1])
                inter = len(set(real) & set(pred))
                f1s.append(2 * inter / (len(real) + len(pred)))
                order = {v: k for k, v in enumerate(real)}
                nc = sum(
                    1
                    for a in range(len(pred))
                    for b in range(a + 1, len(pred))
                    if pred[a] in order and pred[b] in order and order[pred[a]] < order[pred[b]]
                )
                n0 = len(real) * (len(real) - 1) / 2
                n0r = len(pred) * (len(pred) - 1) / 2
                pf1s.append(0 if n0 == 0 or n0r == 0 else 2 * nc / (n0 + n0r))
        # 取平均 F1 和 Pairs-F1
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        mean_pf1 = float(np.mean(pf1s)) if pf1s else 0.0

        # 这里把 loss 位置改成返回 mean_f1，这样 server 日志里的 “loss” 列就是真实的 F1，不再是 0
        return mean_f1, len(f1s), {"f1": mean_f1, "pairs_f1": mean_pf1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--server", default="localhost:8080")
    args = parser.parse_args()

    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)

    fl.client.start_numpy_client(server_address=args.server, client=Client(args.city, args.cid))
