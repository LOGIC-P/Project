# model.py
import tensorflow as tf
from tensorflow.keras import layers


class QueryEncoder(tf.keras.Model):
    def __init__(self, emb_var, k_dim=256):
        super().__init__()
        self.emb = emb_var
        self.t_emb = tf.eye(24, dtype=tf.float32)
        poi_dim = emb_var.shape[1]
        self.fc = layers.Dense(k_dim)

    def call(self, q):
        s_poi = tf.nn.embedding_lookup(self.emb, q[:, 0])
        e_poi = tf.nn.embedding_lookup(self.emb, q[:, 2])
        s_t = tf.nn.embedding_lookup(self.t_emb, q[:, 1])
        e_t = tf.nn.embedding_lookup(self.t_emb, q[:, 3])
        x = tf.concat([s_poi, s_t, e_poi, e_t], 1)
        return tf.nn.leaky_relu(self.fc(x))


class Decoder(tf.keras.Model):
    def __init__(self, emb_var, poi_size, local_mask, dec_units=256):
        super().__init__()
        self.emb = emb_var
        self.poi_size = poi_size

        if isinstance(local_mask, tf.Tensor):
                    self.local_mask = tf.reshape(local_mask, (1, -1))
        else:  # NumPy array
                    self.local_mask = tf.constant(local_mask.reshape(1, -1), dtype=tf.float32)

        self.gru = layers.GRUCell(dec_units)
        self.fc_seq = layers.Dense(poi_size)
        self.fc_dst = layers.Dense(poi_size)

    def call(self, x_id, qv, h):
        xe = tf.squeeze(tf.nn.embedding_lookup(self.emb, x_id), 1)
        out, h2 = self.gru(tf.concat([xe, qv], 1), h)
        seq_log = self.fc_seq(out) - 1e9 * (1 - self.local_mask)
        dst_log = self.fc_dst(out) - 1e9 * (1 - self.local_mask)
        return seq_log, dst_log, h2
