#!/usr/bin/python3
# coding=utf-8

import flwr as fl
import numpy as np
import tensorflow as tf
import time
from input_data import Load_data
from model import QueryModel, Decoder, Hidden_init
from metric import calc_F1, calc_pairsF1

# 启用 GPU 内存增长，确保能够按需分配内存
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print("Error setting GPU memory growth:", e)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, city="Osak", batch_size=32, pretrain_batch_size=32, k=256, dec_units=256,
                 pretrain_epochs=5, train_epochs=5):
        self.city = city
        self.batch_size = batch_size
        self.pretrain_batch_size = pretrain_batch_size
        self.k = k
        self.dec_units = dec_units
        self.pretrain_epochs = pretrain_epochs
        self.train_epochs = train_epochs

        tf.keras.backend.set_floatx('float64')

        self.data = Load_data(city)
        self.poi_embedding, self.poi_size = self.data.self_embedding()

        self.query = QueryModel(self.poi_embedding, k)
        self.decoder = Decoder(self.poi_embedding, self.poi_size, dec_units)
        self.h_state = Hidden_init(dec_units)

        self.train_dataset, self.train_steps = self.data.load_dataset_train(self.batch_size)
        self.test_dataset, self.test_steps = self.data.load_dataset_test(self.batch_size)

        self.pre_train()

    def pre_loss_function(self, sim):
        real = np.eye(self.pretrain_batch_size)
        loss = tf.keras.losses.categorical_crossentropy(real, sim)
        return tf.reduce_mean(loss)

    def pre_train_step(self, pre_que, sample1, sample2, lr=0.0005):
        pre_loss = 0
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        with tf.GradientTape() as tape:
            query_out = self.query(pre_que)
            dec_input1 = sample1[:, 0]
            dec_input2 = sample2[:, 0]
            dec_hidden1 = self.h_state(query_out)
            dec_hidden2 = self.h_state(query_out)
            for t in range(1, sample1.shape[1]):
                output1, dec_hidden1 = self.decoder.pre_train(dec_input1, query_out, dec_hidden1)
                output2, dec_hidden2 = self.decoder.pre_train(dec_input2, query_out, dec_hidden2)
                dec_input1 = sample1[:, t]
                dec_input2 = sample2[:, t]
            sim = tf.matmul(output1, tf.transpose(output2))
            sim = tf.math.softmax(sim)
            pre_loss += self.pre_loss_function(sim)
        variables = self.decoder.trainable_variables + self.query.trainable_variables + self.h_state.trainable_variables
        gradients = tape.gradient(pre_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return pre_loss

    def pre_train(self):
        print("=== Start the pre-training phase ===")
        for epoch in range(self.pretrain_epochs):
            start_time = time.time()
            pre_loss_total = 0
            steps = 0
            for batch, (que, traj) in enumerate(self.train_dataset.take(self.train_steps)):
                pre_que, sample1, sample2 = self.data.load_pretrain_dataset(que, traj)
                batch_loss = self.pre_train_step(pre_que, sample1, sample2)
                pre_loss_total += batch_loss
                steps += 1
            avg_loss = pre_loss_total / steps
            print(f"Pre-training Epoch {epoch+1}: Loss = {avg_loss:.4f}, 用时 {time.time()-start_time:.2f} 秒")
        print("=== Pre-training completed ===")

    def loss_function(self, real, pred, real2, pred2):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        loss2 = loss_object(real2, pred2)
        return tf.reduce_mean(loss_) + tf.reduce_mean(loss2)

    def train_step(self, que, traj, lr=0.1):
        loss = 0
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        with tf.GradientTape() as tape:
            query_out = self.query(que)
            dec_input = traj[:, 0]
            dec_hidden = self.h_state(query_out)
            for t in range(1, traj.shape[1]):
                predictions, predictions2, dec_hidden = self.decoder(dec_input, query_out, dec_hidden)
                loss += self.loss_function(traj[:, t], predictions, que[:, 2], predictions2)
                dec_input = tf.argmax(tf.nn.softmax(predictions), axis=1)
        variables = self.query.trainable_variables + self.decoder.trainable_variables + self.h_state.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def local_train(self, epochs, lr=0.1):
        print("=== Start local training ===")
        for epoch in range(epochs):
            start_time = time.time()
            total_loss = 0
            steps = 0
            for batch, (que, traj) in enumerate(self.train_dataset.take(self.train_steps)):
                batch_loss = self.train_step(que, traj, lr)
                total_loss += batch_loss
                steps += 1
            avg_loss = total_loss / steps
            print(f"Local training Epoch {epoch+1}: Loss = {avg_loss:.4f}, 用时 {time.time()-start_time:.2f} 秒")
        print("=== Local training completed ===")
        return avg_loss

    def evaluate_model(self, dataset, steps):
        total_f1 = 0.0
        total_pairs_f1 = 0.0
        count = 0
        for batch, (que, traj) in enumerate(dataset.take(steps)):
            que1 = tf.expand_dims(que[0], 0)
            traj1 = traj[0]
            traj1_np = traj1.numpy()
            traj1_np = traj1_np[traj1_np != 0]
            traj1 = tf.expand_dims(tf.convert_to_tensor(traj1_np), 0)
            batch_f1, batch_ps_f1, _, _ = self.evaluate_sample(que1, traj1)
            total_f1 += batch_f1
            total_pairs_f1 += batch_ps_f1
            count += 1
        f1_avg = total_f1 / count if count > 0 else 0.0
        pairs_f1_avg = total_pairs_f1 / count if count > 0 else 0.0
        return f1_avg, pairs_f1_avg

    def evaluate_sample(self, que, traj):
        """
        对单个样本进行评估，返回 F1、pair F1 以及真实与预测的轨迹。
        """
        predict_traj = []
        realnum_poi = 0
        query_out = self.query(que)
        dec_input = traj[:, 0]
        for poi in tf.squeeze(traj):
            if poi == 0:
                break
            realnum_poi += 1
        realnum_poi = realnum_poi - 2 if realnum_poi >= 2 else 0
        start_poi = traj[:, 0]
        start_poi = tf.cast(start_poi, dtype=tf.int32)
        if realnum_poi + 1 < traj.shape[1]:
            end_poi = traj[:, realnum_poi + 1]
        else:
            end_poi = traj[:, -1]
        predict_traj.append(start_poi)
        dec_hidden = self.h_state(query_out)
        table = np.ones([self.poi_size], dtype=np.float64)
        table[start_poi.numpy()] = 0.
        table[end_poi.numpy()] = 0.
        table[0] = 0.
        for t in range(realnum_poi):
            self.decoder.set_dropout(0.0)  # 评估时关闭 dropout
            predictions, _, dec_hidden = self.decoder(dec_input, query_out, dec_hidden)
            mask = tf.expand_dims(table, axis=0)
            dec_input = tf.argmax(tf.nn.softmax(predictions * mask), axis=1)
            predict_traj.append(dec_input)
        predict_traj.append(end_poi)
        real_traj = tf.squeeze(traj)[0:realnum_poi + 1].numpy()
        real_traj = np.append(real_traj, end_poi)
        predict_traj = [i.numpy().tolist() for i in predict_traj]
        predict_traj = [i[0] for i in predict_traj]
        batch_f1 = calc_F1(real_traj, predict_traj)
        batch_pairs_f1 = calc_pairsF1(real_traj, predict_traj)
        return batch_f1, batch_pairs_f1, real_traj, predict_traj

    def get_parameters(self, config=None):
        query_weights = self.query.get_weights()
        decoder_weights = self.decoder.get_weights()
        hidden_weights = self.h_state.get_weights()
        return query_weights + decoder_weights + hidden_weights

    def set_parameters(self, parameters):
        n_query = len(self.query.get_weights())
        n_decoder = len(self.decoder.get_weights())
        n_hidden = len(self.h_state.get_weights())
        self.query.set_weights(parameters[0:n_query])
        self.decoder.set_weights(parameters[n_query:n_query+n_decoder])
        self.h_state.set_weights(parameters[n_query+n_decoder:])

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = config.get("local_epochs", self.train_epochs)
        lr = config.get("lr", 0.1)
        final_loss = self.local_train(epochs, lr)
        print(f"Client local training finished. Final average loss: {final_loss:.4f}")
 
        global_params = self.get_parameters()
        param_shapes = [w.shape for w in global_params]
        print(f"Client model parameter shapes: {param_shapes}")
        return self.get_parameters(), self.train_steps, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        f1_avg, pairs_f1_avg = self.evaluate_model(self.test_dataset, self.test_steps)
        print(f"Client evaluation: f1 = {f1_avg:.4f}, pair_f1 = {pairs_f1_avg:.4f}")
        global_params = self.get_parameters()
        param_norms = [np.linalg.norm(w) for w in global_params]
        print(f"Client model parameter norms: {param_norms}")
        return 0.0, self.test_steps, {"f1": f1_avg, "pair_f1": pairs_f1_avg}

if __name__ == "__main__":
    client = FlowerClient()
    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client()
    )
