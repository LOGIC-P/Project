# fl_server.py
import argparse
import flwr as fl
from global_config import create_trainable_embedding
from model import QueryEncoder, Decoder
import tensorflow as tf
from flwr.common import ndarrays_to_parameters


def weighted_avg(metrics):
    total = sum(n for n, _ in metrics)
    agg = {}
    for n, m in metrics:
        for k, v in m.items():
            agg[k] = agg.get(k, 0) + v * n / total
    return agg


def init_parameters():
    emb, _ = create_trainable_embedding()
    q = QueryEncoder(emb, 256)
    d = Decoder(emb, emb.shape[0], tf.ones(emb.shape[0]), 256)
    dense = tf.keras.layers.Dense(256)
    vars_ = q.trainable_variables + d.trainable_variables + dense.trainable_variables + [emb]
    return ndarrays_to_parameters([v.numpy() for v in vars_])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--port", type=str, default="0.0.0.0:8080")
    args = parser.parse_args()

    strategy = fl.server.strategy.FedAvg(
        initial_parameters=init_parameters(),
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        evaluate_metrics_aggregation_fn=weighted_avg,
    )

    fl.server.start_server(
        server_address=args.port,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
