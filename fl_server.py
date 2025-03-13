#!/usr/bin/python3
# coding=utf-8

import flwr as fl
import numpy as np
from flwr.server.strategy import FedAvg

class CustomFedAvgStrategy(FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        print(f"Round {rnd}: Aggregated fit metrics: {aggregated_metrics}")
        if aggregated_parameters is not None:
            weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
            shapes = [w.shape for w in weights]
            norms = [np.linalg.norm(w) for w in weights]
            print(f"Round {rnd}: Global model parameter shapes: {shapes}")
            print(f"Round {rnd}: Global model parameter norms: {norms}")
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)
        print(f"Round {rnd}: Aggregated evaluation metrics: {aggregated_metrics}")
        return aggregated_loss, aggregated_metrics

if __name__ == "__main__":
    strategy = CustomFedAvgStrategy()
    config = fl.server.ServerConfig(num_rounds=3)
    fl.server.start_server(server_address="localhost:8080", config=config, strategy=strategy)
