#!/usr/bin/python3
# coding=utf-8

import flwr as fl
from flwr.server import ServerConfig

if __name__ == "__main__":
    config = ServerConfig(num_rounds=3)
    fl.server.start_server(server_address="localhost:8080", config=config)
