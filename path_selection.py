import numpy as np
from quantum_network import QuantumNetwork

class PathSelection:
    @staticmethod
    def qcast_path_selection(network: QuantumNetwork, sd_pairs):
        print("Starting Q-CAST online path selection using EXT metric.")
        selected_paths = {}
        for sd in sd_pairs:
            s, d = sd
            path, metric = network.extended_dijkstra(s, d)
            if path is not None and network.reserve_resources(path):
                selected_paths[sd] = path
                print(f"S-D pair {sd}: Reserved online path {path} with EXT = {metric:.4f}")
            else:
                print(f"S-D pair {sd}: No contention-free online path found.")
        return selected_paths

    @staticmethod
    def qpass_path_selection(network: QuantumNetwork, sd_pairs, routing_metric):
        print(f"Starting Q-PASS online path selection using {routing_metric} metric.")
        selected_paths = {}
        for sd in sd_pairs:
            s, d = sd
            path = network.select_path(s, d, routing_metric)
            if path is not None and network.reserve_resources(path):
                selected_paths[sd] = path
                print(f"S-D pair {sd}: Reserved online path {path}")
            else:
                print(f"S-D pair {sd}: No contention-free online path found.")
        return selected_paths 