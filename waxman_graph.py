import numpy as np
import networkx as nx
import random
from math import sqrt

def generate_waxman_graph(n, beta, alpha, positions):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    L = max(sqrt((positions[u][0]-positions[v][0])**2 + (positions[u][1]-positions[v][1])**2)
            for u in range(n) for v in range(u+1, n))
    for u in range(n):
        for v in range(u+1, n):
            d = sqrt((positions[u][0]-positions[v][0])**2 + (positions[u][1]-positions[v][1])**2)
            p_edge = beta * np.exp(-d / (alpha * L))
            if random.random() < p_edge:
                G.add_edge(u, v)
    return G 