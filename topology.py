import numpy as np
import networkx as nx
import random
from math import sqrt
from datetime import datetime
import json

class Topology:
    @staticmethod
    def generate_waxman_graph(n: int, beta: float, alpha: float, positions: dict) -> nx.Graph:
        """Generate a Waxman graph with given parameters"""
        G = nx.Graph()
        G.add_nodes_from(range(n))
        L = max(sqrt((positions[u][0]-positions[v][0])**2 + (positions[u][1]-positions[v][1])**2)
                for u in range(n) for v in range(u+1, n))
        
        # Calculate target number of edges based on desired average degree
        target_edges = int(n * (n-1) / 2 * 0.1)  # Start with 10% of possible edges
        
        # Generate edges with probability based on distance
        for u in range(n):
            for v in range(u+1, n):
                d = sqrt((positions[u][0]-positions[v][0])**2 + (positions[u][1]-positions[v][1])**2)
                p_edge = beta * np.exp(-d / (alpha * L))
                if random.random() < p_edge:
                    G.add_edge(u, v)
                    
        return G

    @staticmethod
    def initialize_topology(num_nodes, average_degree, json_file=None):
        if json_file:
            return Topology.load_topology_from_json(num_nodes,json_file)
            
        print("Initializing network topology...")
        d_min = 50.0 / sqrt(num_nodes)
        positions = {}
        for node in range(num_nodes):
            while True:
                pos = (random.uniform(0, 100000), random.uniform(0, 100000))
                if all(sqrt((pos[0]-positions[other][0])**2 + (pos[1]-positions[other][1])**2) >= d_min for other in positions):
                    positions[node] = pos
                    break

        # Calculate target number of edges based on average degree
        target_edges = int(num_nodes * average_degree / 2)
        
        # Try a few alpha values to get close to target degree
        best_G = None
        best_diff = float('inf')
        
        for alpha in [0.2, 0.4, 0.6, 0.8, 1.0]:
            G = Topology.generate_waxman_graph(num_nodes, beta=0.6, alpha=alpha, positions=positions)
            
            # If graph is disconnected, connect it
            if not nx.is_connected(G):
                components = list(nx.connected_components(G))
                while len(components) > 1:
                    # Connect two closest components
                    min_dist = float('inf')
                    edge_to_add = None
                    for i in range(len(components)):
                        for j in range(i+1, len(components)):
                            for u in components[i]:
                                for v in components[j]:
                                    d = sqrt((positions[u][0]-positions[v][0])**2 + (positions[u][1]-positions[v][1])**2)
                                    if d < min_dist:
                                        min_dist = d
                                        edge_to_add = (u, v)
                    if edge_to_add:
                        G.add_edge(*edge_to_add)
                    components = list(nx.connected_components(G))
            
            current_avg_deg = 2 * G.number_of_edges() / num_nodes
            
            # Update best graph if this one is better
            if abs(current_avg_deg - average_degree) < best_diff:
                best_diff = abs(current_avg_deg - average_degree)
                best_G = G.copy()
            
            # If we're close enough to target, break
            if abs(current_avg_deg - average_degree) < 0.5:
                break
        
        # Quick adjustment of edges if needed
        if best_G:
            current_edges = best_G.number_of_edges()
            if current_edges < target_edges:
                # Add some random edges
                possible_edges = [(u, v) for u in range(num_nodes) for v in range(u+1, num_nodes) 
                                if not best_G.has_edge(u, v)]
                random.shuffle(possible_edges)
                for u, v in possible_edges[:min(len(possible_edges), target_edges - current_edges)]:
                    best_G.add_edge(u, v)
            elif current_edges > target_edges:
                # Remove some random edges
                edges_to_remove = random.sample(list(best_G.edges()), min(len(best_G.edges()), current_edges - target_edges))
                best_G.remove_edges_from(edges_to_remove)
        
        for node in best_G.nodes():
            best_G.nodes[node]['pos'] = positions[node]
            
        print(f"Generated topology: {num_nodes} nodes, {best_G.number_of_edges()} edges, average degree ≈ {2*best_G.number_of_edges()/num_nodes:.2f}")
        
        # Store topology in file
        Topology.store_topology(best_G, num_nodes, average_degree)
        
        return best_G

    @staticmethod
    def load_topology_from_json(num_nodes,json_file):
        print(f"Loading topology from {json_file}...")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        G = nx.Graph()
        beta = data.get('beta', 0.6)  # Default to 0.6 if not specified
        alpha = data.get('alpha', 0.4)  # Default to 0.4 if not specified
        
        # Add nodes
        for node in range(num_nodes):
            G.add_node(node)
            if 'positions' in data and str(node) in data['positions']:
                G.nodes[node]['pos'] = tuple(data['positions'][str(node)])
        
       
        for node, neighbors in data['adjacency_list'].items():
            node = int(node)
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        
        print(f"Loaded topology: {num_nodes} nodes, {G.number_of_edges()} edges, average degree ≈ {2*G.number_of_edges()/num_nodes:.2f}")
        print(f"Using parameters: beta={beta}, alpha={alpha}")
        return G

    @staticmethod
    def export_topology_to_json(graph, filename, beta=0.6, alpha=0.4):
        data = {
            'num_nodes': graph.number_of_nodes(),
            'beta': beta,
            'alpha': alpha,
            'adjacency_list': {str(node): list(graph.neighbors(node)) for node in graph.nodes()},
            'positions': {str(node): list(graph.nodes[node]['pos']) for node in graph.nodes() if 'pos' in graph.nodes[node]}
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Topology exported to {filename} with beta={beta}, alpha={alpha}")

    @staticmethod
    def store_topology(graph, num_nodes, average_degree):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = "topology_history.txt"
        
        with open(filename, 'a') as f:
            f.write(f"\n=== Topology Generated at {timestamp} ===\n")
            f.write(f"Parameters: num_nodes={num_nodes}, average_degree={average_degree}\n")
            f.write("Adjacency List Format:\n")
            
            # Write adjacency list
            for node in sorted(graph.nodes()):
                neighbors = sorted(graph.neighbors(node))
                f.write(f"{node}: {neighbors}\n")
            
            # Write node positions
            f.write("\nNode Positions:\n")
            for node in sorted(graph.nodes()):
                if 'pos' in graph.nodes[node]:
                    pos = graph.nodes[node]['pos']
                    f.write(f"{node}: [{pos[0]:.1f}, {pos[1]:.1f}]\n")
            
            # Write additional information
            f.write(f"\nTotal Nodes: {graph.number_of_nodes()}\n")
            f.write(f"Total Edges: {graph.number_of_edges()}\n")
            f.write(f"Average Degree: {2*graph.number_of_edges()/graph.number_of_nodes():.2f}\n")
            f.write("="*50 + "\n")

    @staticmethod
    def adjust_success_probabilities(graph, target_Ep):
        print("Adjusting channel success probabilities to target Ep = {} ± 0.01...".format(target_Ep))
        tol = 0.01
        alpha_factor = 0.5
        while True:
            for u, v in graph.edges():
                prob = min(np.random.uniform(0.1, 0.9) * alpha_factor, 1.0)
                graph[u][v]['success_prob'] = prob
            avg_prob = np.mean([graph[u][v]['success_prob'] for u, v in graph.edges()])
            print(f"Current average Ep = {avg_prob:.3f}")
            if abs(avg_prob - target_Ep) < tol:
                print("Target average Ep achieved.")
                break
            alpha_factor *= 0.95 if avg_prob > target_Ep else 1.05

    @staticmethod
    def determine_hop_count(graph, num_nodes):
        print("Determining maximum hop count (hm)...")
        sample_pairs = [tuple(np.random.choice(num_nodes, size=2, replace=False)) for _ in range(100)]
        hop_counts = []
        for s, d in sample_pairs:
            try:
                path = nx.shortest_path(graph, s, d)
                hop_counts.append(len(path) - 1)
            except nx.NetworkXNoPath:
                continue
        hm = max(hop_counts) if hop_counts else 4
        print(f"Determined hm = {hm}")
        return hm

    @staticmethod
    def initialize_node_resources(graph):
        print("Initializing node resources (qubit capacities from 10 to 14)...")
        initial_qubits = {}
        for node in graph.nodes():
            cap = random.randint(10, 14)
            graph.nodes[node]['qubits'] = cap
            initial_qubits[node] = cap
            print(f"Node {node}: {cap} qubits")
        return initial_qubits

    @staticmethod
    def initialize_edge_channels(graph):
        print("Initializing edge channels (width uniformly from 3 to 7)...")
        for u, v in graph.edges():
            width = random.randint(3, 7)
            graph[u][v]['width'] = width
            graph[u][v]['channels'] = [{'reserved': False, 'entangled': False} for _ in range(width)]
            print(f"Edge ({u},{v}): width = {width}") 