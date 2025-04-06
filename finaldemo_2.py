import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
import copy
from math import comb, sqrt, inf
from itertools import product
from networkx.algorithms.simple_paths import shortest_simple_paths

#####################################
# Helper Functions for EXT Plotting #
#####################################

def compute_EXT_given_parameters(W, h, p, q):
    if h < 1:
        return 0
    P = [[0.0]*(h+1) for _ in range(W+1)]
    for i in range(1, W+1):
        P[i][1] = comb(W, i) * (p ** i) * ((1 - p) ** (W - i))
    for k in range(2, h+1):
        for i in range(1, W+1):
            sum1 = sum(comb(W, l) * (p ** l) * ((1 - p) ** (W - l)) for l in range(i, W+1))
            sum2 = sum(P[l][k-1] for l in range(i+1, W+1))
            P[i][k] = P[i][k-1] * sum1 + (comb(W, i) * (p ** i) * ((1 - p) ** (W - i))) * sum2
    EXT_val = sum(i * P[i][h] for i in range(1, W+1))
    EXT_val *= (q ** (h - 1))
    return EXT_val

def plot_EXT_vs_h(p_values=[0.9, 0.6], q=0.9, widths=[1, 2, 3], h_range=range(1, 11)):
    plt.figure(figsize=(10, 6))
    for p in p_values:
        for W in widths:
            ext_vals = []
            for h in h_range:
                ext = compute_EXT_given_parameters(W, h, p, q)
                ext_vals.append(ext)
            label = f"p={p}, W={W}"
            plt.plot(list(h_range), ext_vals, marker='o', linewidth=2, label=label)
            print(f"Computed EXT for p={p}, W={W}: {ext_vals}")
    plt.title("EXT vs. Hop Count for Different p and Widths")
    plt.xlabel("Hop Count (h)")
    plt.ylabel("Expected Throughput (EXT)")
    plt.grid(True)
    plt.legend()
    plt.show()

#################################
# Custom Waxman Graph Generator #
#################################

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

##############################
# Base Class: QuantumNetwork #
##############################

class QuantumNetwork:
    def __init__(self, num_nodes=100, num_slots=1000, num_requests=10, link_state_range=3,
                 routing_metric="EXT", average_degree=6, target_Ep=0.6, q=0.9):
        self.num_nodes = num_nodes
        self.num_slots = num_slots
        self.num_requests = num_requests
        # For k = infinity, set a very high number.
        self.link_state_range = link_state_range if link_state_range != float('inf') else 1e6
        self.routing_metric = routing_metric  # "EXT", "SumDist", "CR", or "BotCap"
        self.average_degree = average_degree
        self.target_Ep = target_Ep
        self.q = q
        self.graph = nx.Graph()
        self.link_state = {}  # Will hold actual link-state information (up to k hops)
        self.total_entangled_pairs = 0
        self.throughput_per_slot = []
        self.deferred_requests = []
        self.hm = None
        self.initialize_topology()
        self.adjust_success_probabilities()
        self.determine_hop_count()
        self.initialize_node_resources()
        self.initialize_edge_channels()

    def initialize_topology(self):
        print("Initializing network topology...")
        d_min = 50.0 / sqrt(self.num_nodes)
        positions = {}
        for node in range(self.num_nodes):
            while True:
                pos = (random.uniform(0, 100000), random.uniform(0, 100000))
                if all(sqrt((pos[0]-positions[other][0])**2 + (pos[1]-positions[other][1])**2) >= d_min for other in positions):
                    positions[node] = pos
                    break
        a_low, a_high = 0.1, 10.0
        best_G = None
        for _ in range(20):
            alpha_mid = (a_low + a_high) / 2.0
            G_mid = generate_waxman_graph(self.num_nodes, beta=0.6, alpha=alpha_mid, positions=positions)
            avg_deg = 2 * G_mid.number_of_edges() / self.num_nodes
            if abs(avg_deg - self.average_degree) < 0.5:
                best_G = G_mid
                break
            if avg_deg < self.average_degree:
                a_low = alpha_mid
            else:
                a_high = alpha_mid
            best_G = G_mid
        self.graph = best_G
        for node in self.graph.nodes():
            self.graph.nodes[node]['pos'] = positions[node]
        print(f"Generated topology: {self.num_nodes} nodes, {self.graph.number_of_edges()} edges, average degree ≈ {2*self.graph.number_of_edges()/self.num_nodes:.2f}")

    def adjust_success_probabilities(self):
        print("Adjusting channel success probabilities to target Ep = {} ± 0.01...".format(self.target_Ep))
        tol = 0.01
        alpha_factor = 0.5
        while True:
            for u, v in self.graph.edges():
                prob = min(np.random.uniform(0.1, 0.9) * alpha_factor, 1.0)
                self.graph[u][v]['success_prob'] = prob
            avg_prob = np.mean([self.graph[u][v]['success_prob'] for u, v in self.graph.edges()])
            print(f"Current average Ep = {avg_prob:.3f}")
            if abs(avg_prob - self.target_Ep) < tol:
                print("Target average Ep achieved.")
                break
            alpha_factor *= 0.95 if avg_prob > self.target_Ep else 1.05

    def determine_hop_count(self):
        print("Determining maximum hop count (hm)...")
        sample_pairs = [tuple(np.random.choice(self.num_nodes, size=2, replace=False)) for _ in range(100)]
        hop_counts = []
        for s, d in sample_pairs:
            try:
                path = nx.shortest_path(self.graph, s, d)
                hop_counts.append(len(path) - 1)
            except nx.NetworkXNoPath:
                continue
        self.hm = max(hop_counts) if hop_counts else 4
        print(f"Determined hm = {self.hm}")

    def initialize_node_resources(self):
        print("Initializing node resources (qubit capacities from 10 to 14)...")
        self.initial_qubits = {}
        for node in self.graph.nodes():
            cap = random.randint(10, 14)
            self.graph.nodes[node]['qubits'] = cap
            self.initial_qubits[node] = cap
            print(f"Node {node}: {cap} qubits")

    def initialize_edge_channels(self):
        print("Initializing edge channels (width uniformly from 3 to 7)...")
        for u, v in self.graph.edges():
            width = random.randint(3, 7)
            self.graph[u][v]['width'] = width
            self.graph[u][v]['channels'] = [{'reserved': False, 'entangled': False} for _ in range(width)]
            print(f"Edge ({u},{v}): width = {width}")

    def compute_edge_length(self, u, v):
        pos_u = self.graph.nodes[u]['pos']
        pos_v = self.graph.nodes[v]['pos']
        return sqrt((pos_u[0]-pos_v[0])**2 + (pos_u[1]-pos_v[1])**2)

    def compute_SumDist(self, path):
        total = sum(self.compute_edge_length(path[i], path[i+1]) for i in range(len(path)-1))
        print(f"Computed SumDist for path {path}: {total:.4f}")
        return total

    def compute_CR(self, path):
        total = sum(1.0 / self.graph[path[i]][path[i+1]]['success_prob'] if self.graph[path[i]][path[i+1]]['success_prob'] > 0 else float('inf')
                    for i in range(len(path)-1))
        print(f"Computed CR for path {path}: {total:.4f}")
        return total

    def compute_BotCap(self, path):
        widths = [self.graph[u][v]['width'] for u, v in zip(path[:-1], path[1:])]
        bottleneck = min(widths)
        print(f"Computed raw BotCap for path {path}: {-bottleneck}")
        return -bottleneck

    def compute_routing_metric(self, path):
        if self.routing_metric == "EXT":
            val = self.compute_EXT_recursive(path)
            print(f"Using EXT for path {path}: {val:.4f}")
            return val
        elif self.routing_metric == "SumDist":
            sum_dist = self.compute_SumDist(path)
            print(f"Using SumDist for path {path}: {sum_dist:.4f}")
            return sum_dist
        elif self.routing_metric == "CR":
            cr = self.compute_CR(path)
            print(f"Using CR for path {path}: {cr:.4f}")
            return cr
        elif self.routing_metric == "BotCap":
            botcap = self.compute_BotCap(path)
            cr = self.compute_CR(path)
            val = (botcap, cr)
            print(f"Using BotCap for path {path}: {val}")
            return val
        else:
            return self.compute_EXT_recursive(path)


    def link_state_exchange(self):
        print("Exchanging link state information among nodes (up to {}-hop exchange)...".format(self.link_state_range))
        # For each node, perform a BFS up to depth = self.link_state_range.
        for node in self.graph.nodes():
            local_state = {}
            visited = {node}
            queue = [(node, 0)]
            while queue:
                current, depth = queue.pop(0)
                if depth == 0:
                    # Skip the source node itself.
                    pass
                else:
                    # If there is a direct edge between the source and current, record its state.
                    if self.graph.has_edge(node, current):
                        channel_status = []
                        for ch in self.graph[node][current]['channels']:
                            channel_status.append({
                                'reserved': ch['reserved'],
                                'entangled': ch['entangled']
                            })
                        local_state[current] = channel_status
                if depth < self.link_state_range:
                    for nbr in self.graph.neighbors(current):
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append((nbr, depth+1))
            self.link_state[node] = local_state
        for node in self.graph.nodes():
            print(f"Node {node} link state (up to {self.link_state_range}-hop): {self.link_state[node]}")
        time.sleep(0.05)

    def reset_resources_for_new_slot(self):
        print("Resetting node resources and channel reservations for new time slot...")
        for node in self.graph.nodes():
            self.graph.nodes[node]['qubits'] = self.initial_qubits[node]
        for u, v in self.graph.edges():
            for ch in self.graph[u][v]['channels']:
                ch['reserved'] = False
                ch['entangled'] = False
                if 'attempted' in ch:
                    del ch['attempted']

    def reset_entanglements(self):
        print("Resetting entanglement states on all channels...")
        for u, v in self.graph.edges():
            for ch in self.graph[u][v]['channels']:
                ch['entangled'] = False
                if 'attempted' in ch:
                    del ch['attempted']

    def generate_sd_pairs(self):
        sd_pairs = [tuple(np.random.choice(self.num_nodes, size=2, replace=False)) for _ in range(self.num_requests)]
        print(f"Generated S-D pairs: {sd_pairs}")
        return sd_pairs

    def compute_EXT_recursive(self, path):
        h = len(path) - 1
        if h < 1:
            return 0
        widths = []
        p_list = []
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            widths.append(self.graph[u][v]['width'])
            p_list.append(self.graph[u][v]['success_prob'])
        W = min(widths)
        P = [[0.0]*(h+1) for _ in range(W+1)]
        for i in range(1, W+1):
            Q = comb(W, i) * (p_list[0] ** i) * ((1 - p_list[0]) ** (W - i))
            P[i][1] = Q
        for k in range(2, h+1):
            pk = p_list[k-1]
            for i in range(1, W+1):
                Qik = comb(W, i) * (pk ** i) * ((1 - pk) ** (W - i))
                sum1 = sum(comb(W, l) * (pk ** l) * ((1 - pk) ** (W - l)) for l in range(i, W+1))
                sum2 = sum(P[l][k-1] for l in range(i+1, W+1))
                P[i][k] = P[i][k-1] * sum1 + Qik * sum2
        EXT_val = sum(i * P[i][h] for i in range(1, W+1))
        EXT_val *= (self.q ** (h - 1))
        print(f"Computed EXT for path {path}: {EXT_val:.4f}")
        return EXT_val

    def extended_dijkstra(self, source, target):
        print(f"Running Extended Dijkstra from {source} to {target} using {self.routing_metric} metric...")
        if self.routing_metric == "EXT":
            best_val = {node: -float('inf') for node in self.graph.nodes()}
            best_val[source] = 0
        elif self.routing_metric == "BotCap":
            best_val = {node: (float('inf'), float('inf')) for node in self.graph.nodes()}
            best_val[source] = (0, 0)
        else:
            best_val = {node: float('inf') for node in self.graph.nodes()}
            best_val[source] = 0
        best_path = {node: [] for node in self.graph.nodes()}
        best_path[source] = [source]
        visited = set()
        while True:
            if self.routing_metric == "EXT":
                candidates = {node: best_val[node] for node in self.graph.nodes() if node not in visited and best_val[node] > -float('inf')}
                if not candidates:
                    break
                u = max(candidates, key=lambda node: best_val[node])
            elif self.routing_metric == "BotCap":
                candidates = {node: best_val[node] for node in self.graph.nodes() if node not in visited and best_val[node] < (float('inf'), float('inf'))}
                if not candidates:
                    break
                u = min(candidates, key=lambda node: best_val[node])
            else:
                candidates = {node: best_val[node] for node in self.graph.nodes() if node not in visited and best_val[node] < float('inf')}
                if not candidates:
                    break
                u = min(candidates, key=lambda node: best_val[node])
            visited.add(u)
            if u == target:
                break
            for v in self.graph.neighbors(u):
                if self.graph.nodes[v]['qubits'] < 1:
                    continue
                if not any(not ch['reserved'] for ch in self.graph[u][v]['channels']):
                    continue
                new_path = best_path[u] + [v]
                if len(new_path) - 1 > self.hm:
                    continue
                metric_val = self.compute_routing_metric(new_path)
                if self.routing_metric == "EXT":
                    if metric_val > best_val[v]:
                        best_val[v] = metric_val
                        best_path[v] = new_path
                else:
                    if metric_val < best_val[v]:
                        best_val[v] = metric_val
                        best_path[v] = new_path
        if self.routing_metric == "EXT":
            if best_val[target] > -float('inf'):
                print(f"Extended Dijkstra: Best path from {source} to {target} ({self.routing_metric} = {best_val[target]:.4f}): {best_path[target]}")
                return best_path[target], best_val[target]
            else:
                print(f"Extended Dijkstra: No path from {source} to {target}.")
                return None, 0
        else:
            if self.routing_metric == "BotCap":
                if best_val[target] < (float('inf'), float('inf')):
                    print(f"Extended Dijkstra: Best path from {source} to {target} ({self.routing_metric} = {best_val[target]}): {best_path[target]}")
                    return best_path[target], best_val[target]
                else:
                    print(f"Extended Dijkstra: No path from {source} to {target}.")
                    return None, 0
            else:
                if best_val[target] < float('inf'):
                    print(f"Extended Dijkstra: Best path from {source} to {target} ({self.routing_metric} = {best_val[target]}): {best_path[target]}")
                    return best_path[target], best_val[target]
                else:
                    print(f"Extended Dijkstra: No path from {source} to {target}.")
                    return None, 0

    def reserve_resources(self, path):
        print(f"Reserving resources for path: {path}")
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            if self.graph.nodes[u]['qubits'] < 1 or self.graph.nodes[v]['qubits'] < 1:
                print(f"Reservation failed: Insufficient qubits at {u} or {v}.")
                return False
            if not any(not ch['reserved'] for ch in self.graph[u][v]['channels']):
                print(f"Reservation failed: No free channel on edge ({u},{v}).")
                return False
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            self.graph.nodes[u]['qubits'] -= 1
            self.graph.nodes[v]['qubits'] -= 1
            for ch in self.graph[u][v]['channels']:
                if not ch['reserved']:
                    ch['reserved'] = True
                    print(f"Reserved channel on edge ({u},{v}).")
                    break
        return True

    def segment_path(self, path):
        seg_length = self.link_state_range + 1
        segments = []
        for i in range(0, len(path)-1, seg_length - 1):
            seg = path[i: min(i+seg_length, len(path))]
            segments.append(seg)
        print(f"Segmented path {path} into segments: {segments}")
        return segments

    # For Q-PASS, recovery is segmentation-based only.
    def recover_segment(self, segment):
        if len(segment) < 2:
            return segment
        u, v = segment[0], segment[-1]
        original_hm = self.hm
        self.hm = len(segment) - 1
        recovery_path, metric_val = self.extended_dijkstra(u, v)
        self.hm = original_hm
        if recovery_path:
            print(f"Recovered segment from {u} to {v}: {recovery_path} ({self.routing_metric} = {metric_val})")
            return recovery_path
        else:
            print(f"Failed to recover segment from {u} to {v}.")
            return segment

    def xor_operator(self, E1, E2):
        return (E1 | E2) - (E1 & E2)

    def xor_based_full_recovery(self, major_path, s, d):
        print("Starting full XOR-based recovery for major path:", major_path)
        E_major = set()
        for i in range(len(major_path)-1):
            edge = frozenset({major_path[i], major_path[i+1]})
            E_major.add(edge)
        failed_edges = []
        for i in range(len(major_path)-1):
            u, v = major_path[i], major_path[i+1]
            if not any(ch['reserved'] and ch.get('entangled', False) for ch in self.graph[u][v]['channels']):
                failed_edges.append((u, v))
        print("Failed edges in major path:", failed_edges)
        E_recovery_total = set()
        R = 2
        for (u, v) in failed_edges:
            recovery_candidates = []
            try:
                gen = shortest_simple_paths(self.graph, u, v)
                for idx, cand in enumerate(gen):
                    if idx >= 25:
                        break
                    recovery_candidates.append(cand)
            except nx.NetworkXNoPath:
                recovery_candidates = []
            if not recovery_candidates:
                continue
            recovery_candidates = sorted(recovery_candidates, key=lambda p: len(p))[:R]
            for rp in recovery_candidates:
                E_rp = set()
                for j in range(len(rp)-1):
                    E_rp.add(frozenset({rp[j], rp[j+1]}))
                print(f"Recovery candidate from {u} to {v}: {rp}, edges: {E_rp}")
                E_recovery_total = self.xor_operator(E_recovery_total, E_rp)
        print("Total recovery edge set after XOR:", E_recovery_total)
        E_final = self.xor_operator(E_major, E_recovery_total)
        print("Final edge set after applying XOR on major path:", E_final)
        temp_graph = nx.Graph()
        for edge in E_final:
            nodes = list(edge)
            if len(nodes) == 2:
                temp_graph.add_edge(nodes[0], nodes[1])
        for node in major_path:
            temp_graph.add_node(node)
        if nx.has_path(temp_graph, s, d):
            recovered_path = nx.shortest_path(temp_graph, s, d)
            print("XOR-based full recovery succeeded. Recovered path:", recovered_path)
            return recovered_path
        else:
            print("XOR-based full recovery failed. No connected path from", s, "to", d)
            return None

    def perform_entanglement_swapping(self, path):
        print("Performing logarithmic-time entanglement swapping along path:", path)
        h = len(path) - 1
        if h < 2:
            print("Path too short; skipping swapping.")
            return True
        segments = [path]
        iteration = 0
        while any(len(seg) > 2 for seg in segments):
            iteration += 1
            new_segments = []
            print(f"Swapping iteration {iteration}: segments = {segments}")
            for seg in segments:
                if len(seg) <= 2:
                    new_segments.append(seg)
                else:
                    mid = len(seg) // 2
                    seg1 = seg[:mid+1]
                    seg2 = seg[mid:]
                    print(f"Swapping at node {seg[mid]} for segment {seg}: seg1 = {seg1}, seg2 = {seg2}")
                    new_segments.append(seg1)
                    new_segments.append(seg2)
            segments = new_segments
            time.sleep(0.01)
        print(f"Swapping completed after {iteration} iterations.")
        return True

    def attempt_entanglement(self, path):
        print("Attempting entanglement along path:", path)
        success = True
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            channel_success = False
            for ch in self.graph[u][v]['channels']:
                if ch['reserved'] and not ch.get('attempted', False):
                    if random.random() < self.graph[u][v]['success_prob']:
                        ch['entangled'] = True
                        channel_success = True
                        print(f"Entanglement succeeded on edge ({u},{v}).")
                    else:
                        print(f"Entanglement failed on edge ({u},{v}).")
                    ch['attempted'] = True
                    break
            if not channel_success:
                success = False
                print(f"Failure detected on edge ({u},{v}).")
        if success:
            print(f"All entanglements on path {path} succeeded. Proceeding with swapping.")
            if self.perform_entanglement_swapping(path):
                print(f"Swapping succeeded for path {path}.")
            else:
                print(f"Swapping failed for path {path}.")
                success = False
        return success

    def run_slot(self, sd_pairs):
        print(f"\n=== Running Slot with S-D pairs: {sd_pairs} ===")
        self.reset_resources_for_new_slot()
        current_sd = sd_pairs + self.deferred_requests
        print(f"Current S-D pairs: {current_sd}")
        selected_paths = self.qcast_path_selection(current_sd)
        print("---- Resource Reservation Completed ----")
        for sd, path in selected_paths.items():
            print(f"S-D pair {sd}: Selected path: {path}")
        successful_entanglements = 0
        served_sd = set()
        if self.__class__.__name__ in ['QCASTSimulator', 'QCASTRSimulator']:
            if random.randint(0, 9) == 0:
                self.link_state_exchange()
        print("---- Entanglement Phase (P4) ----")
        if self.__class__.__name__ in ['QPASSRSimulator', 'QCASTRSimulator']:
            for sd, path in selected_paths.items():
                if self.attempt_entanglement(path):
                    print(f"Direct entanglement succeeded for {sd}.")
                    successful_entanglements += 1
                    served_sd.add(sd)
                else:
                    print(f"{sd} failed (no recovery attempted).")
        else:
            for sd, path in selected_paths.items():
                s, d = sd
                if self.attempt_entanglement(path):
                    print(f"Direct entanglement succeeded for {sd}.")
                    successful_entanglements += 1
                    served_sd.add(sd)
                else:
                    if self.routing_metric == "EXT":
                        if self.attempt_entanglement_with_recovery(path, s, d):
                            print(f"XOR-based recovery succeeded for {sd}.")
                            successful_entanglements += 1
                            served_sd.add(sd)
                        else:
                            print(f"{sd} failed.")
                    else:
                        if self.attempt_entanglement_with_recovery(path, s, d):
                            print(f"Segmentation-based recovery succeeded for {sd}.")
                            successful_entanglements += 1
                            served_sd.add(sd)
                        else:
                            print(f"{sd} failed.")
        self.total_entangled_pairs += successful_entanglements
        self.throughput_per_slot.append(successful_entanglements)
        print(f"Slot throughput: {successful_entanglements}")
        self.deferred_requests = [sd for sd in current_sd if sd not in served_sd]
        print("Deferred requests for next slot:", self.deferred_requests)
        return successful_entanglements

    def qcast_path_selection(self, sd_pairs):
        print("Starting Q-CAST path selection (P2) for current S-D pairs using routing metric:", self.routing_metric)
        candidate_paths = []
        if self.__class__.__name__ in ['QCASTSimulator', 'QCASTRSimulator']:
            for sd in sd_pairs:
                s, d = sd
                path, metric = self.extended_dijkstra(s, d)
                if path is not None and self.reserve_resources(path):
                    candidate_paths.append((sd, path, metric, min(self.graph[u][v]['width'] for u, v in zip(path[:-1], path[1:]))))
                else:
                    print(f"S-D pair {sd}: No contention-free online path found.")
        else:
            for sd in sd_pairs:
                s, d = sd
                offline_paths = self.offline_candidate_paths(sd)
                for path in offline_paths:
                    offline_width = min(self.graph[u][v]['width'] for u, v in zip(path[:-1], path[1:]))
                    metric = self.compute_routing_metric(path)
                    candidate_paths.append((sd, path, metric, offline_width))
        if self.routing_metric == "EXT":
            candidate_paths.sort(key=lambda x: x[2], reverse=True)
        else:
            candidate_paths.sort(key=lambda x: x[2])
        selected_paths = {}
        unsatisfied = []
        print("Step 1: Major path selection")
        for item in candidate_paths:
            sd, path, metric, offline_width = item
            if sd in selected_paths:
                continue
            if self.reserve_resources(path):
                selected_paths[sd] = path
                print(f"S-D pair {sd}: Reserved major path {path} with metric = {metric}")
            else:
                current_width = min(sum(1 for ch in self.graph[u][v]['channels'] if not ch['reserved'])
                                    for u, v in zip(path[:-1], path[1:]))
                if current_width < offline_width:
                    updated_metric = metric if self.routing_metric == "BotCap" else metric * (current_width / offline_width)
                    unsatisfied.append((sd, path, updated_metric, offline_width, current_width))
                    print(f"S-D pair {sd}: Candidate {path} updated metric to {updated_metric} (available width = {current_width}).")
                else:
                    unsatisfied.append((sd, path, metric, offline_width, current_width))
        print("Step 2: Recovery path selection for unsatisfied candidates")
        if self.routing_metric == "EXT":
            unsatisfied.sort(key=lambda x: x[2], reverse=True)
        else:
            unsatisfied.sort(key=lambda x: x[2])
        for sd, path, metric, offline_width, current_width in unsatisfied:
            if sd in selected_paths:
                continue
            recovered = None
            if self.routing_metric == "EXT":
                recovered = self.xor_based_full_recovery(path, sd[0], sd[1])
            else:
                recovered = self.recover_segment(path)
            if recovered is not None and self.reserve_resources(recovered):
                selected_paths[sd] = recovered
                updated_metric = self.compute_routing_metric(recovered)
                print(f"S-D pair {sd}: Reserved recovery path {recovered} with updated metric = {updated_metric}")
            else:
                print(f"S-D pair {sd}: No candidate path could reserve resources after recovery.")
        return selected_paths

    def offline_candidate_paths(self, sd):
        s, d = sd
        if hasattr(self, 'offline_candidates') and sd in self.offline_candidates:
            return self.offline_candidates[sd]
        candidates = []
        try:
            gen = shortest_simple_paths(self.graph, s, d)
            for idx, path in enumerate(gen):
                candidates.append(path)
                if idx >= 24:
                    break
        except nx.NetworkXNoPath:
            candidates = []
        if not hasattr(self, 'offline_candidates'):
            self.offline_candidates = {}
        self.offline_candidates[sd] = candidates
        return candidates

    # def plot_cdf(self):
    #     plt.figure(figsize=(8, 6))
    #     sorted_throughput = np.sort(self.throughput_per_slot)
    #     cdf = np.arange(1, len(sorted_throughput) + 1) / len(sorted_throughput)
    #     plt.plot(sorted_throughput, cdf, linewidth=2, label=f"{self.__class__.__name__} ({self.routing_metric})")
    #     plt.title("CDF of Throughput (ebits per slot) - " + self.__class__.__name__ + f" ({self.routing_metric})")
    #     plt.xlabel("Throughput (ebits per slot)")
    #     plt.ylabel("CDF")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.show()

###############################
# Simulator Variants          #
###############################

# Q-PASS with segmentation-based recovery (no XOR fallback)
class QPASSSimulator(QuantumNetwork):
    # Inherit segmentation-based recovery from the base class.
    def attempt_entanglement_with_recovery(self, path, s, d):
        print("Attempting entanglement with segmentation-based recovery for path (Q-PASS):", path)
        segments = self.segment_path(path)
        recovered_path = []
        recovery_success = True
        for seg in segments:
            print("Processing segment:", seg)
            seg_success = True
            for i in range(len(seg)-1):
                u, v = seg[i], seg[i+1]
                channel_success = False
                for ch in self.graph[u][v]['channels']:
                    if ch['reserved'] and not ch.get('attempted', False):
                        if random.random() < self.graph[u][v]['success_prob']:
                            ch['entangled'] = True
                            channel_success = True
                            print(f"Segment: Success on edge ({u},{v}).")
                        else:
                            print(f"Segment: Failure on edge ({u},{v}).")
                        ch['attempted'] = True
                        break
                if not channel_success:
                    seg_success = False
                    print(f"Segment: Failure detected on edge ({u},{v}).")
                    break
            if seg_success:
                print(f"Segment {seg} succeeded.")
                if recovered_path:
                    recovered_path.extend(seg[1:])
                else:
                    recovered_path.extend(seg)
            else:
                print(f"Segment {seg} failed. Segmentation-based recovery stops for Q-PASS.")
                recovery_success = False
                break
        if recovery_success:
            if recovered_path[0] != s:
                recovered_path.insert(0, s)
            if recovered_path[-1] != d:
                recovered_path.append(d)
            print("Recovered full path via segmentation-based recovery:", recovered_path)
            return self.attempt_entanglement(recovered_path)
        else:
            return False

    def simulate(self):
        print("\n--- Running Q-PASS Simulation (with segmentation-based recovery) ---")
        slot_throughput = []
        self.deferred_requests = []
        # Q-PASS does not use link state exchange.
        for slot in range(self.num_slots):
            print(f"\n=== Time Slot {slot} (Q-PASS) ===")
            self.reset_resources_for_new_slot()
            sd_pairs = self.generate_sd_pairs()
            current_sd = sd_pairs + self.deferred_requests
            print(f"Current S-D pairs: {current_sd}")
            selected_paths = self.qcast_path_selection(current_sd)
            print("---- Resource Reservation Completed ----")
            for sd, path in selected_paths.items():
                print(f"S-D pair {sd}: Selected path: {path}")
            successful_entanglements = 0
            served_sd = set()
            print("---- Entanglement Phase (P4) ----")
            for sd, path in selected_paths.items():
                s, d = sd
                if self.attempt_entanglement(path):
                    print(f"Direct entanglement succeeded for {sd}.")
                    successful_entanglements += 1
                    served_sd.add(sd)
                else:
                    if self.attempt_entanglement_with_recovery(path, s, d):
                        print(f"Segmentation-based recovery succeeded for {sd}.")
                        successful_entanglements += 1
                        served_sd.add(sd)
                    else:
                        print(f"{sd} failed.")
            slot_throughput.append(successful_entanglements)
            print(f"Time Slot {slot} throughput: {successful_entanglements}")
            self.deferred_requests = [sd for sd in current_sd if sd not in served_sd]
        return slot_throughput

# Q-PASS/R: Recovery-free version for Q-PASS.
class QPASSRSimulator(QuantumNetwork):
    def simulate(self):
        print("\n--- Running Q-PASS/R Simulation (recovery-free) ---")
        slot_throughput = []
        self.deferred_requests = []
        for slot in range(self.num_slots):
            print(f"\n=== Time Slot {slot} (Q-PASS/R) ===")
            self.reset_resources_for_new_slot()
            sd_pairs = self.generate_sd_pairs()
            current_sd = sd_pairs + self.deferred_requests
            print(f"Current S-D pairs: {current_sd}")
            selected_paths = self.qcast_path_selection(current_sd)
            print("---- Resource Reservation Completed ----")
            for sd, path in selected_paths.items():
                print(f"S-D pair {sd}: Selected path: {path}")
            successful_entanglements = 0
            served_sd = set()
            for sd, path in selected_paths.items():
                if self.attempt_entanglement(path):
                    print(f"Direct entanglement succeeded for {sd}.")
                    successful_entanglements += 1
                    served_sd.add(sd)
                else:
                    print(f"{sd} failed (no recovery in Q-PASS/R).")
            slot_throughput.append(successful_entanglements)
            print(f"Time Slot {slot} throughput: {successful_entanglements}")
            self.deferred_requests = [sd for sd in current_sd if sd not in served_sd]
        return slot_throughput

# Q-CAST with XOR-based recovery (online contention-aware path selection)
class QCASTSimulator(QuantumNetwork):
    def qcast_path_selection(self, sd_pairs):
        print("Starting Q-CAST online path selection using EXT metric.")
        selected_paths = {}
        for sd in sd_pairs:
            s, d = sd
            path, metric = self.extended_dijkstra(s, d)
            if path is not None and self.reserve_resources(path):
                selected_paths[sd] = path
                print(f"S-D pair {sd}: Reserved online path {path} with EXT = {metric:.4f}")
            else:
                print(f"S-D pair {sd}: No contention-free online path found.")
        return selected_paths

    def attempt_entanglement_with_recovery(self, path, s, d):
        print("Attempting entanglement with XOR-based recovery for path (Q-CAST):", path)
        recovered_path = self.xor_based_full_recovery(path, s, d)
        if recovered_path and self.attempt_entanglement(recovered_path):
            return True
        else:
            print(f"XOR-based recovery failed for {(s, d)}.")
            return False

    def simulate(self):
        print("\n--- Running Q-CAST Simulation (with XOR-based recovery) ---")
        slot_throughput = []
        self.deferred_requests = []
        # Q-CAST uses link state exchange every 10 slots.
        for slot in range(self.num_slots):
            print(f"\n=== Time Slot {slot} (Q-CAST) ===")
            self.reset_resources_for_new_slot()
            sd_pairs = self.generate_sd_pairs()
            current_sd = sd_pairs + self.deferred_requests
            print(f"Current S-D pairs: {current_sd}")
            selected_paths = self.qcast_path_selection(current_sd)
            served = set()
            successful = 0
            # if slot % 10 == 0:
            self.link_state_exchange()
            for sd, path in selected_paths.items():
                s, d = sd
                if self.attempt_entanglement(path):
                    print(f"Direct entanglement succeeded for {sd}.")
                    successful += 1
                    served.add(sd)
                else:
                    if self.attempt_entanglement_with_recovery(path, s, d):
                        print(f"XOR-based recovery succeeded for {sd}.")
                        successful += 1
                        served.add(sd)
                    else:
                        print(f"{sd} failed.")
            slot_throughput.append(successful)
            print(f"Time Slot {slot} throughput: {successful}")
            self.deferred_requests = [sd for sd in current_sd if sd not in served]
        return slot_throughput

# Q-CAST/R: Recovery-free version for Q-CAST.
class QCASTRSimulator(QuantumNetwork):
    def qcast_path_selection(self, sd_pairs):
        print("Starting Q-CAST online path selection using EXT metric.")
        selected_paths = {}
        for sd in sd_pairs:
            s, d = sd
            path, metric = self.extended_dijkstra(s, d)
            if path is not None and self.reserve_resources(path):
                selected_paths[sd] = path
                print(f"S-D pair {sd}: Reserved online path {path} with EXT = {metric:.4f}")
            else:
                print(f"S-D pair {sd}: No contention-free online path found.")
        return selected_paths

    def simulate(self):
        print("\n--- Running Q-CAST/R Simulation (recovery-free) ---")
        slot_throughput = []
        self.deferred_requests = []
        for slot in range(self.num_slots):
            print(f"\n=== Time Slot {slot} (Q-CAST/R) ===")
            self.reset_resources_for_new_slot()
            sd_pairs = self.generate_sd_pairs()
            current_sd = sd_pairs + self.deferred_requests
            print(f"Current S-D pairs: {current_sd}")
            selected_paths = self.qcast_path_selection(current_sd)
            served = set()
            successful = 0
            if slot % 10 == 0:
                self.link_state_exchange()
            for sd, path in selected_paths.items():
                if self.attempt_entanglement(path):
                    print(f"Direct entanglement succeeded for {sd}.")
                    successful += 1
                    served.add(sd)
                else:
                    print(f"{sd} failed (no recovery in Q-CAST/R).")
            slot_throughput.append(successful)
            print(f"Time Slot {slot} throughput: {successful}")
            self.deferred_requests = [sd for sd in current_sd if sd not in served]
        return slot_throughput

########################################
# Multiple Networks Simulation Wrapper #
########################################

def run_simulation_for_topologies(sim_class, routing_metric, num_topologies, num_nodes, num_slots, num_requests, link_state_range, average_degree, target_Ep, q):
    all_throughputs = []
    for topo in range(num_topologies):
        print(f"\n====== Running simulation on network topology {topo+1} ======")
        sim_obj = sim_class(num_nodes, num_slots, num_requests, link_state_range, routing_metric, average_degree, target_Ep, q)
        throughput = sim_obj.simulate()
        all_throughputs.extend(throughput)
    return all_throughputs

#######################################
# Combined Plot of Simulation Results #
#######################################

def plot_combined_comparison(qcast_data, qpass_cr_data, qpass_sumdist_data, qpass_botcap_data):
    plt.figure(figsize=(10, 6))
    
    sorted_qcast = np.sort(qcast_data)
    cdf_qcast = np.arange(1, len(sorted_qcast) + 1) / len(sorted_qcast)
    plt.plot(sorted_qcast, cdf_qcast, linewidth=2, label="Q-CAST (EXT)")
    
    sorted_qpass_cr = np.sort(qpass_cr_data)
    cdf_qpass_cr = np.arange(1, len(sorted_qpass_cr) + 1) / len(sorted_qpass_cr)
    plt.plot(sorted_qpass_cr, cdf_qpass_cr, linewidth=2, label="Q-PASS (CR)")
    
    sorted_qpass_sumdist = np.sort(qpass_sumdist_data)
    cdf_qpass_sumdist = np.arange(1, len(sorted_qpass_sumdist) + 1) / len(sorted_qpass_sumdist)
    plt.plot(sorted_qpass_sumdist, cdf_qpass_sumdist, linewidth=2, label="Q-PASS (SumDist)")
    
    sorted_qpass_botcap = np.sort(qpass_botcap_data)
    cdf_qpass_botcap = np.arange(1, len(sorted_qpass_botcap) + 1) / len(sorted_qpass_botcap)
    plt.plot(sorted_qpass_botcap, cdf_qpass_botcap, linewidth=2, label="Q-PASS (BotCap)")
    
    plt.title("Aggregated CDF of Throughput (ebits per slot) over 10 Networks")
    plt.xlabel("Throughput (ebits per slot)")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True)
    plt.show()

############################################
# Main Execution with Full Parameter Sweep #
############################################

if __name__ == "__main__":
    print("\n=== Plotting EXT vs. Hop Count Graphs ===")
    plot_EXT_vs_h(p_values=[0.9, 0.6], q=0.9, widths=[1, 2, 3], h_range=range(1, 11))
    
    # Reference parameters as per the paper:
    ref_params = {
        "num_nodes": 10,
        "target_Ep": 0.6,
        "q": 0.9,
        "link_state_range": 5,
        "average_degree": 5,
        "num_requests": 10,
        "num_slots": 30,
        "num_topologies": 1
    }
    
    print("\n=== Running Reference Simulation (n=50, Ep=0.6, q=0.9, k=3, Ed=6, m=10) ===")
    qcast_ext_ref = run_simulation_for_topologies(QCASTSimulator, "EXT",
                                                  ref_params["num_topologies"],
                                                  ref_params["num_nodes"],
                                                  ref_params["num_slots"],
                                                  ref_params["num_requests"],
                                                  ref_params["link_state_range"],
                                                  ref_params["average_degree"],
                                                  ref_params["target_Ep"],
                                                  ref_params["q"])
    qpass_cr_ref = run_simulation_for_topologies(QPASSSimulator, "CR",
                                                 ref_params["num_topologies"],
                                                 ref_params["num_nodes"],
                                                 ref_params["num_slots"],
                                                 ref_params["num_requests"],
                                                 ref_params["link_state_range"],
                                                 ref_params["average_degree"],
                                                 ref_params["target_Ep"],
                                                 ref_params["q"])
    qpass_sumdist_ref = run_simulation_for_topologies(QPASSSimulator, "SumDist",
                                                      ref_params["num_topologies"],
                                                      ref_params["num_nodes"],
                                                      ref_params["num_slots"],
                                                      ref_params["num_requests"],
                                                      ref_params["link_state_range"],
                                                      ref_params["average_degree"],
                                                      ref_params["target_Ep"],
                                                      ref_params["q"])
    qpass_botcap_ref = run_simulation_for_topologies(QPASSSimulator, "BotCap",
                                                     ref_params["num_topologies"],
                                                     ref_params["num_nodes"],
                                                     ref_params["num_slots"],
                                                     ref_params["num_requests"],
                                                     ref_params["link_state_range"],
                                                     ref_params["average_degree"],
                                                     ref_params["target_Ep"],
                                                     ref_params["q"])
    
    plot_combined_comparison(qcast_ext_ref, qpass_cr_ref, qpass_sumdist_ref, qpass_botcap_ref)

    
    # --- Full Parameter Sweep --- (For extension)
    """
    n_values = [50, 100, 200, 400, 800]
    Ep_values = [0.6, 0.3, 0.1]
    q_values = [0.8, 0.9, 1.0]
    k_values = [0, 3, 6, float('inf')]
    Ed_values = [3, 4, 6]99999999999999999
    m_values = list(range(1, 11))
    sweep_results = {}
    for params in product(n_values, Ep_values, q_values, k_values, Ed_values, m_values):
        n, Ep, q, k, Ed, m = params
        print(f"\n==== Running simulation for parameters: n={n}, Ep={Ep}, q={q}, k={k}, Ed={Ed}, m={m} ====")
        data = run_simulation_for_topologies(QCASTSimulator, "EXT",
                                             ref_params["num_topologies"],
                                             n,
                                             ref_params["num_slots"],
                                             m,
                                             k,
                                             Ed,
                                             Ep,
                                             q)
        avg_throughput = np.mean(data)
        print(f"Average throughput for {params}: {avg_throughput:.4f}")
        sweep_results[params] = avg_throughput
    """
