import numpy as np
import networkx as nx
import random
import time
from math import comb, sqrt, inf
from networkx.algorithms.simple_paths import shortest_simple_paths
from topology import Topology
from typing import List

class QuantumNetwork:
    def __init__(self, num_nodes=100, num_slots=1000, num_requests=10, link_state_range=3,
                 routing_metric="EXT", average_degree=6, target_Ep=0.6, q=0.9,json_file=None):
        self.num_nodes = num_nodes
        self.num_slots = num_slots
        self.num_requests = num_requests
        self.link_state_range = link_state_range if link_state_range != float('inf') else 1e6
        self.routing_metric = routing_metric
        self.average_degree = average_degree
        self.target_Ep = target_Ep
        self.q = q
        self.json_file = json_file
        self.graph = None
        self.link_state = {}
        self.total_entangled_pairs = 0
        self.throughput_per_slot = []
        self.deferred_requests = []
        self.hm = None
        self.initialize_network()
        self.adjust_success_probabilities()
        self.determine_hop_count()
        self.initialize_node_resources()
        self.initialize_edge_channels()

    def initialize_network(self):
        self.graph = Topology.initialize_topology(self.num_nodes, self.average_degree,self.json_file)
        Topology.adjust_success_probabilities(self.graph, self.target_Ep)
        self.hm = Topology.determine_hop_count(self.graph, self.num_nodes)
        self.initial_qubits = Topology.initialize_node_resources(self.graph)
        Topology.initialize_edge_channels(self.graph)

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
        for node in self.graph.nodes():
            local_state = {}
            visited = {node}
            queue = [(node, 0)]
            while queue:
                current, depth = queue.pop(0)
                if depth == 0:
                    pass
                else:
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

    def select_path(self, s, d, routing_metric):
        if routing_metric == "CR":
            return self.select_path_cr(s, d)
        elif routing_metric == "SumDist":
            return self.select_path_sumdist(s, d)
        elif routing_metric == "BotCap":
            return self.select_path_botcap(s, d)
        else:
            raise ValueError(f"Unknown routing metric: {routing_metric}")

    def select_path_cr(self, s, d):
        try:
            path = nx.shortest_path(self.graph, source=s, target=d)
            return path
        except nx.NetworkXNoPath:
            return None

    def select_path_sumdist(self, s, d):
        try:
            path = nx.shortest_path(self.graph, source=s, target=d, weight='sumdist')
            return path
        except nx.NetworkXNoPath:
            return None

    def select_path_botcap(self, s, d):
        try:
            path = nx.shortest_path(self.graph, source=s, target=d, weight='botcap')
            return path
        except nx.NetworkXNoPath:
            return None

    def calculate_path_reliability(self, path: List[int]) -> float:
        """Calculate the reliability of a path as the product of link reliabilities"""
        reliability = 1.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            # Get the link reliability from the graph edge attributes
            if self.graph.has_edge(u, v):
                reliability *= self.graph[u][v]['success_prob']
            elif self.graph.has_edge(v, u):
                reliability *= self.graph[v][u]['success_prob']
            else:
                # If link doesn't exist, return 0 reliability
                return 0.0
        return reliability 