import random
from quantum_network import QuantumNetwork
from path_selection import PathSelection
from recovery_strategies import RecoveryStrategies
from typing import List, Dict, Tuple
import numpy as np

class QPASSSimulator(QuantumNetwork):
    def attempt_entanglement_with_recovery(self, path: List[int], s: int, d: int) -> bool:
        """Attempt entanglement with segmentation-based recovery"""
        return RecoveryStrategies.segmentation_based_recovery(self, path, s, d)

    def simulate(self) -> Dict[str, List[float]]:
        """Run Q-PASS simulation"""
        print("\n" + "="*80)
        print("Q-PASS SIMULATION INITIATED")
        print("="*80)
        print("Configuration:")
        print(f"  • Number of Nodes: {self.num_nodes}")
        print(f"  • Number of Time Slots: {self.num_slots}")
        print(f"  • Number of Requests per Slot: {self.num_requests}")
        print(f"  • Link State Range: {self.link_state_range}")
        print(f"  • Average Node Degree: {self.average_degree}")
        print(f"  • Target Entanglement Probability: {self.target_Ep}")
        print(f"  • Quantum Channel Quality (q): {self.q}")
        print("="*80)

        slot_throughput = []
        success_rates = []
        path_reliabilities = []
        recovery_success = []
        self.deferred_requests = []
        
        for slot in range(self.num_slots):
            print(f"\n[Time Slot {slot}] Q-PASS Protocol Execution")
            print("-"*50)
            self.reset_resources_for_new_slot()
            sd_pairs = self.generate_sd_pairs()
            current_sd = sd_pairs + self.deferred_requests
            print(f"Active Source-Destination Pairs: {len(current_sd)}")
            print(f"  • New Requests: {len(sd_pairs)}")
            print(f"  • Deferred Requests: {len(self.deferred_requests)}")
            
            selected_paths = PathSelection.qpass_path_selection(self, current_sd, self.routing_metric)
            print("\nPath Selection Results:")
            print(f"  • Total Paths Selected: {len(selected_paths)}")
            
            successful_entanglements = 0
            recovery_attempts = 0
            recovery_successes = 0
            total_path_reliability = 0
            served_sd = set()
            
            print("\nEntanglement Phase:")
            for sd, path in selected_paths.items():
                s, d = sd
                path_reliability = self.calculate_path_reliability(path)
                total_path_reliability += path_reliability
                
                print(f"\nProcessing S-D pair {sd}:")
                print(f"  • Selected Path: {path}")
                print(f"  • Path Reliability: {path_reliability:.4f}")
                
                if self.attempt_entanglement(path):
                    print(f"  ✓ Direct Entanglement Successful")
                    successful_entanglements += 1
                    served_sd.add(sd)
                else:
                    recovery_attempts += 1
                    print(f"  × Direct Entanglement Failed")
                    if self.attempt_entanglement_with_recovery(path, s, d):
                        print(f"  ✓ Recovery Successful")
                        successful_entanglements += 1
                        recovery_successes += 1
                        served_sd.add(sd)
                    else:
                        print(f"  × Recovery Failed")
            
            slot_throughput.append(successful_entanglements)
            success_rate = len(served_sd) / len(current_sd) * 100
            success_rates.append(success_rate)
            avg_reliability = total_path_reliability / len(selected_paths) if selected_paths else 0
            path_reliabilities.append(avg_reliability)
            recovery_rate = recovery_successes / recovery_attempts * 100 if recovery_attempts > 0 else 0
            recovery_success.append(recovery_rate)
            
            print("\nSlot Summary:")
            print(f"  • Successful Entanglements: {successful_entanglements}")
            print(f"  • Success Rate: {success_rate:.2f}%")
            print(f"  • Average Path Reliability: {avg_reliability:.4f}")
            print(f"  • Recovery Success Rate: {recovery_rate:.2f}%")
            
            self.deferred_requests = [sd for sd in current_sd if sd not in served_sd]
        
        print("\n" + "="*80)
        print("Q-PASS SIMULATION COMPLETED")
        print("="*80)
        print("Final Statistics:")
        print(f"  • Average Throughput: {np.mean(slot_throughput):.2f} EPRs/slot")
        print(f"  • Average Success Rate: {np.mean(success_rates):.2f}%")
        print(f"  • Average Path Reliability: {np.mean(path_reliabilities):.4f}")
        print(f"  • Average Recovery Success: {np.mean(recovery_success):.2f}%")
        print("="*80)
        
        return {
            'throughput': slot_throughput,
            'success_rate': success_rates,
            'path_reliability': path_reliabilities,
            'recovery_success': recovery_success
        }

class QPASSRSimulator(QuantumNetwork):
    def simulate(self) -> Dict[str, List[float]]:
        """Run Q-PASS/R simulation (recovery-free)"""
        print("\n--- Running Q-PASS/R Simulation (recovery-free) ---")
        slot_throughput = []
        self.deferred_requests = []
        for slot in range(self.num_slots):
            print(f"\n=== Time Slot {slot} (Q-PASS/R) ===")
            self.reset_resources_for_new_slot()
            sd_pairs = self.generate_sd_pairs()
            current_sd = sd_pairs + self.deferred_requests
            print(f"Current S-D pairs: {current_sd}")
            selected_paths = PathSelection.qpass_path_selection(self, current_sd, self.routing_metric)
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
        return {'throughput': slot_throughput}

class QCASTSimulator(QuantumNetwork):
    def attempt_entanglement_with_recovery(self, path: List[int], s: int, d: int) -> bool:
        """Attempt entanglement with XOR-based recovery"""
        return RecoveryStrategies.xor_based_recovery(self, path, s, d)

    def simulate(self) -> Dict[str, List[float]]:
        """Run Q-CAST simulation"""
        print("\n" + "="*80)
        print("Q-CAST SIMULATION INITIATED")
        print("="*80)
        print("Configuration:")
        print(f"  • Number of Nodes: {self.num_nodes}")
        print(f"  • Number of Time Slots: {self.num_slots}")
        print(f"  • Number of Requests per Slot: {self.num_requests}")
        print(f"  • Link State Range: {self.link_state_range}")
        print(f"  • Average Node Degree: {self.average_degree}")
        print(f"  • Target Entanglement Probability: {self.target_Ep}")
        print(f"  • Quantum Channel Quality (q): {self.q}")
        print("="*80)

        slot_throughput = []
        success_rates = []
        path_reliabilities = []
        recovery_success = []
        self.deferred_requests = []
        
        for slot in range(self.num_slots):
            print(f"\n[Time Slot {slot}] Q-CAST Protocol Execution")
            print("-"*50)
            self.reset_resources_for_new_slot()
            sd_pairs = self.generate_sd_pairs()
            current_sd = sd_pairs + self.deferred_requests
            print(f"Active Source-Destination Pairs: {len(current_sd)}")
            print(f"  • New Requests: {len(sd_pairs)}")
            print(f"  • Deferred Requests: {len(self.deferred_requests)}")
            
            selected_paths = PathSelection.qcast_path_selection(self, current_sd)
            print("\nPath Selection Results:")
            print(f"  • Total Paths Selected: {len(selected_paths)}")
            
            successful_entanglements = 0
            recovery_attempts = 0
            recovery_successes = 0
            total_path_reliability = 0
            served = set()
            
            if random.randint(0, 9) == 0:
                print("\nPerforming Link State Exchange...")
                self.link_state_exchange()
            
            print("\nEntanglement Phase:")
            for sd, path in selected_paths.items():
                s, d = sd
                path_reliability = self.calculate_path_reliability(path)
                total_path_reliability += path_reliability
                
                print(f"\nProcessing S-D pair {sd}:")
                print(f"  • Selected Path: {path}")
                print(f"  • Path Reliability: {path_reliability:.4f}")
                
                if self.attempt_entanglement(path):
                    print(f"  ✓ Direct Entanglement Successful")
                    successful_entanglements += 1
                    served.add(sd)
                else:
                    recovery_attempts += 1
                    print(f"  × Direct Entanglement Failed")
                    if self.attempt_entanglement_with_recovery(path, s, d):
                        print(f"  ✓ XOR-based Recovery Successful")
                        successful_entanglements += 1
                        recovery_successes += 1
                        served.add(sd)
                    else:
                        print(f"  × Recovery Failed")
            
            slot_throughput.append(successful_entanglements)
            success_rate = len(served) / len(current_sd) * 100
            success_rates.append(success_rate)
            avg_reliability = total_path_reliability / len(selected_paths) if selected_paths else 0
            path_reliabilities.append(avg_reliability)
            recovery_rate = recovery_successes / recovery_attempts * 100 if recovery_attempts > 0 else 0
            recovery_success.append(recovery_rate)
            
            print("\nSlot Summary:")
            print(f"  • Successful Entanglements: {successful_entanglements}")
            print(f"  • Success Rate: {success_rate:.2f}%")
            print(f"  • Average Path Reliability: {avg_reliability:.4f}")
            print(f"  • Recovery Success Rate: {recovery_rate:.2f}%")
            
            self.deferred_requests = [sd for sd in current_sd if sd not in served]
        
        print("\n" + "="*80)
        print("Q-CAST SIMULATION COMPLETED")
        print("="*80)
        print("Final Statistics:")
        print(f"  • Average Throughput: {np.mean(slot_throughput):.2f} EPRs/slot")
        print(f"  • Average Success Rate: {np.mean(success_rates):.2f}%")
        print(f"  • Average Path Reliability: {np.mean(path_reliabilities):.4f}")
        print(f"  • Average Recovery Success: {np.mean(recovery_success):.2f}%")
        print("="*80)
        
        return {
            'throughput': slot_throughput,
            'success_rate': success_rates,
            'path_reliability': path_reliabilities,
            'recovery_success': recovery_success
        }

class QCASTRSimulator(QuantumNetwork):
    def simulate(self) -> Dict[str, List[float]]:
        """Run Q-CAST/R simulation (recovery-free)"""
        print("\n--- Running Q-CAST/R Simulation (recovery-free) ---")
        slot_throughput = []
        self.deferred_requests = []
        for slot in range(self.num_slots):
            print(f"\n=== Time Slot {slot} (Q-CAST/R) ===")
            self.reset_resources_for_new_slot()
            sd_pairs = self.generate_sd_pairs()
            current_sd = sd_pairs + self.deferred_requests
            print(f"Current S-D pairs: {current_sd}")
            selected_paths = PathSelection.qcast_path_selection(self, current_sd)
            served = set()
            successful = 0
            if random.randint(0, 9) == 0:
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
        return {'throughput': slot_throughput}

class QCASTEnhancedSimulator(QuantumNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_history = {}  # Store only last 10 successful paths
        self.entanglement_stats = {}  # Track only recent statistics
        self.max_history_size = 10  # Maximum number of paths to remember
        self.max_stats_age = 100  # Maximum age of statistics in slots
        self.current_slot = 0  # Track current slot for age management

    def update_entanglement_stats(self, path: List[int], success: bool) -> None:
        """Update entanglement success statistics for path segments with cleanup"""
        # Cleanup old statistics based on age
        current_time = self.current_slot
        self.entanglement_stats = {
            k: v for k, v in self.entanglement_stats.items() 
            if current_time - v.get('last_updated', 0) <= self.max_stats_age
        }
        
        # Cleanup based on number of segments
        if len(self.entanglement_stats) > 1000:  # If too many segments tracked
            # Keep only segments with significant history
            self.entanglement_stats = {
                k: v for k, v in self.entanglement_stats.items() 
                if v['total'] > 5  # Keep only segments with more than 5 attempts
            }
        
        # Update statistics for current path
        for i in range(len(path) - 1):
            segment = (path[i], path[i+1])
            if segment not in self.entanglement_stats:
                self.entanglement_stats[segment] = {
                    'success': 0, 
                    'total': 0,
                    'last_updated': current_time
                }
            self.entanglement_stats[segment]['total'] += 1
            if success:
                self.entanglement_stats[segment]['success'] += 1
            self.entanglement_stats[segment]['last_updated'] = current_time

    def get_segment_reliability(self, segment: Tuple[int, int]) -> float:
        """Calculate reliability score for a path segment"""
        if segment in self.entanglement_stats:
            stats = self.entanglement_stats[segment]
            if stats['total'] > 0:
                return stats['success'] / stats['total']
        return 0.5  # Default reliability for unknown segments

    def calculate_path_metrics(self, path: List[int]) -> Dict[str, float]:
        """Calculate essential path metrics for selection"""
        # Basic metrics
        length = len(path) - 1
        reliability = 1.0
        
        # Calculate reliability
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            segment = (u, v)
            reliability *= self.get_segment_reliability(segment)
        
        return {
            'length': length,
            'reliability': reliability
        }

    def enhanced_path_selection(self, sd_pairs: List[Tuple[int, int]]) -> Dict[Tuple[int, int], List[int]]:
        """Enhanced Phase 2: Path Selection with essential metrics"""
        print("Starting Enhanced Q-CAST path selection (Phase 2)")
        selected_paths = {}
        candidate_paths = []

        # First pass: Collect and evaluate paths
        for sd in sd_pairs:
            s, d = sd
            # Check path history first (limited to recent successful paths)
            if sd in self.path_history:
                historical_path = self.path_history[sd]
                if self.reserve_resources(historical_path):
                    selected_paths[sd] = historical_path
                    print(f"S-D pair {sd}: Using successful historical path {historical_path}")
                    continue

            # Find new paths if no historical path available
            path, ext_metric = self.extended_dijkstra(s, d)
            if path is not None:
                metrics = self.calculate_path_metrics(path)
                # Calculate enhanced metric combining EXT and reliability
                enhanced_metric = ext_metric * (1 + 0.5 * metrics['reliability'])  # 50% reliability bonus
                candidate_paths.append((sd, path, enhanced_metric, metrics))

        # Sort paths by enhanced metric
        candidate_paths.sort(key=lambda x: x[2], reverse=True)

        # Second pass: Reserve resources for best paths
        for sd, path, metric, metrics in candidate_paths:
            if sd in selected_paths:
                continue
            if self.reserve_resources(path):
                selected_paths[sd] = path
                print(f"S-D pair {sd}: Reserved path {path} with enhanced metric = {metric:.4f}")
                print(f"Path metrics: {metrics}")

        return selected_paths

    def enhanced_entanglement(self, path: List[int], s: int, d: int) -> Tuple[bool, bool]:
        """Enhanced Phase 4: Entanglement with smart recovery strategy"""
        # Attempt direct entanglement first
        if self.attempt_entanglement(path):
            self.update_entanglement_stats(path, True)
            return True, False  # Success, No recovery needed

        # Calculate path characteristics
        metrics = self.calculate_path_metrics(path)
        
        # Choose recovery strategy based on path characteristics
        if metrics['length'] <= 3 or metrics['reliability'] > 0.7:
            # Try XOR-based recovery for short or reliable paths
            if RecoveryStrategies.xor_based_recovery(self, path, s, d):
                self.update_entanglement_stats(path, True)
                return True, True  # Success with recovery
        else:
            # Try segmentation for longer paths
            if RecoveryStrategies.segmentation_based_recovery(self, path, s, d):
                self.update_entanglement_stats(path, True)
                return True, True  # Success with recovery

        # If all attempts fail, update statistics and return False
        self.update_entanglement_stats(path, False)
        return False, False  # Failed, No recovery success

    def simulate(self) -> Dict[str, List[float]]:
        """Run Enhanced Q-CAST simulation"""
        print("\n" + "="*80)
        print("ENHANCED Q-CAST SIMULATION INITIATED")
        print("="*80)
        print("Configuration:")
        print(f"  • Number of Nodes: {self.num_nodes}")
        print(f"  • Number of Time Slots: {self.num_slots}")
        print(f"  • Number of Requests per Slot: {self.num_requests}")
        print(f"  • Link State Range: {self.link_state_range}")
        print(f"  • Average Node Degree: {self.average_degree}")
        print(f"  • Target Entanglement Probability: {self.target_Ep}")
        print(f"  • Quantum Channel Quality (q): {self.q}")
        print(f"  • Path History Size: {self.max_history_size}")
        print(f"  • Statistics Age Limit: {self.max_stats_age} slots")
        print("="*80)

        slot_throughput = []
        success_rates = []
        path_reliabilities = []
        recovery_success = []
        path_lengths = []
        self.deferred_requests = []

        for slot in range(self.num_slots):
            self.current_slot = slot
            print(f"\n[Time Slot {slot}] Enhanced Q-CAST Protocol Execution")
            print("-"*50)
            self.reset_resources_for_new_slot()
            sd_pairs = self.generate_sd_pairs()
            current_sd = sd_pairs + self.deferred_requests
            print(f"Active Source-Destination Pairs: {len(current_sd)}")
            print(f"  • New Requests: {len(sd_pairs)}")
            print(f"  • Deferred Requests: {len(self.deferred_requests)}")
            
            selected_paths = self.enhanced_path_selection(current_sd)
            print("\nPath Selection Results:")
            print(f"  • Total Paths Selected: {len(selected_paths)}")
            print(f"  • Historical Paths Used: {sum(1 for sd in selected_paths if sd in self.path_history)}")
            
            served = set()
            successful = 0
            recovery_attempts = 0
            recovery_successes = 0
            total_path_reliability = 0
            total_path_length = 0

            print("\nEntanglement Phase:")
            for sd, path in selected_paths.items():
                s, d = sd
                path_reliability = self.calculate_path_reliability(path)
                total_path_reliability += path_reliability
                total_path_length += len(path) - 1

                print(f"\nProcessing S-D pair {sd}:")
                print(f"  • Selected Path: {path}")
                print(f"  • Path Reliability: {path_reliability:.4f}")
                print(f"  • Path Length: {len(path)-1} hops")

                success, used_recovery = self.enhanced_entanglement(path, s, d)
                if success:
                    print(f"  ✓ Enhanced Entanglement Successful")
                    successful += 1
                    served.add(sd)
                    if len(self.path_history) >= self.max_history_size:
                        self.path_history.pop(next(iter(self.path_history)))
                    self.path_history[sd] = path
                    if used_recovery:
                        recovery_attempts += 1
                        recovery_successes += 1
                else:
                    print(f"  × Entanglement Failed")
                    recovery_attempts += 1

            slot_throughput.append(successful)
            success_rate = len(served) / len(current_sd) * 100
            success_rates.append(success_rate)
            avg_reliability = total_path_reliability / len(selected_paths) if selected_paths else 0
            path_reliabilities.append(avg_reliability)
            avg_path_length = total_path_length / len(selected_paths) if selected_paths else 0
            path_lengths.append(avg_path_length)
            recovery_rate = recovery_successes / recovery_attempts * 100 if recovery_attempts > 0 else 0
            recovery_success.append(recovery_rate)

            print("\nSlot Summary:")
            print(f"  • Successful Entanglements: {successful}")
            print(f"  • Success Rate: {success_rate:.2f}%")
            print(f"  • Average Path Reliability: {avg_reliability:.4f}")
            print(f"  • Average Path Length: {avg_path_length:.2f} hops")
            print(f"  • Recovery Success Rate: {recovery_rate:.2f}%")

            self.deferred_requests = [sd for sd in current_sd if sd not in served]

        print("\n" + "="*80)
        print("ENHANCED Q-CAST SIMULATION COMPLETED")
        print("="*80)
        print("Final Statistics:")
        print(f"  • Average Throughput: {np.mean(slot_throughput):.2f} EPRs/slot")
        print(f"  • Average Success Rate: {np.mean(success_rates):.2f}%")
        print(f"  • Average Path Reliability: {np.mean(path_reliabilities):.4f}")
        print(f"  • Average Path Length: {np.mean(path_lengths):.2f} hops")
        print(f"  • Average Recovery Success: {np.mean(recovery_success):.2f}%")
        print("="*80)

        return {
            'throughput': slot_throughput,
            'success_rate': success_rates,
            'path_reliability': path_reliabilities,
            'recovery_success': recovery_success,
            'path_length': path_lengths
        }

    def run_scalability_test(self, node_counts: List[int]) -> Dict[str, List[float]]:
        """Run scalability tests with different node counts"""
        print("\n" + "="*80)
        print("SCALABILITY ANALYSIS INITIATED")
        print("="*80)
        print("Testing with node counts:", node_counts)
        
        results = {
            'node_counts': node_counts,
            'throughput': [],
            'success_rate': [],
            'path_length': [],
            'recovery_overhead': []
        }
        
        for num_nodes in node_counts:
            print(f"\nTesting with {num_nodes} nodes...")
            self.num_nodes = num_nodes
            self.reset_entanglements()
            self.reset_resources_for_new_slot()
            
            metrics = self.simulate()
            
            results['throughput'].append(np.mean(metrics['throughput']))
            results['success_rate'].append(np.mean(metrics['success_rate']))
            results['path_length'].append(np.mean(metrics['path_length']))
            results['recovery_overhead'].append(np.mean(metrics['recovery_success']))
            
            print(f"Results for {num_nodes} nodes:")
            print(f"  • Average Throughput: {results['throughput'][-1]:.2f} EPRs/slot")
            print(f"  • Average Success Rate: {results['success_rate'][-1]:.2f}%")
            print(f"  • Average Path Length: {results['path_length'][-1]:.2f} hops")
            print(f"  • Average Recovery Success: {results['recovery_overhead'][-1]:.2f}%")
        
        print("\n" + "="*80)
        print("SCALABILITY ANALYSIS COMPLETED")
        print("="*80)
        
        return results 