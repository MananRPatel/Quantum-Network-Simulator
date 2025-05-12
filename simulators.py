import random
from quantum_network import QuantumNetwork
from path_selection import PathSelection
from recovery_strategies import RecoveryStrategies

class QPASSSimulator(QuantumNetwork):
    def attempt_entanglement_with_recovery(self, path, s, d):
        return RecoveryStrategies.segmentation_based_recovery(self, path, s, d)

    def simulate(self):
        print("\n--- Running Q-PASS Simulation (with segmentation-based recovery) ---")
        slot_throughput = []
        self.deferred_requests = []
        for slot in range(self.num_slots):
            print(f"\n=== Time Slot {slot} (Q-PASS) ===")
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
        return slot_throughput

class QCASTSimulator(QuantumNetwork):
    def qcast_path_selection(self, sd_pairs):
        return PathSelection.qcast_path_selection(self, sd_pairs)

    def attempt_entanglement_with_recovery(self, path, s, d):
        return RecoveryStrategies.xor_based_recovery(self, path, s, d)

    def simulate(self):
        print("\n--- Running Q-CAST Simulation (with XOR-based recovery) ---")
        slot_throughput = []
        self.deferred_requests = []
        for slot in range(self.num_slots):
            print(f"\n=== Time Slot {slot} (Q-CAST) ===")
            self.reset_resources_for_new_slot()
            sd_pairs = self.generate_sd_pairs()
            current_sd = sd_pairs + self.deferred_requests
            print(f"Current S-D pairs: {current_sd}")
            selected_paths = self.qcast_path_selection(current_sd)
            served = set()
            successful = 0
            if random.randint(0, 9) == 0:
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

class QCASTRSimulator(QuantumNetwork):
    def qcast_path_selection(self, sd_pairs):
        return PathSelection.qcast_path_selection(self, sd_pairs)

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
        return slot_throughput

class QCASTEnhancedSimulator(QuantumNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_history = {}  # Store only last 10 successful paths
        self.entanglement_stats = {}  # Track only recent statistics
        self.max_history_size = 10  # Maximum number of paths to remember
        self.max_stats_age = 100  # Maximum age of statistics in slots
        self.current_slot = 0  # Track current slot for age management

    def update_entanglement_stats(self, path, success):
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

    def get_segment_reliability(self, segment):
        """Calculate reliability score for a path segment"""
        if segment in self.entanglement_stats:
            stats = self.entanglement_stats[segment]
            if stats['total'] > 0:
                return stats['success'] / stats['total']
        return 0.5  # Default reliability for unknown segments

    def calculate_path_metrics(self, path):
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

    def enhanced_path_selection(self, sd_pairs):
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

    def enhanced_entanglement(self, path, s, d):
        """Enhanced Phase 4: Entanglement with smart recovery strategy"""
        # Attempt direct entanglement first
        if self.attempt_entanglement(path):
            self.update_entanglement_stats(path, True)
            return True

        # Calculate path characteristics
        metrics = self.calculate_path_metrics(path)
        
        # Choose recovery strategy based on path characteristics
        if metrics['length'] <= 3 or metrics['reliability'] > 0.7:
            # Try XOR-based recovery for short or reliable paths
            if RecoveryStrategies.xor_based_recovery(self, path, s, d):
                self.update_entanglement_stats(path, True)
                return True
        else:
            # Try segmentation for longer paths
            if RecoveryStrategies.segmentation_based_recovery(self, path, s, d):
                self.update_entanglement_stats(path, True)
                return True

        # If all attempts fail, update statistics and return False
        self.update_entanglement_stats(path, False)
        return False

    def simulate(self):
        print("\n--- Running Enhanced Q-CAST Simulation ---")
        slot_throughput = []
        self.deferred_requests = []

        for slot in range(self.num_slots):
            self.current_slot = slot  # Update current slot
            print(f"\n=== Time Slot {slot} (Enhanced Q-CAST) ===")
            self.reset_resources_for_new_slot()
            sd_pairs = self.generate_sd_pairs()
            current_sd = sd_pairs + self.deferred_requests
            print(f"Current S-D pairs: {current_sd}")

            # Phase 2: Enhanced Path Selection
            selected_paths = self.enhanced_path_selection(current_sd)
            served = set()
            successful = 0

            # Phase 4: Enhanced Entanglement
            for sd, path in selected_paths.items():
                s, d = sd
                if self.enhanced_entanglement(path, s, d):
                    print(f"Entanglement succeeded for {sd}")
                    successful += 1
                    served.add(sd)
                    # Update path history (limited size)
                    if len(self.path_history) >= self.max_history_size:
                        # Remove oldest entry
                        self.path_history.pop(next(iter(self.path_history)))
                    self.path_history[sd] = path
                else:
                    print(f"Entanglement failed for {sd}")

            slot_throughput.append(successful)
            print(f"Time Slot {slot} throughput: {successful}")
            self.deferred_requests = [sd for sd in current_sd if sd not in served]

        return slot_throughput 