import random
from quantum_network import QuantumNetwork
from path_selection import PathSelection
from recovery_strategies import RecoveryStrategies
from typing import List, Dict, Tuple
import numpy as np
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
        path_reliability = []
        recovery_success = []
        self.deferred_requests = []
        
        for slot in range(self.num_slots):
            self.reset_resources_for_new_slot()
            sd_pairs = self.generate_sd_pairs()
            current_sd = sd_pairs + self.deferred_requests
            selected_paths = PathSelection.qpass_path_selection(self, current_sd, self.routing_metric)
            
            successful_entanglements = 0
            recovery_attempts = 0
            recovery_successes = 0
            total_path_reliability = 0
            served_sd = set()
            
            for sd, path in selected_paths.items():
                s, d = sd
                path_reliability = self.calculate_path_reliability(path)
                total_path_reliability += path_reliability
                
                if self.attempt_entanglement(path):
                    successful_entanglements += 1
                    served_sd.add(sd)
                else:
                    recovery_attempts += 1
                    if self.attempt_entanglement_with_recovery(path, s, d):
                        successful_entanglements += 1
                        recovery_successes += 1
                        served_sd.add(sd)
            
            slot_throughput.append(successful_entanglements)
            success_rate = len(served_sd) / len(current_sd) * 100
            success_rates.append(success_rate)
            avg_reliability = total_path_reliability / len(selected_paths) if selected_paths else 0
            path_reliability.append(avg_reliability)
            recovery_rate = recovery_successes / recovery_attempts * 100 if recovery_attempts > 0 else 0
            recovery_success.append(recovery_rate)
            
            print(f"\n[Time Slot {slot}] Summary:")
            print(f"  • Successful Entanglements: {successful_entanglements}")
            print(f"  • Success Rate: {success_rate:.2f}%")
            print(f"  • Recovery Success Rate: {recovery_rate:.2f}%")
            
            self.deferred_requests = [sd for sd in current_sd if sd not in served_sd]
        
        print("\n" + "="*80)
        print("Q-PASS SIMULATION COMPLETED")
        print("="*80)
        print("Final Statistics:")
        print(f"  • Average Throughput: {np.mean(slot_throughput):.2f} EPRs/slot")
        print(f"  • Average Success Rate: {np.mean(success_rates):.2f}%")
        print(f"  • Average Path Reliability: {np.mean(path_reliability):.4f}")
        print(f"  • Average Recovery Success: {np.mean(recovery_success):.2f}%")
        print("="*80)
        
        return {
            'throughput': slot_throughput,
            'success_rate': success_rates,
            'path_reliability': path_reliability,
            'recovery_success': recovery_success
        }

class QPASSRSimulator(QuantumNetwork):
    def simulate(self) -> Dict[str, List[float]]:
        """Run Q-PASS/R simulation (recovery-free)"""
        print("\n" + "="*80)
        print("Q-PASS/R SIMULATION INITIATED")
        print("="*80)
        
        slot_throughput = []
        self.deferred_requests = []
        for slot in range(self.num_slots):
            self.reset_resources_for_new_slot()
            sd_pairs = self.generate_sd_pairs()
            current_sd = sd_pairs + self.deferred_requests
            selected_paths = PathSelection.qpass_path_selection(self, current_sd, self.routing_metric)
            served = set()
            successful = 0
            
            for sd, path in selected_paths.items():
                if self.attempt_entanglement(path):
                    successful += 1
                    served.add(sd)
            
            slot_throughput.append(successful)
            print(f"\n[Time Slot {slot}] Summary:")
            print(f"  • Successful Entanglements: {successful}")
            
            self.deferred_requests = [sd for sd in current_sd if sd not in served]
        
        print("\n" + "="*80)
        print("Q-PASS/R SIMULATION COMPLETED")
        print("="*80)
        print("Final Statistics:")
        print(f"  • Average Throughput: {np.mean(slot_throughput):.2f} EPRs/slot")
        print("="*80)
        
        return {'throughput': slot_throughput}

class QCASTSimulator(QuantumNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_lock = asyncio.Lock()
        self.sleep_completion_times = {}

    def attempt_entanglement_with_recovery(self, path: List[int], s: int, d: int) -> bool:
        """Attempt entanglement with XOR-based recovery"""
        return RecoveryStrategies.xor_based_recovery(self, path, s, d)

    async def _wait_for_turn(self, slot: int) -> None:
        """Wait until it's this slot's turn based on sleep completion time"""
        while True:
            earliest_slot = min(self.sleep_completion_times.items(), key=lambda x: x[1])[0]
            if earliest_slot == slot:
                break
            await asyncio.sleep(0.1)

    async def _process_entanglement_attempts(
        self, 
        selected_paths: Dict[Tuple[int, int], List[int]]
    ) -> Tuple[int, int, int, float, set]:
        """Process all entanglement attempts for a slot"""
        successful_entanglements = 0
        recovery_attempts = 0
        recovery_successes = 0
        total_path_reliability = 0
        served = set()

        for sd, path in selected_paths.items():
            s, d = sd
            path_reliability = self.calculate_path_reliability(path)
            total_path_reliability += path_reliability

            if self.attempt_entanglement(path):
                successful_entanglements += 1
                served.add(sd)
            else:
                recovery_attempts += 1
                if self.attempt_entanglement_with_recovery(path, s, d):
                    successful_entanglements += 1
                    recovery_successes += 1
                    served.add(sd)

        return (successful_entanglements, recovery_attempts, recovery_successes,
                total_path_reliability, served)

    def _print_slot_summary(
        self, 
        slot: int, 
        successful_entanglements: int,
        success_rate: float,
        recovery_rate: float,
        waiting_time: float
    ) -> None:
        """Print summary statistics for a slot"""
        print(f"\n[Time Slot {slot}] Summary:")
        print(f"  • Successful Entanglements: {successful_entanglements}")
        print(f"  • Success Rate: {success_rate:.2f}%")
        print(f"  • Recovery Success Rate: {recovery_rate:.2f}%")
        print(f"  • Waiting Time: {waiting_time:.4f}s")

    async def process_slot(self, slot: int) -> Dict[str, List[float]]:
        """Process a single time slot with concurrency and global lock mechanism"""
        # Initialize slot with random sleep
        sleep_time = random.uniform(1, self.num_slots * 10)
        print(f"\n[Time Slot {slot}] Starting with sleep time: {sleep_time:.2f} seconds")
        
        # Record sleep times
        sleep_start = time.time()
        await asyncio.sleep(sleep_time)
        sleep_completion = time.time()
        self.sleep_completion_times[slot] = sleep_completion
        
        # Wait for turn based on sleep completion
        await self._wait_for_turn(slot)
        
        try:
            # Initialize slot resources and generate requests
            self.reset_resources_for_new_slot()
            sd_pairs = self.generate_sd_pairs()
            current_sd = sd_pairs + self.deferred_requests

            # Record creation time for SD pairs
            sd_creation_time = time.time()

            # Acquire lock and process slot
            await self.global_lock.acquire()
            print(f"\n[Time Slot {slot}] Acquired global lock after sleep completion")
            
            selected_paths = PathSelection.qcast_path_selection(self, current_sd)
            
            
            # Randomly perform link state exchange
            if random.randint(0, 9) == 0:
                self.link_state_exchange()

            
            # Start path selection and record waiting time
            path_selection_start = time.time()
            waiting_time = path_selection_start - sd_creation_time
            
            # Process all entanglement attempts
            (successful_entanglements, recovery_attempts, recovery_successes,
             total_path_reliability, served) = await self._process_entanglement_attempts(
                selected_paths
            )
            
            # Calculate metrics
            slot_throughput = successful_entanglements
            success_rate = len(served) / len(current_sd) * 100
            avg_reliability = total_path_reliability / len(selected_paths) if selected_paths else 0
            recovery_rate = recovery_successes / recovery_attempts * 100 if recovery_attempts > 0 else 0
            
            # Print slot summary
            self._print_slot_summary(
                slot, successful_entanglements, success_rate,
                recovery_rate, waiting_time
            )
            
            # Update deferred requests
            self.deferred_requests = [sd for sd in current_sd if sd not in served]
            
            return {
                'throughput': slot_throughput,
                'success_rate': success_rate,
                'path_reliability': avg_reliability,
                'recovery_success': recovery_rate,
                'waiting_time': waiting_time
            }
            
        finally:
            # Release lock and cleanup
            self.global_lock.release()
            print(f"\n[Time Slot {slot}] Released global lock")
            del self.sleep_completion_times[slot]

    async def simulate_async(self) -> Dict[str, List[float]]:
        """Run Q-CAST simulation with concurrency and global lock mechanism"""
        # Print simulation configuration
        print("\n" + "="*80)
        print("Q-CAST SIMULATION INITIATED (Concurrent Mode with Global Lock)")
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

        # Run all slots concurrently
        tasks = [self.process_slot(slot) for slot in range(self.num_slots)]
        results = await asyncio.gather(*tasks)
        
        # Aggregate results
        metrics = {
            'throughput': [],
            'success_rate': [],
            'path_reliability': [],
            'recovery_success': [],
            'waiting_times': []
        }
        
        for result in results:
            metrics['throughput'].append(result['throughput'])
            metrics['success_rate'].append(result['success_rate'])
            metrics['path_reliability'].append(result['path_reliability'])
            metrics['recovery_success'].append(result['recovery_success'])
            metrics['waiting_times'].append(result['waiting_time'])
        
        # Print final statistics
        print("\n" + "="*80)
        print("Q-CAST SIMULATION COMPLETED")
        print("="*80)
        print("Final Statistics:")
        print(f"  • Average Throughput: {np.mean(metrics['throughput']):.2f} EPRs/slot")
        print(f"  • Average Success Rate: {np.mean(metrics['success_rate']):.2f}%")
        print(f"  • Average Path Reliability: {np.mean(metrics['path_reliability']):.4f}")
        print(f"  • Average Recovery Success: {np.mean(metrics['recovery_success']):.2f}%")
        print(f"  • Average Waiting Time: {np.mean(metrics['waiting_times']):.4f}s")
        print("="*80)
        
        return metrics

    def simulate(self) -> Dict[str, List[float]]:
        """Run Q-CAST simulation (wrapper for async simulation)"""
        return asyncio.run(self.simulate_async())

class QCASTPipeLineSimulator(QuantumNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_lock = asyncio.Lock()
        self.sleep_completion_times = {}

    def attempt_entanglement_with_recovery(self, path: List[int], s: int, d: int) -> bool:
        """Attempt entanglement with XOR-based recovery"""
        return RecoveryStrategies.xor_based_recovery(self, path, s, d)

    async def _wait_for_turn(self, slot: int) -> None:
        """Wait until it's this slot's turn based on sleep completion time"""
        while True:
            earliest_slot = min(self.sleep_completion_times.items(), key=lambda x: x[1])[0]
            if earliest_slot == slot:
                break
            await asyncio.sleep(0.1)

    async def _process_entanglement_attempts(
        self, 
        selected_paths: Dict[Tuple[int, int], List[int]]
    ) -> Tuple[int, int, int, float, set]:
        """Process all entanglement attempts for a slot"""
        successful_entanglements = 0
        recovery_attempts = 0
        recovery_successes = 0
        total_path_reliability = 0
        served = set()

        for sd, path in selected_paths.items():
            s, d = sd
            path_reliability = self.calculate_path_reliability(path)
            total_path_reliability += path_reliability

            if self.attempt_entanglement(path):
                successful_entanglements += 1
                served.add(sd)
            else:
                recovery_attempts += 1
                if self.attempt_entanglement_with_recovery(path, s, d):
                    successful_entanglements += 1
                    recovery_successes += 1
                    served.add(sd)

        return (successful_entanglements, recovery_attempts, recovery_successes,
                total_path_reliability, served)

    def _print_slot_summary(
        self, 
        slot: int, 
        successful_entanglements: int,
        success_rate: float,
        recovery_rate: float,
        waiting_time: float
    ) -> None:
        """Print summary statistics for a slot"""
        print(f"\n[Time Slot {slot}] Summary:")
        print(f"  • Successful Entanglements: {successful_entanglements}")
        print(f"  • Success Rate: {success_rate:.2f}%")
        print(f"  • Recovery Success Rate: {recovery_rate:.2f}%")
        print(f"  • Waiting Time: {waiting_time:.4f}s")

    async def process_slot(self, slot: int) -> Dict[str, List[float]]:
        """Process a single time slot with concurrency and global lock mechanism"""
        # Initialize slot with random sleep
        sleep_time = random.uniform(1, self.num_slots * 10)
        print(f"\n[Time Slot {slot}] Starting with sleep time: {sleep_time:.2f} seconds")
        
        # Record sleep times
        sleep_start = time.time()
        await asyncio.sleep(sleep_time)
        sleep_completion = time.time()
        self.sleep_completion_times[slot] = sleep_completion
        
        # Wait for turn based on sleep completion
        await self._wait_for_turn(slot)
        
        try:
            # Initialize slot resources and generate requests
            self.reset_resources_for_new_slot()
            sd_pairs = self.generate_sd_pairs()
            current_sd = sd_pairs + self.deferred_requests

            # Record creation time for SD pairs
            sd_creation_time = time.time()

            

            selected_paths = PathSelection.qcast_path_selection(self, current_sd)

            # Start path selection and record waiting time
            path_selection_start = time.time()
            waiting_time = path_selection_start - sd_creation_time


            # Acquire lock and process slot
            await self.global_lock.acquire()
            print(f"\n[Time Slot {slot}] Acquired global lock after sleep completion")
            
            # Randomly perform link state exchange
            if random.randint(0, 9) == 0:
                self.link_state_exchange()
            


            # Process all entanglement attempts
            (successful_entanglements, recovery_attempts, recovery_successes,
             total_path_reliability, served) = await self._process_entanglement_attempts(
                selected_paths
            )
            
            # Calculate metrics
            slot_throughput = successful_entanglements
            success_rate = len(served) / len(current_sd) * 100
            avg_reliability = total_path_reliability / len(selected_paths) if selected_paths else 0
            recovery_rate = recovery_successes / recovery_attempts * 100 if recovery_attempts > 0 else 0
            
            # Print slot summary
            self._print_slot_summary(
                slot, successful_entanglements, success_rate,
                recovery_rate, waiting_time
            )
            
            # Update deferred requests
            self.deferred_requests = [sd for sd in current_sd if sd not in served]
            
            return {
                'throughput': slot_throughput,
                'success_rate': success_rate,
                'path_reliability': avg_reliability,
                'recovery_success': recovery_rate,
                'waiting_time': waiting_time
            }
            
        finally:
            # Release lock and cleanup
            self.global_lock.release()
            print(f"\n[Time Slot {slot}] Released global lock")
            del self.sleep_completion_times[slot]

    async def simulate_async(self) -> Dict[str, List[float]]:
        """Run Q-CAST simulation with concurrency and global lock mechanism"""
        # Print simulation configuration
        print("\n" + "="*80)
        print("Q-CAST SIMULATION INITIATED (Concurrent Mode with Global Lock)")
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

        # Run all slots concurrently
        tasks = [self.process_slot(slot) for slot in range(self.num_slots)]
        results = await asyncio.gather(*tasks)
        
        # Aggregate results
        metrics = {
            'throughput': [],
            'success_rate': [],
            'path_reliability': [],
            'recovery_success': [],
            'waiting_times': []
        }
        
        for result in results:
            metrics['throughput'].append(result['throughput'])
            metrics['success_rate'].append(result['success_rate'])
            metrics['path_reliability'].append(result['path_reliability'])
            metrics['recovery_success'].append(result['recovery_success'])
            metrics['waiting_times'].append(result['waiting_time'])
        
        # Print final statistics
        print("\n" + "="*80)
        print("Q-CAST SIMULATION COMPLETED")
        print("="*80)
        print("Final Statistics:")
        print(f"  • Average Throughput: {np.mean(metrics['throughput']):.2f} EPRs/slot")
        print(f"  • Average Success Rate: {np.mean(metrics['success_rate']):.2f}%")
        print(f"  • Average Path Reliability: {np.mean(metrics['path_reliability']):.4f}")
        print(f"  • Average Recovery Success: {np.mean(metrics['recovery_success']):.2f}%")
        print(f"  • Average Waiting Time: {np.mean(metrics['waiting_times']):.4f}s")
        print("="*80)
        
        return metrics

    def simulate(self) -> Dict[str, List[float]]:
        """Run Q-CAST simulation (wrapper for async simulation)"""
        return asyncio.run(self.simulate_async())

class QCASTRSimulator(QuantumNetwork):
    def simulate(self) -> Dict[str, List[float]]:
        """Run Q-CAST/R simulation (recovery-free)"""
        print("\n" + "="*80)
        print("Q-CAST/R SIMULATION INITIATED")
        print("="*80)
        
        slot_throughput = []
        self.deferred_requests = []
        for slot in range(self.num_slots):
            self.reset_resources_for_new_slot()
            sd_pairs = self.generate_sd_pairs()
            current_sd = sd_pairs + self.deferred_requests
            selected_paths = PathSelection.qcast_path_selection(self, current_sd)
            served = set()
            successful = 0
            
            if random.randint(0, 9) == 0:
                self.link_state_exchange()
                
            for sd, path in selected_paths.items():
                if self.attempt_entanglement(path):
                    successful += 1
                    served.add(sd)
            
            slot_throughput.append(successful)
            print(f"\n[Time Slot {slot}] Summary:")
            print(f"  • Successful Entanglements: {successful}")
            
            self.deferred_requests = [sd for sd in current_sd if sd not in served]
        
        print("\n" + "="*80)
        print("Q-CAST/R SIMULATION COMPLETED")
        print("="*80)
        print("Final Statistics:")
        print(f"  • Average Throughput: {np.mean(slot_throughput):.2f} EPRs/slot")
        print("="*80)
        
        return {'throughput': slot_throughput}

class QCASTEnhancedSimulator(QuantumNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_history = {}
        self.entanglement_stats = {}
        self.max_history_size = 10
        self.max_stats_age = 100
        self.current_slot = 0

    def update_entanglement_stats(self, path: List[int], success: bool) -> None:
        current_time = self.current_slot
        self.entanglement_stats = {
            k: v for k, v in self.entanglement_stats.items() 
            if current_time - v.get('last_updated', 0) <= self.max_stats_age
        }
        
        if len(self.entanglement_stats) > 1000:
            self.entanglement_stats = {
                k: v for k, v in self.entanglement_stats.items() 
                if v['total'] > 5
            }
        
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
        if segment in self.entanglement_stats:
            stats = self.entanglement_stats[segment]
            if stats['total'] > 0:
                return stats['success'] / stats['total']
        return 0.5

    def calculate_path_metrics(self, path: List[int]) -> Dict[str, float]:
        length = len(path) - 1
        reliability = 1.0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            segment = (u, v)
            reliability *= self.get_segment_reliability(segment)
        
        return {
            'length': length,
            'reliability': reliability
        }

    def enhanced_path_selection(self, sd_pairs: List[Tuple[int, int]]) -> Dict[Tuple[int, int], List[int]]:
        selected_paths = {}
        candidate_paths = []

        for sd in sd_pairs:
            s, d = sd
            if sd in self.path_history:
                historical_path = self.path_history[sd]
                if self.reserve_resources(historical_path):
                    selected_paths[sd] = historical_path
                    continue

            path, ext_metric = self.extended_dijkstra(s, d)
            if path is not None:
                metrics = self.calculate_path_metrics(path)
                enhanced_metric = ext_metric * (1 + 0.5 * metrics['reliability'])
                candidate_paths.append((sd, path, enhanced_metric, metrics))

        candidate_paths.sort(key=lambda x: x[2], reverse=True)

        for sd, path, metric, metrics in candidate_paths:
            if sd in selected_paths:
                continue
            if self.reserve_resources(path):
                selected_paths[sd] = path

        return selected_paths

    def enhanced_entanglement(self, path: List[int], s: int, d: int) -> Tuple[bool, bool]:
        if self.attempt_entanglement(path):
            self.update_entanglement_stats(path, True)
            return True, False

        metrics = self.calculate_path_metrics(path)
        
        if metrics['length'] <= 3 or metrics['reliability'] > 0.7:
            if RecoveryStrategies.xor_based_recovery(self, path, s, d):
                self.update_entanglement_stats(path, True)
                return True, True
        else:
            if RecoveryStrategies.segmentation_based_recovery(self, path, s, d):
                self.update_entanglement_stats(path, True)
                return True, True

        self.update_entanglement_stats(path, False)
        return False, False

    def simulate(self) -> Dict[str, List[float]]:
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
        path_reliability = []
        recovery_success = []
        path_lengths = []
        self.deferred_requests = []

        for slot in range(self.num_slots):
            self.current_slot = slot
            self.reset_resources_for_new_slot()
            sd_pairs = self.generate_sd_pairs()
            current_sd = sd_pairs + self.deferred_requests
            
            selected_paths = self.enhanced_path_selection(current_sd)
            
            served = set()
            successful = 0
            recovery_attempts = 0
            recovery_successes = 0
            total_path_reliability = 0
            total_path_length = 0

            for sd, path in selected_paths.items():
                s, d = sd
                path_reliability = self.calculate_path_reliability(path)
                total_path_reliability += path_reliability
                total_path_length += len(path) - 1

                success, used_recovery = self.enhanced_entanglement(path, s, d)
                if success:
                    successful += 1
                    served.add(sd)
                    if len(self.path_history) >= self.max_history_size:
                        self.path_history.pop(next(iter(self.path_history)))
                    self.path_history[sd] = path
                    if used_recovery:
                        recovery_attempts += 1
                        recovery_successes += 1
                else:
                    recovery_attempts += 1

            slot_throughput.append(successful)
            success_rate = len(served) / len(current_sd) * 100
            success_rates.append(success_rate)
            avg_reliability = total_path_reliability / len(selected_paths) if selected_paths else 0
            path_reliability.append(avg_reliability)
            avg_path_length = total_path_length / len(selected_paths) if selected_paths else 0
            path_lengths.append(avg_path_length)
            recovery_rate = recovery_successes / recovery_attempts * 100 if recovery_attempts > 0 else 0
            recovery_success.append(recovery_rate)

            print(f"\n[Time Slot {slot}] Summary:")
            print(f"  • Successful Entanglements: {successful}")
            print(f"  • Success Rate: {success_rate:.2f}%")
            print(f"  • Average Path Length: {avg_path_length:.2f} hops")
            print(f"  • Recovery Success Rate: {recovery_rate:.2f}%")

            self.deferred_requests = [sd for sd in current_sd if sd not in served]

        print("\n" + "="*80)
        print("ENHANCED Q-CAST SIMULATION COMPLETED")
        print("="*80)
        print("Final Statistics:")
        print(f"  • Average Throughput: {np.mean(slot_throughput):.2f} EPRs/slot")
        print(f"  • Average Success Rate: {np.mean(success_rates):.2f}%")
        print(f"  • Average Path Reliability: {np.mean(path_reliability):.4f}")
        print(f"  • Average Path Length: {np.mean(path_lengths):.2f} hops")
        print(f"  • Average Recovery Success: {np.mean(recovery_success):.2f}%")
        print("="*80)

        return {
            'throughput': slot_throughput,
            'success_rate': success_rates,
            'path_reliability': path_reliability,
            'recovery_success': recovery_success,
            'path_length': path_lengths
        }

    def run_scalability_test(self, node_counts: List[int]) -> Dict[str, List[float]]:
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
            self.num_nodes = num_nodes
            self.reset_entanglements()
            self.reset_resources_for_new_slot()
            
            metrics = self.simulate()
            
            results['throughput'].append(np.mean(metrics['throughput']))
            results['success_rate'].append(np.mean(metrics['success_rate']))
            results['path_length'].append(np.mean(metrics['path_length']))
            results['recovery_overhead'].append(np.mean(metrics['recovery_success']))
            
            print(f"\nResults for {num_nodes} nodes:")
            print(f"  • Average Throughput: {results['throughput'][-1]:.2f} EPRs/slot")
            print(f"  • Average Success Rate: {results['success_rate'][-1]:.2f}%")
            print(f"  • Average Path Length: {results['path_length'][-1]:.2f} hops")
            print(f"  • Average Recovery Success: {results['recovery_overhead'][-1]:.2f}%")
        
        print("\n" + "="*80)
        print("SCALABILITY ANALYSIS COMPLETED")
        print("="*80)
        
        return results 