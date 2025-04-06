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