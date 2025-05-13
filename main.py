from dataclasses import dataclass
from typing import List, Optional, Type, Protocol
import numpy as np
from ext_plotting import (
    plot_EXT_vs_h,
    save_and_plot_metrics
)
from simulators import QCASTSimulator, QPASSSimulator
from config import SimulationConfig, SimulationType, SimulationFactory, SIMULATION_TYPES, DEFAULT_CONFIG

class Simulator(Protocol):
    """Protocol defining the interface for all simulators"""
    def simulate(self) -> List[float]:
        """Run the simulation and return throughput results"""
        ...

@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""
    def to_dict(self) -> dict:
        """Convert config to dictionary for compatibility"""
        return {
            "num_nodes": self.num_nodes,
            "target_Ep": self.target_Ep,
            "q": self.q,
            "link_state_range": self.link_state_range,
            "average_degree": self.average_degree,
            "num_requests": self.num_requests,
            "num_slots": self.num_slots,
            "num_topologies": self.num_topologies,
            "use_json_topology": self.use_json_topology
        }

@dataclass
class SimulationType:
    """Class representing a simulation type and its configuration"""
    name: str
    simulator_class: Type[Simulator]
    routing_metric: str
    display_name: str

class SimulationFactory:
    """Factory class to create simulator instances"""
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.params = config.to_dict()

    def create_simulator(self, sim_type: SimulationType) -> Simulator:
        """Create a simulator instance with the given configuration"""
        sim_args = {
            "num_nodes": self.params["num_nodes"],
            "num_slots": self.params["num_slots"],
            "num_requests": self.params["num_requests"],
            "link_state_range": self.params["link_state_range"],
            "routing_metric": sim_type.routing_metric,
            "average_degree": self.params["average_degree"],
            "target_Ep": self.params["target_Ep"],
            "q": self.params["q"]
        }
        
        return sim_type.simulator_class(**sim_args)

def run_simulation_for_topologies(
    sim_type: SimulationType,
    factory: SimulationFactory
) -> List[float]:
    """Run simulation across multiple topologies"""
    all_throughputs = []
    
    for topo in range(factory.config.num_topologies):
        print(f"\n====== Running {sim_type.display_name} on network topology {topo+1} ======")
        simulator = factory.create_simulator(sim_type)
        throughput = simulator.simulate()
        all_throughputs.extend(throughput)
        
    return all_throughputs

if __name__ == "__main__":
    # Plot EXT vs Hop Count
    print("\n=== Plotting EXT vs. Hop Count Graphs ===")
    plot_EXT_vs_h(p_values=[0.9, 0.6], q=0.9, widths=[1, 2, 3], h_range=range(1, 11))
    
    # Run simulations with random topology and collect metrics
    print("\n=== Running Time-based Metrics Analysis ===")
    time_metrics = {}
    for sim_type in SIMULATION_TYPES:
        print(f"\nRunning {sim_type.display_name} simulation...")
        simulator = SimulationFactory(DEFAULT_CONFIG).create_simulator(sim_type)
        time_metrics[sim_type.display_name] = simulator.simulate()
    
    # Save time metrics to JSON and plot
    save_and_plot_metrics(time_metrics, "time_metrics.json", "time_based_metrics.png")
    