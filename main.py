from dataclasses import dataclass
from typing import List, Optional, Type, Protocol
import numpy as np
import matplotlib.pyplot as plt
from ext_plotting import plot_EXT_vs_h
from simulators import QCASTSimulator, QPASSSimulator
import json
from config import SimulationConfig, SimulationType, SimulationFactory, SIMULATION_TYPES, DEFAULT_CONFIG, JSON_CONFIG

class Simulator(Protocol):
    """Protocol defining the interface for all simulators"""
    def simulate(self) -> List[float]:
        """Run the simulation and return throughput results"""
        ...

@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""
    num_nodes: int = 50
    target_Ep: float = 0.6
    q: float = 0.9
    link_state_range: int = 5
    average_degree: int = 5
    num_requests: int = 10
    num_slots: int = 30
    num_topologies: int = 5
    use_json_topology: bool = False

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

def plot_cdf(data: List[float], label: str, linewidth: int = 2) -> None:
    """Helper function to plot CDF for a single dataset"""
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, linewidth=linewidth, label=label)

def plot_combined_comparison(results: dict[str, List[float]]) -> None:
    """Plot CDF comparison of different simulation results"""
    plt.figure(figsize=(10, 6))
    
    # Plot each dataset
    for sim_type, data in results.items():
        plot_cdf(data, sim_type)
    
    plt.title("Aggregated CDF of Throughput (ebits per slot) over 10 Networks")
    plt.xlabel("Throughput (ebits per slot)")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True)
    plt.show()

def run_simulations(
    sim_types: List[SimulationType],
    config: SimulationConfig
) -> dict[str, List[float]]:
    """Run all simulations with given configuration"""
    print(f"\n=== Running {'JSON' if config.use_json_topology else 'Random'} Topology Simulation ===")
    
    factory = SimulationFactory(config)
    results = {}
    
    for sim_type in sim_types:
        results[sim_type.display_name] = run_simulation_for_topologies(
            sim_type, factory
        )
    
    return results

def save_plot_data_to_json(data: dict[str, List[float]], filename: str = "plot_data.json") -> None:
    """Save plot data to a JSON file"""
    plot_data = {}
    for label, values in data.items():
        sorted_data = np.sort(values)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plot_data[label] = {
            "x": sorted_data.tolist(),
            "y": cdf.tolist()
        }
    
    with open(filename, 'w') as f:
        json.dump(plot_data, f, indent=4)
    print(f"Plot data saved to {filename}")

def load_plot_data_from_json(filename: str = "plot_data.json") -> dict[str, dict]:
    """Load plot data from a JSON file"""
    with open(filename, 'r') as f:
        plot_data = json.load(f)
    return plot_data

def plot_from_json(filename: str = "plot_data.json", linewidth: int = 2) -> None:
    """Plot CDF comparison from JSON data"""
    plot_data = load_plot_data_from_json(filename)
    
    plt.figure(figsize=(10, 6))
    
    # Plot each dataset
    for label, data in plot_data.items():
        plt.plot(data["x"], data["y"], linewidth=linewidth, label=label)
    
    plt.title("Aggregated CDF of Throughput (ebits per slot) over 10 Networks")
    plt.xlabel("Throughput (ebits per slot)")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Plot EXT vs Hop Count
    print("\n=== Plotting EXT vs. Hop Count Graphs ===")
    plot_EXT_vs_h(p_values=[0.9, 0.6], q=0.9, widths=[1, 2, 3], h_range=range(1, 11))
    
    # Run simulations with random topology
    random_results = run_simulations(SIMULATION_TYPES, DEFAULT_CONFIG)
    save_plot_data_to_json(random_results, "random_topology_plot.json")
    plot_from_json("random_topology_plot.json")
    
    # Run simulations with JSON topology
    json_results = run_simulations(SIMULATION_TYPES, JSON_CONFIG)
    save_plot_data_to_json(json_results, "json_topology_plot.json")
    plot_from_json("json_topology_plot.json")