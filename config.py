from dataclasses import dataclass
from typing import List, Optional, Type, Protocol
from simulators import QCASTSimulator, QPASSSimulator, QCASTEnhancedSimulator

class Simulator(Protocol):
    """Protocol defining the interface for all simulators"""
    def simulate(self) -> List[float]:
        """Run the simulation and return throughput results"""
        ...
@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""
    num_nodes: int = 100
    target_Ep: float = 0.6
    q: float = 0.9
    link_state_range: int = 2
    average_degree: int = 6
    num_requests: int = 50
    num_slots: int = 100
    use_json_topology: bool = False  # Whether to use JSON topology
    json_file: Optional[str] = None  # Path to JSON topology file

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
            "use_json_topology": self.use_json_topology,
            "json_file": self.json_file
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
        
        if self.params["use_json_topology"] and self.params["json_file"]:
            sim_args["json_file"] = self.params["json_file"]
            
        return sim_type.simulator_class(**sim_args)

# Define available simulation types
SIMULATION_TYPES = [
    SimulationType(
        name="QCAST",
        simulator_class=QCASTSimulator,
        routing_metric="EXT",
        display_name="Q-CAST (EXT)"
    ),
    SimulationType(
        name="QCAST_ENHANCED",
        simulator_class=QCASTEnhancedSimulator,
        routing_metric="EXT",
        display_name="Q-CAST (Enhanced)"
    ),
    SimulationType(
        name="QPASS_CR",
        simulator_class=QPASSSimulator,
        routing_metric="CR",
        display_name="Q-PASS (CR)"
    ),
    SimulationType(
        name="QPASS_SumDist",
        simulator_class=QPASSSimulator,
        routing_metric="SumDist",
        display_name="Q-PASS (SumDist)"
    ),
    SimulationType(
        name="QPASS_BotCap",
        simulator_class=QPASSSimulator,
        routing_metric="BotCap",
        display_name="Q-PASS (BotCap)"
    )
]

# Default configuration for random topology
DEFAULT_CONFIG = SimulationConfig()

# Configuration for JSON topology
JSON_CONFIG = SimulationConfig(
    use_json_topology=True,
    json_file="test_topology.json"
) 