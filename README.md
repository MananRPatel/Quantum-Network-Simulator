# Quantum Network Simulation Framework

A framework for simulating quantum network protocols including Q-CAST and Q-PASS.

## Installation

```bash
git clone git@github.com:quantum-qphase-developers/QPhase.git
pip install numpy networkx matplotlib
```

## Quick Start



### Custom Configuration
```python
config = SimulationConfig(
    num_nodes=100,        # Number of nodes in network
    target_Ep=0.6,        # Target entanglement probability
    q=0.9,               # Quantum memory quality
    link_state_range=3,   # Link state exchange range
    average_degree=6,     # Average node degree
    num_requests=10,      # Number of S-D pairs
    num_slots=50         # Time slots for simulation
)
results = run_simulations(SIMULATION_TYPES, config)
```

## Available Simulators

1. **Q-CAST (EXT)**
   - Uses Extended Dijkstra's Algorithm
   - Implements XOR-based recovery
   - Optimized for quantum memory efficiency

2. **Q-CAST Enhanced**
   - Advanced path selection
   - Adaptive recovery strategies
   - Historical path optimization

3. **Q-PASS Variants**
   - CR (Channel Reliability)
   - SumDist (Sum of distances)
   - BotCap (Bottleneck Capacity)

## Adding a New Simulator

To add your own simulator class to the framework:

1. **Create Your Simulator Class**
```python
from quantum_network import QuantumNetwork

class MyCustomSimulator(QuantumNetwork):
    def simulate(self) -> Dict[str, List[float]]:
        """Implement your simulation logic here"""
        # Your implementation
        return {
            'throughput': throughput_list,
            'success_rate': success_rates,
            'path_reliability': reliabilities
        }
```

2. **Add to config.py**
```python
# In config.py
from simulators import MyCustomSimulator

# Add to SIMULATION_TYPES list
SIMULATION_TYPES = [
    # ... existing simulators ...
    SimulationType(
        name="MY_CUSTOM",
        simulator_class=MyCustomSimulator,
        routing_metric="YOUR_METRIC",  # e.g., "EXT", "CR", "SumDist"
        display_name="My Custom Simulator"
    )
]
```

3. **Required Methods**
Your simulator class must implement:
- `simulate()`: Main simulation method
- `attempt_entanglement(path)`: Entanglement attempt logic
- `attempt_entanglement_with_recovery(path, s, d)`: Recovery mechanism

4. **Return Format**
The `simulate()` method should return a dictionary with metrics:
```python
{
    'throughput': List[float],      # Entanglements per slot
    'success_rate': List[float],    # Success rates
    'path_reliability': List[float] # Path reliabilities
}
```

## Project Structure

```
├── main.py                  # Main entry point
├── quantum_network.py       # Core network implementation
├── simulators.py           # Simulation implementations
├── topology.py             # Topology management
├── config.py               # Configuration settings
├── path_selection.py       # Path selection algorithms
├── recovery_strategies.py  # Recovery implementations
├── ext_plotting.py        # External plotting utilities
```
