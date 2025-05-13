# Quantum Network Simulation Framework

A comprehensive simulation framework for quantum networks that implements various quantum communication protocols, routing strategies, and entanglement distribution mechanisms.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Usage](#usage)
- [Configuration](#configuration)
- [Simulation Metrics](#simulation-metrics)
- [Advanced Features](#advanced-features)
- [Contributing](#contributing)

## Overview

This framework provides a sophisticated simulation environment for quantum networks, focusing on:
- Entanglement distribution
- Quantum routing protocols
- Resource management
- Network topology optimization
- Performance analysis

## Features

### Network Configuration
- Configurable network size (number of nodes)
- Adjustable quantum memory (qubits per node)
- Customizable channel capacities
- Flexible topology generation
- Link state management

### Routing Protocols
1. **Path Selection Algorithms**
   - Extended Dijkstra's Algorithm
   - QCAST Path Selection
   - Multiple Candidate Paths
   - Dynamic Route Updates

2. **Routing Metrics**
   - EXT (Expected number of Transmission attempts)
   - SumDist (Sum of distances)
   - CR (Channel Reliability)
   - BotCap (Bottleneck Capacity)

### Quantum Operations
- Entanglement Generation
- Entanglement Swapping
- Resource Reservation
- Memory Management
- Channel State Tracking

### Recovery Mechanisms
- XOR-based Recovery
- Segment-based Recovery
- Dynamic Resource Reallocation
- Failed Path Recovery

## Installation

```bash
# Clone the repository
git clone [repository-url]

# Install required dependencies
pip install numpy networkx matplotlib
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
├── waxman_graph.py        # Waxman graph generation
└── various JSON files     # Topology and plot data storage
```

## Core Components

### 1. Quantum Network (`quantum_network.py`)
- Network initialization
- Resource management
- Entanglement operations
- Path selection
- Link state exchange

### 2. Simulators (`simulators.py`)
- QCAST Simulator
- QPASS Simulator
- Performance metrics
- Throughput calculation

### 3. Topology Management (`topology.py`)
- Random topology generation
- JSON topology loading
- Network structure modification
- Link probability adjustment

### 4. Configuration (`config.py`)
- Simulation parameters
- Network settings
- Protocol configurations
- Performance thresholds

## Usage

### Basic Simulation

```python
from config import SimulationConfig, SIMULATION_TYPES
from main import run_simulations

# Configure simulation parameters
config = SimulationConfig(
    num_nodes=100,
    target_Ep=0.6,
    q=0.9,
    link_state_range=3,
    average_degree=6,
    num_requests=10,
    num_slots=50,
    num_topologies=3
)

# Run simulation
results = run_simulations(SIMULATION_TYPES, config)
```

### Custom Topology

```python
# Load custom topology from JSON
config = SimulationConfig(
    use_json_topology=True,
    num_nodes=100,
    # ... other parameters
)

results = run_simulations(SIMULATION_TYPES, config)
```

### Visualization

```python
from main import plot_combined_comparison

# Plot results
plot_combined_comparison(results)
```

## Configuration

### Network Parameters
```python
SimulationConfig(
    num_nodes=100,        # Number of network nodes
    target_Ep=0.6,        # Target entanglement probability
    q=0.9,               # Quantum memory quality
    link_state_range=3,   # Link state exchange range
    average_degree=6,     # Average node degree
    num_requests=10,      # Number of connection requests
    num_slots=50,        # Time slots for simulation
    num_topologies=3     # Number of topologies to simulate
)
```

### Node Resources
- Qubit capacity: 10-14 qubits per node
- Channel width: 3-7 channels per edge
- Dynamic resource allocation

## Simulation Metrics

### Performance Measurements
1. **Throughput**
   - Successful entanglement rate
   - Resource utilization
   - Network capacity

2. **Quality Metrics**
   - Entanglement fidelity
   - Success probability
   - Path reliability

3. **Resource Efficiency**
   - Memory utilization
   - Channel occupation
   - Request satisfaction rate

### Analysis Tools
- CDF (Cumulative Distribution Function) plotting
- Performance comparison
- Resource usage statistics
- Network state visualization

## Advanced Features

### 1. Link State Management
- Dynamic state updates
- Multi-hop information exchange
- Reliability tracking

### 2. Path Recovery
- Alternative path selection
- Resource reallocation
- Failure handling

### 3. Resource Optimization
- Dynamic memory allocation
- Channel scheduling
- Request prioritization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Specify License]

## Contact

[Specify Contact Information] 