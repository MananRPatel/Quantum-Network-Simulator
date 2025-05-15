# Quantum Network Protocol Simulation - Comprehensive Documentation

## 1. `quantum_network.py`

### Class: QuantumNetwork
Base class implementing core quantum network functionality. This class serves as the foundation for all quantum network simulations, providing essential network operations and metrics calculations.

#### Variables:
- `num_nodes` (int): 
  - Number of nodes in the network
  - Range: 1 to infinity
  - Default: 100
  - Used for network initialization and topology creation

- `num_slots` (int):
  - Number of time slots for simulation
  - Range: 1 to infinity
  - Default: 1000
  - Determines simulation duration and time-based metrics

- `num_requests` (int):
  - Number of requests per time slot
  - Range: 1 to infinity
  - Default: 10
  - Controls simulation load and throughput calculation

- `link_state_range` (int):
  - Range for link state information exchange
  - Range: 1 to infinity
  - Default: 3
  - Affects routing decisions and path selection

- `routing_metric` (str):
  - Metric used for path selection
  - Values: "EXT", "SumDist", "CR", "BotCap"
  - Default: "EXT"
  - Determines path selection strategy

- `average_degree` (int):
  - Average node degree in the network
  - Range: 1 to num_nodes-1
  - Default: 6
  - Controls network connectivity

- `target_Ep` (float):
  - Target entanglement probability
  - Range: 0.0 to 1.0
  - Default: 0.6
  - Used for success probability calculations

- `q` (float):
  - Quantum channel quality parameter
  - Range: 0.0 to 1.0
  - Default: 0.9
  - Affects channel reliability

- `graph` (nx.Graph):
  - NetworkX graph representing the network
  - Type: Undirected graph
  - Contains nodes, edges, and their attributes
  - Used for all network operations

- `link_state` (dict):
  - Stores link state information for each node
  - Format: {node_id: {neighbor_id: state_info}}
  - Updated during simulation
  - Used for routing decisions

- `total_entangled_pairs` (int):
  - Count of successful entanglements
  - Incremented on successful entanglement
  - Used for throughput calculation

- `throughput_per_slot` (list):
  - Throughput values for each time slot
  - Length equals num_slots
  - Used for performance analysis

- `deferred_requests` (list):
  - Requests deferred to next time slot
  - Format: [(source, destination, path), ...]
  - Handles failed requests

- `hm` (dict):
  - Hop matrix for path calculations
  - Format: {(source, destination): hop_count}
  - Used for path metric calculations

#### Methods:

1. `__init__(self, num_nodes=100, num_slots=1000, num_requests=10, link_state_range=3, routing_metric="EXT", average_degree=6, target_Ep=0.6, q=0.9, json_file=None)`
   - Purpose: Initializes network parameters and topology
   - Parameters:
     - `num_nodes`: Number of nodes in network
     - `num_slots`: Simulation duration
     - `num_requests`: Requests per slot
     - `link_state_range`: Link state exchange range
     - `routing_metric`: Path selection metric
     - `average_degree`: Average node connections
     - `target_Ep`: Target entanglement probability
     - `q`: Channel quality
     - `json_file`: Optional topology file
   - Side Effects:
     - Creates network graph
     - Initializes link states
     - Sets up quantum channels

2. `initialize_network(self)`
   - Purpose: Sets up network graph and resources
   - Operations:
     - Creates network topology
     - Initializes success probabilities
     - Sets up quantum channels
     - Initializes link states
   - Returns: None
   - Side Effects:
     - Modifies graph structure
     - Initializes network resources

3. `initialize_edge_channels(self)`
   - Purpose: Creates quantum channels between nodes
   - Operations:
     - Creates channels for each edge
     - Sets channel width
     - Initializes channel states
   - Returns: None
   - Side Effects:
     - Adds channel attributes to edges

4. `compute_routing_metric(self, path)`
   - Purpose: Calculates path metric based on selected routing strategy
   - Parameters:
     - `path`: List of nodes representing the path
   - Returns: float
   - Metric Types:
     - EXT: Extended Throughput
     - SumDist: Sum of Distances
     - CR: Capacity Ratio
     - BotCap: Bottleneck Capacity

5. `calculate_path_reliability(self, path)`
   - Purpose: Computes path reliability as product of link reliabilities
   - Parameters:
     - `path`: List of nodes representing the path
   - Returns: float between 0 and 1
   - Calculation:
     - Product of success probabilities for each link
     - Returns 0 if any link doesn't exist

## 2. `simulators.py`

### Class: QPASSSimulator
Implements Q-PASS protocol with segmentation-based recovery. This simulator focuses on reliable quantum communication using path segmentation.

#### Methods:

1. `simulate(self)`
   - Purpose: Runs Q-PASS simulation
   - Operations:
     - Processes requests for each time slot
     - Attempts entanglement
     - Handles recovery if needed
   - Returns: dict with metrics:
     - throughput: float
     - success_rate: float
     - path_reliability: float
     - recovery_success: float

2. `attempt_entanglement_with_recovery(self, path, s, d)`
   - Purpose: Attempts entanglement with segmentation-based recovery
   - Parameters:
     - `path`: List of nodes
     - `s`: Source node
     - `d`: Destination node
   - Returns: bool indicating success
   - Recovery Strategy:
     - Segments path into smaller parts
     - Attempts recovery for each segment

### Class: QCASTSimulator
Implements Q-CAST protocol with XOR-based recovery. This simulator focuses on efficient quantum communication using XOR operations.

#### Methods:

1. `simulate(self)`
   - Purpose: Runs Q-CAST simulation
   - Operations:
     - Processes requests for each time slot
     - Attempts entanglement
     - Handles recovery if needed
   - Returns: dict with metrics:
     - throughput: float
     - success_rate: float
     - path_reliability: float
     - recovery_success: float

2. `attempt_entanglement_with_recovery(self, path, s, d)`
   - Purpose: Attempts entanglement with XOR-based recovery
   - Parameters:
     - `path`: List of nodes
     - `s`: Source node
     - `d`: Destination node
   - Returns: bool indicating success
   - Recovery Strategy:
     - Uses XOR operations for recovery
     - More efficient for certain network conditions

### Class: QCASTEnhancedSimulator
Enhanced Q-CAST with additional features for improved performance and analysis.

#### Variables:

- `path_history` (dict):
  - Stores successful paths
  - Format: {path_id: {path: [...], success_count: int}}
  - Used for path optimization

- `entanglement_stats` (dict):
  - Tracks entanglement statistics
  - Format: {path_id: {success: int, failure: int}}
  - Used for performance analysis

- `max_history_size` (int):
  - Maximum paths to remember
  - Default: 1000
  - Controls memory usage

- `max_stats_age` (int):
  - Maximum age of statistics
  - Default: 100
  - Controls statistics relevance

- `current_slot` (int):
  - Current simulation slot
  - Used for time-based operations

#### Methods:

1. `simulate(self)`
   - Purpose: Runs enhanced Q-CAST simulation
   - Operations:
     - Processes requests
     - Updates path history
     - Tracks statistics
   - Returns: dict with enhanced metrics

2. `run_scalability_test(self, node_counts)`
   - Purpose: Tests performance with varying node counts
   - Parameters:
     - `node_counts`: List of node counts to test
   - Returns: dict with scalability metrics

3. `enhanced_entanglement(self, path, s, d)`
   - Purpose: Smart recovery strategy selection
   - Parameters:
     - `path`: List of nodes
     - `s`: Source node
     - `d`: Destination node
   - Returns: (success, used_recovery) tuple
   - Strategy Selection:
     - Based on path characteristics
     - Uses historical data

## 3. `recovery_strategies.py`

### Class: RecoveryStrategies
Static methods for recovery strategies. This class provides different recovery mechanisms for quantum networks.

#### Methods:

1. `segmentation_based_recovery(network, path, s, d)`
   - Purpose: Implements Q-PASS recovery
   - Parameters:
     - `network`: QuantumNetwork instance
     - `path`: List of nodes
     - `s`: Source node
     - `d`: Destination node
   - Returns: bool indicating success
   - Strategy:
     - Breaks path into segments
     - Attempts recovery for each segment
     - Combines successful segments

2. `xor_based_recovery(network, path, s, d)`
   - Purpose: Implements Q-CAST recovery
   - Parameters:
     - `network`: QuantumNetwork instance
     - `path`: List of nodes
     - `s`: Source node
     - `d`: Destination node
   - Returns: bool indicating success
   - Strategy:
     - Uses XOR operations
     - More efficient for certain conditions
     - Handles multiple failures

## 4. `metrics_plotting.py`

### Functions:

1. `plot_time_based_metrics(all_results, save_path)`
   - Purpose: Creates 2x2 subplot of time-based metrics
   - Parameters:
     - `all_results`: dict of simulation results
     - `save_path`: path to save plot
   - Plots:
     - End-to-End Throughput
     - Success Rate
     - Path Reliability
     - Recovery Efficiency

2. `plot_scalability_metrics(enhanced_results, save_path)`
   - Purpose: Creates 2x2 subplot of scalability metrics
   - Parameters:
     - `enhanced_results`: dict of scalability results
     - `save_path`: path to save plot
   - Plots:
     - Throughput vs. Nodes
     - Success Rate vs. Nodes
     - Path Length vs. Nodes
     - Recovery Overhead vs. Nodes

3. `save_metrics_to_json(time_metrics, scalability_metrics, time_file, scalability_file)`
   - Purpose: Saves metrics to JSON files
   - Parameters:
     - `time_metrics`: dict of time-based metrics
     - `scalability_metrics`: dict of scalability metrics
     - `time_file`: path for time metrics
     - `scalability_file`: path for scalability metrics

4. `load_metrics_from_json(time_file, scalability_file)`
   - Purpose: Loads metrics from JSON files
   - Parameters:
     - `time_file`: path to time metrics
     - `scalability_file`: path to scalability metrics
   - Returns: tuple of (time_metrics, scalability_metrics)

## 5. `config.py`

### Class: SimulationConfig
Configuration data class for simulation parameters.

#### Variables:

- `num_nodes` (int):
  - Network size
  - Range: 1 to infinity
  - Default: 100

- `target_Ep` (float):
  - Target entanglement probability
  - Range: 0.0 to 1.0
  - Default: 0.6

- `q` (float):
  - Channel quality
  - Range: 0.0 to 1.0
  - Default: 0.9

- `link_state_range` (int):
  - Link state exchange range
  - Range: 1 to infinity
  - Default: 3

- `average_degree` (int):
  - Average node connections
  - Range: 1 to num_nodes-1
  - Default: 6

- `num_requests` (int):
  - Requests per slot
  - Range: 1 to infinity
  - Default: 10

- `num_slots` (int):
  - Simulation duration
  - Range: 1 to infinity
  - Default: 1000

- `use_json_topology` (bool):
  - Whether to use JSON topology
  - Default: False

### Class: SimulationType
Defines simulation types with their characteristics.

#### Variables:

- `name` (str):
  - Protocol name
  - Used for identification

- `simulator_class` (Type):
  - Simulator class
  - Used for instantiation

- `routing_metric` (str):
  - Path selection metric
  - Values: "EXT", "SumDist", "CR", "BotCap"

- `display_name` (str):
  - Display name for plots
  - Used in visualization

### Class: SimulationFactory
Creates simulator instances with proper configuration.

#### Methods:

1. `create_simulator(self, sim_type)`
   - Purpose: Creates simulator with proper configuration
   - Parameters:
     - `sim_type`: SimulationType instance
   - Returns: Simulator instance
   - Operations:
     - Creates simulator
     - Configures parameters
     - Initializes network

## 6. `topology.py`

### Class: Topology
Handles network topology operations.

#### Methods:

1. `initialize_topology(num_nodes, average_degree, json_file)`
   - Purpose: Creates network topology
   - Parameters:
     - `num_nodes`: Number of nodes
     - `average_degree`: Average node degree
     - `json_file`: Optional topology file
   - Returns: NetworkX graph
   - Operations:
     - Creates random topology
     - Sets edge attributes
     - Initializes resources

2. `load_topology_from_json(num_nodes, json_file)`
   - Purpose: Loads topology from JSON file
   - Parameters:
     - `num_nodes`: Number of nodes
     - `json_file`: Path to JSON file
   - Returns: NetworkX graph
   - Operations:
     - Reads JSON file
     - Creates graph
     - Sets attributes

3. `export_topology_to_json(graph, filename, beta, alpha)`
   - Purpose: Exports topology to JSON file
   - Parameters:
     - `graph`: NetworkX graph
     - `filename`: Output file path
     - `beta`: Beta parameter
     - `alpha`: Alpha parameter
   - Operations:
     - Converts graph to JSON
     - Saves to file

## 7. `main.py`

### Functions:

1. `run_simulation_for_topologies(sim_type, factory)`
   - Purpose: Runs simulation across multiple topologies
   - Parameters:
     - `sim_type`: SimulationType instance
     - `factory`: SimulationFactory instance
   - Returns: list of throughput values
   - Operations:
     - Creates topologies
     - Runs simulations
     - Collects results

2. `plot_cdf(data, label, linewidth)`
   - Purpose: Plots CDF for single dataset
   - Parameters:
     - `data`: List of values
     - `label`: Plot label
     - `linewidth`: Line width
   - Operations:
     - Calculates CDF
     - Creates plot

3. `plot_combined_comparison(results)`
   - Purpose: Plots CDF comparison of different protocols
   - Parameters:
     - `results`: dict of results
   - Operations:
     - Creates subplots
     - Plots CDFs
     - Adds labels

4. `run_simulations(sim_types, config)`
   - Purpose: Runs all simulations with given configuration
   - Parameters:
     - `sim_types`: List of SimulationType instances
     - `config`: SimulationConfig instance
   - Returns: dict of results
   - Operations:
     - Creates simulators
     - Runs simulations
     - Collects results

5. `save_plot_data_to_json(data, filename)`
   - Purpose: Saves plot data to JSON file
   - Parameters:
     - `data`: dict of plot data
     - `filename`: Output file path
   - Operations:
     - Converts data to JSON
     - Saves to file

6. `load_plot_data_from_json(filename)`
   - Purpose: Loads plot data from JSON file
   - Parameters:
     - `filename`: Input file path
   - Returns: dict of plot data
   - Operations:
     - Reads JSON file
     - Converts to data

7. `plot_from_json(filename, linewidth)`
   - Purpose: Plots CDF comparison from JSON data
   - Parameters:
     - `filename`: Input file path
     - `linewidth`: Line width
   - Operations:
     - Loads data
     - Creates plot

### Main Execution Flow:
1. Plot EXT vs. Hop Count
   - Creates initial plot
   - Shows relationship between EXT and hop count

2. Run time-based metrics analysis
   - Creates simulators
   - Runs simulations
   - Collects metrics

3. Generate and save plots
   - Creates time-based plots
   - Saves to files
   - Displays plots

4. Optional scalability analysis
   - Tests different node counts
   - Collects scalability metrics
   - Creates scalability plots

5. Save metrics to JSON files
   - Saves time-based metrics
   - Saves scalability metrics
   - Enables later analysis

## Dependencies
- numpy: Numerical computations
- matplotlib: Plotting and visualization
- networkx: Graph operations
- json: Data serialization
- typing: Type hints
- dataclasses: Data class support 