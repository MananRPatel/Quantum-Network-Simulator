from ext_plotting import (
    plot_EXT_vs_h,
    save_and_plot_metrics
)
from config import SimulationFactory, SIMULATION_TYPES, DEFAULT_CONFIG


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
    