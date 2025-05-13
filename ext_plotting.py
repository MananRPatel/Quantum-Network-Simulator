import numpy as np
import matplotlib.pyplot as plt
from math import comb
from typing import Dict, List, Tuple
import json

def compute_EXT_given_parameters(W, h, p, q):
    if h < 1:
        return 0
    P = [[0.0]*(h+1) for _ in range(W+1)]
    for i in range(1, W+1):
        P[i][1] = comb(W, i) * (p ** i) * ((1 - p) ** (W - i))
    for k in range(2, h+1):
        for i in range(1, W+1):
            sum1 = sum(comb(W, l) * (p ** l) * ((1 - p) ** (W - l)) for l in range(i, W+1))
            sum2 = sum(P[l][k-1] for l in range(i+1, W+1))
            P[i][k] = P[i][k-1] * sum1 + (comb(W, i) * (p ** i) * ((1 - p) ** (W - i))) * sum2
    EXT_val = sum(i * P[i][h] for i in range(1, W+1))
    EXT_val *= (q ** (h - 1))
    return EXT_val

def plot_EXT_vs_h(p_values=[0.9, 0.6], q=0.9, widths=[1, 2, 3], h_range=range(1, 11)):
    """Plot EXT vs hop count for different parameters"""
    plt.figure(figsize=(10, 6))
    for p in p_values:
        for W in widths:
            ext_vals = []
            for h in h_range:
                ext = compute_EXT_given_parameters(W, h, p, q)
                ext_vals.append(ext)
            label = f"p={p}, W={W}"
            plt.plot(list(h_range), ext_vals, marker='o', linewidth=2, label=label)
            print(f"Computed EXT for p={p}, W={W}: {ext_vals}")
    plt.title("EXT vs. Hop Count for Different p and Widths")
    plt.xlabel("Hop Count (h)")
    plt.ylabel("Expected Throughput (EXT)")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_time_based_metrics(all_results: Dict[str, Dict[str, List[float]]], save_path: str = "time_based_metrics.png"):
    """Plot time-based metrics for all protocols"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: End-to-End Throughput
    for protocol, metrics in all_results.items():
        ax1.plot(metrics['throughput'], label=protocol)
    ax1.set_title('End-to-End Throughput')
    ax1.set_xlabel('Time Slot')
    ax1.set_ylabel('EPRs/slot')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Success Rate
    protocols = list(all_results.keys())
    success_rates = [np.mean(metrics['success_rate']) for metrics in all_results.values()]
    ax2.bar(protocols, success_rates)
    ax2.set_title('Success Rate of Routing Requests')
    ax2.set_xlabel('Protocol')
    ax2.set_ylabel('Success Rate (%)')
    ax2.grid(True)
    
    # Plot 3: Average Path Reliability
    for protocol, metrics in all_results.items():
        ax3.plot(metrics['path_reliability'], label=protocol)
    ax3.set_title('Average Path Reliability')
    ax3.set_xlabel('Time Slot')
    ax3.set_ylabel('Reliability Score')
    ax3.grid(True)
    ax3.legend()
    
    # Plot 4: Recovery Efficiency
    recovery_rates = [np.mean(metrics['recovery_success']) for metrics in all_results.values()]
    ax4.bar(protocols, recovery_rates)
    ax4.set_title('Recovery Efficiency')
    ax4.set_xlabel('Protocol')
    ax4.set_ylabel('Recovery Success Rate (%)')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    throughput_data = {name: metrics['throughput'] for name, metrics in all_results.items()}

    cdf_throughput(throughput_data, "throughput_cdf.json", True, False, True, 2)



def cdf_throughput(data: dict[str, List[float]] = None, 
                    filename: str = "plot_data.json",
                    save: bool = False,
                    load: bool = False,
                    plot: bool = False,
                    linewidth: int = 2) -> dict[str, dict]:
    """
    Unified function to handle plot data operations (save, load, plot)
    
    Args:
        data: Input data dictionary with labels as keys and lists of values
        filename: JSON file to save/load data from
        save: Whether to save data to JSON
        load: Whether to load data from JSON
        plot: Whether to plot the data
        linewidth: Line width for plotting
        
    Returns:
        The plot data dictionary
    """
    plot_data = {}
    
    # Save data if provided and save flag is True
    if data is not None and save:
        for label, values in data.items():
            sorted_data = np.sort(values)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            plot_data[label] = {
                "x": sorted_data.tolist(),
                "y": cdf.tolist()
            }
    # Plot data if plot flag is True
    if plot:
        plt.figure(figsize=(10, 6))
        for label, data in plot_data.items():
            plt.plot(data["x"], data["y"], linewidth=linewidth, label=label)
        plt.title("Aggregated CDF of Throughput (ebits per slot) over 10 Networks")
        plt.xlabel("Throughput (ebits per slot)")
        plt.ylabel("CDF")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return plot_data

def save_metrics_to_json(time_metrics: Dict[str, Dict[str, List[float]]], 
                        scalability_metrics: Dict[str, List[float]],
                        time_file: str = "time_metrics.json",
                        scalability_file: str = "scalability_metrics.json") -> None:
    """Save metrics to JSON files"""
    with open(time_file, 'w') as f:
        json.dump(time_metrics, f, indent=4)
    
    if scalability_metrics:
        with open(scalability_file, 'w') as f:
            json.dump(scalability_metrics, f, indent=4)

def load_metrics_from_json(time_file: str = "time_metrics.json",
                          scalability_file: str = "scalability_metrics.json") -> Tuple[Dict, Dict]:
    """Load metrics from JSON files"""
    with open(time_file, 'r') as f:
        time_metrics = json.load(f)
    
    scalability_metrics = None
    if scalability_file:
        try:
            with open(scalability_file, 'r') as f:
                scalability_metrics = json.load(f)
        except FileNotFoundError:
            pass
    
    return time_metrics, scalability_metrics

def save_and_plot_metrics(time_metrics: Dict[str, Dict[str, List[float]]], 
                         time_file: str = "time_metrics.json",
                         plot_file: str = "time_based_metrics.png") -> None:
    """Save metrics to JSON and immediately plot them"""
    # Save metrics to JSON
    save_metrics_to_json(time_metrics, None, time_file)
    print(f"Time metrics saved to {time_file}")
    
    # Load metrics and plot
    loaded_metrics, _ = load_metrics_from_json(time_file)
    plot_time_based_metrics(loaded_metrics, plot_file)
    print(f"Plot saved to {plot_file}")

def save_plot_data_to_json(data: dict[str, List[float]], filename: str = "plot_data.json") -> None:
    """Save plot data to JSON file"""
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