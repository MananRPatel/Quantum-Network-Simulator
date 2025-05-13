import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json

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

def plot_scalability_metrics(enhanced_results: Dict[str, List[float]], save_path: str = "scalability_metrics.png"):
    """Plot scalability metrics for QCAST-Enhanced"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Throughput vs Number of Nodes
    ax1.plot(enhanced_results['node_counts'], enhanced_results['throughput'], 'o-')
    ax1.set_title('Throughput vs Number of Nodes')
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Average EPRs/slot')
    ax1.grid(True)
    
    # Plot 2: Success Rate vs Number of Nodes
    ax2.plot(enhanced_results['node_counts'], enhanced_results['success_rate'], 'o-')
    ax2.set_title('Success Rate vs Number of Nodes')
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Success Rate (%)')
    ax2.grid(True)
    
    # Plot 3: Average Path Length vs Number of Nodes
    ax3.plot(enhanced_results['node_counts'], enhanced_results['path_length'], 'o-')
    ax3.set_title('Average Path Length vs Number of Nodes')
    ax3.set_xlabel('Number of Nodes')
    ax3.set_ylabel('Average Hop Count')
    ax3.grid(True)
    
    # Plot 4: Recovery Overhead vs Number of Nodes
    ax4.plot(enhanced_results['node_counts'], enhanced_results['recovery_overhead'], 'o-')
    ax4.set_title('Recovery Overhead vs Number of Nodes')
    ax4.set_xlabel('Number of Nodes')
    ax4.set_ylabel('Recovery Success Rate (%)')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_metrics_to_json(time_metrics: Dict[str, Dict[str, List[float]]], 
                        scalability_metrics: Dict[str, List[float]],
                        time_file: str = "time_metrics.json",
                        scalability_file: str = "scalability_metrics.json"):
    """Save metrics to JSON files"""
    with open(time_file, 'w') as f:
        json.dump(time_metrics, f, indent=4)
    
    with open(scalability_file, 'w') as f:
        json.dump(scalability_metrics, f, indent=4)

def load_metrics_from_json(time_file: str = "time_metrics.json",
                          scalability_file: str = "scalability_metrics.json"):
    """Load metrics from JSON files"""
    with open(time_file, 'r') as f:
        time_metrics = json.load(f)
    
    with open(scalability_file, 'r') as f:
        scalability_metrics = json.load(f)
    
    return time_metrics, scalability_metrics 