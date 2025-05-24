from simulators import QCASTEnhancedSimulator
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Define node counts to test
    node_counts = [100, 200, 300, 400, 500]
    
    # Metrics for QCASTEnhanced
    enhanced_metrics = {
        'reliability': [],
        'success_rate': [],
        'throughput': []
    }
    
    print(f"\n{'='*80}")
    print(f"Testing Q-CAST Enhanced")
    print(f"{'='*80}")
    
    for num_nodes in node_counts:
        print(f"\nInitializing Q-CAST Enhanced simulator for {num_nodes} nodes...")
        simulator = QCASTEnhancedSimulator(
            num_nodes=num_nodes,
            num_slots=30,
            num_requests=10,
            link_state_range=3,
            average_degree=4,
            target_Ep=0.6,
            q=0.9,
            routing_metric='EXT'
        )
        
        # Run simulation for this network size
        print(f"Running Q-CAST Enhanced simulation for {num_nodes} nodes...")
        metrics = simulator.simulate()
        
        # Store metrics for QCASTEnhanced
        enhanced_metrics['reliability'].append(np.mean(metrics['path_reliability']))
        enhanced_metrics['success_rate'].append(np.mean(metrics['success_rate']))
        enhanced_metrics['throughput'].append(np.mean(metrics['throughput']))
        
        print(f"Results for {num_nodes} nodes:")
        print(f"  • Average Throughput: {np.mean(metrics['throughput']):.2f} EPRs/slot")
        print(f"  • Average Success Rate: {np.mean(metrics['success_rate']):.2f}%")
        print(f"  • Average Path Length: {np.mean(metrics['path_length']):.2f} hops")
        print(f"  • Average Recovery Success: {np.mean(metrics['recovery_success']):.2f}%")

    # Create throughput vs nodes graph for QCASTEnhanced
    plt.figure(figsize=(10, 6))
    plt.plot(node_counts, enhanced_metrics['throughput'], 'go-', linewidth=2, markersize=8)
    plt.title('QCASTEnhanced: Throughput vs Network Size', fontsize=14)
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Throughput (EPRs/slot)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add data labels
    for i, throughput in enumerate(enhanced_metrics['throughput']):
        plt.text(node_counts[i], throughput, f'{throughput:.2f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('qcastenhanced_throughput.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create reliability vs nodes graph for QCASTEnhanced
    plt.figure(figsize=(10, 6))
    plt.plot(node_counts, enhanced_metrics['reliability'], 'bo-', linewidth=2, markersize=8)
    plt.title('QCASTEnhanced: Path Reliability vs Network Size', fontsize=14)
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Path Reliability', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add data labels
    for i, reliability in enumerate(enhanced_metrics['reliability']):
        plt.text(node_counts[i], reliability, f'{reliability:.4f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('qcastenhanced_reliability.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create success rate vs nodes graph for QCASTEnhanced
    plt.figure(figsize=(10, 6))
    plt.plot(node_counts, enhanced_metrics['success_rate'], 'ro-', linewidth=2, markersize=8)
    plt.title('QCASTEnhanced: Success Rate vs Network Size', fontsize=14)
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add data labels
    for i, success_rate in enumerate(enhanced_metrics['success_rate']):
        plt.text(node_counts[i], success_rate, f'{success_rate:.2f}%', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('qcastenhanced_success_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 