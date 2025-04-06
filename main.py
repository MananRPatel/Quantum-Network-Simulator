import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from ext_plotting import plot_EXT_vs_h
from simulators import QCASTSimulator, QPASSSimulator

def get_reference_parameters():
    return {
        "num_nodes": 5,
        "target_Ep": 0.6,
        "q": 0.9,
        "link_state_range": 5,
        "average_degree": 5,
        "num_requests": 5,
        "num_slots": 30,
        "num_topologies": 1
    }

def run_simulation_for_topologies(sim_class, routing_metric, num_topologies, num_slots, num_requests, 
                                link_state_range, average_degree, target_Ep, q, num_nodes, json_file=None):
    all_throughputs = []
    for topo in range(num_topologies):
        print(f"\n====== Running simulation on network topology {topo+1} ======")
        # If using JSON, only num_nodes is optional
        if json_file:
            sim_obj = sim_class(num_nodes, num_slots, num_requests, link_state_range, 
                              routing_metric, average_degree, target_Ep, q, json_file)
        else:
            sim_obj = sim_class(num_nodes, num_slots, num_requests, link_state_range, 
                              routing_metric, average_degree, target_Ep, q)
        throughput = sim_obj.simulate()
        all_throughputs.extend(throughput)
    return all_throughputs

def plot_combined_comparison(qcast_data, qpass_cr_data, qpass_sumdist_data, qpass_botcap_data):
    plt.figure(figsize=(10, 6))
    
    sorted_qcast = np.sort(qcast_data)
    cdf_qcast = np.arange(1, len(sorted_qcast) + 1) / len(sorted_qcast)
    plt.plot(sorted_qcast, cdf_qcast, linewidth=2, label="Q-CAST (EXT)")
    
    sorted_qpass_cr = np.sort(qpass_cr_data)
    cdf_qpass_cr = np.arange(1, len(sorted_qpass_cr) + 1) / len(sorted_qpass_cr)
    plt.plot(sorted_qpass_cr, cdf_qpass_cr, linewidth=2, label="Q-PASS (CR)")
    
    sorted_qpass_sumdist = np.sort(qpass_sumdist_data)
    cdf_qpass_sumdist = np.arange(1, len(sorted_qpass_sumdist) + 1) / len(sorted_qpass_sumdist)
    plt.plot(sorted_qpass_sumdist, cdf_qpass_sumdist, linewidth=2, label="Q-PASS (SumDist)")
    
    sorted_qpass_botcap = np.sort(qpass_botcap_data)
    cdf_qpass_botcap = np.arange(1, len(sorted_qpass_botcap) + 1) / len(sorted_qpass_botcap)
    plt.plot(sorted_qpass_botcap, cdf_qpass_botcap, linewidth=2, label="Q-PASS (BotCap)")
    
    plt.title("Aggregated CDF of Throughput (ebits per slot) over 10 Networks")
    plt.xlabel("Throughput (ebits per slot)")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("\n=== Plotting EXT vs. Hop Count Graphs ===")
    plot_EXT_vs_h(p_values=[0.9, 0.6], q=0.9, widths=[1, 2, 3], h_range=range(1, 11))
    
    # Get reference parameters
    ref_params = get_reference_parameters()
    
    # Run simulations with random topology
    print("\n=== Running Reference Simulation with Random Topology ===")
    qcast_ext_ref = run_simulation_for_topologies(QCASTSimulator, "EXT",
                                                  ref_params["num_topologies"],
                                                  ref_params["num_slots"],
                                                  ref_params["num_requests"],
                                                  ref_params["link_state_range"],
                                                  ref_params["average_degree"],
                                                  ref_params["target_Ep"],
                                                  ref_params["q"],
                                                  ref_params["num_nodes"])
    
    qpass_cr_ref = run_simulation_for_topologies(QPASSSimulator, "CR",
                                                 ref_params["num_topologies"],
                                                 ref_params["num_slots"],
                                                 ref_params["num_requests"],
                                                 ref_params["link_state_range"],
                                                 ref_params["average_degree"],
                                                 ref_params["target_Ep"],
                                                 ref_params["q"],
                                                 ref_params["num_nodes"])
    
    qpass_sumdist_ref = run_simulation_for_topologies(QPASSSimulator, "SumDist",
                                                      ref_params["num_topologies"],
                                                      ref_params["num_slots"],
                                                      ref_params["num_requests"],
                                                      ref_params["link_state_range"],
                                                      ref_params["average_degree"],
                                                      ref_params["target_Ep"],
                                                      ref_params["q"],
                                                      ref_params["num_nodes"])
    
    qpass_botcap_ref = run_simulation_for_topologies(QPASSSimulator, "BotCap",
                                                     ref_params["num_topologies"],
                                                     ref_params["num_slots"],
                                                     ref_params["num_requests"],
                                                     ref_params["link_state_range"],
                                                     ref_params["average_degree"],
                                                     ref_params["target_Ep"],
                                                     ref_params["q"],
                                                     ref_params["num_nodes"])
    
    # Run simulations with JSON topology
    print("\n=== Running Reference Simulation with JSON Topology ===")
    qcast_ext_json = run_simulation_for_topologies(QCASTSimulator, "EXT",
                                                  ref_params["num_topologies"],
                                                  ref_params["num_slots"],
                                                  ref_params["num_requests"],
                                                  ref_params["link_state_range"],
                                                  ref_params["average_degree"],
                                                  ref_params["target_Ep"],
                                                  ref_params["q"],
                                                  ref_params["num_nodes"],
                                                  json_file="test_topology.json")
    
    qpass_cr_json = run_simulation_for_topologies(QPASSSimulator, "CR",
                                                 ref_params["num_topologies"],
                                                 ref_params["num_slots"],
                                                 ref_params["num_requests"],
                                                 ref_params["link_state_range"],
                                                 ref_params["average_degree"],
                                                 ref_params["target_Ep"],
                                                 ref_params["q"],
                                                 ref_params["num_nodes"],
                                                 json_file="test_topology.json")
    
    qpass_sumdist_json = run_simulation_for_topologies(QPASSSimulator, "SumDist",
                                                      ref_params["num_topologies"],
                                                      ref_params["num_slots"],
                                                      ref_params["num_requests"],
                                                      ref_params["link_state_range"],
                                                      ref_params["average_degree"],
                                                      ref_params["target_Ep"],
                                                      ref_params["q"],
                                                      ref_params["num_nodes"],
                                                      json_file="test_topology.json")
    
    qpass_botcap_json = run_simulation_for_topologies(QPASSSimulator, "BotCap",
                                                     ref_params["num_topologies"],
                                                     ref_params["num_slots"],
                                                     ref_params["num_requests"],
                                                     ref_params["link_state_range"],
                                                     ref_params["average_degree"],
                                                     ref_params["target_Ep"],
                                                     ref_params["q"],
                                                     ref_params["num_nodes"],                                                     
                                                     json_file="test_topology.json")
    
    # Plot comparisons
    print("\n=== Plotting Results for Random Topology ===")
    plot_combined_comparison(qcast_ext_ref, qpass_cr_ref, qpass_sumdist_ref, qpass_botcap_ref)
    
    print("\n=== Plotting Results for JSON Topology ===")
    plot_combined_comparison(qcast_ext_json, qpass_cr_json, qpass_sumdist_json, qpass_botcap_json)