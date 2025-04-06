import numpy as np
import matplotlib.pyplot as plt
from math import comb

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