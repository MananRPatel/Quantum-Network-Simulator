import random
import networkx

class RecoveryStrategies:
    @staticmethod
    def segmentation_based_recovery(network, path, s, d):
        print("Attempting entanglement with segmentation-based recovery for path (Q-PASS):", path)
        segments = network.segment_path(path)
        recovered_path = []
        recovery_success = True
        for seg in segments:
            print("Processing segment:", seg)
            seg_success = True
            for i in range(len(seg)-1):
                u, v = seg[i], seg[i+1]
                channel_success = False
                for ch in network.graph[u][v]['channels']:
                    if ch['reserved'] and not ch.get('attempted', False):
                        if random.random() < network.graph[u][v]['success_prob']:
                            ch['entangled'] = True
                            channel_success = True
                            print(f"Segment: Success on edge ({u},{v}).")
                        else:
                            print(f"Segment: Failure on edge ({u},{v}).")
                        ch['attempted'] = True
                        break
                if not channel_success:
                    seg_success = False
                    print(f"Segment: Failure detected on edge ({u},{v}).")
                    break
            if seg_success:
                print(f"Segment {seg} succeeded.")
                if recovered_path:
                    recovered_path.extend(seg[1:])
                else:
                    recovered_path.extend(seg)
            else:
                print(f"Segment {seg} failed. Segmentation-based recovery stops for Q-PASS.")
                recovery_success = False
                break
        if recovery_success:
            if recovered_path[0] != s:
                recovered_path.insert(0, s)
            if recovered_path[-1] != d:
                recovered_path.append(d)
            print("Recovered full path via segmentation-based recovery:", recovered_path)
            return network.attempt_entanglement(recovered_path)
        else:
            return False

    @staticmethod
    def xor_based_recovery(network, path, s, d):
        print("Attempting entanglement with XOR-based recovery for path (Q-CAST):", path)
        recovered_path = network.xor_based_full_recovery(path, s, d)
        if recovered_path and network.attempt_entanglement(recovered_path):
            return True
        else:
            print(f"XOR-based recovery failed for {(s, d)}.")
            return False 