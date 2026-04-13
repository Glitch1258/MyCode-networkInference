import random
import numpy as np
from collections import defaultdict
from math import log

class NETINF:
    def __init__(self, epsilon=0.01, rate=1.0):
        self.epsilon = epsilon  # Probability for non-edge transmissions
        self.rate = rate        # Exponential rate parameter
        self.inferred_edges = []
    
    def transmission_prob(self, time_diff):
        """Exponential transmission probability P(dt) = rate * exp(-rate * dt)"""
        if time_diff <= 0:
            return 0.0
        return self.rate * np.exp(-self.rate * time_diff)
    
    def most_likely_tree(self, cascade):
        """Build the most likely propagation tree for a single cascade"""
        # cascade: list of (node, timestamp) sorted by time
        if len(cascade) <= 1:
            return [], 0.0
        
        tree = []
        total_log_likelihood = 0.0
        
        for i, (node, t_i) in enumerate(cascade):
            if i == 0:
                continue  # First node has no parent
            
            best_parent = None
            best_prob = -1.0
            
            # Consider all earlier nodes as possible parents
            for j in range(i):
                parent_node, t_j = cascade[j]
                dt = t_i - t_j
                
                if dt <= 0:
                    continue
                
                # Check if edge exists in current inferred graph
                edge = (parent_node, node)
                if edge in self.inferred_edges:
                    prob = self.transmission_prob(dt)
                else:
                    prob = self.epsilon * np.exp(-dt / 1000000)  # Scale dt for email timestamps
                
                if prob > best_prob:
                    best_prob = prob
                    best_parent = parent_node
            
            if best_parent is not None:
                tree.append((best_parent, node))
                total_log_likelihood += log(best_prob + 1e-10)
        
        return tree, total_log_likelihood
    
    def log_likelihood(self, cascades):
        """Total log-likelihood of all cascades given current graph"""
        total = 0.0
        for cascade in cascades:
            _, ll = self.most_likely_tree(cascade)
            total += ll
        return total
    
    def fit(self, cascades, max_edges=100):
        """Greedy edge addition"""
        self.inferred_edges = []
        
        # Extract all unique nodes
        all_nodes = set()
        for c in cascades:
            all_nodes.update([n for n, _ in c])
        all_nodes = list(all_nodes)
        
        for iteration in range(max_edges):
            best_edge = None
            best_gain = -np.inf
            current_ll = self.log_likelihood(cascades)
            
            print(f"Iteration {iteration+1}: trying candidate edges...")
            
            # Try all possible directed edges
            edge_count = 0# change the line below to speed up
            for u in all_nodes[:50]:  # Limit to first 50 nodes for speed
                for v in all_nodes[:50]:
                    if u == v:
                        continue
                    edge = (u, v)
                    if edge in self.inferred_edges:
                        continue
                    
                    edge_count += 1
                    # Test adding this edge
                    self.inferred_edges.append(edge)
                    new_ll = self.log_likelihood(cascades)
                    gain = new_ll - current_ll
                    self.inferred_edges.pop()
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_edge = edge
            
            print(f"  Tested {edge_count} edges, best gain: {best_gain:.4f}")
            
            if best_gain <= 1e-6:
                print("No improving edge found. Stopping.")
                break
            
            self.inferred_edges.append(best_edge)
            print(f"Iteration {iteration+1}: added {best_edge} with gain {best_gain:.4f}")
        
        return self.inferred_edges


def load_email_data(filepath):
    """Load email-Eu-core-temporal data file with three columns: src dst timestamp"""
    edges = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                u = int(parts[0])
                v = int(parts[1])
                t = int(parts[2])
                edges.append((u, v, t))
    return edges


def group_into_cascades(edges, window_size=604800):  # 7 days in seconds
    """Group email edges into time-window cascades"""
    if not edges:
        return []
    
    edges_sorted = sorted(edges, key=lambda x: x[2])
    cascades = []
    current_window = []
    start_time = edges_sorted[0][2]
    
    for u, v, t in edges_sorted:
        if t - start_time <= window_size:
            current_window.append((u, t))
            current_window.append((v, t))
        else:
            if current_window:
                # Group by node, keep earliest timestamp per node
                node_times = {}
                for node, ts in current_window:
                    if node not in node_times or ts < node_times[node]:
                        node_times[node] = ts
                cascade = sorted([(n, ts) for n, ts in node_times.items()], key=lambda x: x[1])
                cascades.append(cascade)
            current_window = [(u, t), (v, t)]
            start_time = t
    
    # Add the last window
    if current_window:
        node_times = {}
        for node, ts in current_window:
            if node not in node_times or ts < node_times[node]:
                node_times[node] = ts
        cascade = sorted([(n, ts) for n, ts in node_times.items()], key=lambda x: x[1])
        cascades.append(cascade)
    
    return cascades


def compute_metrics(inferred, ground_truth_edges):
    """Precision, recall, F1 against ground truth (undirected comparison)"""
    # Build ground truth set of undirected edges from email data
    truth_set = set()
    for u, v, t in ground_truth_edges:
        truth_set.add((min(u, v), max(u, v)))
    
    # Build inferred set of undirected edges
    inferred_set = set()
    for u, v in inferred:
        inferred_set.add((min(u, v), max(u, v)))
    
    tp = len(inferred_set & truth_set)
    fp = len(inferred_set - truth_set)
    fn = len(truth_set - inferred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


# Main execution
if __name__ == "__main__":
    # Load the email dataset
    print("Loading email-Eu-core-temporal-Dept1.txt...")
    edges = load_email_data("email-Eu-core-temporal-Dept1.txt")
    print(f"Loaded {len(edges)} directed timestamped edges")
    
    # Group edges into cascades by 7-day windows
    print("Grouping edges into cascades...")
    cascades = group_into_cascades(edges, window_size=(60))#reduce to make fast
    print(f"Created {len(cascades)} cascades")
    
    # Print sample cascade
    if cascades:
        print(f"\nSample cascade (first 5 events):")
        for node, ts in cascades[0][:5]:
            print(f"  Node {node} at time {ts}")
    
    # Run NETINF
    print("\nRunning NETINF...")
    model = NETINF(epsilon=0.01, rate=1.0)
    inferred = model.fit(cascades, max_edges=50)#reduce to make fast
    print(f"\nInferred {len(inferred)} directed edges")
    
    # Evaluate
    precision, recall, f1 = compute_metrics(inferred, edges)
    print(f"\nResults (comparing inferred to email edges):")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Show top inferred edges
    print("\nTop 10 inferred edges:")
    for i, (u, v) in enumerate(inferred[:10]):
        print(f"  {i+1}. {u} -> {v}")
