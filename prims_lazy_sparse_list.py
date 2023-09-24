import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict
import heapq

nodes = 5000

# Create an empty random tree
def random_tree(adj_list, a):
    while True:
        if a == nodes - 1:
            break
        else:
            b = random.randint(0, nodes - 1)
            weight = random.randint(1, 100)
            if b == a:
                continue
            else:
                if b not in adj_list[a]:
                    adj_list[a].append((b, weight))
                    adj_list[b].append((a, weight))
                    a = a + 1
                else:
                    continue

    return adj_list

# Add edges for the tree
def add_edges(adj_list, n):
    while True:
        if n == 0:
            break
        else:
            a = random.randint(0, nodes - 1)
            b = random.randint(0, nodes - 1)
            weight = random.randint(1, 100)
            if a == b:
                continue
            else:
                if b not in adj_list[a]:
                    adj_list[a].append((b, weight))
                    adj_list[b].append((a, weight))
                    n = n - 1
                else:
                    continue

    return adj_list

# Create the graph from the generated tree
def sparse_graph(adj_list, a):
    adj_list = defaultdict(list)
    adj_list = random_tree(adj_list, a)
    extra_edges = nodes // 2
    adj_list = add_edges(adj_list, extra_edges)

    return adj_list

# Plot the graph
def graph_plot(adj_list, mst_edges=None):
    G = nx.Graph()
    for node in adj_list:
        for neighbor, weight in adj_list[node]:
            G.add_edge(node, neighbor, weight=weight)

    position = nx.spring_layout(G)
    nx.draw(G, position, with_labels=True, node_color='red', font_weight='bold')

    # Draw only MST edges in a different color
    if mst_edges:
        for u, v, weight in mst_edges:
            nx.draw_networkx_edges(G, position, edgelist=[(u, v)], edge_color='blue', width=2)

    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, position, edge_labels=labels)
    plt.show()

# Initialize lists to keep track of the selected vertices and their corresponding edges
def initialize_prim_lazy_vars(adj_list):
    num_nodes = len(adj_list)
    selected = [False] * num_nodes
    parent = [-1] * num_nodes
    key = [float('inf')] * num_nodes
    candidate_edges = []

    return selected, parent, key, candidate_edges

# Add edges to the candidate list
def add_candidate_edges(adj_list, vertex, selected, key, candidate_edges):
    for neighbor, weight in adj_list[vertex]:
        if not selected[neighbor] and weight < key[neighbor]:
            key[neighbor] = weight
            heapq.heappush(candidate_edges, (key[neighbor], vertex, neighbor))

# Prim's lazy algorithm
def prim_lazy(adj_list):
    num_nodes = len(adj_list)
    selected, parent, key, candidate_edges = initialize_prim_lazy_vars(adj_list)

    # Choose the starting node as the first vertex
    start_node = 0
    key[start_node] = 0
    add_candidate_edges(adj_list, start_node, selected, key, candidate_edges)

    result = []
    while candidate_edges:
        weight, u, v = heapq.heappop(candidate_edges)
        if not selected[v]:
            selected[v] = True
            parent[v] = u
            result.append((u, v, weight))
            add_candidate_edges(adj_list, v, selected, key, candidate_edges)

    return result

# Plot the result graph
def plot_final_mst(result):
    G = nx.Graph()
    for node1, node2, weight in result:
        G.add_edge(node1, node2, weight=weight)

    position = nx.spring_layout(G)
    nx.draw(G, position, with_labels=True, node_color='red', font_weight='bold')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, position, edge_labels=labels)
    plt.show()

if __name__ == "__main__":
    adjacency_list = defaultdict(list)
    generated_list = sparse_graph(adjacency_list, 0)

    graph_plot(generated_list)

    result_prim_lazy = prim_lazy(generated_list)
    print("Resultant MST (Prim's Lazy Algorithm):", result_prim_lazy)

    plot_final_mst(result_prim_lazy)
