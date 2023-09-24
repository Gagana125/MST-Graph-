import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict

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

# Create the dense graph directly from the sparse tree using adjacency lists
def dense_graph(adj_list):
    num_nodes = len(adj_list)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if i != j and not any(x[0] == j for x in adj_list[i]):
                weight = random.randint(1, 100)
                adj_list[i].append((j, weight))
                adj_list[j].append((i, weight))
    return adj_list

# Plot the graph
def graph_plot(adj_list):
    G = nx.Graph()
    for node, edges in adj_list.items():
        G.add_node(node)
        for neighbor, weight in edges:
            G.add_edge(node, neighbor, weight=weight)

    position = nx.spring_layout(G)
    nx.draw(G, position, with_labels=True, node_color='red', font_weight='bold')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, position, edge_labels=labels)
    plt.show()

# Find parent
def kruskal(adj_list):
    result = []
    edges = []

    # Create a list of all edges
    for node, neighbors in adj_list.items():
        for neighbor, weight in neighbors:
            edges.append((node, neighbor, weight))

    # Sort edges by weight
    edges.sort(key=lambda x: x[2])

    parent = {node: node for node in adj_list}

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)
        parent[root1] = root2

    for edge in edges:
        node1, node2, weight = edge
        if find(node1) != find(node2):
            result.append((node1, node2, weight))
            union(node1, node2)

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
    adjacency_list = random_tree(adjacency_list, 0)

    # Create the dense graph directly from the sparse tree using adjacency lists
    adjacency_list = dense_graph(adjacency_list)

    graph_plot(adjacency_list)

    result = kruskal(adjacency_list)

    print("Resultant MST : ", result)

    plot_final_mst(result)
