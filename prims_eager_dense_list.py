import random
import matplotlib.pyplot as plt
import networkx as nx
import sys

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

# Create the dense graph directly from the sparse tree
def dense_graph(adj_list):
    for i in range(nodes):
        for j in range(i + 1, nodes):
            if i != j:
                weight = random.randint(1, 100)
                adj_list[i].append((j, weight))
                adj_list[j].append((i, weight))

    return adj_list

# Plot the graph
def graph_plot(adj_list):
    G = nx.Graph()

    for node, neighbors in enumerate(adj_list):
        for neighbor, weight in neighbors:
            G.add_edge(node, neighbor, weight=weight)

    position = nx.spring_layout(G)
    nx.draw(G, position, with_labels=True, node_color='red', font_weight='bold')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, position, edge_labels=labels)
    plt.show()

# Find the minimum key vertex
def min_key_vertex(key, mst_set):
    min_key = sys.maxsize
    min_vertex = -1
    for v in range(nodes):
        if not mst_set[v] and key[v] < min_key:
            min_key = key[v]
            min_vertex = v
    return min_vertex

# Prim's Eager Algorithm
def prim_eager(adj_list):
    parent = [-1] * nodes  # Parent array to store the MST
    key = [sys.maxsize] * nodes  # Key values to track minimum edge weights
    mst_set = [False] * nodes  # MST set to keep track of included vertices

    key[0] = 0  # Start with the first vertex

    for _ in range(nodes):
        u = min_key_vertex(key, mst_set)
        mst_set[u] = True

        # Update key values of adjacent vertices
        for neighbor, weight in adj_list[u]:
            if not mst_set[neighbor] and weight < key[neighbor]:
                parent[neighbor] = u
                key[neighbor] = weight

    # Create the MST result
    mst = []
    for v in range(1, nodes):
        u = parent[v]
        weight = key[v]
        mst.append((u, v, weight))

    return mst

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
    adjacency_list = [[] for _ in range(nodes)]
    generated_adj_list = dense_graph(adjacency_list)

    graph_plot(generated_adj_list)

    result = prim_eager(generated_adj_list)

    print("Resultant MST:")
    for u, v, weight in result:
        print(f"Edge: ({u}, {v}), Weight: {weight}")

    plot_final_mst(result)
