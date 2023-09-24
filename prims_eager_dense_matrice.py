import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys

nodes = 5000

# Create an empty random tree
def random_tree(matrix, a):
    while(True):
        if a == nodes-1:
            break
        else:
            b = random.randint(0, nodes-1)
            weight = random.randint(1, 100)
            if b == a:
                continue
            else:
                if matrix[a][b] == 0:
                    matrix[a][b] = weight
                    matrix[b][a] = weight
                    a = a + 1
                else:
                    continue

    return matrix

# Create the dense graph directly from the sparse tree
def dense_graph(matrix):
    num_nodes = len(matrix)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if i != j and matrix[i][j] == 0:
                weight = random.randint(1, 100)
                matrix[i][j] = weight
                matrix[j][i] = weight

    return matrix

# Plot the graph
def graph_plot(matrix):
    numpy_matrix = np.array(matrix)
    G = nx.Graph()
    num_nodes = numpy_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if numpy_matrix[i][j] != 0:
                G.add_edge(i, j, weight=numpy_matrix[i][j])

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
def prim_eager(matrix):
    parent = [-1] * nodes  # Parent array to store the MST
    key = [sys.maxsize] * nodes  # Key values to track minimum edge weights
    mst_set = [False] * nodes  # MST set to keep track of included vertices

    key[0] = 0  # Start with the first vertex

    for _ in range(nodes):
        u = min_key_vertex(key, mst_set)
        mst_set[u] = True

        # Update key values of adjacent vertices
        for v in range(nodes):
            if matrix[u][v] and not mst_set[v] and matrix[u][v] < key[v]:
                parent[v] = u
                key[v] = matrix[u][v]

    # Create the MST result
    mst = []
    for v in range(1, nodes):
        mst.append([parent[v], v, matrix[v][parent[v]]])

    return mst

# Plot the MST
def plot_mst(matrix, mst_edges):
    G = nx.Graph()

    for i in range(nodes):
        for j in range(i + 1, nodes):
            if matrix[i][j] != 0:
                G.add_edge(i, j, weight=matrix[i][j], color='gray')

    position = nx.spring_layout(G)
    edge_colors = ['gray' if not G[i][j]['weight'] in [edge[2] for edge in mst_edges] else 'red' for i, j in G.edges()]
    edge_labels = {(i, j): G[i][j]['weight'] for i, j in G.edges()}

    nx.draw_networkx_nodes(G, position, node_color='red', node_size=500)
    nx.draw_networkx_labels(G, position)
    nx.draw_networkx_edges(G, position, edge_color=edge_colors)
    nx.draw_networkx_edge_labels(G, position, edge_labels=edge_labels)
    plt.show()


if __name__ == "__main__":
    adjacency_matrix = [[0 for x in range(nodes)] for y in range(nodes)]
    generated_matrix = dense_graph(adjacency_matrix)

    graph_plot(generated_matrix)

    result = prim_eager(generated_matrix)

    print("Resultant MST : ", result)

    # Plot the final MST
    final_matrix = [[0 for x in range(nodes)] for y in range(nodes)]

    for i in range(len(result)):
        final_matrix[result[i][0]][result[i][1]] = result[i][2]
        final_matrix[result[i][1]][result[i][0]] = result[i][2]

    plot_mst(final_matrix, result)
