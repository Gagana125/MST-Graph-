import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import heapq

nodes = 5000

# Create an empty random tree
def random_tree(matrix, a):
    while True:
        if a == nodes - 1:
            break
        else:
            b = random.randint(0, nodes - 1)
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

# Add edges for the tree
def add_edges(matrix, n):
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
                if matrix[a][b] == 0:
                    matrix[a][b] = weight
                    matrix[b][a] = weight
                    n = n - 1
                else:
                    continue

    return matrix

# Create the graph from the generated tree
def sparse_graph(matrix, a):
    matrix = random_tree(matrix, a)
    extra_edges = nodes // 2
    matrix = add_edges(matrix, extra_edges)

    return matrix

# Plot the graph
def graph_plot(matrix, mst_edges=None):
    numpy_matrix = np.array(matrix)
    G = nx.Graph()
    num_nodes = numpy_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if numpy_matrix[i][j] != 0:
                G.add_edge(i, j, weight=numpy_matrix[i][j])

    position = nx.spring_layout(G)
    nx.draw(G, position, with_labels=True, node_color='red', font_weight='bold')

    # Draw only MST edges in a different color
    if mst_edges:
        mst_edges = [(edge[0], edge[1]) for edge in mst_edges]
        mst_edges = [(edge[1], edge[0]) for edge in mst_edges]
        nx.draw_networkx_edges(G, position, edgelist=mst_edges, edge_color='blue', width=2)

    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, position, edge_labels=labels)
    plt.show()

# Initialize lists to keep track of the selected vertices and their corresponding edges
def initialize_prim_vars(matrix):
    num_nodes = len(matrix)
    selected = [False] * num_nodes
    key = [float('inf')] * num_nodes
    candidate_edges = []

    return selected, key, candidate_edges

# Add edges to the candidate list
def add_candidate_edges(matrix, vertex, selected, key, candidate_edges):
    for i in range(len(matrix)):
        if not selected[i] and matrix[vertex][i] != 0 and matrix[vertex][i] < key[i]:
            key[i] = matrix[vertex][i]
            heapq.heappush(candidate_edges, (key[i], vertex, i))

# Prim's lazy algorithm
def prim_lazy(matrix):
    num_nodes = len(matrix)
    selected, key, candidate_edges = initialize_prim_vars(matrix)

    # Choose the starting node as the first vertex
    start_node = 0
    key[start_node] = 0
    add_candidate_edges(matrix, start_node, selected, key, candidate_edges)

    result = []
    while candidate_edges:
        weight, u, v = heapq.heappop(candidate_edges)
        if not selected[v]:
            selected[v] = True
            result.append([u, v, weight])
            add_candidate_edges(matrix, v, selected, key, candidate_edges)

    return result

# Plot the result matrix
def plot_final_mst(result, original_matrix):
    num_nodes = len(original_matrix)
    final_matrix = [[0 for x in range(num_nodes)] for y in range(num_nodes)]

    for i in range(len(result)):
        u, v, weight = result[i]
        final_matrix[u][v] = weight
        final_matrix[v][u] = weight

    graph_plot(final_matrix)

if __name__ == "__main__":
    adjacency_matrix = [[0 for x in range(nodes)] for y in range(nodes)]
    generated_matrix = sparse_graph(adjacency_matrix, 0)

    graph_plot(generated_matrix)

    result_prim_lazy = prim_lazy(generated_matrix)

    print("Resultant MST (Prim's Lazy Algorithm):", result_prim_lazy)

    plot_final_mst(result_prim_lazy, generated_matrix)
