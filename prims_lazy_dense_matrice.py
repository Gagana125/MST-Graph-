import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
import heapq

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

# Prim's Lazy Algorithm for Dense Graphs
def prim_lazy(matrix):
    num_nodes = len(matrix)
    parent = [-1] * num_nodes
    key = [sys.maxsize] * num_nodes
    mst_set = [False] * num_nodes

    key[0] = 0

    candidate_edges = []  # Priority queue for candidate edges

    for i in range(num_nodes):
        candidate_edges.append((key[i], i))

    heapq.heapify(candidate_edges)

    while candidate_edges:
        _, u = heapq.heappop(candidate_edges)
        mst_set[u] = True

        for v in range(num_nodes):
            if matrix[u][v] and not mst_set[v] and matrix[u][v] < key[v]:
                parent[v] = u
                key[v] = matrix[u][v]

                # Update the priority queue with the new key value
                for i in range(len(candidate_edges)):
                    if candidate_edges[i][1] == v:
                        candidate_edges[i] = (key[v], v)
                        break

                heapq.heapify(candidate_edges)

    mst = []
    for v in range(1, num_nodes):
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

    result = prim_lazy(generated_matrix)

    print("Resultant MST (Prim's Lazy Algorithm):", result)

    # Plot the final MST
    final_matrix = [[0 for x in range(nodes)] for y in range(nodes)]

    for i in range(len(result)):
        final_matrix[result[i][0]][result[i][1]] = result[i][2]
        final_matrix[result[i][1]][result[i][0]] = result[i][2]

    plot_mst(final_matrix, result)
