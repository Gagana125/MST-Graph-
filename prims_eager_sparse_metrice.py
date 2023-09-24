import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

nodes = 5000

# Create an empty random tree
def random_tree(matrix,a):
    while(True):
        if a == nodes-1:
            break
        else:
            b = random.randint(0,nodes-1)
            weight = random.randint(1,100)
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
def add_edges(matrix,n):
    while(True):
        if n == 0:
            break
        else:
            a = random.randint(0, nodes-1)
            b = random.randint(0, nodes-1)
            weight = random.randint(1,100)
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

# Find parent
def find(parent, i):
    if parent[i] == -1:
        return i
    else:
        return find(parent, parent[i])

# Get the union
def union(parent, x, y):
    x_root = find(parent, x)
    y_root = find(parent, y)
    parent[x_root] = y_root

def prim(matrix):
    num_nodes = len(matrix)
    
    # Initialize lists to keep track of the selected vertices and their corresponding edges
    selected = [False] * num_nodes
    parent = [-1] * num_nodes
    key = [float('inf')] * num_nodes
    
    # Choose the starting node as the first vertex
    start_node = 0
    key[start_node] = 0
    
    for _ in range(num_nodes):
        # Find the vertex with the minimum key value among the vertices not yet included in MST
        min_key = float('inf')
        min_index = -1
        for i in range(num_nodes):
            if not selected[i] and key[i] < min_key:
                min_key = key[i]
                min_index = i
        
        # Add the selected vertex to the MST
        selected[min_index] = True
        
        # Update key values and parent for adjacent vertices
        for i in range(num_nodes):
            if matrix[min_index][i] != 0 and not selected[i] and matrix[min_index][i] < key[i]:
                parent[i] = min_index
                key[i] = matrix[min_index][i]
    
    result = []
    for i in range(1, num_nodes):
        result.append([parent[i], i, matrix[i][parent[i]]])
    
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

    result_prim = prim(generated_matrix)

    print("Resultant MST (Prim's Eager Algorithm):", result_prim)

    plot_final_mst(result_prim, generated_matrix)
