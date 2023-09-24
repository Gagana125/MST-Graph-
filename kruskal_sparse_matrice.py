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
def graph_plot(matrix):
    numpy_matrix = np.array(matrix)
    G = nx.Graph()
    num_nodes = numpy_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if numpy_matrix[i][j] != 0:
                G.add_edge(i,j,weight = numpy_matrix[i][j])
    
    position = nx.spring_layout(G)
    nx.draw(G, position, with_labels = True, node_color = 'red',font_weight = 'bold')
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

def kruskal(matrix):
    result = []
    i = 0
    e = 0
    parent = [-1 for x in range(nodes)]
    while e < nodes-1:
        min = 999
        for i in range(nodes):
            for j in range(nodes):
                if find(parent, i) != find(parent, j) and matrix[i][j] < min and matrix[i][j] != 0:
                    min = matrix[i][j]
                    x = i
                    y = j
        
        union(parent, x, y)
        result.append([x,y,matrix[x][y]])
        e = e + 1
    
    return result

# Plot the result matrix
def plot_final_mst(result):
    final_matrix = [[0 for x in range(nodes)] for y in range(nodes)]

    for i in range(len(result)):
        final_matrix[result[i][0]][result[i][1]] = result[i][2]
        final_matrix[result[i][1]][result[i][0]] = result[i][2]
    
    graph_plot(final_matrix)


if __name__ == "__main__":
    adjacency_matrix = [[0 for x in range(nodes)] for y in range(nodes)]
    generated_matrix = sparse_graph(adjacency_matrix, 0)

    # graph_plot(generated_matrix)

    result = kruskal(generated_matrix)

    print("Resultant MST : ", result)

    # plot_final_mst(result)
