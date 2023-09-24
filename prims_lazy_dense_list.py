import random
import matplotlib.pyplot as plt
import networkx as nx
import sys
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

# Prim's Lazy Algorithm for Dense Graphs
def prim_lazy(adj_list):
    num_nodes = len(adj_list)
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

        for v, weight in adj_list[u]:
            if not mst_set[v] and weight < key[v]:
                parent[v] = u
                key[v] = weight

                # Update the priority queue with the new key value
                for i in range(len(candidate_edges)):
                    if candidate_edges[i][1] == v:
                        candidate_edges[i] = (key[v], v)
                        break

                heapq.heapify(candidate_edges)

    mst = []
    for v in range(1, num_nodes):
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

    result = prim_lazy(generated_adj_list)

    print("Resultant MST:")
    for u, v, weight in result:
        print(f"Edge: ({u}, {v}), Weight: {weight}")

    plot_final_mst(result)
