import numpy as np
from collections import deque, defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches


class GraphCut:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        # capacity [u][v] represents capacity for the edge from node u to node v
        self.capacity = defaultdict(dict)
        # flow[u][v] represents the current flow through the edge from node u to node v
        self.flow = defaultdict(dict)
        # adj_list[u] is a list of neighbors of node u
        self.adj_list = defaultdict(list)

    def add_node(self):
        node = self.num_nodes
        self.num_nodes += 1
        self.adj_list[node] = []
        return node

    def add_edge(self, u, v, w):
        self.capacity[u][v] = w
        self.flow[u][v] = 0
        self.adj_list[u].append(v)
        # Add reverse edge with capacity 0 for residual graph
        self.capacity[v][u] = 0
        self.flow[v][u] = 0
        self.adj_list[v].append(u)

    def bfs(self, source, sink, parent):
        visited = np.zeros(self.num_nodes, dtype=bool)
        queue = deque([source])  # use deque for more efficiency
        visited[source] = True

        while queue:
            u = queue.popleft()
            for v in self.adj_list[u]:
                if not visited[v] and self.capacity[u][v] - self.flow[u][v] > 0:
                    queue.append(v)
                    visited[v] = True
                    parent[v] = u
                    if v == sink:
                        return True
        return False

    def ford_fulkerson(self, source, sink):
        self.source = source
        self.sink = sink
        parent = np.full(self.num_nodes, -1, dtype=int)
        max_flow = 0

        while self.bfs(source, sink, parent):
            path_flow = np.inf
            s = sink
            while s != source:
                path_flow = min(
                    path_flow, self.capacity[parent[s]][s] - self.flow[parent[s]][s]
                )
                s = parent[s]

            v = sink
            while v != source:
                u = parent[v]
                self.flow[u][v] += path_flow
                self.flow[v][u] -= path_flow
                v = parent[v]

            max_flow += path_flow

        return max_flow

    # first call ford_fulkerson to find max flow and then call min_cut with the new flow to find the partition
    def min_cut(self, source):
        visited = np.zeros(self.num_nodes, dtype=bool)
        queue = deque([source])
        visited[source] = True

        while queue:
            u = queue.popleft()
            for v in self.adj_list[u]:
                if not visited[v] and self.capacity[u][v] - self.flow[u][v] > 0:
                    queue.append(v)
                    visited[v] = True

        return visited

    def display_graph_and_min_cut(self, source, sink):
        # Create a directed graph
        G = nx.DiGraph()

        # Add edges with capacities and flows
        for u in range(self.num_nodes):
            for v in self.adj_list[u]:
                if self.capacity[u][v] > 0:
                    G.add_edge(u, v, capacity=self.capacity[u][v], flow=self.flow[u][v])

        # Compute min cut
        min_cut_nodes = self.min_cut(source)

        # Determine the frontier nodes and edges
        frontier_edges = []
        for u in range(self.num_nodes):
            for v in self.adj_list[u]:
                if min_cut_nodes[u] and not min_cut_nodes[v]:
                    frontier_edges.append((u, v))

        # Draw the graph
        pos = nx.circular_layout(G)
        # Color nodes based on partition
        node_colors = [
            "lightblue" if min_cut_nodes[v] else "lightgreen" for v in G.nodes()
        ]
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=500,
            font_size=10,
        )

        # Highlight frontier edges
        nx.draw_networkx_edges(G, pos, edgelist=frontier_edges, edge_color="r", width=2)

        # Create edge labels with flow/capacity
        edge_labels = {
            (u, v): f"{G[u][v]['flow']:.1f}/{G[u][v]['capacity']:.1f}"
            for u, v in G.edges()
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # Create legend
        source_patch = mpatches.Patch(color="lightblue", label="Source Side")
        sink_patch = mpatches.Patch(color="lightgreen", label="Sink Side")
        plt.legend(handles=[source_patch, sink_patch], title="Partitions")

        plt.title("Graph with Minimum Cut Frontier Highlighted")
        plt.show()
    
    def get_segment(self, node):
        ## return 0 if the node belogs to the source partition and 1 if it belongs to the sink partition
        if not self.source or not self.sink:
            raise ValueError("Call ford_fulkerson before get_segment")
        visited = self.min_cut(self.source)
        if visited[node]:
            return 0
        else:
            return 1

