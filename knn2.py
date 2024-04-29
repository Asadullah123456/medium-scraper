import numpy as np
from graphs import build_graphs_from_files
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
from CNN import find_common_subgraphs
def graph_features(G):
    features = []
    features.append(len(G.nodes()))  # Node Count
    features.append(len(G.edges()))  # Edge Count
    degrees = np.mean([deg for n, deg in G.degree()])  # Average Degree
    features.append(degrees)
    density = nx.density(G)  # Graph Density
    features.append(density)
    if nx.is_connected(G.to_undirected()):  # This requires the graph to be undirected
        diameter = nx.diameter(G)
        avg_path_len = nx.average_shortest_path_length(G)
    else:
        diameter = -1  # indicating disconnected graph
        avg_path_len = -1
    features.append(diameter)
    features.append(avg_path_len)
    return features
document_graphs=[]
directory_path1 = r'D:\semester 6\GT\food_data'  # Using a raw string
directory_path2 = r'D:\semester 6\GT\sports_data'
# Generate graphs
document_graphs1 = build_graphs_from_files(directory_path1)
document_graphs2 = build_graphs_from_files(directory_path2)

document_graphs[0]=directory_path1
document_graphs[1]=directory_path2
# Assuming 'document_graphs' contains at least two graphs
if len(document_graphs) >= 2:
    common_subs = find_common_subgraphs(document_graphs[:2])
# Example of preparing a dataset
feature_vectors = [graph_features(g) for g in document_graphs]
