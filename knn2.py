import numpy as np
import networkx as nx
from graphs import build_graphs_from_files  # Ensure this module correctly returns a list of graph objects
from CNN import find_common_subgraphs  # Ensure this module's function is implemented
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def graph_features(G):
    features = []
    features.append(len(G.nodes()))
    features.append(len(G.edges()))
    degrees = np.mean([deg for n, deg in G.degree()])
    features.append(degrees)
    density = nx.density(G)
    features.append(density)
    if nx.is_connected(G.to_undirected()):
        diameter = nx.diameter(G)
        avg_path_len = nx.average_shortest_path_length(G)
    else:
        diameter = -1
        avg_path_len = -1
    features.append(diameter)
    features.append(avg_path_len)
    return features

# Load or generate graphs
directory_path1 = r'D:\semester 6\GT\food_data'
directory_path2 = r'D:\semester 6\GT\sports_data'
document_graphs1 = build_graphs_from_files(directory_path1)
document_graphs2 = build_graphs_from_files(directory_path2)
document_graphs = document_graphs1 + document_graphs2
labels = ['food'] * len(document_graphs1) + ['sports'] * len(document_graphs2)  # Example labels
topics = ['food', 'sports']  # Classification topics
feature_vectors = [graph_features(g) for g in document_graphs]
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Print Classification Report
print("Classification Report for Graph-Based Classification:")
print(classification_report(y_test, y_pred, target_names=topics, zero_division=1))

# Find common subgraphs between first two graphs (if applicable)
if len(document_graphs) >= 2:
    common_subs = find_common_subgraphs(document_graphs[:2])

# Extract features from all graphs


# Assuming labels are available or generated here for each graph
