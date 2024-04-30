import numpy as np
import networkx as nx
from graphs import build_graphs_from_files
from CNN import find_common_subgraphs
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


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
labels = ['food'] * len(document_graphs1) + ['sports'] * len(document_graphs2)
topics = ['food', 'sports']
feature_vectors = [graph_features(g) for g in document_graphs]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.3, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Metrics
print("Classification Report for Graph-Based Classification:")
print(classification_report(y_test, y_pred, target_names=topics, zero_division=1))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=topics, yticklabels=topics)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Find common subgraphs between first two graphs (if applicable)
if len(document_graphs) >= 2:
    common_subs = find_common_subgraphs(document_graphs[:2])
