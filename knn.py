import os
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# nltk resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to process text
def process_text(text):
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    stemmer = PorterStemmer()
    words = word_tokenize(text.lower())
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words and word.isalnum()]
    return filtered_words

# Function to create a graph from a list of words
def create_graph(words):
    G = nx.DiGraph()
    for i in range(len(words) - 1):
        if words[i] not in G:
            G.add_node(words[i])
        if words[i+1] not in G:
            G.add_node(words[i+1])
        if G.has_edge(words[i], words[i+1]):
            G[words[i]][words[i+1]]['weight'] += 1
        else:
            G.add_edge(words[i], words[i+1], weight=1)
    return G

# Extract graph features
def extract_features(graph):
    return {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'average_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
        'density': nx.density(graph)
    }

# Build graphs from files and extract features
def build_features_and_graphs(directory):
    graphs = []
    features = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            path = os.path.join(directory, filename)
            with open(path, 'r', encoding='utf-8') as file:
                text = file.read()
                words = process_text(text)
                graph = create_graph(words)
                graphs.append(graph)
                features.append(extract_features(graph))
    return graphs, features

# Visualization function
def visualize_graph(G):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500, alpha=0.6)
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
    plt.title('Graph Representation of Text')
    plt.axis('off')
    plt.show()

# Directory containing your files
directory_path = r'D:\semester 6\GT\food_data'

# Generate graphs and features
document_graphs, features_list = build_features_and_graphs(directory_path)
features_df = pd.DataFrame(features_list)

# Mock labels (you need actual labels here)
labels = [1 if i % 2 == 0 else 0 for i in range(len(features_list))]  # Example: binary classification

# Split data
X_train, X_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.2, random_state=42)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Visualize first graph if available
if document_graphs:
    visualize_graph(document_graphs[0])

