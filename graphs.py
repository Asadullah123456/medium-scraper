import os
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import matplotlib.pyplot as plt


# Function to process text: tokenize, remove stopwords, stem
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

# Main function to process documents and build graphs
def build_graphs_from_files(directory):
    graphs = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):  # Assuming text files
            path = os.path.join(directory, filename)
            with open(path, 'r', encoding='utf-8') as file:
                text = file.read()
                words = process_text(text)
                graph = create_graph(words)
                graphs.append(graph)
    return graphs
def visualize_graph(G):
    plt.figure(figsize=(12, 8))  # Set the size of the graph display
    pos = nx.spring_layout(G)  # Compute the layout for a better visual representation
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500, alpha=0.6)
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
    plt.title('Graph Representation of Text')
    plt.axis('off')  # Turn off the axes
    plt.show()

# Directory containing your files
directory_path = r'D:\semester 6\GT\food_data'  # Using a raw string

# Generate graphs
document_graphs = build_graphs_from_files(directory_path)
if document_graphs:
    visualize_graph(document_graphs[0])
# Optionally, print the graph nodes and edges for the first document
#print("Nodes:", document_graphs[0].nodes())
#print("Edges:", document_graphs[0].edges(data=True))
