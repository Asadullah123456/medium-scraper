import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
import iso  # Ensure iso is properly imported or defined
from graphs import build_graphs_from_files
def generate_subgraphs(G, max_size=3):
    """Generate all subgraphs up to a certain size from a given graph."""
    subgraphs = []
    for size in range(2, max_size + 1):  # Considering subgraphs of at least size 2
        for node_subset in combinations(G.nodes(), size):
            sg = G.subgraph(node_subset)
            if nx.is_weakly_connected(sg):
                subgraphs.append(sg)
    return subgraphs

def compare_graphs(g1, g2):
    """Compare two graphs and check if they are isomorphic considering weights."""
    nm = iso.numerical_edge_match('weight', 1)  # Assuming default weight is 1
    return nx.is_isomorphic(g1, g2, edge_match=nm)

def find_common_subgraphs(graphs, max_size=3):
    """Find common subgraphs between two graphs."""
    G1, G2 = graphs
    subgraphs1 = generate_subgraphs(G1, max_size)
    subgraphs2 = generate_subgraphs(G2, max_size)

    common_subgraphs = []
    for sg1 in subgraphs1:
        for sg2 in subgraphs2:
            if compare_graphs(sg1, sg2):
                common_subgraphs.append(sg1)  # Or some canonical form of sg1
                break
    return common_subgraphs
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
    for idx, sg in enumerate(common_subs, 1):
        print(f"Common Subgraph {idx}:")
        print("Nodes:", sg.nodes())
        print("Edges:", sg.edges(data=True))
        plt.figure(figsize=(8, 6))
        nx.draw(sg, with_labels=True, node_color='skyblue', edge_color='gray')
        plt.title(f"Common Subgraph {idx}")
        plt.show()
