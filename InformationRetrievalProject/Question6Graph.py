import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Load data
df = pd.read_csv('Greek_Parliament_Proceedings_1989_2020_refined_cleaned.csv')

# Group speeches by member
grouped = df.groupby('member_name')['cleaned_speech'].apply(' '.join)

# Extract feature vectors using TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(grouped)

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a graph
G = nx.Graph()

# Add nodes for each member of parliament
for member in grouped.index:
    G.add_node(member)

# Add edges between nodes if similarity is above threshold
threshold = 0.85
for i, member1 in enumerate(grouped.index):
    for j, member2 in enumerate(grouped.index):
        if i < j:
            sim = cosine_sim[i, j]
            if sim > threshold:
                G.add_edge(member1, member2, weight=sim)

# Remove isolated nodes
isolated_nodes = list(nx.isolates(G))
G.remove_nodes_from(isolated_nodes)

# Calculate node centrality measures
degree_centrality = nx.degree_centrality(G)

# Set a constant for the edge width
edge_width = 2

# Use a different layout if spring_layout doesn't give enough separation
#pos = nx.kamada_kawai_layout(G)

# Use spring_layout with an adjusted 'k' parameter
pos = nx.spring_layout(G, k=0.5, iterations=30)  # Adjust 'k' as needed


# Define color categories based on the number of edges
color_categories = {
    1: 'blue',
    2: 'green',
    3: 'yellow',
    4: 'orange',
    5: 'red'
}

# Determine node colors based on the number of edges
node_colors = []
for node in G.nodes():
    num_edges = len(G.edges(node))
    color = color_categories.get(num_edges, 'red')  # Default to red for nodes with 5 or more edges
    node_colors.append(color)

# Draw the graph without node labels
plt.figure(figsize=(12, 10))
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300)
nx.draw_networkx_edges(G, pos, width=edge_width, edge_color='black')

# Adding numerical labels next to nodes
node_labels = {node: str(idx + 1) for idx, node in enumerate(G.nodes())}
for node, (x, y) in pos.items():
    plt.text(x, y + 0.05, s=node_labels[node], horizontalalignment='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

# Create a legend for the color categories
legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'{num} edges', markersize=10, markerfacecolor=color)
                   for num, color in color_categories.items()]
plt.legend(handles=legend_elements, loc='lower right', title="Number of edges")

plt.title('Network Visualization with Degree Centrality, Numerical Labels, and Color Legend')
plt.axis('off')
plt.savefig('network_degree_centrality_no_labels50_with_out_labes.png')
plt.show()

# Printing paths and similarities for each node (This part remains the same as the previous explanation)

for node in G.nodes():
    # Retrieve the numerical label for the current node
    node_label = node_labels[node]
    print(f"Paths and Similarities from '{node}' (Node {node_label}):")
    for neighbor in G.neighbors(node):
        # Retrieve the numerical label for the neighbor node
        neighbor_label = node_labels[neighbor]
        similarity = G[node][neighbor]['weight']
        print(f"Node {node_label} ({node}) -> Node {neighbor_label} ({neighbor}), Similarity: {similarity:.2f}")
    print("\n")  # New line for readability