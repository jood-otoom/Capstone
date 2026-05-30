import networkx as nx
import matplotlib.pyplot as plt

# 1. Initialize the graph based on your graph_logic.py
graph = nx.DiGraph()

# Add Edges (Node A -> Node B means A Yields to B)
graph.add_edge("Side Road", "Main Road")
graph.add_edge("Dirt Road", "Paved Road")
graph.add_edge("Entering Roundabout", "Inside Roundabout")
graph.add_edge("Yield Sign", "Main Road")
graph.add_edge("Stop Sign", "Main Road")
graph.add_edge("Stop Sign", "Yield Sign") 
graph.add_edge("Red Light", "Green Light")

# 2. Setup the Matplotlib Plot
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(graph, seed=42, k=1.5) # Force-directed layout

# 3. Draw Nodes and Edges
nx.draw_networkx_nodes(graph, pos, node_size=3500, node_color='#1a73e8', edgecolors='black', linewidths=2)
nx.draw_networkx_edges(graph, pos, width=2.5, arrowstyle='->', arrowsize=25, edge_color='#5f6368', connectionstyle='arc3,rad=0.1')

# 4. Add Labels
nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold', font_color='white')

# 5. Add Titles and Save
plt.title("AcciEye TALE: Deterministic Right-of-Way Knowledge Graph", fontsize=16, fontweight='bold', pad=20)
plt.text(0.5, 0.95, "Direction of arrow indicates 'Owes Right of Way' (e.g., Stop Sign yields to Main Road)", 
         horizontalalignment='center', fontsize=12, transform=plt.gca().transAxes, color='gray')

plt.axis('off')
plt.tight_layout()
plt.savefig('tale_logic_graph.png', dpi=300)
print("Graph saved as 'tale_logic_graph.png'")