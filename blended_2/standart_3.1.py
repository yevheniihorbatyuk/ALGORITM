import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Define the names of the nodes in the graph
terminals = ["Terminal 1", "Terminal 2"]
warehouses = ["Warehouse 1", "Warehouse 2", "Warehouse 3", "Warehouse 4"]
stores = ["Store 1", "Store 2", "Store 3", "Store 4", "Store 5",
          "Store 6", "Store 7", "Store 8", "Store 9", "Store 10",
          "Store 11", "Store 12", "Store 13", "Store 14"]

# All edges of the graph with capacities
edges = [
    ("Terminal 1", "Warehouse 1", 25),
    ("Terminal 1", "Warehouse 2", 20),
    ("Terminal 1", "Warehouse 3", 15),
    ("Terminal 2", "Warehouse 3", 15),
    ("Terminal 2", "Warehouse 4", 30),
    ("Terminal 2", "Warehouse 2", 10),
    ("Warehouse 1", "Store 1", 15),
    ("Warehouse 1", "Store 2", 10),
    ("Warehouse 1", "Store 3", 20),
    ("Warehouse 2", "Store 4", 15),
    ("Warehouse 2", "Store 5", 10),
    ("Warehouse 2", "Store 6", 25),
    ("Warehouse 3", "Store 7", 20),
    ("Warehouse 3", "Store 8", 15),
    ("Warehouse 3", "Store 9", 10),
    ("Warehouse 4", "Store 10", 20),
    ("Warehouse 4", "Store 11", 10),
    ("Warehouse 4", "Store 12", 15),
    ("Warehouse 4", "Store 13", 5),
    ("Warehouse 4", "Store 14", 10),
]

# Add artificial nodes (source and sink)
source = "Source"
sink = "Sink"

# Build the graph
G = nx.DiGraph()

# Add the real connections
for start, end, capacity in edges:
    G.add_edge(start, end, capacity=capacity)

# Add artificial connections from the source to the terminals
for terminal in terminals:
    G.add_edge(source, terminal, capacity=float("inf"))

# Add artificial connections from the stores to the sink
for store in stores:
    G.add_edge(store, sink, capacity=float("inf"))

# Function to find the maximum flow (Edmonds-Karp)
def max_flow_edmonds_karp(G, source, sink):
    flow_value, flow_dict = nx.maximum_flow(G, source, sink, flow_func=nx.algorithms.flow.edmonds_karp)
    return flow_value, flow_dict

# Calculate the maximum flow
max_flow, flow_distribution = max_flow_edmonds_karp(G, source, sink)


plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)

# Draw the base graph
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=1000, font_size=10, font_color='black',
        arrowsize=20, arrowstyle='-|>',
        connectionstyle="arc3,rad=0.2") 

# Draw capacities on edges
edge_labels = { (u, v): f"{data['capacity']}" for u, v, data in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

# Draw flow on edges (only if flow > 0)
for u, v, data in G.edges(data=True):
    flow = flow_distribution.get(u, {}).get(v, 0)  # Prevent KeyError
    if flow > 0:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                               edge_color='red', width=3,
                               arrowsize=20, arrowstyle='-|>',
                               connectionstyle="arc3,rad=0.2")  # Highlight edges carrying flow
        # Adjust flow label position
        x, y = (pos[u][0] + pos[v][0]) / 2 + 0.05, (pos[u][1] + pos[v][1]) / 2
        plt.text(x, y, f"{flow}", color='red', fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)) 

plt.title(f"Logistics Network with Capacities and Flow (Max Flow: {max_flow})", fontsize=14)
plt.show()


print(f"🔹 Maximum Flow Through the Network: {max_flow}\n")
print("📊 Flow Distribution Between Nodes:")

# Organize flow distribution for better reporting
flow_summary = defaultdict(dict)
for start, flows in flow_distribution.items():
    for end, flow in flows.items():
        if flow > 0:
            flow_summary[start][end] = flow

for start, destinations in flow_summary.items():
    print(f"\n➡️ {start}:")
    for end, flow in destinations.items():
        print(f"    → {end}: {flow}") 

# Calculate and report utilization
print("\n📈 Edge Utilization:")
for u, v, data in G.edges(data=True):
    capacity = data['capacity']
    flow = flow_distribution.get(u, {}).get(v, 0)  # Prevent KeyError
    utilization = (flow / capacity) * 100 if capacity > 0 else 0
    print(f"    {u} → {v}: {utilization:.2f}% ({flow}/{capacity})")
