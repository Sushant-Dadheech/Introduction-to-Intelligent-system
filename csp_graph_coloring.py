"""
Constraint Satisfaction Problem (CSP) - Graph Coloring
Using Google OR-Tools and NetworkX for visualization.
"""

import time
import sys
from ortools.sat.python import cp_model
import networkx as nx
import matplotlib.pyplot as plt

def typing_print(text, delay=0.015):
    """Outputs text with a typewriter effect."""
    for char in str(text):
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def loading_animation(message="Loading", dots=3, speed=0.3):
    """Displays a simple loading animation."""
    sys.stdout.write(message)
    sys.stdout.flush()
    for _ in range(dots):
        time.sleep(speed)
        sys.stdout.write(".")
        sys.stdout.flush()
    print()

def graph_coloring_visual():
    typing_print("=== 🎨 Graph Coloring CSP Initialization ===", delay=0.03)
    loading_animation("⚙️ Building CP Model", dots=3, speed=0.2)
    
    # step 1 : create model
    model = cp_model.CpModel()
    
    # step 2 : Define graph
    nodes = ['A', 'B', 'C', 'D', 'E']
    edges = [('A','B'), ('A','C'), ('B','D'), ('C','D'), ('B','E'), ('C','E')]
    num_colors = 3 # maximum colors
    
    typing_print(f"📍 Nodes: {nodes}", delay=0.01)
    typing_print(f"🔗 Edges: {edges}", delay=0.01)
    typing_print(f"🎨 Max Colors Allowed: {num_colors}\n", delay=0.01)
    
    # Step 3 : create variables
    node_vars = {n: model.new_int_var(0, num_colors - 1, n) for n in nodes}
    
    # step 4 : Add constraints (important)
    for u, v in edges:
        model.add(node_vars[u] != node_vars[v])
        
    # step 5 : Solve
    loading_animation("🔍 Solving Constraint Satisfaction Problem", dots=4, speed=0.1)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    # step 6 : Check solution
    if status != cp_model.FEASIBLE and status != cp_model.OPTIMAL:
        typing_print("❌ No solution found!", delay=0.03)
        return
        
    color_map = []
    typing_print("✅ Solution Found:\n", delay=0.03)
    for n in nodes:
        color = solver.Value(node_vars[n])
        time.sleep(0.2)
        typing_print(f"  {n} -> Color {color}", delay=0.02)
        color_map.append(color)
        
    print()
    loading_animation("📈 Rendering Graph Visualization", dots=3, speed=0.2)
    
    # Step 8 : Create graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    # Step 9 : Draw graph
    pos = nx.spring_layout(G, seed=42) # fixed layout
    plt.figure(figsize=(6,7))          # Fixed typo: plt.Figure -> plt.figure
    nx.draw(G, pos,
            with_labels=True,
            node_color=color_map,
            cmap=plt.cm.Set2,          # better color
            node_size=2000,
            font_size=14,
            font_weight='bold')
    plt.title("Graph Coloring Visualization (CSP)")
    plt.show()

if __name__ == "__main__":
    graph_coloring_visual()
