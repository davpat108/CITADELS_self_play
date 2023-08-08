import networkx as nx
import matplotlib.pyplot as plt

def build_graph(node, G=None, parent_name=None):
    """
    Recursively walks through the CFRNode tree and builds a networkx graph.
    """
    if G is None:
        G = nx.DiGraph()

    # Create a unique identifier for the node. Here, I use id(node) to ensure uniqueness.
    # You could modify this depending on your structure.
    node_name = str(id(node)) + "\n" + str(node.current_player_id)
    
    G.add_node(node_name)

    if parent_name is not None:
        G.add_edge(parent_name, node_name)

    for _, child_node in node.children:
        build_graph(child_node, G, node_name)

    return G

def visualize_cfr_tree(root_node):
    """
    Visualizes the given CFRNode tree using networkx and matplotlib.
    """
    G = build_graph(root_node)
    pos = nx.spring_layout(G)  # Positioning of nodes. You can try other layouts too.
    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True, node_size=4, node_color='skyblue', font_size=10, ax=ax)
    plt.show()

# Example usage:
# root = CFRNode(...)  # Your root node
# visualize_cfr_tree(root)







