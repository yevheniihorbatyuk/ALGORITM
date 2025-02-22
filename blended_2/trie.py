import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class TrieNode:
    """
    A node in the trie structure.
    Each node contains:
    - a dictionary of children nodes
    - a flag indicating if it's the end of a word
    - a value associated with the word (if it's an end node)
    """
    def __init__(self):
        self._children = {}  # Dictionary to store child nodes
        self._is_end = False  # Flag to mark end of word
        self._value = None   # Value associated with the word

    def is_end(self) -> bool:
        """Returns True if the node represents the end of a word."""
        return self._is_end

    def set_end(self, is_end: bool):
        """Sets the end flag for this node."""
        self._is_end = is_end

    def get_value(self):
        """Returns the value associated with this node."""
        return self._value

    def set_value(self, value):
        """Sets the value associated with this node."""
        self._value = value

    def get_children(self) -> dict:
        """Returns the dictionary of child nodes."""
        return self._children


class Trie:
    """
    Trie data structure implementation.
    Supports:
    - Inserting words with associated values
    - Searching for words
    - Checking if words exist
    - Finding words with common prefixes
    """
    def __init__(self):
        """Initialize the trie with a root node."""
        self._root = TrieNode()

    def get_root(self) -> TrieNode:
        """Returns the root node of the trie."""
        return self._root

    def put(self, key: str, value) -> None:
        """
        Insert a word into the trie with associated value.
        
        Args:
            key (str): The word to insert
            value: Value to associate with the word
            
        Raises:
            TypeError: If key is not a string
            ValueError: If key is empty
        """
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        if not key:
            raise ValueError("Key cannot be empty")

        current = self._root
        
        # Traverse/create the path for each character
        for char in key:
            if char not in current._children:
                current._children[char] = TrieNode()
            current = current._children[char]
            
        # Mark end of word and set value
        current.set_end(True)
        current.set_value(value)

    def get(self, key: str):
        """
        Get the value associated with a word.
        
        Args:
            key (str): The word to look up
            
        Returns:
            The value associated with the word or None if not found
            
        Raises:
            TypeError: If key is not a string
            ValueError: If key is empty
        """
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        if not key:
            raise ValueError("Key cannot be empty")

        current = self._root
        
        # Follow the path for the word
        for char in key:
            if char not in current._children:
                return None
            current = current._children[char]
            
        # Return value only if it's a complete word
        return current.get_value() if current.is_end() else None

    def contains(self, key: str) -> bool:
        """
        Check if a word exists in the trie.
        
        Args:
            key (str): The word to check
            
        Returns:
            bool: True if the word exists, False otherwise
            
        Raises:
            TypeError: If key is not a string
            ValueError: If key is empty
        """
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        if not key:
            raise ValueError("Key cannot be empty")

        current = self._root
        
        # Follow the path for the word
        for char in key:
            if char not in current._children:
                return False
            current = current._children[char]
            
        return current.is_end()

    def delete(self, key: str) -> bool:
        """
        Remove a word from the trie.
        
        Args:
            key (str): The word to remove
            
        Returns:
            bool: True if the word was deleted, False if not found
            
        Raises:
            TypeError: If key is not a string
            ValueError: If key is empty
        """
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        if not key:
            raise ValueError("Key cannot be empty")

        def _delete_helper(node: TrieNode, key: str, depth: int) -> bool:
            # If we've processed all characters
            if depth == len(key):
                # If it's not a word, nothing to delete
                if not node.is_end():
                    return False
                
                # Clear end flag and value but keep the node
                # as it might be part of another word
                node.set_end(False)
                node.set_value(None)
                return True

            char = key[depth]
            if char not in node._children:
                return False

            should_delete_current = _delete_helper(node._children[char], key, depth + 1)

            # Delete the child node only if:
            # 1. The deletion was successful
            # 2. The child has no children of its own
            # 3. The child is not marking end of another word
            if (should_delete_current and 
                not node._children[char]._children and 
                not node._children[char].is_end()):
                del node._children[char]

            return should_delete_current

        return _delete_helper(self._root, key, 0)






class TrieVisualizer:
    """Helper class to visualize Trie structure using NetworkX"""
    
    def __init__(self):
        self.node_count = 0
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))  # Color palette
        
    def create_graph(self, trie_instance):
        """Convert trie to NetworkX graph"""
        G = nx.DiGraph()
        self.node_count = 0
        node_positions = {}
        
        def add_nodes(node, node_id="root", x=0, y=0, layer=0):
            # Create current node
            label = "ROOT" if node_id == "root" else node_id.split("_")[-1]
            color = self.colors[layer % len(self.colors)]
            
            # Add node with its attributes
            G.add_node(node_id, 
                      label=label,
                      is_end=node.is_end(),
                      color=color)
            node_positions[node_id] = (x, -layer)  # Negative y for top-to-bottom layout
            
            # Calculate positions for children
            children = list(node.get_children().items())
            width = max(len(children), 1) * 2
            
            for i, (char, child) in enumerate(children):
                self.node_count += 1
                child_id = f"node_{self.node_count}"
                
                # Calculate child position
                child_x = x - width/2 + i * 2 + 1
                child_y = -layer - 1
                
                # Add child recursively
                add_nodes(child, child_id, child_x, child_y, layer + 1)
                
                # Add edge with label
                G.add_edge(node_id, child_id, label=char)
        
        # Build the graph
        add_nodes(trie_instance.get_root())
        return G, node_positions
    
    def visualize_trie(self, trie_instance, title="Trie Visualization", figsize=(12, 8), highlight_path=None, highlight_nodes=None):
        """Creates and displays a visualization of the trie structure"""
        G = nx.DiGraph()
        self.node_count = 0
        pos = {}
        
        def add_nodes(node, node_id="root", x=0, y=0, layer=0):
            # Create current node
            label = "ROOT" if node_id == "root" else node_id.split("_")[-1]
            G.add_node(node_id, 
                      label=label,
                      is_end=node.is_end(),
                      layer=layer)
            pos[node_id] = (x, -layer)
            
            children = list(node.get_children().items())
            width = max(len(children), 1) * 2
            
            for i, (char, child) in enumerate(children):
                self.node_count += 1
                child_id = f"node_{self.node_count}"
                child_x = x - width/2 + i * 2 + 1
                child_y = -layer - 1
                
                # Add child node
                add_nodes(child, child_id, child_x, child_y, layer + 1)
                # Add edge with character
                G.add_edge(node_id, child_id, label=char)
        
        # Build the graph
        add_nodes(trie_instance.get_root())
        
        # Create the plot
        plt.figure(figsize=figsize)
        plt.title(title, pad=20)
        
        # Draw nodes
        node_colors = []
        for node in G.nodes():
            if highlight_nodes and node in highlight_nodes:
                node_colors.append('yellow')
            else:
                node_colors.append('lightblue')
        
        nx.draw_networkx_nodes(G, pos, 
                             node_color=node_colors,
                             node_size=200)
        
        # Draw edges
        edge_colors = []
        edge_widths = []
        for u, v in G.edges():
            if highlight_path and u in highlight_path and v in highlight_path:
                edge_colors.append('red')
                edge_widths.append(2.0)
            else:
                edge_colors.append('black')
                edge_widths.append(1.0)
        
        nx.draw_networkx_edges(G, pos,
                             edge_color=edge_colors,
                             width=edge_widths)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, 
                              labels=nx.get_node_attributes(G, 'label'))
        
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, 
                                   edge_labels=edge_labels,
                                   font_size=10)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    def run_test(name, test_func):
        """Helper function to run tests and print results"""
        try:
            test_func()
            print(f"✓ {name} - Passed")
        except AssertionError as e:
            print(f"✗ {name} - Failed: {str(e)}")
        except Exception as e:
            print(f"✗ {name} - Error: {str(e)}")
        print()

    def test_basic_operations():
        trie = Trie()
        # Test insertion
        trie.put("app", 1)
        assert trie.contains("app") == True, "Word 'app' should exist after insertion"
        assert trie.get("app") == 1, "Value for 'app' should be 1"
        
        # Test non-existent word
        assert trie.contains("apple") == False, "Word 'apple' should not exist"
        assert trie.get("apple") is None, "Get should return None for non-existent word"

    def test_nested_words():
        trie = Trie()
        # Insert nested words
        trie.put("app", 1)
        trie.put("apple", 2)
        trie.put("apricot", 3)
        
        assert trie.contains("app") == True, "'app' should exist"
        assert trie.contains("apple") == True, "'apple' should exist"
        assert trie.get("app") == 1, "Value for 'app' should be 1"
        assert trie.get("apple") == 2, "Value for 'apple' should be 2"

    def test_deletion():
        trie = Trie()
        # Setup
        trie.put("app", 1)
        trie.put("apple", 2)
        
        # Test deletion
        assert trie.delete("apple") == True, "Delete should return True for existing word"
        assert trie.contains("apple") == False, "'apple' should be deleted"
        assert trie.contains("app") == True, "'app' should still exist"
        assert trie.get("app") == 1, "Value for 'app' should still be 1"

    def test_error_handling():
        trie = Trie()
        
        # Test None input
        try:
            trie.put(None, 1)
            assert False, "Should raise TypeError for None key"
        except TypeError:
            print("✓ Successfully caught TypeError for None key")
        
        # Test empty string
        try:
            trie.put("", 1)
            assert False, "Should raise ValueError for empty key"
        except ValueError:
            print("✓ Successfully caught ValueError for empty key")
        
        # Test invalid operations on empty trie
        assert trie.get("test") is None, "Get should return None for empty trie"
        assert trie.contains("test") == False, "Contains should return False for empty trie"
        assert trie.delete("test") == False, "Delete should return False for empty trie"

    # Run all tests
    print("Running Trie Tests...")
    print("-" * 50)
    run_test("Basic Operations", test_basic_operations)
    run_test("Nested Words", test_nested_words)
    run_test("Deletion", test_deletion)
    run_test("Error Handling", test_error_handling)
    print("-" * 50)
    print("Test suite completed.")