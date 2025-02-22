from trie import Trie

class Homework(Trie):
    def count_words_with_suffix(self, pattern) -> int:
        """
        Counts words ending with the given pattern (case-sensitive)
        
        Args:
            pattern (str): The suffix pattern to search for
            
        Returns:
            int: Number of words ending with the pattern
        """
        # Input validation
        if not isinstance(pattern, str):
            raise TypeError("Pattern must be a string")
        if not pattern:
            raise ValueError("Pattern cannot be empty")
            
        def _traverse_and_count(node, current_word):
            count = 0
            
            # If we found a complete word, check if it ends with our pattern
            if node.is_end():
                if current_word.endswith(pattern):
                    print(f"Found word ending with '{pattern}': {current_word}")
                    count += 1
                
            # Check all child nodes recursively
            for char, child_node in node.get_children().items():
                count += _traverse_and_count(child_node, current_word + char)
                
            return count
            
        return _traverse_and_count(self.get_root(), "")

    def has_prefix(self, prefix) -> bool:
        """
        Checks if any word in the trie starts with the given prefix (case-sensitive)
        
        Args:
            prefix (str): The prefix to search for
            
        Returns:
            bool: True if prefix exists, False otherwise
        """
        # Input validation
        if not isinstance(prefix, str):
            raise TypeError("Prefix must be a string")
        if not prefix:
            raise ValueError("Prefix cannot be empty")
            
        current = self.get_root()
        path = []
        
        # Follow the prefix path in the trie
        for char in prefix:
            if char not in current.get_children():
                print(f"Prefix '{prefix}' not found: stopped at '{''.join(path)}', '{char}' not found")
                return False
            current = current.get_children()[char]
            path.append(char)
            
        print(f"Found prefix '{prefix}' in trie")
        return True

if __name__ == "__main__":
    # Create a new trie and add some words
    print("Creating trie with example words...")
    trie = Homework()
    words = ["apple", "application", "banana", "cat", "appropriate", "app"]
    
    print("\nAdding words to trie:")
    for i, word in enumerate(words):
        print(f"Adding: {word}")
        trie.put(word, i)
    
    print("\n--- Demonstrating suffix counting ---")
    print("\nLooking for words ending with 'e':")
    count = trie.count_words_with_suffix("e")
    print(f"Total words ending with 'e': {count}")
    
    print("\nLooking for words ending with 'ion':")
    count = trie.count_words_with_suffix("ion")
    print(f"Total words ending with 'ion': {count}")
    
    print("\nLooking for words ending with 'p':")
    count = trie.count_words_with_suffix("p")
    print(f"Total words ending with 'p': {count}")
    
    print("\n--- Demonstrating prefix checking ---")
    print("\nChecking for prefix 'app':")
    exists = trie.has_prefix("app")
    print(f"Words with prefix 'app' exist: {exists}")
    
    print("\nChecking for prefix 'xyz':")
    exists = trie.has_prefix("xyz")
    print(f"Words with prefix 'xyz' exist: {exists}")
    
    print("\nChecking for prefix 'ban':")
    exists = trie.has_prefix("ban")
    print(f"Words with prefix 'ban' exist: {exists}")
    
    print("\n--- Demonstrating case sensitivity ---")
    print("\nChecking for prefix 'App' (uppercase 'A'):")
    exists = trie.has_prefix("App")
    print(f"Words with prefix 'App' exist: {exists}")
    
    print("\nLooking for words ending with 'E' (uppercase):")
    count = trie.count_words_with_suffix("E")
    print(f"Total words ending with 'E': {count}")