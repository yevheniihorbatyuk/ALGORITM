from trie import Trie

class LongestCommonWord(Trie):
    def find_longest_common_word(self, strings) -> str:
        """
        Finds the longest common prefix among all strings in the input array.
        
        Args:
            strings (List[str]): Array of strings to find common prefix from
            
        Returns:
            str: Longest common prefix, or empty string if none exists
        """
        # Input validation with detailed feedback
        if not isinstance(strings, list):
            print(f"Error: Expected list input, got {type(strings)}")
            raise TypeError("Input must be a list of strings")
            
        if not strings:
            print("Input list is empty, returning empty string")
            return ""
            
        # Validate all elements are strings
        non_strings = [str(x) for x in strings if not isinstance(x, str)]
        if non_strings:
            print(f"Error: Found non-string elements: {non_strings}")
            raise TypeError("All elements must be strings")
            
        print(f"\nProcessing input strings: {strings}")
        
        # Insert all strings into the trie
        for i, word in enumerate(strings):
            print(f"Adding word to trie: {word}")
            self.put(word, i)
            
        def _find_common_prefix(node, prefix, depth=0):
            indent = "  " * depth  # For prettier output formatting
            
            # Check for word end or branching point
            if node.is_end():
                print(f"{indent}Found word end at: '{prefix}'")
                return prefix
                
            child_count = len(node.get_children())
            if child_count > 1:
                print(f"{indent}Found branching point at '{prefix}' with {child_count} children:")
                for char in node.get_children().keys():
                    print(f"{indent}- Branch: '{char}'")
                return prefix
                
            # If node has exactly one child, continue traversing
            if child_count == 1:
                char, child = next(iter(node.get_children().items()))
                print(f"{indent}Following single path: '{char}'")
                return _find_common_prefix(child, prefix + char, depth + 1)
                
            print(f"{indent}No children found at '{prefix}'")
            return prefix
            
        # Start search from root
        print("\nSearching for longest common prefix...")
        result = _find_common_prefix(self.get_root(), "")
        print(f"\nFound longest common prefix: '{result}'\n")
        return result

def demonstrate_with_example(example_name, strings):
    """Helper function to demonstrate the algorithm with different examples"""
    print(f"\n{'='*20} {example_name} {'='*20}")
    trie = LongestCommonWord()
    result = trie.find_longest_common_word(strings)
    print(f"Final result for {example_name}: '{result}'")
    print("="*60)

if __name__ == "__main__":
    print("Demonstrating LongestCommonWord functionality\n")
    
    # Example 1: Basic case with common prefix
    demonstrate_with_example(
        "Basic Example", 
        ["flower", "flow", "flight"]
    )
    
    # Example 2: Longer common prefix
    demonstrate_with_example(
        "Longer Common Prefix",
        ["interspecies", "interstellar", "interstate"]
    )
    
    # Example 3: No common prefix
    demonstrate_with_example(
        "No Common Prefix",
        ["dog", "racecar", "car"]
    )
    
    # Example 4: Edge case - single character difference
    demonstrate_with_example(
        "Single Character Difference",
        ["prefix", "prefer", "prevent"]
    )
    
    # Example 5: Complete word as prefix
    demonstrate_with_example(
        "Complete Word as Prefix",
        ["cat", "catalog", "category"]
    )
    
    print("\nDemonstrating error handling:")
    
    # Empty list
    print("\nTesting with empty list:")
    trie = LongestCommonWord()
    result = trie.find_longest_common_word([])
    print(f"Result with empty list: '{result}'")
    
    # Invalid input types
    print("\nTesting with invalid input types:")
    try:
        trie.find_longest_common_word(None)
    except TypeError as e:
        print(f"✓ Caught TypeError for None input: {e}")
        
    try:
        trie.find_longest_common_word([1, 2, 3])
    except TypeError as e:
        print(f"✓ Caught TypeError for non-string elements: {e}")