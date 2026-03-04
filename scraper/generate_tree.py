from anytree import Node, RenderTree
import json
from urllib.parse import urlparse
from collections import defaultdict

def get_clean_data():
    # Specify the JSON file path
    json_file_path = 'history.json'
    data = []

    # Open and read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    clean_data = [a for [a, _] in data]

    return clean_data

def build_url_tree(urls, base_url="https://www.healthlinkbc.ca"):
    """
    Build a tree structure from a list of URLs.
    
    Args:
        urls: List of URL strings
        base_url: The base domain (default: "www.mywebsite.com")
    
    Returns:
        The root node of the tree
    """
    # Create root node
    root = Node(base_url)
    
    # Dictionary to keep track of created nodes by their path
    node_map = {(): root}
    
    for url in urls:
        # Parse the URL and get the path
        parsed = urlparse(url if '://' in url else f'http://{url}')
        path = parsed.path.strip('/')
        
        # Skip empty paths (just the base URL)
        if not path:
            continue
        
        # Split path into segments
        segments = path.split('/')
        
        # Build the tree incrementally
        for i in range(len(segments)):
            path_tuple = tuple(segments[:i+1])
            
            # If this path doesn't exist yet, create it
            if path_tuple not in node_map:
                parent_path = tuple(segments[:i]) if i > 0 else ()
                parent_node = node_map[parent_path]
                new_node = Node(segments[i], parent=parent_node)
                node_map[path_tuple] = new_node
    
    return root


def print_tree(root):
    """Print the tree in a readable format."""
    for pre, _, node in RenderTree(root):
        print(f"{pre}{node.name}")

def main():
    clean_data = get_clean_data()

    print("Building tree from URLs...")
    print()
    
    root = build_url_tree(clean_data)
    
    print("URL Tree Structure:")
    print("=" * 50)
    print_tree(root)
    
    print("\n" + "=" * 50)
    print(f"Total nodes in tree: {len(root.descendants) + 1}")

main()