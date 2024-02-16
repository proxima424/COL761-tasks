import os
import numpy as np
def convert_to_gspan_input(input_file, output_path):
    """
    Convert the given graph data from the input file to the format required by gSpan.

    Parameters:
    - input_file (str): Path to the input file containing graph data.
    - output_path_class0 (str): Path to save the converted gSpan input for class 0.
    - output_path_class1 (str): Path to save the converted gSpan input for class 1.

    Returns:
    - None
    """
    order = []
    with open(input_file, 'r') as input_file:
        lines = input_file.readlines()

    with open(output_path, 'w') as output_file:
        current_graph_id = None

        for line in lines:
            if line.startswith('#'):
                current_graph_id = line.split()[1]
                label = int(line.split()[2])
                current_graph_id = str(current_graph_id)
                output_file.write(f't # {current_graph_id}\n')
                order.append(label)
            elif line.startswith('v'):
                parts = line.split()
                node_id = parts[1]
                label = parts[2]
                output_file.write(f'v {node_id} {label}\n')
            elif line.startswith('e'):
                parts = line.split()
                source_id = parts[1]
                target_id = parts[2]
                label = parts[3]
                output_file.write(f'e {source_id} {target_id} {label}\n')
    return order


def run_gspan(input_path, support_threshold):
    """
    Run gSpan algorithm on the input graph dataset.

    Parameters:
    - input_path (str): Path to the input graph dataset.
    - output_path (str): Path to save the output document containing frequent subgraphs.
    - support_threshold (float): Support threshold for gSpan.

    Returns:
    - None
    """
    gspan_command = f'./gSpan-64 -f {input_path} -s {support_threshold} -o -i'
    os.system(gspan_command)

def process_subgraph_file(subgraph_file, top_k = 100):
    """
    Process the subgraph file, select the top-k subgraphs based on support.

    Parameters:
    - subgraph_file (str): Path to the subgraph file.
    - top_k (int): Number of top subgraphs to select.

    Returns:
    - List of top-k subgraphs as strings.
    """
    with open(subgraph_file, 'r') as subgraph_file:
        lines = subgraph_file.readlines()

    subgraphs = []
    for line in lines:
        if line.startswith('x'):
            subgraph = line.split()[1:]  # Extract values after 'x'
            subgraphs.append(subgraph)

    # Sort subgraphs based on support
    sorted_subgraphs = sorted(subgraphs, key=lambda x: len(x), reverse=True)

    if len(sorted_subgraphs) < top_k:
        return sorted_subgraphs

    return sorted_subgraphs[:top_k]


def generate_feature_vectors(subgraphs, n):
    k = len(subgraphs)
    feature_vectors = np.zeros((k, n), dtype=int)  # Initialize a 2D array with zeros

    for i in range(k):
        # Assuming subgraphs is a list of 50 subgraphs represented as strings
        current_subgraph_str = subgraphs[i]
        
        # Parse the string representation into a list of integers
        current_subgraph = list(map(int, current_subgraph_str))

        # Assume 'n' is the number of features
        for j in range(n):
            # If a specific feature is present in the current subgraph, set the corresponding entry to 1
            if j in current_subgraph:
                feature_vectors[i, j] = 1

    return feature_vectors
