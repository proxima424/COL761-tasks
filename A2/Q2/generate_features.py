from argparse import ArgumentParser
from functions import convert_to_gspan_input, run_gspan, process_subgraph_file, generate_feature_vectors
import numpy as np

parser = ArgumentParser()
parser.add_argument("-g", "--graphs", type=str, required=True, 
                    help="Path to graph.txt, e.g., ../dataset/AIDS/graph.txt")
parser.add_argument("-f", "--features", type=str, required=True, 
                    help="Path to features_kerberosid.txt,\
                    e.g., ../dataset/AIDS/features_csz228001.txt")

args = parser.parse_args()

# Define the input and output paths
input_graph_file_path = args.graphs
converted_file_path = "data_for_gspan.txt"

# Convert the input graph dataset to gSpan input format
order = convert_to_gspan_input(input_graph_file_path, converted_file_path)
order_np = np.array(order)

run_gspan(converted_file_path, 0.36)
converted_subgraph_file_path = "data_for_gspan.txt.fp"

top_subgraphs = process_subgraph_file(converted_subgraph_file_path, 100)
feature_vectors = generate_feature_vectors(top_subgraphs, len(order))

# Output file path
output_file_path = args.features

with open(output_file_path, 'w') as output_file:
    for i in range(len(order)):
            output_file.write(f"{i} # {' '.join(map(str, feature_vectors[:, i]))}\n")
