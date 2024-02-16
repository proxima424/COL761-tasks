from argparse import ArgumentParser
from functions import convert_to_gspan_input, run_gspan, process_subgraph_file, generate_feature_vectors

parser = ArgumentParser()
parser.add_argument("-g", "--graphs", type=str, required=True, 
                    help="Path to graph.txt, e.g., ../dataset/AIDS/graph.txt")
parser.add_argument("-f", "--features", type=str, required=True, 
                    help="Path to features_kerberosid.txt,\
                    e.g., ../dataset/AIDS/features_csz228001.txt")

args = parser.parse_args()

# Define the input and output paths
input_graph_file_path = args.graphs
class_0_file_path = "class_0.txt"
class_1_file_path = "class_1.txt"

# Convert the input graph dataset to gSpan input format
order = convert_to_gspan_input(input_graph_file_path, class_0_file_path, class_1_file_path)

num_class_0 = 0
num_class_1 = 0

for labels_present in order:
    if labels_present == 0:
        num_class_0 = num_class_0 + 1
    else:
        num_class_1 = num_class_1 + 1

run_gspan(class_0_file_path, 0.5)
run_gspan(class_1_file_path, 0.5)

class_0_subgraph_file_path = "class_0.txt.fp"
class_1_subgraph_file_path = "class_1.txt.fp"
# Process subgraph file and select top 50 subgraphs
top_50_subgraphs_class_0 = process_subgraph_file(class_0_subgraph_file_path, top_k = 50)
# Generate feature vectors for each graph based on the selected subgraphs
feature_vectors_class_0 = generate_feature_vectors(top_50_subgraphs_class_0, num_class_0)

# Process subgraph file and select top 50 subgraphs
top_50_subgraphs_class_1 = process_subgraph_file(class_1_subgraph_file_path, top_k = 50)
# Generate feature vectors for each graph based on the selected subgraphs
feature_vectors_class_1 = generate_feature_vectors(top_50_subgraphs_class_1, num_class_1)

num_features_class_0 = len(top_50_subgraphs_class_0)
num_features_class_1 = len(top_50_subgraphs_class_1)

# Output file path
output_file_path = args.features

p = 0 # class 0 indexing
q = 0 # class 1 indexing

with open(output_file_path, 'w') as output_file:
    for i in range(len(order)):
        if order[i] == 0:
            # Print from feature_vectors_class_0
            output_file.write(f"{i} # {' '.join(map(str, feature_vectors_class_0[:, p]))} {' '.join(['0'] * num_features_class_1)}\n")
            p = p + 1
        else:
            # Print 50 zeros then from feature_vectors_class_1
            output_file.write(f"{i} # {' '.join(['0'] * num_features_class_0)} {' '.join(map(str, feature_vectors_class_1[:, q]))}\n")
            q = q + 1
