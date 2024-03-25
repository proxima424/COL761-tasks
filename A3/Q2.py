import numpy as np
from sklearn.decomposition import PCA
import lshashpy3
from argparse import ArgumentParser
import timeit

import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree,BallTree

parser = ArgumentParser()
parser.add_argument("-s", "--subpart", type=str, required=True)
parser.add_argument("-p", "--path", type=str, required=True)

args = parser.parse_args()
num_tables = 5
k=5
k_values = [1, 5, 10, 50, 100, 500]
n_queries=100
def load_dataset(file_path):
    return np.loadtxt(file_path)

def reduce_dimensions(dataset, dimensions):
    reduced_datasets = []
    for dim in dimensions:
        pca = PCA(n_components=dim, random_state=42)
        reduced_dataset = pca.fit_transform(dataset)
        reduced_datasets.append(reduced_dataset)
    return reduced_datasets

def build_lsh_index(dataset, num_tables):
    lsh = lshashpy3.LSHash(5, input_dim = dataset.shape[1], num_hashtables= num_tables)
    for data in dataset:
        lsh.index(data)
    return lsh

def knn_query_lsh(query, lsh, k):
    return [result[0] for result in lsh.query(query, num_results=k)]

def knn_query_sequential(query, dataset, k):
    distances = np.linalg.norm(dataset - query, axis=1)
    indices = np.argsort(distances)[:k]
    return dataset[indices]

def sequential_scan(query_point, data, k):
   
    distances = []  # List to store distances and indices
    for i in range(len(data)):
        distance = np.linalg.norm(query_point - data[i])  # Computing L2 distance
        distances.append((distance, i))  # Storing distance and index of ith pt
    
    distances.sort()  # Sort distances in ascending order
    indices = [idx for _, idx in distances[:k]]  # Get indices of k-nearest neighbors
    distances = [dist for dist, _ in distances[:k]]  # Get distances of k-nearest neighbors
    
    return indices, distances

def compute_jaccard_index(true_neighbors, predicted_neighbors):
    predicted_neighbors = [a for a, _ in predicted_neighbors]
    true_set = set(tuple(arr) for arr in true_neighbors)
    predicted_set = set(predicted_neighbors)
    intersection = true_set.intersection(predicted_set)
    union = true_set.union(predicted_set)
    return len(intersection) / len(union)

## For LSH time plus accuracies evaluation
def evaluate(dataset, queries, k, lsh):
    lsh_times = []
    seq_times = []
    accuracies = []
    for query in queries:
        start_time = timeit.default_timer()
        lsh_neighbors = knn_query_lsh(query, lsh, k)
        lsh_time = timeit.default_timer() - start_time
        lsh_times.append(lsh_time)

        start_time = timeit.default_timer()
        seq_neighbors = knn_query_sequential(query, dataset, k)
        seq_time = timeit.default_timer() - start_time
        seq_times.append(seq_time)

        accuracy = compute_jaccard_index(seq_neighbors, lsh_neighbors)
        accuracies.append(accuracy)

    avg_lsh_time = np.mean(lsh_times)
    std_lsh_time = np.std(lsh_times)
    avg_seq_time = np.mean(seq_times)
    std_seq_time = np.std(seq_times)
    accuracy = np.mean(accuracies)

    return avg_lsh_time, std_lsh_time, avg_seq_time, std_seq_time, accuracy

def evaluate2(dataset,k,queries,tree):
    times= []
    dim=dataset.shape[1]
    if (tree=="KDTREE"):

        index_tree = KDTree(dataset, metric='euclidean')
        for q in range(n_queries):
            query_point = [dataset[np.random.choice(dataset.shape[0])]]
            start_time = timeit.default_timer()
            index_tree.query(query_point, k)
            end_time= timeit.default_timer() - start_time
            times.append(end_time) 
        
        avg_time=np.mean(times)
        std_time=np.std(times)

        return avg_time,std_time
    else:
        index_tree = BallTree(dataset, metric='euclidean')
        for q in range(n_queries):
            query_point = [dataset[np.random.choice(dataset.shape[0])]]
            start_time = timeit.default_timer()
            index_tree.query(query_point, k)
            end_time= timeit.default_timer() - start_time
            times.append(end_time) 
        
        avg_time=np.mean(times)
        std_time=np.std(times)

        return avg_time,std_time


if __name__ == "__main__":
    file_path=args.path
    dataset=load_dataset(file_path)
    
    dimensions = [2, 4, 10, 20]
    reduced_datasets = reduce_dimensions(dataset, dimensions)
    queries = [reduced_datasets[i][np.random.choice(dataset.shape[0], 100, replace=False)] for i in range(len(reduced_datasets))]

    
    lsh_avg_times = []
    seq_avg_times = []
    lsh_std_times = []
    seq_std_times = []

    avg_time_kd_tree = []
    std_dev_kd_tree = []
    avg_time_m_tree = []
    std_dev_m_tree = []
    
    lsh_knn_accuracies = []
    lsh_indices = [build_lsh_index(reduced_dataset, num_tables) for reduced_dataset in reduced_datasets]
    subpart=args.subpart
    if (subpart=='c'):
        ## Draw running time plots
        ## KD-Tree
        print("KDTree computation")
        for i, dim in enumerate(dimensions):
            print(f"Processing dimension {dim}...")
            avg_tree,std_tree=evaluate2(reduced_datasets[i], n_queries, k,"KDTREE")
            avg_time_kd_tree.append(avg_tree)
            std_dev_kd_tree.append(std_tree)
        ## LSH Part + Sequential Scan
            
        print("LSH and Sequential scan Computation")
        for i, dim in enumerate(dimensions):
            print(f"Processing dimension {dim}...")
            avg_lsh_time, std_lsh_time, avg_seq_time, std_seq_time, accuracy = evaluate(reduced_datasets[i], queries[i], k, lsh_indices[i])
            print(i, dim,avg_lsh_time, std_lsh_time, avg_seq_time, std_seq_time, accuracy)
            
            lsh_avg_times.append(avg_lsh_time)
            seq_avg_times.append(avg_seq_time)
            lsh_std_times.append(std_lsh_time)
            seq_std_times.append(std_seq_time)
            lsh_knn_accuracies.append(accuracy)
        
        

        ## For M-Tree
        print("MTree computation")    
        for i, dim in enumerate(dimensions):
            print(f"Processing dimension {dim}...")
            avg_tree,std_tree=evaluate2(reduced_datasets[i], n_queries, k,"MTREE")
            avg_time_m_tree.append(avg_tree)
            std_dev_m_tree.append(std_tree)
        
        plt.figure()
        plt.errorbar(dimensions, lsh_avg_times, yerr=lsh_std_times, label='LSH', marker='o')
        plt.xlabel('Dimensions')
        plt.ylabel('Average Running Time (s)')
        plt.title('Average Running Time of 5-NN Query (LSH)')
        plt.legend()
        plt.savefig("LSH.png")

        plt.figure()
        plt.errorbar(dimensions, seq_avg_times, yerr=seq_std_times, label='Sequential Scan', marker='o')
        plt.xlabel('Dimensions')
        plt.ylabel('Average Running Time (s)')
        plt.title('Average Running Time of Sequential Scan')
        plt.legend()
        plt.savefig("Sequential.png")

        plt.figure()
        plt.errorbar(dimensions, avg_time_kd_tree, yerr=std_dev_kd_tree, label='KD Tree', marker='o')
        plt.xlabel('Dimensions')
        plt.ylabel('Average Running Time (s)')
        plt.title('Average Running Time of  5-NN Query')
        plt.legend()
        plt.savefig("Kdtree.png")

        plt.figure()
        plt.errorbar(dimensions, avg_time_m_tree, yerr=std_dev_m_tree, label='M Tree', marker='o')
        plt.xlabel('Dimensions')
        plt.ylabel('Average Running Time (s)')
        plt.title('Average Running Time of  5-NN Query')
        plt.legend()
        plt.savefig("Mtree.png")
    
    else:
        for i, dim in enumerate(dimensions):
            knn_accuracies = []
            for k in k_values:
                print(f"Processing dimension {dim}, k = {k}...")
                avg_lsh_time, std_lsh_time, avg_seq_time, std_seq_time, accuracy = evaluate(reduced_datasets[i], queries[i], k, lsh_indices[i])
                print(i, dim,avg_lsh_time, std_lsh_time, avg_seq_time, std_seq_time, accuracy)
                knn_accuracies.append(accuracy)
            
            plt.figure()
            # Plot the results for this dimension
            plt.plot(k_values, knn_accuracies, marker='o', label=f'Dimension {dim}')

            plt.xlabel('k')
            plt.ylabel('Accuracy (Jaccard Index)')
            plt.title('Accuracy of LSH against k for Different Dimensions')
            plt.legend()
            plt.savefig(f"{dim}-{k}-Jaccard.png")  # Fixed error in file name generation

