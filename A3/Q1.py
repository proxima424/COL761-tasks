import numpy as np
import matplotlib.pyplot as plt

# Function to generate random dataset in d-dimensional space
def generate_dataset(dimension, num_points):
    return np.random.uniform(0, 1000, size=(num_points, dimension))

# Function to compute L1, L2, and Linf distances between two points
def compute_distances(point1, point2):
    l1_distance = np.sum(np.abs(point1 - point2))
    l2_distance = np.linalg.norm(point1 - point2)
    linf_distance = np.max(np.abs(point1 - point2))
    return l1_distance, l2_distance, linf_distance

# Function to find farthest and nearest points from a query point
def find_farthest_and_nearest(query_point, data_points):
    differences = np.abs(query_point - data_points)
    L1 = np.sum(differences, axis=1)
    L2 = np.linalg.norm(differences, axis=1)
    Linf = np.max(differences, axis=1)

    return np.max(L1) / np.min(L1), np.max(L2) / np.min(L2), np.max(Linf) / np.min(Linf)

# Function to compute average ratio of farthest and nearest distances for each dimension
def compute_average_ratio(d_values, num_queries, num_points_per_dimension):
    average_ratios = {'L1': [], 'L2': [], 'Linf': []}
    for i, d in enumerate(d_values):
        print(f"Computing for dimension {d} ({i+1}/{len(d_values)})...")
        dataset = generate_dataset(d, num_points_per_dimension)
        query_points = np.random.choice(len(dataset), num_queries, replace=False)
        farthest_nearest_ratios = {'L1': [], 'L2': [], 'Linf': []}
        for j, query_point in enumerate(query_points):
            print(f"\tQuery {j+1}/{num_queries}...")
            dataset_without_query = np.delete(dataset, query_point, axis=0)  # Precompute dataset without query points
            L1_ratio, L2_ratio, L_inf_ratio = find_farthest_and_nearest(dataset[query_point], dataset_without_query)
            farthest_nearest_ratios['L1'].append(L1_ratio)
            farthest_nearest_ratios['L2'].append(L2_ratio)
            farthest_nearest_ratios['Linf'].append(L_inf_ratio)
        for distance_measure in farthest_nearest_ratios:
            average_ratios[distance_measure].append(np.mean(farthest_nearest_ratios[distance_measure]))
    return average_ratios

if __name__ == "__main__":
    # Parameters
    d_values = [1, 2, 4, 8, 16, 32, 64]  # dimensions
    num_queries = 100
    num_points_per_dimension = 1000000  # total dataset size for each dimension

    # Compute average ratios
    average_ratios = compute_average_ratio(d_values, num_queries, num_points_per_dimension)

    # Plot the average ratio versus d for the three distance measures
    plt.figure(figsize=(10, 6))
    for distance_measure in average_ratios:
        plt.plot(d_values, average_ratios[distance_measure], label=distance_measure)
    plt.title('Average Ratio of Farthest and Nearest Distances vs Dimensionality')
    plt.xlabel('Dimensionality (d)')
    plt.ylabel('Average Ratio')
    plt.xticks(d_values)
    plt.legend()
    plt.grid(True)
    plt.savefig("Fig1.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    for distance_measure in average_ratios:
        plt.plot(d_values, average_ratios[distance_measure], label=distance_measure)
    plt.title('Average Ratio of Farthest and Nearest Distances vs Dimensionality')
    plt.xlabel('Dimensionality (d)')
    plt.ylabel('Average Ratio')
    plt.xticks(d_values)
    plt.yscale('log')  # set logarithmic scale for y-axis
    plt.legend()
    plt.grid(True)
    plt.savefig("Fig2.png")
    plt.close()
