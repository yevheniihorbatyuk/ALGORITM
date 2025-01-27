import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist

# Step 1: Generate synthetic data
def generate_data(n_samples=300, n_features=2, n_clusters=4, cluster_std=1.0, random_state=333):
    data, labels = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=cluster_std, n_features=n_features, random_state=random_state)
    return data

# Step 2: Implement k-means++ initialization
def kmeans_plus_plus(data, k, random_state=42):
    np.random.seed(random_state)
    n_samples, n_features = data.shape

    # Randomly select the first centroid
    centroids = [data[np.random.choice(range(n_samples))]]

    # Select the remaining k-1 centroids
    for _ in range(1, k):
        # Compute the distance from each point to the nearest centroid
        distances = np.min(cdist(data, np.array(centroids)), axis=1)
        probabilities = distances / distances.sum()

        # Choose the next centroid probabilistically
        next_centroid_idx = np.random.choice(range(n_samples), p=probabilities)
        centroids.append(data[next_centroid_idx])

    return np.array(centroids)

# Step 3: Plot the data and centroids for visualization
def plot_clusters(data, centroids, title="k-means++ Initialization"):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], s=30, alpha=0.6, label="Data Points")
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, label="Centroids")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()

# Parameters
n_samples = 300
n_features = 2
n_clusters = 4

# Generate data
data = generate_data(n_samples=n_samples, n_features=n_features, n_clusters=n_clusters)

# Perform k-means++ initialization
centroids = kmeans_plus_plus(data, n_clusters)

# Visualize results
plot_clusters(data, centroids, title="k-means++ Initialization")

