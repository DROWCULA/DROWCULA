import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from transformers import AutoModel

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix

import matplotlib.pyplot as plt




def load_features(filename):
    features = np.load(filename)
    print(f"Features loaded from {filename}, shape: {features.shape}")
    return features


def ari_nmi(labels , my_labels):
    ari = adjusted_rand_score(my_labels, labels)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    
    nmi = normalized_mutual_info_score(my_labels, labels)
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")




def clustering_accuracy(true_labels, cluster_labels):
    # Build the contingency matrix
    cont_matrix = contingency_matrix(true_labels, cluster_labels)
    # Find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(-cont_matrix)
    # Compute accuracy
    accuracy = cont_matrix[row_ind, col_ind].sum() / cont_matrix.sum()
    return accuracy


def minibatch_euclidean_kmeans(features, n_clusters, max_iter=100, batch_size=100):
    np.random.seed(0)
    initial_centroids_indices = np.random.choice(range(features.shape[0]), n_clusters, replace=False)
    centroids = features[initial_centroids_indices]
    
    for iteration in range(max_iter):
        # Select a random batch of samples
        batch_indices = np.random.choice(range(features.shape[0]), batch_size, replace=False)
        batch = features[batch_indices]
        
        # Assign each point in the batch to the nearest centroid
        distances = np.linalg.norm(batch[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Update centroids based on the batch assignments
        new_centroids = np.array([batch[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                                  for i in range(n_clusters)])
        
        # Check for convergence
        if np.allclose(centroids, new_centroids, atol=1e-6):
            print(f"Converged after {iteration} iterations")
            break
        centroids = new_centroids
    
    # Compute final labels for the entire dataset
    final_distances = np.linalg.norm(features[:, np.newaxis] - centroids, axis=2)
    final_labels = np.argmin(final_distances, axis=1)
    
    silhouette_avg = silhouette_score(features, final_labels, metric='euclidean')
    print(f"The Silhouette score with Euclidean distance is: {silhouette_avg:.4f}")
    return final_labels, silhouette_avg




def minibatch_cosine_kmeans(features, n_clusters, silhouette = "cosine" , max_iter=100):
    np.random.seed(0)
    initial_centroids_indices = np.random.choice(range(features.shape[0]), n_clusters, replace=False)
    centroids = features[initial_centroids_indices]
    
    for iteration in range(max_iter):
        similarities = cosine_similarity(features, centroids)
        labels = np.argmax(similarities, axis=1)  # Higher similarity means closer cluster

        new_centroids = np.array([features[labels == i].mean(axis=0) for i in range(n_clusters)])
        
        # Check for convergence 
        if np.allclose(centroids, new_centroids, atol=1e-6):
            print(f"Converged after {iteration} iterations")
            break
        centroids = new_centroids

    silhouette_avg = silhouette_score(features, labels, metric=silhouette)
    print(f"The Silhouette score with cosine similarity is: {silhouette_avg:.4f}")
    return labels, silhouette_avg


