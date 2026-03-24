import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def calculate_typicality(features, k=20):
    """
    Calculates Typicality as the inverse of the average distance 
    to K nearest neighbors.
    """
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(features)
    
    distances, _ = nn.kneighbors(features)
    
    avg_dist = np.mean(distances, axis=1)
    typicality = 1.0 / (avg_dist + 1e-10)
    return typicality

def select_queries(features, budget_b, already_labeled_indices=[]):
    """
    Implements the Typical Clustering strategy.
    """
    typicality_scores = calculate_typicality(features)
    
    # Step 2: Clustering for diversity
    num_clusters = len(already_labeled_indices) + budget_b
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    # Step 3: Querying typical examples from uncovered clusters
    labeled_clusters = set(cluster_labels[already_labeled_indices])
    all_clusters = set(range(num_clusters))
    uncovered_clusters = list(all_clusters - labeled_clusters)
    
    # Sort uncovered clusters by size (largest first)
    uncovered_clusters.sort(key=lambda c: np.sum(cluster_labels == c), reverse=True)
    
    selected_indices = []
    for i in range(min(budget_b, len(uncovered_clusters))):
        cluster_idx = uncovered_clusters[i]
        in_cluster_indices = np.where(cluster_labels == cluster_idx)[0]
        
        # Find the index with the highest typicality within this specific cluster
        best_in_cluster = in_cluster_indices[np.argmax(typicality_scores[in_cluster_indices])]
        selected_indices.append(best_in_cluster)
        
    return selected_indices # Corrected: Added return statement
