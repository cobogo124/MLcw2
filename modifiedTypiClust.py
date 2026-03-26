import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def calculate_typicality(features, k=20):
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(features)
    distances, _ = nn.kneighbors(features)
    avg_dist = np.mean(distances, axis=1)
    return 1.0 / (avg_dist + 1e-10)

def select_queries_modified(features, budget_b, already_labeled_indices=[]):
    typicality_scores = calculate_typicality(features)
    num_clusters = len(already_labeled_indices) + budget_b
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    # NEW: Calculate cluster sizes to identify "significant" regions
    cluster_sizes = np.array([np.sum(cluster_labels == i) for i in range(num_clusters)])
    
    labeled_clusters = set(cluster_labels[already_labeled_indices])
    uncovered_clusters = list(set(range(num_clusters)) - labeled_clusters)
    
    # Sort by size to prioritize large, diverse regions first
    uncovered_clusters.sort(key=lambda c: cluster_sizes[c], reverse=True)
    
    selected_indices = []
    for i in range(min(budget_b, len(uncovered_clusters))):
        c_idx = uncovered_clusters[i]
        indices = np.where(cluster_labels == c_idx)[0]
        
        # MODIFICATION: Weighted Typicality
        # Instead of just Argmax(Typicality), we scale by the log of cluster size.
        # This prevents picking typical points from tiny, irrelevant "noise" clusters.
        size_weight = np.log1p(cluster_sizes[c_idx]) 
        weighted_scores = typicality_scores[indices] * size_weight
        
        best_idx = indices[np.argmax(weighted_scores)]
        selected_indices.append(best_idx)
        
    return selected_indices