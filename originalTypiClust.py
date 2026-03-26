import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def calculate_typicality(features, k=20):
    # Setting n_jobs=1 avoids the Windows wmic subprocess error
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=1)
    nn.fit(features)
    distances, _ = nn.kneighbors(features)
    avg_dist = np.mean(distances, axis=1)
    return 1.0 / (avg_dist + 1e-10)

def select_queries(features, budget_b, already_labeled_indices=[]):
    typicality_scores = calculate_typicality(features)
    num_clusters = len(already_labeled_indices) + budget_b
    
    # n_init=3 speeds up the process for Task 3 evaluation
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=3)
    cluster_labels = kmeans.fit_predict(features)
    
    labeled_clusters = set(cluster_labels[already_labeled_indices])
    uncovered_clusters = [c for c in range(num_clusters) if c not in labeled_clusters]
    
    # Standard diversity: prioritize largest clusters first
    uncovered_clusters.sort(key=lambda c: np.sum(cluster_labels == c), reverse=True)
    
    selected_indices = []
    for i in range(min(budget_b, len(uncovered_clusters))):
        c_idx = uncovered_clusters[i]
        indices = np.where(cluster_labels == c_idx)[0]
        # Pick the most typical point in the cluster
        best_idx = indices[np.argmax(typicality_scores[indices])]
        selected_indices.append(best_idx)
        
    return selected_indices