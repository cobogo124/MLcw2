import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def calculate_typicality(features, k=20):
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=1)
    nn.fit(features)
    distances, _ = nn.kneighbors(features)
    avg_dist = np.mean(distances, axis=1)
    return 1.0 / (avg_dist + 1e-10)

def select_queries_modified(features, budget_b, already_labeled_indices=[]):
    typicality_scores = calculate_typicality(features)
    num_clusters = len(already_labeled_indices) + int(budget_b * 1.2)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=3)
    cluster_labels = kmeans.fit_predict(features)
    
    cluster_sizes = np.bincount(cluster_labels, minlength=num_clusters)
    cluster_weights = np.sqrt(cluster_sizes) # Using sqrt to balance density and diversity
    
    # points are ranked by typicality AND the size of their neighborhood
    weighted_scores = typicality_scores * cluster_weights[cluster_labels]
    
    selected_indices = []
    labeled_clusters = set(cluster_labels[already_labeled_indices])
    
    # sort all points by the new global metric
    potential_indices = np.argsort(weighted_scores)[::-1]
    
    for idx in potential_indices:
        if len(selected_indices) >= budget_b:
            break
        
        c_id = cluster_labels[idx]
        # try to cover new clusters first
        if c_id not in labeled_clusters:
            selected_indices.append(idx)
            labeled_clusters.add(c_id)
            
    if len(selected_indices) < budget_b:
        for idx in potential_indices:
            if len(selected_indices) >= budget_b: break
            if idx not in selected_indices:
                selected_indices.append(idx)
        
    return selected_indices