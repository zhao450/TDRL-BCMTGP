from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = '1'
import random



def cluster(population, pc, toolbox,offspring_num ,k):
    pc_array = np.array(pc)
    if pc_array.ndim == 1:
        pc_array = pc_array.reshape(-1, 1)  

    kmeans = KMeans(n_clusters=min(k, len(population)), n_init=20,random_state=42)
    cluster_labels = kmeans.fit_predict(pc_array)

    cluster_individuals = {}
    
    for cluster_id in range(kmeans.n_clusters):
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_individuals[cluster_id] = cluster_indices
    total_to_select = offspring_num

    individuals_per_cluster = total_to_select // kmeans.n_clusters
    remainder = total_to_select % kmeans.n_clusters

    actual_individuals_per_cluster = {}
    for cluster_id in range(kmeans.n_clusters):
        if cluster_id < remainder:
            actual_individuals_per_cluster[cluster_id] = individuals_per_cluster + 1
        else:
            actual_individuals_per_cluster[cluster_id] = individuals_per_cluster
        
        actual_individuals_per_cluster[cluster_id] = min(len(cluster_individuals[cluster_id]), actual_individuals_per_cluster[cluster_id])
    selected_indices = set()

    for cluster_id in range(kmeans.n_clusters):
        cluster_indices = cluster_individuals[cluster_id]
        num_to_select = actual_individuals_per_cluster[cluster_id]
        top=True
        if top:
            if num_to_select > 0 and cluster_indices:
                cluster_fitness = [population[idx].fitness.values[0] for idx in cluster_indices]
                top_indices = sorted(range(len(cluster_fitness)), key=lambda i: cluster_fitness[i])[:num_to_select]
                for i in top_indices:
                    original_idx = cluster_indices[i]
                    selected_indices.add(original_idx)
        else:
            if num_to_select>0 and cluster_indices:
                if len(cluster_indices)<=num_to_select:
                    selected_from_cluster = cluster_indices
                else:
                    selected_from_cluster = random.sample(cluster_indices, num_to_select)

            for original_idx in selected_from_cluster:
                selected_indices.add(original_idx)

    selected_population = []
    
    for i in range(len(population)):
        if i in selected_indices:
            selected_population.append(population[i])

    if len(selected_indices) < offspring_num:
        remaining_to_select = offspring_num - len(selected_indices)
        remaining_indices = [i for i in range(len(population)) if i not in selected_indices]
        remaining_fitness = [(i, population[i].fitness.values[0]) for i in remaining_indices]
        remaining_fitness.sort(key=lambda x: x[1])

        for i in range(min(remaining_to_select, len(remaining_fitness))):
            selected_indices.add(remaining_fitness[i][0])

    selected_population = [population[i] for i in selected_indices]
    while len(selected_population) < offspring_num:
        selected_population.append(random.choice(selected_population))
    
    return selected_population


def adaptive_clustering(population, pc, toolbox, offspring_num,k_range=(3, 10)):
    
    pc_array = np.array(pc)
    if pc_array.ndim == 1:
        pc_array = pc_array.reshape(-1, 1)
    
    best_k = k_range[0]
    best_score = -1

    for k in range(k_range[0], min(k_range[1] + 1, len(pc_array))):
        if len(pc_array) <= k:
            break
            
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(pc_array)
        
        cluster_sizes = [sum(1 for l in labels if l == i) for i in range(k)]
        if min(cluster_sizes) < 2:
            continue
            
        try:
            score = silhouette_score(pc_array, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except:
            continue

    return cluster(population, pc, toolbox,offspring_num ,k=best_k)