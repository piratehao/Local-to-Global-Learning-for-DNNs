import os
from copy import copy
import numpy as np
import random




def clusters_chosen_random(all_clusters, used_clusters, num_cluster_choose):
    all_clusters = np.arange(all_clusters)
    cluster_remain = np.delete(all_clusters, used_clusters)
    cluster = random.sample(list(cluster_remain), num_cluster_choose)
    return cluster