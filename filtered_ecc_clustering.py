#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 21:58:03 2021

@author: kesaprm
"""

from matplotlib import pyplot as plt
import pandas as pd

filtered_df = pd.read_csv("filteresEcc.csv") 

plt.hist(filtered_df['compactness'], bins=10, color='g',
                            alpha=0.6, rwidth=0.5)

x = filtered_df['index']
y1 = filtered_df['compactness']
y2 = filtered_df['solidity']

plt.scatter(x,y1)
plt.xlabel("Cells")
plt.ylabel("Compactness");
plt.axis('tight')
plt.grid(True)
plt.title('Compactness of cells having eccentricity >0.5 and <0.85')

plt.scatter(x,y2)
plt.xlabel("Cells")
plt.ylabel("Solidity");
plt.axis('tight')
plt.grid(True)
plt.title('Solidity of cells having eccentricity >0.5 and <0.85')


ecc = filtered_df['eccentricity']
sol = filtered_df['solidity']
e0 = ecc[sol < 0.7]
e1 = ecc[sol >= 0.7]

plt.hist(ecc, bins=30, color='y',
                            alpha=0.6, rwidth=0.5,label = 'All values')

nall, binsAll, patchesAll = plt.hist(e0, bins=30, color='b',
                            alpha=0.6, rwidth=0.5 , label='solidity < 0.7')
nall1, binsAll1, patchesAll2 = plt.hist(e1, bins=30, color='r',
                            alpha=0.6, rwidth=0.5, label='solidity >= 0.7')

plt.grid(axis='y', alpha=0.75)
plt.xlabel('Eccentricity')
plt.ylabel('Frequency')
plt.title('Eccentricity values[0.5-0.85] clustered based on solidity')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)




# ## Need to fix clustering
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from sklearn import metrics
# from scipy.spatial.distance import cdist
# import numpy as np

# sil = []
# kmax = 10

# all_df = filtered_df[['eccentricity','solidity']]
# # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
# for k in range(2, kmax+1):
#   kmeans = KMeans(n_clusters = k).fit(all_df)
#   labels = kmeans.labels_
#   sil.append(silhouette_score(all_df, labels, metric = 'euclidean'))
 

# plt.plot(range(2, kmax+1), sil, 'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('Silhouette score')
# plt.title('The Elbow Method using Silhouette score')
# plt.show()

# fig, ax = plt.subplots(figsize=(12, 12))

# centers = kmeans.cluster_centers_
# plt.scatter(centers[ 0], centers[ 1], marker="x", s=169, linewidths=3,
#             color="w", zorder=10)
# ##elbow method to get the optimal k value
# distortions = []
# inertias = []
# mapping1 = {}
# mapping2 = {}
# for k in range(2, kmax+1):
#     # Building and fitting the model
#     kmeanModel = KMeans(n_clusters=k).fit(all_df)
#     kmeanModel.fit(all_df)
 
#     distortions.append(sum(np.min(cdist(all_df, kmeanModel.cluster_centers_,
#                                         'euclidean'), axis=1)) / all_df.shape[0])
#     inertias.append(kmeanModel.inertia_)
 
#     mapping1[k] = sum(np.min(cdist(all_df, kmeanModel.cluster_centers_,
#                                    'euclidean'), axis=1)) / all_df.shape[0]
#     mapping2[k] = kmeanModel.inertia_

# plt.plot(range(2, kmax+1), distortions, 'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method using Distortion')
# plt.show()