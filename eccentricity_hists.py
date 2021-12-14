#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 08:54:04 2021

@author: kesaprm
"""
from matplotlib import pyplot as plt
import pandas as pd


m0_df = pd.read_csv("M0.csv", index_col=0) 
m1_df = pd.read_csv("M1.csv", index_col=0) 
m2_df = pd.read_csv("M2.csv", index_col=0) 

all_df = pd.read_csv("Allcells.csv") 
all_df.drop(all_df.loc[all_df['eccentricity']=='eccentricity'].index, inplace=True)
#eccentricity histograms
n, bins, patches = plt.hist(m0_df['eccentricity'], bins='auto', color='b',
                            alpha=0.8, rwidth=0.5,label='M0 image')
n1, bins1, patches1 = plt.hist(m1_df['eccentricity'], bins='auto', color='y',
                            alpha=0.8, rwidth=0.5,label='M1 image')
n2, bins2, patches2 = plt.hist(m2_df['eccentricity'], bins='auto', color='r',
                            alpha=0.6, rwidth=0.5,label='M2 image')

# nall, binsAll, patchesAll = plt.hist(all_df['eccentricity'], bins='sqrt', color='g',
#                             alpha=0.6, rwidth=0.5)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Eccentricity')
plt.ylabel('Frequency')
plt.title('Eccentricity values of M0 vs M1 vs M2 images')
#maxfreq = n.max()
# Set a clean upper y-axis limit.
#plt.ylim(ymax=maxfreq)
#plt.legend( loc='upper left')
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


## clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np

sil = []
kmax = 10

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters = k).fit(all_df)
  labels = kmeans.labels_
  sil.append(silhouette_score(all_df, labels, metric = 'euclidean'))
 

plt.plot(range(2, kmax+1), sil, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score')
plt.title('The Elbow Method using Silhouette score')
plt.show()

fig, ax = plt.subplots(figsize=(12, 12))

centers = kmeans.cluster_centers_
plt.scatter(centers[ 0], centers[ 1], marker="x", s=169, linewidths=3,
            color="w", zorder=10)
##elbow method to get the optimal k value
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
for k in range(2, kmax+1):
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(all_df)
    kmeanModel.fit(all_df)
 
    distortions.append(sum(np.min(cdist(all_df, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / all_df.shape[0])
    inertias.append(kmeanModel.inertia_)
 
    mapping1[k] = sum(np.min(cdist(all_df, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / all_df.shape[0]
    mapping2[k] = kmeanModel.inertia_

plt.plot(range(2, kmax+1), distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()