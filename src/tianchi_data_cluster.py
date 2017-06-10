import numpy as np
import sklearn.cluster as scicluster

idx_ys_train = np.loadtxt('../data/ys_train.csv', delimiter=',')
# ys_train = np.reshape(ys_train,(-1,2))
# print(ys_train[:100,:])

# i = 1
# while True:
#     kmeans = scicluster.KMeans(n_clusters= i)
#     kmeans.fit(ys_train)
#
#     centroids = kmeans.cluster_centers_
#     labels = kmeans.labels_
#     loss = kmeans.inertia_/10000.0
#
#     print(centroids)
#     print(loss)
#     print(labels)
#
#     for j in range(i):
#         print(np.sum(labels == j))
#
#     if(loss < 12.04):
#         break
#
#     i += 1

k = 8
kmeans = scicluster.KMeans(n_clusters= k)
ys_train = np.reshape(idx_ys_train[:, 1], (-1, 1))
kmeans.fit(ys_train)

centroids = kmeans.cluster_centers_
labels = np.reshape(kmeans.labels_, (-1, 1))
loss = kmeans.inertia_/10000.0

print(centroids)
print(loss)
# print(labels)

for j in range(k):
    print(np.sum(labels == j))

idx_ys_centroids_train = np.concatenate((idx_ys_train, labels), axis = 1)
np.savetxt('../data/idx_ys_centroids_train_k_%d.csv'%(k), idx_ys_centroids_train, delimiter=',')
print(idx_ys_centroids_train)