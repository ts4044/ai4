import random

import numpy
import numpy as np


### Assignment 4 ###

class MLP:
    def __init__(self, w1, b1, w2, b2, lr):
        self.l1 = FCLayer(w1, b1, lr)
        self.a1 = Sigmoid()
        self.l2 = FCLayer(w2, b2, lr)
        self.a2 = Sigmoid()

    def MSE(self, prediction, target):
        return np.square(target - prediction).sum()

    def MSEGrad(self, prediction, target):
        return - 2.0 * (target - prediction)

    def shuffle(self, X, y):
        idxs = np.arange(y.size)
        np.random.shuffle(idxs)
        return X[idxs], y[idxs]

    def train(self, X, y, steps):
        for s in range(steps):
            i = s % y.size
            if (i == 0):
                X, y = self.shuffle(X, y)
            xi = np.expand_dims(X[i], axis=0)
            yi = np.expand_dims(y[i], axis=0)

            pred = self.l1.forward(xi)
            pred = self.a1.forward(pred)
            pred = self.l2.forward(pred)
            pred = self.a2.forward(pred)
            loss = self.MSE(pred, yi)
            # print(loss)

            grad = self.MSEGrad(pred, yi)
            grad = self.a2.backward(grad)
            grad = self.l2.backward(grad)
            grad = self.a1.backward(grad)
            grad = self.l1.backward(grad)

    def predict(self, X):
        pred = self.l1.forward(X)
        pred = self.a1.forward(pred)
        pred = self.l2.forward(pred)
        pred = self.a2.forward(pred)
        pred = np.round(pred)
        return np.ravel(pred)


class FCLayer:

    def __init__(self, w, b, lr):
        self.lr = lr
        self.w = w  # Each column represents all the weights going into an output node
        self.b = b
        self.stored_output = None

    def forward(self, input):
        # Write forward pass here
        ouput = np.dot(input, self.w) + self.b
        self.stored_output = ouput
        return ouput

    def backward(self, gradients):
        # Write backward pass here
        layer_output = self.stored_output
        activation = np.dot(self.w, gradients[0])
        self.w -= self.lr * np.dot(layer_output, gradients[0])
        return activation


class Sigmoid:

    def __init__(self):
        self.stored_output = None

    def forward(self, input):
        # Write forward pass here
        sigmoid = 1 / (1 + np.exp(-input))
        self.stored_output = sigmoid
        return sigmoid

    def backward(self, gradients):
        # Write backward pass here
        sigmoid = self.stored_output
        derivative = gradients * (sigmoid * (1 - sigmoid))
        return derivative


class K_MEANS:

    def __init__(self, k, t):
        # k_means state here
        # Feel free to add methods
        # t is max number of iterations
        # k is the number of clusters
        self.k = k
        self.t = t

    def distance(self, centroids, datapoint):
        diffs = (centroids - datapoint) ** 2
        return np.sqrt(diffs.sum(axis=1))

    # This function takes a cluster, and finds a new mean by finding mean of each column
    def mean(self, cluster):
        new_centroid: list = []
        for i in range(cluster.shape[1]):
            column_values: list = cluster[:, i]
            new_centroid.append(np.mean(column_values))
        return new_centroid

    def train(self, X):
        # training logic here
        # input is array of features (no labels)

        # Finding k random centroids using the input data
        random_indexes: list = random.sample(list(range(X.shape[1])), self.k)
        centroids: numpy.ndarray = np.array(X[random_indexes])

        # Updating the k random centroids for t number of iterations
        # Stores the id of the cluster to which the corresponding data point belongs to
        cluster_ids: list = []
        for _ in range(self.t):
            # At every iteration, we are forming the cluster ids again, hence re-initialize
            cluster_ids = []

            # For each datapoint, calculate what cluster they belong to  by taking least distance centroid
            for i in range(X.shape[0]):
                index: int = i
                distances: list = self.distance(centroids, X[index])
                cluster_ids.append(np.argmin(distances))

            # Once the clusters are formed, calculate the new centroid by calculating the mean and update
            for i in range(self.k):
                cluster_members_indexes: list = [j for j in range(X.shape[0]) if cluster_ids[j] == i]
                cluster_members: numpy.ndarray = X[cluster_members_indexes]
                new_centroid: list = self.mean(cluster_members)
                centroids[i] = new_centroid

        return np.array(cluster_ids)


# return array with cluster id corresponding to each item in dataset


class AGNES:
    # Use single link method(distance between cluster a and b = distance between closest
    # members of clusters a and b
    def __init__(self, k):
        # agnes state here
        # Feel free to add methods
        # k is the number of clusters
        self.k = k

    def distance(self, a, b):
        diffs = (a - b) ** 2
        return np.sqrt(diffs.sum())

    def train(self, X):
        # training logic here
        # input is array of features (no labels)

        # Initially, each datapoint is in it's own cluster
        clusters = [[i] for i in range(X.shape[0])]

        # Calculate the bottom half of distance matrix since the distance is undirected, infinity for everything else
        distance_matrix = [[self.distance(X[i], X[j]) if j < i else float('inf') for j in range(X.shape[0])] for i in
                           range(X.shape[0])]

        # Do while the number of clusters is not k. The number of clusters reduces at each step due to merge
        while len(clusters) != self.k:
            minimum = float('inf')
            cluster_to_add = -1
            cluster_to_add_to = -1

            # Find least value in the distance matrix
            # clusters[i][0] gives the index of the first element in the cluster - using which I track the distances in
            # matrix. This is because my matrix is in the bottom triangle, j<i, so I add an element to the cluster
            # of an element with larger index always.
            for i in range(len(clusters)):
                for j in range(i):
                    if distance_matrix[clusters[i][0]][clusters[j][0]] < minimum:
                        minimum = distance_matrix[clusters[i][0]][clusters[j][0]]
                        cluster_to_add_to = i
                        cluster_to_add = j

            # Form single link - since we update the links at every merge, when merging two clusters, the least distance
            # is always considered
            for i in range(len(clusters[cluster_to_add])):
                if distance_matrix[cluster_to_add][i] < distance_matrix[cluster_to_add_to][i]:
                    distance_matrix[cluster_to_add_to][i] = distance_matrix[cluster_to_add][i]

            # Set the distance matrix of the row we already added to another row as infinity to avoid duplication
            distance_matrix[cluster_to_add] = [float('inf') for _ in distance_matrix[cluster_to_add]]

            # Merge the clusters
            clusters[cluster_to_add_to].extend(clusters[cluster_to_add])
            clusters.pop(cluster_to_add)

        # For every element within a cluster, set the corresponding cluster id in a list
        result = np.zeros(X.shape[0])
        for cluster_id, cluster in enumerate(clusters):
            for i in range(len(cluster)):
                result[cluster[i]] = cluster_id

        return result
# return array with cluster id corresponding to each item in dataset
