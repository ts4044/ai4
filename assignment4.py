import random
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
        self.stored_input = None

    def forward(self, input):
        # Write forward pass here
        output = np.dot(input, self.w) + self.b
        self.stored_input = input
        return output

    def backward(self, gradients):
        # Write backward pass here
        x_dash = np.dot(gradients, np.transpose(self.w))
        w_dash = np.dot(np.transpose(self.stored_input), gradients)
        self.w = self.w - (self.lr * w_dash)
        self.b = self.b - (self.lr * gradients)
        return x_dash


class Sigmoid:

    def __init__(self):
        self.stored_sigmoid = None

    def forward(self, input):
        # Write forward pass here
        sigmoid = 1 / (1 + np.exp(-input))
        self.stored_sigmoid = sigmoid
        return sigmoid

    def backward(self, gradients):
        # Write backward pass here
        sigmoid = self.stored_sigmoid
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

    def train(self, X):
        # training logic here
        # input is array of features (no labels)

        # Finding k random centroids using the input data
        random_indexes = random.sample(range(X.shape[0]), self.k)
        centroids = np.array(X[random_indexes])

        # Updating the k random centroids for t number of iterations
        # Stores the id of the cluster to which the corresponding data point belongs to
        cluster_ids = []
        for _ in range(self.t):
            # At every iteration, we are forming the cluster ids again, hence re-initialize
            cluster_ids = []

            # For each datapoint, calculate what cluster they belong to by taking least distance centroid
            for i in range(X.shape[0]):
                index = i
                distances = self.distance(centroids, X[index])
                cluster_ids.append(np.argmin(distances))

            # Once the clusters are formed, calculate the new centroid by calculating the mean and update
            centroid_changed = False
            for i in range(self.k):
                cluster_members_indexes = [j for j in range(X.shape[0]) if cluster_ids[j] == i]
                cluster_members = X[cluster_members_indexes]

                new_centroid = np.mean(cluster_members, axis=0)

                for k in range(new_centroid.shape[0]):
                    if new_centroid[k] != centroids[i][k]:
                        centroids[i] = new_centroid
                        centroid_changed = True
                        break

            if not centroid_changed:
                break

        # Return array with cluster id corresponding to each item in dataset
        return np.array(cluster_ids)


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
        cluster_id = [i for i in range(X.shape[0])]

        # This is used to track the number of clusters
        num_clusters = X.shape[0]

        # Calculate the distances between each pair of datapoints
        pairs = []
        distances = []
        for i in range(X.shape[0]):
            for j in range(i):
                # If the indexes are same, skip - since the distance will be 0 anyway
                if i == j:
                    continue
                # The key represents the pair of points
                # Since i > j always, this is like calculating only the bottom half of the distance matrix
                # The greater index of the two points always forms the x in the x,y pair
                pairs.append(str(i) + "," + str(j))
                distances.append(self.distance(X[i], X[j]))

        # Sort the distances in ascending order - get the index
        sorted_distances = np.argsort(distances)
        # Pointer to go through the sorted_distances
        pointer = 0

        # Do while the number of clusters is not k
        # The number of clusters reduces by 1 at each step where we merge clusters.
        while num_clusters != self.k:
            # Get the minimum distance from the sorted list
            # This points to the corresponding index of the pair of datapoints in pairs array
            index = sorted_distances[pointer]
            pointer += 1

            # The pop returns a tuple of ("x,y", distance) where "x,y" is the datapoint pair, and their distance
            pair = pairs[index]
            datapoints = pair.split(",")

            # Cluster_1 holds the cluster id to which we are adding Cluster_2 to
            cluster_1 = cluster_id[int(datapoints[0])]
            cluster_2 = cluster_id[int(datapoints[1])]

            # Update the cluster id for all the datapoints belonging to cluster_2
            if cluster_1 != cluster_2:
                for i, cid in enumerate(cluster_id):
                    if cid == cluster_2:
                        cluster_id[i] = cluster_1
                num_clusters -= 1

        # Return array with cluster id corresponding to each item in dataset
        return np.array(cluster_id)
