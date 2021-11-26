import random

import numpy
import numpy as np
import math


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
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi)
			#print(loss)

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
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b

	def forward(self, input):
		#Write forward pass here
		return None

	def backward(self, gradients):
		#Write backward pass here
		return None


class Sigmoid:

	def __init__(self):
		None

	def forward(self, input):
		#Write forward pass here
		ez = math.pow(2.71, -1 * input)
		return 1 / (1 + ez)

	def backward(self, gradients):
		#Write backward pass here
		ez = math.pow(2.71, -1 * gradients)
		sigmoid = 1 / (1 + ez)
		derivative = sigmoid * (1 - sigmoid)
		return derivative


class K_MEANS:

	def __init__(self, k, t):
		#k_means state here
		#Feel free to add methods
		# t is max number of iterations
		# k is the number of clusters
		self.k = k
		self.t = t

	def distance(self, centroids, datapoint):
		diffs = (centroids - datapoint)**2
		return np.sqrt(diffs.sum(axis=1))

	def mean(self, cluster):
		newcentroid = []
		for i in range(cluster.shape[1]):
			column = cluster[:, i]
			newcentroid.append(np.mean(column))
		return newcentroid

	def train(self, X):
		#training logic here
		#input is array of features (no labels)

		# Finding k random centroids using the input data
		randompoints = random.sample(list(range(X.shape[1])), self.k)
		centroids = np.array(X[randompoints])

		# Updating the k random centroids for t number of iterations
		clusterids = []
		for _ in range(self.t):
			clusterids = []
			for i in range(X.shape[0]):
				index: int = i
				distances = self.distance(centroids, X[index])
				clusterids.append(np.argmin(distances))

			for i in range(self.k):
				clustermembers = [j for j in range(X.shape[0]) if clusterids[j] == i]
				datapoints = X[clustermembers]
				newcentroid = self.mean(datapoints)
				centroids[i] = newcentroid

		return np.array(clusterids)
		#return array with cluster id corresponding to each item in dataset


class AGNES:
	#Use single link method(distance between cluster a and b = distance between closest
	#members of clusters a and b
	def __init__(self, k):
		#agnes state here
		#Feel free to add methods
		# k is the number of clusters
		self.k = k

	def distance(self, a, b):
		diffs = (a - b)**2
		return np.sqrt(diffs.sum())

	def train(self, X):
		#training logic here
		#input is array of features (no labels)


		return self.cluster
		#return array with cluster id corresponding to each item in dataset

