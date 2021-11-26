import assignment4 as models
import numpy as np
import sys
from sklearn import metrics

if(sys.version_info[0] < 3):
	raise Exception("This assignment must be completed using Python 3")

#==========================================================Data==========================================================
# Number of Instances:	
# 653
# Number of Attributes:
# 35 numeric, predictive attributes and the class

# Attribute Information:

# We have 35 variables for 653 counties, including demographics, covid info, previous election 
# results, work related information.
# percentage16_Donald_Trump	
# percentage16_Hillary_Clinton	
# total_votes20	
# latitude	
# longitude	
# Covid Cases/Pop	
# Covid Deads/Cases	
# TotalPop	
# Women/Men
# Hispanic
# White	
# Black	
# Native	
# Asian	
# Pacific	
# VotingAgeCitizen	
# Income	
# ChildPoverty	
# Professional	
# Service	
# Office	
# Construction	
# Production	
# Drive	
# Carpool	
# Transit	
# Walk	
# OtherTransp	
# WorkAtHome	
# MeanCommute	
# Employed	
# PrivateWork	
# SelfEmployed	
# FamilyWork	
# Unemployment


# Class Distribution:
# 328 - Candidate A (1), 325 - Candidate B (0)
#========================================================================================================================

def train_test_split(X, y, test_ratio):
	tr = int(y.size*test_ratio)
	return X[:tr], X[tr:], y[:tr], y[tr:]

def load_data(path):
	data = np.genfromtxt(path, delimiter=',', dtype=float)
	return data[:,:-1], data[:,-1].astype(int)

def load_cluster(path):
	data = np.genfromtxt(path, delimiter='\n', dtype=int)
	return data

#for testing the clustering algorithms
def silhouette(X,cluster):
    return metrics.silhouette_score(X, cluster, metric='euclidean')

X, y = load_data("county_statistics.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, 0.75)

y_cluster = load_cluster("real_county_cluster.txt")
# print(y_cluster)

#Initialization
#MLP
lr = .0001
w1 = np.random.normal(0, .1, size=(X_train.shape[1], 10))
w2 = np.random.normal(0, .1, size=(10,1))
b1 = np.random.normal(0, .1, size=(1,10))
b2 = np.random.normal(0, .1, size=(1,1))
# mlp = models.MLP(w1, b1, w2, b2, lr)
#
# #Train
# steps = 100*y_train.size
# mlp.train(X_train, y_train, steps)

#Check weights (For grading)
# mlp.w1
# mlp.b1
# mlp.w2
# mlp.b2

#Evaluate
def evaluate(solutions, real):
	if(solutions.shape != real.shape):
		raise ValueError("Output is wrong shape.")
	predictions = np.array(solutions)
	labels = np.array(real)
	return (predictions == labels).sum() / float(labels.size)

# solutions = mlp.predict(X_test)
# print(f"MLP acc: {evaluate(solutions, y_test)*100.0}%\n")


#Initialization
#k_means
k = 3
t = 50 #max iterations
k_means = models.K_MEANS(k, t)

#train
KM_cluster = k_means.train(X)

#evaluate
print(f"K-Means silhouette: {silhouette(X, KM_cluster)*100.0}%")


#AGNES
k = 3
agnes = models.AGNES(k)

#train
AG_cluster = agnes.train(X)

#evaluate
print(f"AGNES silhouette: {silhouette(X, AG_cluster)*100.0}%")



