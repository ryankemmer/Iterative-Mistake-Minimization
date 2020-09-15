import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import math


class Node:
    
    def __init__(self):

        self.left = None
        self.right = None
        self.feature = None
        self.value = None
        self.cluster = None

class imm(object):
    
    def __init__(self,k):
        self.k = k
        self.tree = None
    
    def fit(self,x):
        
        #perform kmeans clustering on dataset        
        clt = KMeans(n_clusters = self.k, random_state = 0, n_jobs = -1)
        clusterK = clt.fit(x)
        
        u = np.array(clusterK.cluster_centers_)
        y = np.array(clusterK.labels_)
            
        self.tree = self.build_tree(x, y, u)
            
    def build_tree(self,x,y,u):
        
        #check if array is homogenous
        first = y[0]
        count = 0
        for label in y:
            if label ==first:
                count += 1
        
        if count == len(y):
            leaf = Node()
            leaf.cluster = first
            return leaf
        
        else:
            
            #populate arrays of r and l
            l = np.zeros(len(x[0]))
            r = np.zeros(len(x[0]))
            
            for i in range(len(x[0])):
                
                arr = np.zeros(len(x))
                for j in range(len(x)):
                    arr[j] = u[y[j]][i]
                
                l[i] = np.amin(arr)
                r[i] = np.amax(arr)
            
            mistakes = []
            cutoffList = np.vstack([l[:], r[:]]).mean(axis = 0)
            
            #iterate through features
            for i in range(len(cutoffList)):
                sum = 0
                for j in range(len(x)):
                    sum += self.mistake(x[j],u[y[j]], i, cutoffList[i])
                mistakes.append(sum)
            
            i = np.argmin(mistakes)
            theta = cutoffList[i]
            
            M = []
            L = []
            R = []
            
            for j in range(len(x[0])):
                if self.mistake(x[j],u[y[j]], i, theta) == 1:
                    M.append(j)
                elif x[j][i] <= theta:
                    L.append(j)
                elif x[j][i] > theta:
                    R.append(j)
            
            leftx = []
            lefty = []
            
            for e in range(len(x)):
                if e in L:
                    leftx.append(x[e])
                    lefty.append(y[e])
            
            rightx = []
            righty = []
            
            for e in range(len(x)):
                if e in R:
                    rightx.append(x[e])
                    righty.append(y[e])
            
            node = Node()
            node.feature = i
            node.value = theta
            node.left = self.build_tree(leftx, lefty, u)
            node.right = self.build_tree(rightx, righty, u)
            
            return node
             
    def mistake(self, x , u, i, theta):
            
        if (x[i] <= theta) != (u[i] <= theta):
            return 1
        else:
            return 0
    
    def predict(self, x):
        
        if self.tree is None:
            raise TypeError('Model is untrained.')
        else:
            return self.traverse(self.tree, x) 
        
    def traverse(self, tree, x):
        
        feature = tree.feature
        value = tree.value
        
        if tree.cluster is not None:
            return tree.cluster
        
        if x[feature] <= value:
            return self.traverse(tree.left, x)
        else:
            return self.traverse(tree.right, x)