import numpy as np
from collections import Counter

def euclidean_distance(x1,x2):
  distance = np.sqrt(np.sum((x1 - x2)**2))
  return distance

class KNN:
  def __init__(self,k=3):
    self.k = k

  def fit(self,X,y):
    self.X_train = X
    self.y_train = y

  def predict(self,X):
    prediction = [self._predict(x) for x in X]
    return prediction

  def _predict(self,x):
    #compute the distance
    distance = [euclidean_distance(x,x_train) for x_train in self.X_train]
    #get the closest k
    k_sorted_idx = np.argsort(distance)[:self.k]
    k_closest_label = [self.y_train[i] for i in k_sorted_idx]

    #get the majority vote
    most_common = Counter(k_closest_label).most_common()
    return most_common[0][0]