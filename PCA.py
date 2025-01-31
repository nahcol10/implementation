import numpy as np

class PCA:
  def __init(self,n_components):
    self.n_components = n_components
    self.components = None
    self.mean = None

  def fit(self,X):
    #shift to zero
    self.mean = np.mean(X,axis=0)
    X = X - self.mean

    #calculate the covariance
    cov = np.cov(X.T)

    #eigen vector and eigen value
    eigen_vector , eigen_value = np.linalg.eig(cov)
    
    eigen_vector = eigen_vector.T #converting to column vector
    idx = np.argsort(eigen_value)[::-1] #arranging in descending order

    eigen_value = eigen_value[idx]
    eigen_vector = eigen_vector[idx]

    self.components = eigen_vector[:self.n_components]

  def transform(self,X):
    X = X - self.mean
    return X @ self.components