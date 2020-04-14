from numpy import linalg as LA
import numpy as np

def pca(X, d):
    X_bar = np.mean(X, axis=0)
    X_tilde = X-X_bar
    S = np.dot(X_tilde.T, X_tilde) / X_tilde.shape[0]
    v, u = LA.eig(S)
    # sort the eigenvectors according their corresponding eigenvalue
    idx = v.argsort()[::-1]   
    v = v[idx]
    u = u[:,idx]
    # project
    u_tilde = u[:,0:d]
    return np.dot(X, u_tilde)