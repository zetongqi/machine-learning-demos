from numpy import linalg as LA
import numpy as np

def pca(X):
    X_bar = np.mean(X, axis=0)
    X_tilde = X-X_bar
    S = np.matmul(np.transpose(X_tilde), X_tilde) / X_tilde.shape[0]
    v, u = LA.eig(S)
    u_tilde = u[:,0:3]
    return np.matmul(X, u_tilde)
