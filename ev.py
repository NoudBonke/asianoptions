# %%
import numpy as np 
T = 1
N = 2**8
ts = [j/N for j in range(1, N+1)]

def get_cov(ts):
    matr = np.zeros((len(ts), len(ts)))
    for i in range(len(ts)):
        for j in range(len(ts)):
            matr[i][j] = min(ts[i], ts[j])
    
    return matr

def get_ev(ts):
    matr = get_cov(ts)
    
    # get eigenvector
    ev = np.linalg.eig(matr)[1][:,0]
    ev = np.linalg.eig(matr)

    return ev
