import numpy as np
import scipy.sparse
import scipy.linalg
import Libs.GLOBAL as G

def createSparseMatrix():
    N = G.DM
    # percentage = G.DENSITY
    percentage = 0.01
    n_non_zero = int(np.round((N**2) * percentage))
    non_diag_rows = np.random.randint(N, size=(1,n_non_zero)) + 1
    non_diag_cols = np.random.randint(N, size=(1,n_non_zero)) + 1
    non_diag_vals = np.random.random((1, n_non_zero))

    diag_rows = np.array([i for i in range(1,N+1)]).reshape((1,N))
    diag_vals = np.array([n_non_zero for i in range(1,N+1)]).reshape((1,N))

    # print(diag_rows.shape, diag_vals.shape)

    a = np.append(non_diag_rows, diag_rows, axis=1)[0]
    b = np.append(non_diag_cols, diag_rows, axis=1)[0]
    c = np.append(non_diag_vals, diag_vals).reshape((1,len(b)))[0]

    #https://stackoverflow.com/questions/40890960/numpy-scipy-equivalent-of-matlabs-sparse-function
    # I=a, J=b, SV=c

    m = scipy.sparse.csr_matrix((c ,(a,b)), shape=(N+1, N+1))
    A = m
    # A = m.todense()
#     print(A.shape, scipy.linalg.svdvals(A), sep='\n')
    return A

def makeSparseWellConditioned(W):
    zero_rows = []
    zero_cols = []
    # Determine zero lines
    W_test = W.todense()
    for i in range(W_test.shape[0]):
        if np.all((W_test[i,:] == 0)):
            zero_rows.append(i)
        if np.all((W_test[:,i] == 0)):
            zero_cols.append(i)

    # print(zero_rows, zero_cols, sep='\n')

    W_test = np.delete(W_test, zero_rows, axis=0)
    W_test = np.delete(W_test, zero_cols, axis=1)

#     print('')
#     print(A.shape, scipy.linalg.svdvals(A), sep='\n')
#     print(np.linalg.cond(A))
    return W_test

    
def scaleMatrix(W):
    #1. Mean=0 already
    
    #2. Scale: [-0.5,0.5]
    raw_max = np.max(np.abs(W), axis=0)
    # print(W.shape, raw_max.shape)
    data = (W/raw_max)*0.5
    W = scipy.sparse.csr_matrix(W, dtype=np.float64, copy=True)
    
    return data

def printStats(W, header=""):
    print(header)
    max_ = max(scipy.linalg.svdvals(W))
    min_ = min(scipy.linalg.svdvals(W))
    print("Shape {s}".format(s=W.shape), "SVD Max: {ma} \tSVD Min: {mi}".format(ma=max_, mi=min_), sep='\n')
    print("Condition number ", np.linalg.cond(W))
    # print("Mean ", np.round(np.mean(W,axis=0)[0:10]))
    # print("Max ", np.round(np.max(np.abs(W), axis=0)[0:10], 5))
    # print("Min", np.round(np.min(np.abs(W), axis=0)[0:10], 5))