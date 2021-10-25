"""
Class used to define the Network.
"""

import numpy as np 
import scipy.sparse as ss

import Libs.GLOBAL as G

class Network:
    def __init__(self, dimension, a=0.5, alpha=0.99,seed=12345):
        self._a=a
        self._alpha=alpha
        self._dim=dimension
        rng=np.random.default_rng(seed)# Pad with zeros  #range: [0.0,1.0)

        ################################
        ## Generate Internal Matrices
        ################################

        if G.SPARSE_IMPLEMENTATION:
            #Input matrix: 
            import Libs.createSparse_WellC_Scaled as CSWS
            A = CSWS.createSparseMatrix()
            # CSWS.printStats(A, "CREATE SPARSE")
            A = CSWS.makeSparseWellConditioned(A)
            # CSWS.printStats(A, "\nCONDITION NUMBER")
            self._W_in = CSWS.scaleMatrix(A)
            if G.verbosity:
                CSWS.printStats(self._W_in, "\nCONDITIONED, SCALED SPARSE INPUT MATRIX")
            self._W_in = ss.csr_matrix(self._W_in)
            print( ss.csr_matrix.count_nonzero( self._W_in))

            #Network Matrix
            W=rng.random((self._dim,self._dim))-0.5
            print("Determining W's eigenvalues")
            largestEig = max(ss.linalg.eigs(W, which='LM', return_eigenvectors=False))
            
            self._W = (1/largestEig) * W #Spectral Radius = 1
            self._W = (self._W).astype(np.float64)
        
        else:
            #Input Matrix
            scale = 0.5
            self._W_in = ss.random(self._dim, self._dim, density=G.DENSITY, format='csr', dtype=float, random_state=seed)
            self._W_in.data += -0.5
            self._W_in.data *= scale
            self._W_in.data = (self._W_in.data - 0.5)*scale

            # self._W_in=(rng.random((self._dim,self._dim))-0.5)*0.5 #previous

            #Network Matrix:
            W=rng.random((self._dim,self._dim))-0.5
            print("Determining W's eigenvalues")

            # self._W=1/max(abs(np.linalg.eigvals(W))) * W #Spectral Radius =1   #previous
            largestEig = max(ss.linalg.eigs(W, which='LM', return_eigenvectors=False))
            
            self._W = (1/largestEig) * W #Spectral Radius = 1
            self._W = (self._W).astype(np.float64)



    def g(self,u,x):
        # Pad u with zeros
        u_=np.zeros(self._dim) 
        u_[0:len(u)]=u

        #W_in = A, _W = B
        x_1 = (1-self._a)*x +self._a*np.tanh(self._W_in@u_+self._alpha*self._W@x)
        # return (np.asarray(x_1)).flatten()
        return np.asarray(x_1)        