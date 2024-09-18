import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2
import math
from tqdm import tqdm
from tqdm.std import trange
from torch.utils.data import dataloader
from scipy.sparse import csr_matrix, linalg


def create_T(N):
    """
    Create the inner blocks of the A matrix (excluding boundaries)
    
    Args
    ----------
    N : int
        The size of PDE domain is (N,N)
    """
    m = np.zeros((N,N))

    for i in range(0,N):
        m[i,i] = 4.
        
    for i in range(0,N-2):
        m[i+1,i] = -1.
        m[i+1,i+2] = -1.

    m[0,1] = -2.
    m[N-1,N-2] = -2.
    
    return m


def create_rest(N):
    """
    Create the outer blocks of the A matrix (at boundaries)
    
    Args
    ----------
    N : int
        The size of PDE domain is (N,N)
    """
    m = - np.eye(N)
    
    return m


def create_A(N):
    """
    Generate the matrix A for the equation Au = b using finite difference method
    
    Args
    ----------
    N : int
        The size of PDE domain is (N,N)
        
    Outputs
    ----------
    A : ndarray of size (N**2,N**2)
        The matrix A
    """
    
    A = np.zeros((N**2,N**2))
    
    # Set the top left and bottom right N*N block to identity matrix
    A[0:N,0:N] = np.eye(N)
    A[N*(N-1):N*(N-1)+N,N*(N-1):N*(N-1)+N] = np.eye(N)

    # Set the inner blocks of matrix A
    for i in range(1,N-1):
        A[N*i:N*i+N,N*i:N*i+N] = create_T(N)
    
    # Set the boundary blocks of matrix A
    for i in range(0,N-2):
        A[N*(i+1):N*(i+1)+N,N*i:N*i+N] = create_rest(N)
        A[N*(i+1):N*(i+1)+N,N*(i+2):N*(i+2)+N] = create_rest(N)
    
    return A


def create_forcing_term(N, a, b, c, d):
    """
    Generate the forcing term
    
    Args
    ----------
    N : int
        The size of PDE domain is (N,N)
        
    Outputs
    ----------
    r : ndarray of size (N**2,1)
        The forcing term b
    """
    
    h = 6/(N-1)
    N2 = (N-2)*(N-2)
    x = np.arange(-3.0001,3.0001,h)
    y = np.arange(-3.0001,3.0001,h)

    r = np.zeros(N**2)

    for i in range(0,N):
        r[i] = d
    
    for i in range((N-1)*N,N*N):
        r[i] = d

    for i in range(1,N-1):
        for j in range(0,N):
            r[i*N+j] = h**2 * (a*np.sin(b*x[i])*np.cos(c*y[j])+b*np.cos(a*x[i])*np.sin(c*y[j])+c*np.exp(a*np.cos(b*x[i])*np.sin(c*y[j]))+(a*x[i]**3-b*y[j]**3)/(x[i]**2+c*y[j]**2+1))
        
    return r


def generate_data(N,a,b,c,d):
    """
    Generate the PDE data u which follows the equation Au = b using finite difference method
    
    Args
    ----------
    N : int
        The size of PDE domain is (N,N)
        
    Outputs
    ----------
    w : ndarray of size (N,N)
        The PDE solution u for the equation Au = b 
    r : ndarray of size (N**2,1)
        The forcing term b
    A : ndarray of size (N**2,N**2)
        The matrix A
    x,y : ndarray of size (N,1)
        x and y coordinates
    """
    
    # Initialisation
    h = 6/(N-1)
    x = np.arange(-3.0001,3.0001,h)
    y = np.arange(-3.0001,3.0001,h)
    
    # Work out the matrix A
    A = create_A(N)
        
    # Work out the forcing term r
    r = create_forcing_term(N,a,b,c,d)

    A_sparse = csr_matrix(A)
    w = linalg.spsolve(A_sparse,r).reshape((N,N))
        
    # Work out w where Aw = r
    # w = np.linalg.solve(A,r).reshape((N,N))
    
    return w, r, A, x, y
