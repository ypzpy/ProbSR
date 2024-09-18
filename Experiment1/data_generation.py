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
    m[0,0] = 1.
    m[1,0] = 1.
    m[N-1,N-1] = 1.
    m[N-2,N-1] = 1.
    
    for i in range(1,N-1):
        m[i,i] = -4.
        
    for i in range(1,N-2):
        m[i,i+1] = 1.
        m[i+1,i] = 1.
    
    return m


def create_rest(N):
    """
    Create the outer blocks of the A matrix (at boundaries)
    
    Args
    ----------
    N : int
        The size of PDE domain is (N,N)
    """
    m = np.zeros((N,N))
    m[1:N-1,1:N-1] = np.eye(N-2)
    
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


def create_forcing_term(N, a, b, c):
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
    
    h = 1/(N-1)
    N2 = (N-2)*(N-2)
    x = np.arange(0,1.0001,h)
    y = np.arange(0,1.0001,h)
    
    r_middle = np.zeros(N2)

    for i in range (0,N-2):
        for j in range (0,N-2):
            r_middle[i+(N-2)*j] = -(a*np.pi**2) * np.sin(b*np.pi*x[i+1]) * np.sin(c*np.pi*y[j+1])*h**2
                
    r = np.zeros(N**2)
    for i in range(N-2):
        r[N*(i+1)+1:N*(i+1)+N-1] = r_middle[(N-2)*i:(N-2)*i+(N-2)]
        
    return r


def generate_data(N,a,b,c):
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
    h = 1/(N-1)
    x = np.arange(0,1.0001,h)
    y = np.arange(0,1.0001,h)
    
    # Work out the matrix A
    A = create_A(N)
        
    # Work out the forcing term r
    r = create_forcing_term(N,a,b,c)

    A_sparse = csr_matrix(A)
    w = linalg.spsolve(A_sparse,r).reshape((N,N))
        
    # Work out w where Aw = r
    # w = np.linalg.solve(A,r).reshape((N,N))
    
    return w, r, A, x, y


def gaussian_kernal(x,y,l,sigma,N):
    """
    Work out the covariance of GP for the forcing term
    
    Args
    ----------
    x, y: ndarray
        x and y coordinates
    l, sigma: float
        hyperparameters of the covariance kernal
    N : int
        The size of PDE domain is (N,N)
    """
    m = N*N
    n = N*N
    dist_matrix = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            dist_matrix[i][j] = (y[i%N]-y[j%N])**2 + (x[i//N]-x[j//N])**2
    
    return sigma ** 2 * np.exp(-0.5 / l ** 2 * dist_matrix)


def u_covariance(l,sigma,N):
    """
    Work out the mean and covariance matrix of the prior of u
    
    Args
    ----------
    l, sigma: float
        hyperparameters of the covariance kernal
    N : int
        The size of PDE domain is (N,N)
    """
    h = 1/(N-1)
    x = np.arange(0,1.0001,h)
    y = np.arange(0,1.0001,h)
    
    A = create_A(N)
    G = gaussian_kernal(x,y,l,sigma,N)
    covariance_u = np.matmul(np.linalg.solve(A,G),np.linalg.inv(A).T)
    
    return covariance_u
