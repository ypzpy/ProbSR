import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2
import math
from tqdm import tqdm
from tqdm.std import trange
from torch.utils.data import dataloader
from scipy.sparse import csr_matrix, linalg, lil_matrix


def create_subA(N):

    subA = lil_matrix((N,N))

    subA[0,0] = 1
    subA[-1,-1] = 1

    for i in range(1,N-1):
        subA[i,i-1] = -1
        subA[i,i] = 6
        subA[i,i+1] = -1

    return subA


def create_subB(N):
    subB = lil_matrix((N,N))

    for i in range(1,N-1):
        subB[i,i] = -1

    return subB


def create_subI(N):

    subI = lil_matrix((N,N))

    for i in range(N):
        subI[i,i] = 1

    return subI


def create_bigA(N):

    bigA = lil_matrix((N**2,N**2))

    bigA[0:N,0:N] = create_subI(N)
    bigA[N*(N-1):N*N,N*(N-1):N*N] = create_subI(N)

    for i in range(1,N-1):
        bigA[N*i:N*(i+1),N*(i-1):N*i] = create_subB(N)
        bigA[N*i:N*(i+1),N*i:N*(i+1)] = create_subA(N)
        bigA[N*i:N*(i+1),N*(i+1):N*(i+2)] = create_subB(N)

    return bigA


def create_bigB(N):

    bigB = lil_matrix((N**2,N**2))

    for i in range(1,N-1):
        bigB[N*i:N*(i+1),N*i:N*(i+1)] = create_subB(N)

    return bigB


def create_bigI(N):

    bigI = lil_matrix((N**2,N**2))

    for i in range(N**2):
        bigI[i,i] = 1

    return bigI


def create_A(N):

    A = lil_matrix((N**3,N**3))

    A[0:N**2,0:N**2] = create_bigI(N)
    A[(N**2)*(N-1):N**3,(N**2)*(N-1):N**3] = create_bigI(N)

    bigA = create_bigA(N)
    bigB = create_bigB(N)

    for i in range(1,N-1):
        A[(N**2)*i:(N**2)*(i+1),(N**2)*(i-1):(N**2)*i] = bigB
        A[(N**2)*i:(N**2)*(i+1),(N**2)*i:(N**2)*(i+1)] = bigA
        A[(N**2)*i:(N**2)*(i+1),(N**2)*(i+1):(N**2)*(i+2)] = bigB

    return A


def create_forcing_term(N, a, b, c, d):

    h = 1/(N-1)
    x = np.arange(0,1.0001,h)
    y = np.arange(0,1.0001,h)
    z = np.arange(0,1.0001,h)

    r = np.ones(N**3) * d

    for i in range(1,N-1):
        for j in range(1,N-1):
            for k in range(1,N-1):
                position = N**2 * i + N * j + k
                r[position] = h**2 * (np.sin(a*np.pi*x[i]) * np.cos(b*np.pi*y[j]) * np.exp(-c*z[k]) + np.sin(b*np.pi*y[j]) * np.cos(c*np.pi*z[k]) * np.exp(-a*x[i]) + np.sin(c*np.pi*z[k]) * np.cos(a*np.pi*x[i]) * np.exp(-b*y[j]))

    return r


def generate_data(N,a,b,c,d):

    # Initialisation
    h = 1/(N-1)
    x = np.arange(0,1.0001,h)
    y = np.arange(0,1.0001,h)
    z = np.arange(0,1.0001,h)
    
    # Work out the matrix A
    A = create_A(N)
        
    # Work out the forcing term r
    r = create_forcing_term(N,a,b,c,d)

    w = linalg.spsolve(A,r).reshape((N,N,N))
    
    return w, r, A, x, y, z
