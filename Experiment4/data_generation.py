import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2
import math
from tqdm import tqdm
from torch.utils.data import dataloader
import scipy
from scipy.sparse import csr_matrix, linalg, lil_matrix
from scipy.sparse.linalg import spsolve


def create_I(N):
    
    bigI = lil_matrix((N,N))

    for i in range(N):
        bigI[i,i] = 1

    return bigI

def create_minusI(N):

    minusI = lil_matrix((N,N))

    for i in range(1,N-1):
        minusI[i,i] = -1

    return minusI

def create_minusAI(N,empty_rows):

    minusAI = lil_matrix((N,N))

    num_values = (N-2-empty_rows-2)//2

    for i in range(1,1+num_values):
        minusAI[i,i] = -1
        minusAI[-(i+1),-(i+1)] = -1

    return minusAI

def create_minusBI(N,empty_rows):

    minusBI = lil_matrix((N,N-empty_rows))

    num_values = (N-2-empty_rows-2)//2

    for i in range(1,1+num_values):
        minusBI[i,i] = -1
        minusBI[-(i+1),-(i+1)] = -1

    return minusBI

def create_minusCI(N,empty_rows):

    minusCI = lil_matrix((N-empty_rows,N))

    num_values = (N-2-empty_rows-2)//2

    for i in range(1,1+num_values):
        minusCI[i,i] = -1
        minusCI[-(i+1),-(i+1)] = -1

    return minusCI


def create_minusDI(N,empty_rows):

    minusDI = lil_matrix((N-empty_rows,N-empty_rows))

    num_values = (N-2-empty_rows-2)//2

    for i in range(1,1+num_values):
        minusDI[i,i] = -1
        minusDI[-(i+1),-(i+1)] = -1

    return minusDI


def create_A(N):

    A = lil_matrix((N,N))

    A[0,0] = 1
    A[-1,-1] = 1
    
    for i in range(1,N-1):
        A[i,i] = 4
        A[i,i-1] = -1
        A[i,i+1] = -1

    return A

def create_B(N,empty_rows):

    B = lil_matrix((N,N))

    B[0,0] = 1
    B[-1,-1] = 1

    num_values = (N-2-empty_rows-2)//2

    for i in range(1,1+num_values):

        B[i,i] = 4
        B[i,i-1] = -1
        B[i,i+1] = -1

        B[-(i+1),-(i+1)] = 4
        B[-(i+1),-i] = -1
        B[-(i+1),-(i+2)] = -1

    for i in range(1+num_values,1+num_values+2+empty_rows):
        B[i,i] = 1

    return B

def create_C(N,empty_rows):

    C = lil_matrix((N-empty_rows,N-empty_rows))

    C[0,0] = 1
    C[-1,-1] = 1

    num_values = (N-2-empty_rows-2)//2

    for i in range(1,1+num_values):

        C[i,i] = 4
        C[i,i-1] = -1
        C[i,i+1] = -1

        C[-(i+1),-(i+1)] = 4
        C[-(i+1),-i] = -1
        C[-(i+1),-(i+2)] = -1

    for i in range(1+num_values,1+num_values+2):

        C[i,i] = 1

    return C

def create_bigA(N):

    empty_rows = ((N-1) // 8) * 2 - 1
    dimension = N*(N-empty_rows) + (N-empty_rows)*empty_rows
    bigA = lil_matrix((dimension,dimension))

    I = create_I(N)
    minusI = create_minusI(N)
    minusAI = create_minusAI(N,empty_rows)
    minusBI = create_minusBI(N,empty_rows)
    minusCI = create_minusCI(N,empty_rows)
    minusDI = create_minusDI(N,empty_rows)
    A = create_A(N)
    B = create_B(N,empty_rows)
    C = create_C(N,empty_rows)

    bigA[0:N,0:N] = I

    num_values = (N-2-empty_rows-2)//2

    for i in range(num_values):
        bigA[(i+1)*N:(i+2)*N,i*N:(i+1)*N] = minusI
        bigA[(i+1)*N:(i+2)*N,(i+1)*N:(i+2)*N] = A
        bigA[(i+1)*N:(i+2)*N,(i+2)*N:(i+3)*N] = minusI

    bigA[(1+num_values)*N:(2+num_values)*N,num_values*N:(1+num_values)*N] = minusAI
    bigA[(1+num_values)*N:(2+num_values)*N,(1+num_values)*N:(2+num_values)*N] = B
    bigA[(1+num_values)*N:(2+num_values)*N,(2+num_values)*N:(2+num_values)*N+N-empty_rows] = minusBI

    if empty_rows == 1:
        
        bigA[(2+num_values)*N:(2+num_values)*N+N-empty_rows,(1+num_values)*N:(2+num_values)*N] = minusCI
        bigA[(2+num_values)*N:(2+num_values)*N+N-empty_rows,(2+num_values)*N:(2+num_values)*N+N-empty_rows] = C
        bigA[(2+num_values)*N:(2+num_values)*N+N-empty_rows,(2+num_values)*N+N-empty_rows:(3+num_values)*N+N-empty_rows] = minusCI

    else:

        bigA[(2+num_values)*N:(2+num_values)*N+N-empty_rows,(1+num_values)*N:(2+num_values)*N] = minusCI
        bigA[(2+num_values)*N:(2+num_values)*N+N-empty_rows,(2+num_values)*N:(2+num_values)*N+N-empty_rows] = C
        bigA[(2+num_values)*N:(2+num_values)*N+N-empty_rows,(2+num_values)*N+N-empty_rows:(2+num_values)*N+2*(N-empty_rows)] = minusDI

        for i in range(empty_rows-2):

            bigA[(2+num_values)*N+(i+1)*(N-empty_rows):(2+num_values)*N+(i+2)*(N-empty_rows),(2+num_values)*N+i*(N-empty_rows):(2+num_values)*N+(i+1)*(N-empty_rows)] = minusDI
            bigA[(2+num_values)*N+(i+1)*(N-empty_rows):(2+num_values)*N+(i+2)*(N-empty_rows),(2+num_values)*N+(i+1)*(N-empty_rows):(2+num_values)*N+(i+2)*(N-empty_rows)] = C
            bigA[(2+num_values)*N+(i+1)*(N-empty_rows):(2+num_values)*N+(i+2)*(N-empty_rows),(2+num_values)*N+(i+2)*(N-empty_rows):(2+num_values)*N+(i+3)*(N-empty_rows)] = minusDI

        bigA[(2+num_values)*N+(empty_rows-1)*(N-empty_rows):(2+num_values)*N+empty_rows*(N-empty_rows),(2+num_values)*N+(empty_rows-2)*(N-empty_rows):(2+num_values)*N+(empty_rows-1)*(N-empty_rows)] = minusDI
        bigA[(2+num_values)*N+(empty_rows-1)*(N-empty_rows):(2+num_values)*N+empty_rows*(N-empty_rows),(2+num_values)*N+(empty_rows-1)*(N-empty_rows):(2+num_values)*N+empty_rows*(N-empty_rows)] = C
        bigA[(2+num_values)*N+(empty_rows-1)*(N-empty_rows):(2+num_values)*N+empty_rows*(N-empty_rows),(2+num_values)*N+empty_rows*(N-empty_rows):(3+num_values)*N+empty_rows*(N-empty_rows)] = minusCI

    bigA[(2+num_values)*N+empty_rows*(N-empty_rows):(3+num_values)*N+empty_rows*(N-empty_rows),(2+num_values)*N+(empty_rows-1)*(N-empty_rows):(2+num_values)*N+empty_rows*(N-empty_rows)] = minusBI
    bigA[(2+num_values)*N+empty_rows*(N-empty_rows):(3+num_values)*N+empty_rows*(N-empty_rows),(2+num_values)*N+empty_rows*(N-empty_rows):(3+num_values)*N+empty_rows*(N-empty_rows)] = B
    bigA[(2+num_values)*N+empty_rows*(N-empty_rows):(3+num_values)*N+empty_rows*(N-empty_rows),(3+num_values)*N+empty_rows*(N-empty_rows):(4+num_values)*N+empty_rows*(N-empty_rows)] = minusAI

    for i in range(num_values):
        bigA[(2+num_values)*N+empty_rows*(N-empty_rows)+(i+1)*N:(2+num_values)*N+empty_rows*(N-empty_rows)+(i+2)*N,(2+num_values)*N+empty_rows*(N-empty_rows)+i*N:(2+num_values)*N+empty_rows*(N-empty_rows)+(i+1)*N] = minusI
        bigA[(2+num_values)*N+empty_rows*(N-empty_rows)+(i+1)*N:(2+num_values)*N+empty_rows*(N-empty_rows)+(i+2)*N,(2+num_values)*N+empty_rows*(N-empty_rows)+(i+1)*N:(2+num_values)*N+empty_rows*(N-empty_rows)+(i+2)*N] = A
        bigA[(2+num_values)*N+empty_rows*(N-empty_rows)+(i+1)*N:(2+num_values)*N+empty_rows*(N-empty_rows)+(i+2)*N,(2+num_values)*N+empty_rows*(N-empty_rows)+(i+2)*N:(2+num_values)*N+empty_rows*(N-empty_rows)+(i+3)*N] = minusI

    bigA[dimension-N:dimension,dimension-N:dimension] = I

    return bigA


def create_forcing_term(N,a,b,c,d):
    
    h = 1/(N-1)
    x = np.arange(0,1.0001,h)
    y = np.arange(0,1.0001,h)

    empty_rows = ((N-1) // 8) * 2 - 1
    dimension = N*(N-empty_rows) + (N-empty_rows)*empty_rows
    num_values = (N-2-empty_rows-2)//2

    r = np.zeros(dimension)

    # top row
    i = 1+num_values
    for j in range(1+num_values,1+num_values+empty_rows+2):
        position = i * N + j
        coordinate = (j-(1+num_values))*h
        r[position] = -10 * coordinate * (coordinate-0.25)

    # middle rows
    for k in range(empty_rows):
        i = 1 + num_values + k + 1
        j1 = 1+num_values
        j2 = 1+num_values+empty_rows+1

        coordinate = (i-(1+num_values))*h

        position1 = (2+num_values) * N + k * (N-empty_rows) + 1 + num_values
        position2 = (2+num_values) * N + k * (N-empty_rows) + 1 + num_values + 1

        r[position1] = - 15 * coordinate * (coordinate-0.25)
        r[position2] = - 10 * coordinate * (coordinate-0.25)

    # bottom row
    i = 1 + num_values + 1 + empty_rows
    for j in range(1+num_values,1+num_values+empty_rows+2):
        position = (2+num_values) * N + empty_rows*(N-empty_rows) + j
        coordinate = (j-(1+num_values))*h
        r[position] = - 12 * coordinate * (coordinate-0.25)

    return r


def transfer_to_u_whole(N,u):

    h = 1/(N-1)
    x = np.arange(0,1.0001,h)
    y = np.arange(0,1.0001,h)

    # Top rows
    empty_rows = ((N-1) // 8) * 2 - 1
    dimension = N*(N-empty_rows) + (N-empty_rows)*empty_rows
    num_values = (N-2-empty_rows-2)//2

    u_whole = np.zeros((N,N))

    for i in range(2+num_values):
        for j in range(N):
            position = i * N + j
            u_whole[i][j] = u[position]

    # middle rows
    for k in range(empty_rows):
        i = 1 + num_values + k + 1
        
        for j in range(0,2+num_values):
            position = (2+num_values) * N + k * (N-empty_rows) + j
            u_whole[i][j] = u[position]

        for j in range(2+num_values+empty_rows,N):
            position = (2+num_values) * N + k * (N-empty_rows) + j - empty_rows
            u_whole[i][j] = u[position]

    # bottom rows
    for i in range(2+num_values+empty_rows,N):
        for j in range(N):
            position = i * N - empty_rows**2 + j

            u_whole[i][j] = u[position]

    return u_whole
        

def generate_data(N,a,b,c,d):
    
    # Initialisation
    h = 1/(N-1)
    x = np.arange(0,1.0001,h)
    y = np.arange(0,1.0001,h)

    A = create_bigA(N)
    r = create_forcing_term(N,a,b,c,d)
    u = linalg.spsolve(A,r)
    
    return u, r, A


def create_H(N_low,N_high):

    empty_rows_low = ((N_low-1) // 8) * 2 - 1
    num_values_low = (N_low-2-empty_rows_low-2)//2
    dimension_low = N_low*(N_low-empty_rows_low) + (N_low-empty_rows_low)*empty_rows_low
    empty_rows_high = ((N_high-1) // 8) * 2 - 1
    num_values_high = (N_high-2-empty_rows_high-2)//2
    dimension_high = N_high*(N_high-empty_rows_high) + (N_high-empty_rows_high)*empty_rows_high
    H = lil_matrix((dimension_low,dimension_high))

    A = lil_matrix((N_low,N_high))
    for i in range(N_low):
        A[i,4*i] = 1

    B = lil_matrix((N_low-empty_rows_low,N_high-empty_rows_high))
    for i in range(2+num_values_low):
        B[i,4*i] = 1
    
    for j in range((N_low-empty_rows_low)//2):
        B[i+1+j,4*i+1+4*j] = 1

    for i in range(2+num_values_low):
        H[i*N_low:(i+1)*N_low,4*i*N_high:(4*i+1)*N_high] = A

    for i in range(empty_rows_low):
        H[(2+num_values_low)*N_low+i*(N_low-empty_rows_low):(2+num_values_low)*N_low+(i+1)*(N_low-empty_rows_low),(2+num_values_high)*N_high+(4*i+3)*(N_high-empty_rows_high):(2+num_values_high)*N_high+(4*i+4)*(N_high-empty_rows_high)] = B

    for i in range(2+num_values_low):
        H[(2+num_values_low)*N_low+empty_rows_low*(N_low-empty_rows_low)+i*N_low:(2+num_values_low)*N_low+empty_rows_low*(N_low-empty_rows_low)+(i+1)*N_low,(2+num_values_high)*N_high+empty_rows_high*(N_high-empty_rows_high)+4*i*N_high:(2+num_values_high)*N_high+empty_rows_high*(N_high-empty_rows_high)+(4*i+1)*N_high] = A

    return H


def partial_bicubic(N_low,N_high,u):

    empty_rows_low = ((N_low-1) // 8) * 2 - 1
    num_values_low = (N_low-2-empty_rows_low-2)//2

    empty_rows_high = ((N_high-1) // 8) * 2 - 1
    num_values_high = (N_high-2-empty_rows_high-2)//2

    u_top_low = u[:(num_values_low+2)*N_low].reshape((num_values_low+2),N_low)

    u_bottom_low = u[-(num_values_low+2)*N_low:].reshape((num_values_low+2),N_low)

    u_left_low = np.zeros((empty_rows_low,2+num_values_low))
    u_right_low = np.zeros((empty_rows_low,2+num_values_low))

    for i in range(empty_rows_low):
        for j in range(2+num_values_low):
            u_left_low[i][j] = u[N_low*(2+num_values_low)+i*(N_low-empty_rows_low)+j] 
            u_right_low[i][j] = u[N_low*(2+num_values_low)+i*(N_low-empty_rows_low)+(N_low-empty_rows_low)//2+j]

    u_top_sr = scipy.ndimage.zoom(u_top_low,((2+num_values_high)/(2+num_values_low),N_high/N_low))
    u_bottom_sr = scipy.ndimage.zoom(u_bottom_low,((2+num_values_high)/(2+num_values_low),N_high/N_low))
    u_left_sr = scipy.ndimage.zoom(u_left_low,(empty_rows_high/empty_rows_low,(2+num_values_high)/(2+num_values_low)))
    u_right_sr = scipy.ndimage.zoom(u_right_low,(empty_rows_high/empty_rows_low,(2+num_values_high)/(2+num_values_low)))

    u_sr = np.zeros(((N_high*(N_high-empty_rows_high) + (N_high-empty_rows_high)*empty_rows_high),1))

    # top
    u_sr[:(num_values_high+2)*N_high] = u_top_sr.reshape(-1,1)

    # bottom
    u_sr[-(num_values_high+2)*N_high:] = u_bottom_sr.reshape(-1,1)

    # left
    for i in range(empty_rows_high):
        for j in range(2+num_values_high):
            u_sr[N_high*(2+num_values_high)+i*(N_high-empty_rows_high)+j] = u_left_sr[i][j]
            u_sr[N_high*(2+num_values_high)+i*(N_high-empty_rows_high)+(N_high-empty_rows_high)//2+j] = u_right_sr[i][j]

    return u_sr