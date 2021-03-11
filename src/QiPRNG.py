#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:31:29 2021

Relevant paper: "Novel pseudo-random number generator based on quantum random walks"

@author: Aaron Gregory
"""

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import struct

def find_principal_eig(A):
    state = np.random.get_state()
    # make the solver deterministic
    np.random.seed(np.uint32(hash(str(A))))
    
    # select a random normalized starting vector
    x = np.random.random(A.shape[0]).astype(A.dtype) - 0.5
    x += np.random.random(A.shape[0]).astype(A.dtype) * (0+1j)
    x /= np.linalg.norm(x,2)
    
    # restore the previous state of numpy's PRNG
    np.random.set_state(state)
    
    # solving by gradient descent
#    for i in range(100):
#        grad = 2 * (A.dot(x) - np.conjugate(x).dot(A.dot(x)) * x)
#        x += grad / (i+1)
#        x /= np.linalg.norm(x,2)
    
    # solving by power iteration
    for i in range(100):
        x = A.dot(x)
        x /= np.linalg.norm(x,2)
    
    return np.conjugate(x).dot(A.dot(x)), x

# TODO: Implement the Lanczos algorithm
# The returned values will be a sparse matrix H_tridiag
# and a dense change of basis matrix X such that XHX^{-1} = H_tridiag
def Lanczos(H):
    return H, np.eye(H.shape[0])

# Quantum-inspired PRNG supporting dense Hamiltonians
def QiPRNG_dense(v0, H, M, verbosity = 0):
    # the dimension of the walk
    N = H.shape[0]
    
    # building abs(H)
    A = abs(H)
    
    # POTENTIAL PROBLEM: scipy.sparse.linalg.eigsh exhibits nondeterministic
    # behavior due to random starting points for iteration; the relevant code is
    # buried somewhere in the fortran of ARPACK. Here we use a custom method
    # instead. Less accuracy and efficiency, but deterministic.
    
    # finding |abs(H)| and |d> for equation (4)
    A_norm, d = find_principal_eig(A)
    
    # constructing T from equation (6)
    T = np.array( (N**2, N), dtype=np.complex128)
    for j in range(N):
        for k in range(max(0, j-1), min(j+2, N)):
            T[j * N + k,j] = np.sqrt((np.conjugate(H[j,k]) * d[k]) / (d[j] * A_norm))
    
    # constructing S from equation (5)
    S = np.array( (N**2, N**2), dtype=np.complex128)
    for j in range(N):
        for k in range(N):
            S[j * N + k, k * N + j] = 1
    
    # projector onto the |psi_j> space
    P = T.dot(T.conjugate(True).transpose())
    
    # reflector across the |psi_j> space
    R = 2 * P - np.eye(N**2)
    
    # the walk operator (just above equation (7))
    W = S.dot(R)
    
    if verbosity >= 1:
        # measuring how close to unitarity we are
        dev = np.max(np.abs(W.dot(W.transpose().conjugate(True)).todense() - np.eye(25)))
        print("Finished constructing walk operator")
        print("Deviation from unitarity: ", dev)
    
    # tridiagonalize W for efficient computation
    W_tridiag, X = Lanczos(W)
    
    # constructing the initial state in the span{ |psi_j> } space
    initial_state = T.dot(v0)
    current_state = initial_state
    
    # evolve the current state up to time N so we have
    # some nonzero amplitude on every state
    for _ in range(N):
        current_state = W.dot(current_state)
    
    # M is the basis we'll be measuring in
    # we push it to the { |psi_j> } space here
    M = M.dot(T.transpose().conjugate(True))
    
    # the core loop: evolving the state and yielding the probabilities
    while True:
        # evolve to the next timestep
        current_state = W.dot(current_state)
        
        # find the amplitudes in our chosen basis
        amps = M.dot(current_state)
        for j in range(N):
            # get the probabilities
            prob_j = np.real(amps[j])**2 + np.imag(amps[j])**2
            
            # get bits with a little-endian arrangement
            b = struct.pack("<f", prob_j)
            
            # yield the less significant half of the bytes
            for k in range(len(b) // 2):
                yield b[k]

# Quantum-inspired PRNG supporting Hamiltonians that have been tridiagonalized
def QiPRNG_tridiag(v0, alpha, beta, M, verbosity = 0):
    # the dimension of the walk
    N = len(alpha)
    
    # the Hamiltonian
    H = sp.sparse.diags([np.conj(beta),alpha,beta], [-1,0,1], dtype=np.complex128).tocsr()
    
    # building abs(H)
    abs_alpha = list(map(abs, alpha))
    abs_beta = list(map(abs, beta))
    A = sp.sparse.diags([abs_beta, abs_alpha, abs_beta], [-1,0,1], dtype=np.complex128)
    
    # POTENTIAL PROBLEM: scipy.sparse.linalg.eigsh exhibits nondeterministic
    # behavior due to random starting points for iteration; the relevant code is
    # buried somewhere in the fortran of ARPACK. Here we use a custom method
    # instead. Less accuracy and efficiency, but deterministic.
    
    # finding |abs(H)| and |d> for equation (4)
    A_norm, d = find_principal_eig(A)
    
    # constructing T from equation (6)
    T = sp.sparse.csr_matrix( (N**2, N), dtype=np.complex128)
    for j in range(N):
        for k in range(max(0, j-1), min(j+2, N)):
            T[j * N + k,j] = np.sqrt((np.conjugate(H[j,k]) * d[k]) / (d[j] * A_norm))
    
    # constructing S from equation (5)
    S = sp.sparse.csr_matrix( (N**2, N**2), dtype=np.complex128)
    for j in range(N):
        for k in range(N):
            S[j * N + k, k * N + j] = 1
    
    # projector onto the |psi_j> space
    P = T.dot(T.conjugate(True).transpose())
    
    # reflector across the |psi_j> space
    R = 2 * P - sp.sparse.eye(N**2)
    
    # the walk operator (just above equation (7))
    W = S.dot(R)
    
    if verbosity >= 1:
        # measuring how close to unitarity we are
        dev = np.max(np.abs(W.dot(W.transpose().conjugate(True)).todense() - np.eye(25)))
        print("Finished constructing walk operator")
        print("Deviation from unitarity: ", dev)
    
    # constructing the initial state in the span{ |psi_j> } space
    initial_state = T.dot(v0)
    current_state = initial_state
    
    # evolve the current state up to time N so we have
    # some nonzero amplitude on every state
    for _ in range(N):
        current_state = W.dot(current_state)
    
    # M is the basis we'll be measuring in
    # we push it to the { |psi_j> } space here
    M = M.dot(T.transpose().conjugate(True).todense()).getA()
    
    # the core loop: evolving the state and yielding the probabilities
    while True:
        # evolve to the next timestep
        current_state = W.dot(current_state)
        
        # find the amplitudes in our chosen basis
        amps = M.dot(current_state)
        
        for j in range(N):
            # get the probabilities
            prob_j = np.real(amps[j])**2 + np.imag(amps[j])**2
            
            # get bits with a little-endian arrangement
            b = struct.pack("<f", prob_j)
            
            # yield the less significant half of the bytes
            for k in range(len(b) // 2):
                yield b[k]

# Quantum-inspired PRNG supporting diagonal Hamiltonians
def QiPRNG_diag(v0, eigs, M, verbosity = 0):
    # the dimension of the walk
    N = len(eigs)
    
    # the diagonal elements of the walk operator
    W = np.exp(np.array(eigs, dtype=np.complex128) * (0+1j))
    
    # constructing the initial state in the span{ |psi_j> } space
    initial_state = v0
    current_state = initial_state
    
    # the core loop: evolving the state and yielding the probabilities
    while True:
        # evolve to the next timestep
        current_state = W * current_state
        
        # find the amplitudes in the given basis
        amps = M.dot(current_state)
        for j in range(N):
            # get the probabilities
            prob_j = np.real(amps[j])**2 + np.imag(amps[j])**2
            
            # get bits with a little-endian arrangement
            b = struct.pack("<f", prob_j)
            
            # yield the less significant half of the bytes
            for k in range(len(b) // 2):
                yield b[k]

def generate_datafile(filename, generator, num_bytes):
    print("generating %s:" % filename)
    update_period = num_bytes // 100
    with open(filename, 'wb') as f:
        for i in range(num_bytes):
            b = generator.__next__()
            f.write(b.to_bytes(1, 'big'))
            if i % update_period == 0:
                print('\r%3d%% complete' % (i // update_period), end="")
        print("\r100% complete")

# the initial state (human selected)
v0 = np.array([1, 2, 1, 1, 2], dtype=np.complex128)
v0 /= np.linalg.norm(v0)

# from Lanczos' algorithm, the diagonals in H
# these numbers have no particular meaning
# (i.e. they were written at random by a human)
alpha = [1, 2, 3, 4, 5]
beta = [1, 9, 8, 7 + 7j]

state = np.random.get_state()
# make the code deterministic
np.random.seed(1337)
    
# we need a measurement basis. Normally it should be unitary,
# but any matrix can be used.
M = np.random.random((5,5))

# restore the previous state of numpy's PRNG
np.random.set_state(state)

import time
t = time.time()
generate_datafile("data_diag_1e3.bin", QiPRNG_diag(v0, alpha, M), 1000)
generate_datafile("data_diag_1e4.bin", QiPRNG_diag(v0, alpha, M), 10000)
generate_datafile("data_diag_1e5.bin", QiPRNG_diag(v0, alpha, M), 100000)
generate_datafile("data_diag_1e6.bin", QiPRNG_diag(v0, alpha, M), 1000000)
print("Time elapsed: %.3f" % (time.time() - t))

t = time.time()
generate_datafile("data_tridiag_1e3.bin", QiPRNG_tridiag(v0, alpha, beta, M), 1000)
generate_datafile("data_tridiag_1e4.bin", QiPRNG_tridiag(v0, alpha, beta, M), 10000)
generate_datafile("data_tridiag_1e5.bin", QiPRNG_tridiag(v0, alpha, beta, M), 100000)
generate_datafile("data_tridiag_1e6.bin", QiPRNG_tridiag(v0, alpha, beta, M), 1000000)
print("Time elapsed: %.3f" % (time.time() - t))

