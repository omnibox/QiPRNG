#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2021 Aaron Gregory
This program is distributed under the terms of the GNU General Public License.

This file contains four distinct implementations of QiPRNG, a quantum-inspired
pseudorangom number generator. All are mathematically equivalent in exact
arithmetic, varying only in the effect finite precision math  has on their
output. This is determined by the format of the Hamiltonians: we give dense,
tridiagonal, diagonal, and exact versions.

QiPRNG is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

QiPRNG is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with QiPRNG. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import struct

def find_principal_eig(A):
    """
    Computes the largest eigenvector/value pair for a given matrix.
    
    This function is needed because ARPACK uses random initialization
    points before applying Krylov iterations, and that leads to
    nondeterministic behaviour (not a great thing for a PRNG). Here we
    use a pseudorandom starting point by seeding numpy's PRNG with a hash
    from the input matrix.

    Parameters
    ----------
    A : 2D numpy or scipy array
        The matrix to find the largest eigenvector/value pair for.

    Returns
    -------
    lambda : float
        The largest eigenvalue of A.
    x : numpy array
        The eigenvector of A with eigenvalue lambda.

    """
    state = np.random.get_state()
    # make the solver deterministic
    np.random.seed(np.uint32(hash(str(A))))
    
    # select a random normalized starting vector
    x = np.random.random(A.shape[0]).astype(np.complex128) - 0.5
    x += np.random.random(A.shape[0]).astype(np.complex128) * (0+1j) - (0 + 0.5j)
    x /= np.linalg.norm(x,2)
    
    # restore the previous state of numpy's PRNG
    np.random.set_state(state)
    
    # solving by 100 steps of power iteration
    for i in range(100):
        x = A.dot(x)
        x /= np.linalg.norm(x,2)
    
    return np.conjugate(x).dot(A.dot(x)), x

# The returned values will be a sparse matrix H_tridiag
# and a dense change of basis matrix X such that XHX^{-1} = H_tridiag
# WARNING: The Lanczos algorithm is numerically unstable.
# Householder rotations are the preferred method of tridiagonalization.
def Lanczos(H, verbosity = 0):
    state = np.random.get_state()
    # make the algorithm deterministic
    np.random.seed(np.uint32(hash(str(H))))
    
    # select a random normalized starting vector
    v = np.random.random(H.shape[0]).astype(np.complex128) - 0.5
    v += np.random.random(H.shape[0]).astype(np.complex128) * (0+1j) - (0 + 0.5j)
    v /= np.linalg.norm(v,2)
    
    alpha = np.zeros(H.shape[0], dtype=np.complex128)
    beta = np.zeros(H.shape[0] - 1, dtype=np.complex128)
    X = np.zeros(H.shape, dtype=np.complex128)
    X[:,0] = v
    
    wp = H.dot(X[:,0])
    alpha[0] = wp.conjugate().dot(X[:,0])
    w = wp - alpha[0] * X[:,0]
    
    for j in range(1, H.shape[0]):
        beta[j-1] = np.linalg.norm(w)
        if beta[j-1] != 0:
            if verbosity >= 2:
                print("Warning: beta[%d] = 0 found in Lanczos(...)" % (j-1))
            
            X[:,j] = w / beta[j-1]
        else:
            v = np.random.random(H.shape[0]).astype(np.complex128) - 0.5
            v += np.random.random(H.shape[0]).astype(np.complex128) * (0+1j) - (0 + 0.5j)
            for k in range(j):
                v -= X[:,k] * X[:,k].conjugate().dot(v)
            
            v /= np.linalg.norm(v,2)
            X[:,j] = v
        
        wp = H.dot(X[:,j])
        alpha[j] = wp.conjugate().dot(X[:,j])
        w = wp - alpha[j] * X[:,j] - beta[j-1] * X[:,j-1]
    
    # restore the previous state of numpy's PRNG
    np.random.set_state(state)
    
    H_tridiag = sp.sparse.diags([np.conj(beta),alpha,beta], [-1,0,1], dtype=np.complex128).tocsr()
    
    if verbosity >= 1:
        unit_dev = np.max(X.conjugate().transpose().dot(X) - np.eye(H.shape[0])).real
        trid_dev = np.max(X.conjugate().transpose().dot(H.dot(X)) - H_tridiag).real
        print("Deviation from unitarity:", unit_dev)
        print("Deviation from tridiagonality:", trid_dev)
    
    return H_tridiag, X

def Householder(H, verbosity = 0):
    v = H[:,0].copy()
    v[0] = 0
    v[1] += np.linalg.norm(H[1:,0]) * H[1,0] / abs(H[1,0])
    v /= np.linalg.norm(v)
    
    P = np.eye(H.shape[0]) - 2 * np.outer(v,v.conjugate())
    A = P.dot(H.dot(P))
    X = P
    
    for k in range(1,H.shape[0]-2):
        v = A[:,k].copy()
        v[:k+1] = 0
        v[k+1] += np.linalg.norm(A[k+1:,k]) * A[k+1,k] / abs(A[k+1,k])
        v /= np.linalg.norm(v)
        
        P = np.eye(A.shape[0]) - 2 * np.outer(v,v.conjugate())
        A = P.dot(A.dot(P))
        X = P.dot(X)
    
    alpha = np.zeros(H.shape[0], np.complex128)
    beta = np.zeros(H.shape[0]-1, np.complex128)
    alpha[0] = A[0,0]
    for j in range(1,H.shape[0]):
        alpha[j] = A[j,j]
        beta[j-1] = A[j-1,j]
    
    H_tridiag = sp.sparse.diags([np.conj(beta),alpha,beta], [-1,0,1], dtype=np.complex128).tocsr()
    
    if verbosity >= 1:
        unit_dev = np.max(X.dot(X.conjugate().transpose()) - np.eye(H.shape[0])).real
        trid_dev = np.max(X.dot(H.dot(X.conjugate().transpose())) - H_tridiag).real
        print("Deviation from unitarity:", unit_dev)
        print("Deviation from tridiagonality:", trid_dev)
    
    return H_tridiag, X

# Quantum-inspired PRNG supporting dense Hamiltonians
def QiPRNG_dense(v0, H, M, verbosity = 0):
    """
    Implementation of QiPRNG for dense Hamiltonians. Given information
    specifying a quantum system, constructs a DTQW and yields the least
    significant bits from the measurement probabilities.

    Parameters
    ----------
    v0 : 1D numpy array
        The initial state for the walk.
    H : 2D numpy array
        The Hamiltonian for the walk.
    M : 2D numpy array
        The measurement basis.
    verbosity : int, optional
        Level of messages to print. The default is 0, which means silence.

    Yields
    ------
    int
        A stream of pseudorandom bytes.

    """
    # the dimension of the walk
    N = H.shape[0]
    
    # building abs(H)
    A = abs(H)
    
    # POTENTIAL PROBLEM: scipy.sparse.linalg.eigsh exhibits nondeterministic
    # behavior due to random starting points for iteration; the relevant code is
    # buried somewhere in the fortran of ARPACK. Here we use a custom method
    # instead. Lower accuracy and efficiency, but deterministic.
    
    # finding |abs(H)| and |d> for equation (4)
    A_norm, d = find_principal_eig(A)
    
    # constructing T from equation (6)
    T = np.zeros( (N**2, N), dtype=np.complex128)
    for j in range(N):
        for k in range(max(0, j-1), min(j+2, N)):
            T[j * N + k,j] = np.sqrt((np.conjugate(H[j,k]) * d[k]) / (d[j] * A_norm))
    
    # constructing S from equation (5)
    S = np.zeros( (N**2, N**2), dtype=np.complex128)
    for j in range(N):
        for k in range(N):
            S[j * N + k, k * N + j] = 1
    
    # projector onto the |psi_j> space
    P = T.dot(T.conjugate().transpose())
    
    # reflector across the |psi_j> space
    R = 2 * P - np.eye(N**2)
    
    # the walk operator (just above equation (7))
    W = S.dot(R)
    
    if verbosity >= 1:
        # measuring how close to unitarity we are
        dev = np.max(np.abs(W.dot(W.transpose().conjugate()).todense() - np.eye(25)))
        print("Finished constructing walk operator")
        print("Deviation from unitarity: ", dev)
    
    # tridiagonalize W for efficient computation
    # W_tridiag = XWX^{-1}
    W_tridiag, X = Lanczos(W)
    # TODO: Currently this function is called, but it is not used.
    
    # constructing the initial state in the span{ |psi_j> } space
    initial_state = T.dot(v0)
    current_state = initial_state
    
    # evolve the current state up to time N so we have
    # some nonzero amplitude on every state
    for _ in range(N):
        current_state = W.dot(current_state)
    
    # M is the basis we'll be measuring in
    # we push it to the { |psi_j> } space here
    M = M.dot(T.transpose().conjugate())
    
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

# # the initial state (human selected)
# v0 = np.array([1, 2, 1, 1, 2], dtype=np.complex128)
# v0 /= np.linalg.norm(v0)

# # from Lanczos' algorithm, the diagonals in H
# # these numbers have no particular meaning
# # (i.e. they were written at random by a human)
# alpha = [1, 2, 3, 4, 5]
# beta = [1, 9, 8, 7 + 7j]

# state = np.random.get_state()
# # make the code deterministic
# np.random.seed(1337)
    
# # we need a measurement basis. Normally it should be unitary,
# # but any matrix can be used.
# M = np.random.random((5,5))

# # Let's choose a random dense Hamiltonian while we're here
# H = np.random.random((5,5))
# H += H.transpose()

# # restore the previous state of numpy's PRNG
# np.random.set_state(state)


# # Now we generate the binary files

# import time
# t = time.time()
# generate_datafile("data_diag_1e3.bin", QiPRNG_diag(v0, alpha, M), 1000)
# generate_datafile("data_diag_1e4.bin", QiPRNG_diag(v0, alpha, M), 10000)
# generate_datafile("data_diag_1e5.bin", QiPRNG_diag(v0, alpha, M), 100000)
# generate_datafile("data_diag_1e6.bin", QiPRNG_diag(v0, alpha, M), 1000000)
# print("Time elapsed: %.3f" % (time.time() - t))

# t = time.time()
# generate_datafile("data_tridiag_1e3.bin", QiPRNG_tridiag(v0, alpha, beta, M), 1000)
# generate_datafile("data_tridiag_1e4.bin", QiPRNG_tridiag(v0, alpha, beta, M), 10000)
# generate_datafile("data_tridiag_1e5.bin", QiPRNG_tridiag(v0, alpha, beta, M), 100000)
# generate_datafile("data_tridiag_1e6.bin", QiPRNG_tridiag(v0, alpha, beta, M), 1000000)
# print("Time elapsed: %.3f" % (time.time() - t))

# t = time.time()
# generate_datafile("data_dense_1e3.bin", QiPRNG_dense(v0, H, M), 1000)
# generate_datafile("data_dense_1e4.bin", QiPRNG_dense(v0, H, M), 10000)
# generate_datafile("data_dense_1e5.bin", QiPRNG_dense(v0, H, M), 100000)
# generate_datafile("data_dense_1e6.bin", QiPRNG_dense(v0, H, M), 1000000)
# print("Time elapsed: %.3f" % (time.time() - t))


# # And run some tests on one of the sequences generated

# import sys
# sys.path.append('../sp800_22_tests_python3')
# from sp800_22_tests import run_tests

# results = run_tests("data_dense_1e3.bin")

H = np.random.random((4,4)).astype(np.complex128)
H += np.random.random((4,4)).astype(np.complex128) * (0 + 1j)
H += H.conjugate().transpose()
H_t, X = Householder(H, 1)

