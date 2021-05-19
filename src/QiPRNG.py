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
import scipy.stats
import hashlib

def find_principal_eig(A):
    """
    Computes the largest eigenvector/value pair for a given matrix.
    
    This function is needed because ARPACK uses random initialization
    points before applying Krylov iterations, and that leads to
    nondeterministic behaviour (not a great thing for a PRNG). Here we
    use a random orthonormal basis as pseudorandom starting points by
    seeding numpy's PRNG with a hash from the input matrix.

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
    print(A)
    state = np.random.get_state()
    # make the solver deterministic
    h = hashlib.sha256(str(A).encode('utf-8'))
    s = int.from_bytes(h.digest()[:8], "big")
    np.random.seed(np.uint32(s))
    
    # select a random orthonormal set of starting vectors
    xs = sp.stats.unitary_group.rvs(A.shape[0])
    
    # restore the previous state of numpy's PRNG
    np.random.set_state(state)
    
    # solving by 100 steps of power iteration for each starting vector
    for i in range(A.shape[0]):
        for j in range(100):
            xs[i] = A.dot(xs[i])
            xs[i] /= np.linalg.norm(xs[i],2)
    
    # find the best approximation of the principal eigenvector
    best_vec = xs[0]
    best_val = np.conjugate(best_vec).dot(A.dot(best_vec))
    for i in range(A.shape[0]):
        val = np.conjugate(xs[i]).dot(A.dot(xs[i]))
        if val > best_val:
            best_vec = xs[i]
            best_val = val
    
    return best_val, best_vec

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
    
    # constructing the initial state in the span{ |psi_j> } space
    current_state = T.dot(v0)
    
    # evolve the current state up to time N so we have
    # some nonzero amplitude on every state
    for _ in range(N):
        current_state = W.dot(current_state)
        current_state /= np.linalg.norm(current_state, 2)
    
    # M is the basis we'll be measuring in
    # we push it to the { |psi_j> } space here
    M = M.dot(T.transpose().conjugate())
    
    # the core loop: evolving the state and yielding the probabilities
    while True:
        # evolve to the next timestep
        current_state = W.dot(current_state)
        current_state /= np.linalg.norm(current_state, 2)
        
        # find the amplitudes in our chosen basis
        amps = M.dot(current_state)
        for j in range(N):
            # get the probabilities
            prob_j = np.real(amps[j])**2 + np.imag(amps[j])**2
            
            # get bits with a little-endian arrangement
            b = struct.pack("<d", prob_j)
            
            # yield the less significant half of the bytes
            for k in range(1, len(b) // 2):
                yield b[k]

# Quantum-inspired PRNG supporting Hamiltonians that have been tridiagonalized
def QiPRNG_tridiag(v0, alpha, beta, M, verbosity = 0):
    # the dimension of the walk
    N = len(alpha)
    
    # the Hamiltonian
    H = sp.sparse.diags([np.conj(beta),alpha,beta], [-1,0,1], dtype=np.complex128).tocsr()
    print(alpha)
    print(M)
    
    # building abs(H)
    abs_alpha = list(map(abs, alpha))
    abs_beta = list(map(abs, beta))
    A = sp.sparse.diags([abs_beta, abs_alpha, abs_beta], [-1,0,1], dtype=np.complex128)
    
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
        dev = np.max(np.abs(W.dot(W.transpose().conjugate(True)).todense() - np.eye(W.shape[0])))
        print("Finished constructing walk operator")
        print("Deviation from unitarity: ", dev)
    
    # constructing the initial state in the span{ |psi_j> } space
    current_state = T.dot(v0)
    
    # evolve the current state up to time N so we have
    # some nonzero amplitude on every state
    for _ in range(N):
        current_state = W.dot(current_state)
        current_state /= np.linalg.norm(current_state, 2)
    
    # M is the basis we'll be measuring in
    # we push it to the { |psi_j> } space here
    M = M.dot(T.transpose().conjugate(True).todense()).getA()
    
    # the core loop: evolving the state and yielding the probabilities
    while True:
        # evolve to the next timestep
        current_state = W.dot(current_state)
        current_state /= np.linalg.norm(current_state, 2)
        
        # find the amplitudes in our chosen basis
        amps = M.dot(current_state)
        
        for j in range(N):
            # get the probabilities
            prob_j = np.real(amps[j])**2 + np.imag(amps[j])**2
            
            # get bits with a little-endian arrangement
            b = struct.pack("<d", prob_j)
            
            # yield the less significant half of the bytes
            for k in range(1, len(b) // 2):
                yield b[k]

# Quantum-inspired PRNG supporting diagonal Hamiltonians
def QiPRNG_diag(v0, eigs, M, verbosity = 0):
    # the diagonal elements of the walk operator
    W = np.exp(np.array(eigs, dtype=np.complex128) * (0+1j))
    
    # the initial state
    current_state = v0
    
    # the core loop: evolving the state and yielding the probabilities
    while True:
        # evolve to the next timestep
        current_state = W * current_state
        current_state /= np.linalg.norm(current_state, 2)
        
        # find the amplitudes in the given basis
        amps = M.dot(current_state)
        for j in range(len(v0)):
            # get the probabilities
            prob_j = np.real(amps[j])**2 + np.imag(amps[j])**2
            
            # get bits with a little-endian arrangement
            b = struct.pack("<d", prob_j)
            
            # yield the less significant half of the bytes
            for k in range(1, len(b) // 2):
                yield b[k]

# Quantum-inspired PRNG supporting diagonal Hamiltonians
def QiPRNG_exact(v0, eigs, M, verbosity = 0):
    # the initial state
    initial_state = v0
    T = 0
    
    # the core loop: evolving the state and yielding the probabilities
    while True:
        # increment the time
        T += 1
        
        # compute the diagonal elements of the walk operator to the Tth power
        W = np.exp(np.array(eigs, dtype=np.complex128) * T * (0+1j))
    
        # evolve to the current timestep
        current_state = W * initial_state
        
        # find the amplitudes in the given basis
        amps = M.dot(current_state)
        for j in range(len(v0)):
            # get the probabilities
            prob_j = np.real(amps[j])**2 + np.imag(amps[j])**2
            
            # get bytes with a little-endian arrangement
            b = struct.pack("<d", prob_j)
            
            # yield the less significant half of the bytes
            # we don't yield the least significant byte though, since for
            # some reason the first bit in it is zero 60% of the time
            for k in range(1, len(b) // 2):
                yield b[k]

