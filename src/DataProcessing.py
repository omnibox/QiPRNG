#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2021 Aaron Gregory
This program is distributed under the terms of the GNU General Public License.

This file contains code to generate files with output from QiPRNG, and then to
perform statistical analysis on those files. Generation is from a seed which is
used to initial numpy's internal PRNG; this allows for repeatability.

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
import scipy.stats
from QiPRNG import QiPRNG_diag, QiPRNG_tridiag, QiPRNG_dense

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
    
    if verbosity >= 1:
        H_tridiag = sp.sparse.diags([np.conj(beta),alpha,beta], [-1,0,1], dtype=np.complex128).tocsr()
        
        unit_dev = np.max(X.conjugate().transpose().dot(X) - np.eye(H.shape[0])).real
        trid_dev = np.max(X.conjugate().transpose().dot(H.dot(X)) - H_tridiag).real
        print("Deviation from unitarity:", unit_dev)
        print("Deviation from tridiagonality:", trid_dev)
    
    return alpha, beta, X

def Householder(H, verbosity = 0):
    v = H[:,0].copy()
    v[0] = 0
    v[1] += np.linalg.norm(H[1:,0]) * H[1,0] / abs(H[1,0])
    v /= np.linalg.norm(v)
    
    R = np.eye(H.shape[0]) - 2 * np.outer(v,v.conjugate())
    A = R.dot(H.dot(R))
    X = R
    
    for k in range(1,H.shape[0]-2):
        v = A[:,k].copy()
        v[:k+1] = 0
        v[k+1] += np.linalg.norm(A[k+1:,k]) * A[k+1,k] / abs(A[k+1,k])
        v /= np.linalg.norm(v)
        
        R = np.eye(A.shape[0]) - 2 * np.outer(v,v.conjugate())
        A = R.dot(A.dot(R))
        X = R.dot(X)
    
    alpha = np.zeros(H.shape[0], np.complex128)
    beta = np.zeros(H.shape[0]-1, np.complex128)
    alpha[0] = A[0,0]
    for j in range(1,H.shape[0]):
        alpha[j] = A[j,j]
        beta[j-1] = A[j-1,j]
    
    if verbosity >= 1:
        H_tridiag = sp.sparse.diags([np.conj(beta),alpha,beta], [-1,0,1], dtype=np.complex128).tocsr()
        
        unit_dev = np.max(X.dot(X.conjugate().transpose()) - np.eye(H.shape[0])).real
        trid_dev = np.max(X.dot(H.dot(X.conjugate().transpose())) - H_tridiag).real
        print("Deviation from unitarity:", unit_dev)
        print("Deviation from tridiagonality:", trid_dev)
    
    return alpha, beta, X

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

def construct_PRNG_tuple(seed, n, verbose = 0):
    # save the state of numpy's PRNG
    state = np.random.get_state()
    
    # make the code deterministic
    np.random.seed(1357)
    
    # select a random normalized starting vector
    # since elements are normally distrbuted there is no angular bias
    v0 = np.random.normal(size = n).astype(np.complex128) - 0.5
    v0 += np.random.normal(size = n).astype(np.complex128) * (0+1j) - (0 + 0.5j)
    v0 /= np.linalg.norm(v0,2)
    
    # generating the eigenvalues in [0,1) for the diagonal system
    eigs = np.random.uniform(size = n).astype(np.complex128)
    
    # we need a measurement basis
    M = sp.stats.unitary_group.rvs(n)
    
    # enerating corresponding dense and tridiagonal Hamiltonians
    X = sp.stats.unitary_group.rvs(n)
    H_dense = X.dot(sp.sparse.diags([eigs], [0]).dot(X.conjugate().transpose()))
    M_dense = M.dot(X.conjugate().transpose())
    v0_dense = X.dot(v0)
    
    alpha, beta, X_tri = Householder(H_dense, 0)
    M_tridiag = M_dense.dot(X_tri.conjugate().transpose())
    v0_tridiag = X_tri.dot(v0_dense)
    
    # restore the previous state of numpy's PRNG
    np.random.set_state(state)
    
    if verbose >= 1:
        mes_diag = M.dot(sp.sparse.diags([eigs], [0]).dot(M.conjugate().transpose()))
        mes_dense = M_dense.dot(H_dense.dot(M_dense.conjugate().transpose()))
        
        H_tridiag = sp.sparse.diags([np.conj(beta),alpha,beta], [-1,0,1], dtype=np.complex128).tocsr()
        mes_tridiag = M_tridiag.dot(H_tridiag.dot(M_tridiag.conjugate().transpose()))
        
        dev_dense = np.max(abs(mes_diag - mes_dense))
        dev_tridiag = np.max(abs(mes_diag - mes_tridiag))
        print("Deviation in dense Hamiltonian:", dev_dense)
        print("Deviation in tridiag Hamiltonian:", dev_tridiag)
        
        step_diag = M.dot(sp.sparse.diags([eigs], [0]).dot(v0))
        step_dense = M_dense.dot(H_dense.dot(v0_dense))
        step_tridiag = M_tridiag.dot(H_tridiag.dot(v0_tridiag))
        
        dev_step_dense = np.max(abs(step_diag - step_dense))
        dev_step_tridiag = np.max(abs(step_diag - step_tridiag))
        print("Deviation in measurement of dense step:", dev_step_dense)
        print("Deviation in measurement of tridiag step:", dev_step_tridiag)
    
    # construct the PRNGs
    gen_diag = QiPRNG_diag(v0, eigs, M)
    gen_tridiag = QiPRNG_tridiag(v0_tridiag, alpha, beta, M_tridiag)
    gen_dense = QiPRNG_dense(v0_dense, H_dense, M_dense)
    
    return gen_diag, gen_tridiag, gen_dense

import sys
sys.path.append('sp800_22_tests_python3')
from sp800_22_tests import run_tests
import csv
import os

def generate_and_test(seed, n_dims, n_bits, results_filename, delete_after = False):
    gen_diag, gen_tridiag, gen_dense = construct_PRNG_tuple(seed, n_dims)
    row_dict = {"seed":seed}
    
    # Now we generate the binary files
    
    filename_diag = "data_%d_%d_%d_diag.bin" % (seed, n_dims, n_bits)
    generate_datafile(filename_diag, gen_diag, n_bits)
    
    results = run_tests(filename_diag)
    for test_name,p_value,passed in results:
        row_dict[test_name + "_diag"] = p_value
    
    if delete_after:
        os.remove(filename_diag)
    
    filename_tridiag = "data_%d_%d_%d_tridiag.bin" % (seed, n_dims, n_bits)
    generate_datafile(filename_tridiag, gen_tridiag, n_bits)
    
    results = run_tests(filename_tridiag)
    for test_name,p_value,passed in results:
        row_dict[test_name + "_tridiag"] = p_value
    
    if delete_after:
        os.remove(filename_tridiag)
    
    filename_dense = "data_%d_%d_%d_dense.bin" % (seed, n_dims, n_bits)
    generate_datafile(filename_dense, gen_dense, n_bits)
    
    results = run_tests(filename_dense)
    for test_name,p_value,passed in results:
        row_dict[test_name + "_dense"] = p_value
    
    if delete_after:
        os.remove(filename_dense)
    
    make_hdr = not os.path.isfile(results_filename)
    with open(results_filename, 'a') as f:
        writer = csv.DictWriter(f, fieldnames = row_dict.keys())
        if make_hdr:
            writer.writeheader()
        
        writer.writerow(row_dict)

for i in range(100):
    generate_and_test(i, 5, 10000, "results.csv", True)

# def plot_results(results_filename):
#     # and plot the results
#     import matplotlib.pyplot as plt
    
#     for i in range(3):
#         plt.bar(np.arange(len(probs[:,i]))+0.2 * i, probs[:,i], width=0.2, label=filenames[i])
    
#     plt.ylabel("probability (low means fail)")
#     plt.xlabel("test index")
#     plt.legend()
#     plt.show()