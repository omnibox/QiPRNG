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
from QiPRNG import QiPRNG_exact, QiPRNG_diag, QiPRNG_tridiag, QiPRNG_dense

DELETE_AFTER = True
N_DIMS = 10
N_BYTES = 1000000
RESULTS_FILENAME = "../data/results.csv"
BINARY_DATA_DIR = "../data/binary_data/"
STATS_DATA_DIR = "../data/stats_output/"
USE_C_IMPLEMENTATION = True # only set this true when N_BYTES is 1e6
# remember to compile sp800.so with
# cc -fPIC -shared -o sp800.so ./sp800_22_tests_c/src/*


def Lanczos(H, verbosity = 0):
    """
    Tridiagonalizes H via the Lanczos algorithm.
    
    The returned values will be two diagonals from a sparse matrix H_tridiag
    and a dense change of basis matrix X such that XHX^{-1} = H_tridiag

    WARNING: The Lanczos algorithm is numerically unstable.
    Householder rotations are the preferred method of tridiagonalization,
    and Lanczos is normally used only when Householder fails.

    Parameters
    ----------
    H : 2d numpy array
        The Hamiltonian to be tridiagonalized.
    verbosity : int, optional
        What level of diagnostic messages to print. The default is 0.

    Returns
    -------
    alpha : 1d numpy array
        The diagonal elements of H_tridiag.
    beta : 1d numpy array
        The elements above the diagonal in H_tridiag.
    X : 2d numpy array
        The change of basis matrix the tridiagonalizes H into H_tridiag.
    """
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
    """
    Tridiagonalizes H via the Householder transformations.
    
    The returned values will be two diagonals from a sparse matrix H_tridiag
    and a dense change of basis matrix X such that XHX^{-1} = H_tridiag

    Parameters
    ----------
    H : 2d numpy array
        The Hamiltonian to be tridiagonalized.
    verbosity : int, optional
        What level of diagnostic messages to print. The default is 0.

    Returns
    -------
    alpha : 1d numpy array
        The diagonal elements of H_tridiag.
    beta : 1d numpy array
        The elements above the diagonal in H_tridiag.
    X : 2d numpy array
        The change of basis matrix the tridiagonalizes H into H_tridiag.
    """
    # Householder transforms are reflections that, up to a phase factor,
    # map the first column of H (sans top element) to a vector pointing
    # in the x_1 direction. We then have a similar subproblem for H[1:,1:].
    
    # construct the vector to reflect across
    v = H[:,0].copy()
    v[0] = 0
    v[1] += np.linalg.norm(H[1:,0]) * H[1,0] / abs(H[1,0])
    v /= np.linalg.norm(v)
    
    # compute the reflector R
    R = np.eye(H.shape[0]) - 2 * np.outer(v,v.conjugate())
    
    # perform the reflection
    A = R.dot(H.dot(R))
    
    # and add it to the change of basis matrix
    X = R
    
    for k in range(1,H.shape[0]-2):
        # construct the vector to reflect across (now for a subsystem)
        v = A[:,k].copy()
        v[:k+1] = 0
        v[k+1] += np.linalg.norm(A[k+1:,k]) * A[k+1,k] / abs(A[k+1,k])
        v /= np.linalg.norm(v)
        
        # compute the reflector R
        R = np.eye(A.shape[0]) - 2 * np.outer(v,v.conjugate())
        
        # perform the reflection
        A = R.dot(A.dot(R))
        
        # add it to the change of basis matrix
        X = R.dot(X)
    
    # extract the diagonals
    alpha = np.zeros(H.shape[0], np.complex128)
    beta = np.zeros(H.shape[0]-1, np.complex128)
    alpha[0] = A[0,0]
    for j in range(1,H.shape[0]):
        alpha[j] = A[j,j]
        beta[j-1] = A[j-1,j]
    
    # printing how much deviation we have from mathematically expected results
    if verbosity >= 1:
        # construct the tridiagonal matrix
        H_tridiag = sp.sparse.diags([np.conj(beta),alpha,beta], [-1,0,1], dtype=np.complex128).tocsr()
        
        # check that X is unitary
        unit_dev = np.max(X.dot(X.conjugate().transpose()) - np.eye(H.shape[0])).real
        
        # and check that XHX^{-1} = H_tridiag
        trid_dev = np.max(X.dot(H.dot(X.conjugate().transpose())) - H_tridiag).real
        print("Deviation from unitarity:", unit_dev)
        print("Deviation from tridiagonality:", trid_dev)
    
    return alpha, beta, X

def generate_datafile(filename, generator, num_bytes, verbosity = 0):
    """
    Streams bytes from a generator to a binary file.

    Parameters
    ----------
    filename : string
        The file to write to.
    generator : iterable
        The source of data to be written to the file. Should yield single bytes.
    num_bytes : int
        The number of bytes to write to the file.
    verbosity : int, optional
        How much to print while working. Default is 0. Value of 1 will result
        in messages announcing the stream's start and end. A value of 2 will
        result in percentage updates.

    Returns
    -------
    None.

    """
    # announce that we're trying to access the file
    if verbosity >= 1:
        print("attempting to open %s..." % filename)
    
    update_period = num_bytes // 100
    with open(BINARY_DATA_DIR + filename, 'wb') as f:
        # announce that the file is open
        if verbosity >= 2:
            print("generating %s..." % filename, end="")
        
        # stream out the data
        for i in range(num_bytes):
            # generate and write a byte
            f.write(generator.__next__().to_bytes(1, 'big'))
            
            # maybe print a status update
            if i % update_period == 0 and verbosity >= 2:
                print("\rgenerating %s... %3d%% complete" % (filename, i // update_period), end="")
    
    # announce that generation is done after the file is closed
    if verbosity >= 1:
        print("\rgenerating %s... 100%% complete" % filename)

def construct_PRNG_tuple(seed, n_dims, verbosity = 0):
    """
    Constructs four QiPRNG instances that all simulate the same quantum system.
    
    The system being simulated is generated by seeding numpy's PRNG with seed
    and randomly sampling H, M, and v0. These then act as a key for QiPRNG.
    
    Parameters
    ----------
    seed : int
        The seed for numpy's PRNG. This allows an unbiased but deterministic
        selection of system to simulate, which in turn is the key for our QiPRNGs.
    n_dims : int
        The dimension of the quantum system to simulate.
    verbosity : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    gen_exact : iterable
        A QiPRNG[exact] instance.
    gen_diag : iterable
        A QiPRNG[diagonal] instance.
    gen_tridiag : iterable
        A QiPRNG[tridiagonal] instance.
    gen_dense : iterable
        A QiPRNG[dense] instance.
    """
    # save the state of numpy's PRNG so we can restore it later
    state = np.random.get_state()
    
    # make the code deterministic
    np.random.seed(seed)
    
    # select a random normalized starting vector
    # since elements are normally distrbuted there is no angular bias
    v0 = np.random.normal(size = n_dims).astype(np.complex128) - 0.5
    v0 += np.random.normal(size = n_dims).astype(np.complex128) * (0+1j) - (0 + 0.5j)
    v0 /= np.linalg.norm(v0,2)
    
    # generating the eigenvalues in [0,1) for the diagonal system
    eigs = np.random.uniform(size = n_dims).astype(np.complex128)
    
    # we need a measurement basis
    M = sp.stats.unitary_group.rvs(n_dims)
    
    # generating corresponding dense and tridiagonal Hamiltonians
    X = sp.stats.unitary_group.rvs(n_dims)
    H_dense = X.dot(sp.sparse.diags([eigs], [0]).dot(X.conjugate().transpose()))
    M_dense = M.dot(X.conjugate().transpose())
    v0_dense = X.dot(v0)
    
    alpha, beta, X_tri = Householder(H_dense, 0)
    M_tridiag = M_dense.dot(X_tri.conjugate().transpose())
    v0_tridiag = X_tri.dot(v0_dense)
    
    # restore the previous state of numpy's PRNG
    np.random.set_state(state)
    
    # print the deviation between different constructions
    # since these are all the same system in different bases, we are
    # printing a measure of how much finite precision error has crept in
    if verbosity >= 1:
        # first we compute MHM^\dagger, i.e. H's elements in the measurement basis
        mes_diag = M.dot(sp.sparse.diags([eigs], [0]).dot(M.conjugate().transpose()))
        mes_dense = M_dense.dot(H_dense.dot(M_dense.conjugate().transpose()))
        
        H_tridiag = sp.sparse.diags([np.conj(beta),alpha,beta], [-1,0,1], dtype=np.complex128).tocsr()
        mes_tridiag = M_tridiag.dot(H_tridiag.dot(M_tridiag.conjugate().transpose()))
        
        # compute the max elementwise differences, comparing against the diagonal basis
        dev_dense = np.max(abs(mes_diag - mes_dense))
        dev_tridiag = np.max(abs(mes_diag - mes_tridiag))
        print("Deviation in dense Hamiltonian:", dev_dense)
        print("Deviation in tridiag Hamiltonian:", dev_tridiag)
        
        # second we compute MH|v0> to check that v0 has transforme properly
        step_diag = M.dot(sp.sparse.diags([eigs], [0]).dot(v0))
        step_dense = M_dense.dot(H_dense.dot(v0_dense))
        step_tridiag = M_tridiag.dot(H_tridiag.dot(v0_tridiag))
        
        # again the diagonal basis in the baseline
        # and comparisons are maximum elementwise difference
        dev_step_dense = np.max(abs(step_diag - step_dense))
        dev_step_tridiag = np.max(abs(step_diag - step_tridiag))
        print("Deviation in measurement of dense step:", dev_step_dense)
        print("Deviation in measurement of tridiag step:", dev_step_tridiag)
    
    # construct the PRNGs
    gen_exact = QiPRNG_exact(v0, eigs, M)
    gen_diag = QiPRNG_diag(v0, eigs, M)
    gen_tridiag = QiPRNG_tridiag(v0_tridiag, alpha, beta, M_tridiag)
    gen_dense = QiPRNG_dense(v0_dense, H_dense, M_dense)
    
    return gen_exact, gen_diag, gen_tridiag, gen_dense

import sys
import shutil
if USE_C_IMPLEMENTATION:
    import ctypes
    NUMOFPVALS = 188
    sp800 = ctypes.CDLL("./sp800.so")
    sp800.run_tests.argtypes = [type(ctypes.pointer((ctypes.c_double * NUMOFPVALS)())), ctypes.c_char_p]
else:
    sys.path.append('sp800_22_tests_python3')
    from sp800_22_tests import run_tests_python
import csv
import os

def generate_and_test(generator, suffix, seed, n_dims, n_bytes, results_dict, delete_after = False):
    """
    Generates a binary data file from a given QiPRNG instance, and then runs
    statistical tests on the file's contents. The results of the test are
    stored as entries in the results_dict dictionary.

    Parameters
    ----------
    generator : iterable
        A QiPRNG instance to generate data with.
    suffix : string
        The type of QiPRNG that has been passed. Acceptable options are
        "exact", "diag", "tridiag", and "dense".
    seed : int
        ID number for the generator. This will be used in the name of the file
        where the binary data is stored.
    n_dims : int
        The dimension of the system the QiPRNG instance is based on.
    n_bytes : int
        The number of bytes to test on.
    results_dict : dictionary
        This is where results of the statistical tests will be written.
    delete_after : boolean, optional
        Whether to delete the binary data files after testing is complete.
        The default is False.
    Returns
    -------
    None.

    """
    # generate the binary data to test on
    filename = "data_%d_%d_%d_%s.bin" % (seed, n_dims, n_bytes, suffix)
    generate_datafile(filename, generator, n_bytes)
    
    if USE_C_IMPLEMENTATION:
        # get the filename without the suffix
        key = filename.split(".")[0]
        
        # build the arguments and pass them to the c library
        pvals_buffer = ctypes.pointer((ctypes.c_double * NUMOFPVALS)())
        filename_buffer = ctypes.create_string_buffer(len(key) + 1)
        filename_buffer.value = key.encode("utf-8")
        sp800.run_tests(pvals_buffer, filename_buffer)
        
        print(pvals_buffer.contents[:])
        
        # save the results to the library
        for index,p_value in enumerate(pvals_buffer.contents):
            results_dict["test_%03d_%s" % (index, suffix)] = p_value
        
        # remove the directory used by the C implementation
        if delete_after:
            shutil.rmtree(STATS_DATA_DIR + key)
    else:
        # run tests on the pseudorandom datafile
        results = run_tests_python(BINARY_DATA_DIR + filename)
        
        # write the results to the dictionary
        for test_name,p_value,passed in results:
            results_dict[test_name + "_" + suffix] = p_value
    
    # remove the datafile
    if delete_after:
        os.remove(BINARY_DATA_DIR + filename)

def generate_batch_and_save(seed, n_dims, n_bytes, results_filename, delete_after = False):
    """
    Constructs a set of four equivalent QiPRNG instances, runs statistical
    tests on all of them, and stores the results as a single line in a cvs file.

    Parameters
    ----------
    seed : int
        The seed that will be used to generate the system for the QiPRNGs.
    n_dims : int
        The dimension of the quantum system for the QiPRNGs.
    n_bytes : int
        The number of bytes that will be tested on from each generator.
    results_filename : string
        The file where the results will be appended.
    delete_after : boolean, optional
        Whether the binary datafiles should be deleted after the statistical
        tests are complete. The default is False.

    Returns
    -------
    None.
    """
    # construct the generators
    gen_exact, gen_diag, gen_tridiag, gen_dense = construct_PRNG_tuple(seed, n_dims)
    row_dict = {"seed":seed, "n_dims":n_dims, "n_bytes":n_bytes}
    
    # generate and process the data
    generate_and_test(gen_exact  , "exact"  , seed, n_dims, n_bytes, row_dict, delete_after)
    generate_and_test(gen_diag   , "diag"   , seed, n_dims, n_bytes, row_dict, delete_after)
    generate_and_test(gen_tridiag, "tridiag", seed, n_dims, n_bytes, row_dict, delete_after)
    generate_and_test(gen_dense  , "dense"  , seed, n_dims, n_bytes, row_dict, delete_after)
    
    # write the results to a file
    make_hdr = not os.path.isfile(results_filename)
    with open(results_filename, 'a') as f:
        writer = csv.DictWriter(f, fieldnames = row_dict.keys())
        if make_hdr:
            writer.writeheader()
        
        writer.writerow(row_dict)

def max_seed_completed(n_dims, n_bytes, results_filename):
    """
    Scans through a results file to find the highest
    seed value that's been tested on.

    Parameters
    ----------
    n_dims : int
        The dimension of the quantum system for the QiPRNG.
    n_bytes : int
        The number of bytes that was generated to test on.
    results_filename : string
        The name of the file to search for past results.

    Returns
    -------
    int
        The max seed that we've already generated results for.

    """
    # generation starts at max_seed + 1,
    # so if the file isn't there return -1
    if not os.path.isfile(results_filename):
        return -1;
    
    max_seed = -1
    with open(results_filename, 'r') as f:
        reader = csv.DictReader(f)
        # scan through the file line by line
        for row in reader:
            # and collect the maximum seed w ith matching n_dims and n_bytes
            if n_dims == int(row["n_dims"]) and n_bytes == int(row["n_bytes"]):
                max_seed = max(max_seed, int(row["seed"]))
    
    return max_seed

import multiprocessing as mp
def run_from_seed(seed):
    """
    A wrapper function to make parallelization easier.
    Calls generate_batch_and_save with seed and the global arguments.

    Parameters
    ----------
    seed : int
        The seed to specify keys for QiPRNG.

    Returns
    -------
    None.
    """
    generate_batch_and_save(seed, N_DIMS, N_BYTES, RESULTS_FILENAME, DELETE_AFTER)

def generate_parallel(num_seeds):
    """
    Runs and stores statistical analysis for multiple sets of QiPRNGs.
    Each set of 4 QiPRNGs built around the same system is sceduled
    indipendently and in parallel if possible.
    
    Seed selection starts above the highest completed so far, so this function
    can be called multiple times in serial and new data will be created.

    Parameters
    ----------
    num_seeds : int
        The number of lines of data to add to the results file.

    Returns
    -------
    None.
    """
    # get preliminary information; number of CPUs, number of past seeds
    print("Found %d CPUs" % (mp.cpu_count()))
    max_seed = max_seed_completed(N_DIMS, N_BYTES, RESULTS_FILENAME)
    print("Found results file with seeds up to %d" % (max_seed))
    
    # construct and execute a pool of tasks
    pool = mp.Pool(mp.cpu_count())
    pool.map(run_from_seed, np.arange(num_seeds) + max_seed + 1)
    pool.close()

def generate_serial(num_seeds):
    """
    Runs and stores statistical analysis for multiple sets of QiPRNGs.
    
    Seed selection starts above the highest completed so far, so this function
    can be called multiple times in serial and new data will be created.

    Parameters
    ----------
    num_seeds : int
        The number of lines of data to add to the results file.

    Returns
    -------
    None.

    """
    # find out how many seeds have already been run
    max_seed = max_seed_completed(N_DIMS, N_BYTES, RESULTS_FILENAME)
    print("Found results file with seeds up to %d" % (max_seed))
    
    # go through num_seeds more and generate results for each
    seeds_to_run = np.arange(num_seeds) + max_seed + 1
    for seed in seeds_to_run:
        generate_batch_and_save(seed, N_DIMS, N_BYTES, RESULTS_FILENAME, DELETE_AFTER)

# TODO: plot results
# def plot_results(results_filename):
#     # and plot the results
#     import matplotlib.pyplot as plt
    
#     for i in range(3):
#         plt.bar(np.arange(len(probs[:,i]))+0.2 * i, probs[:,i], width=0.2, label=filenames[i])
    
#     plt.ylabel("probability (low means fail)")
#     plt.xlabel("test index")
#     plt.legend()
#     plt.show()


import time

# import scipy.stats as ss
if __name__ == "__main__":
    generate_parallel(10000)
#     seed = 2
#     n_dims = 50
#     N = 100000
#     update_period = N // 100
#     gen_exact, gen_diag, gen_tridiag, gen_dense = construct_PRNG_tuple(seed, n_dims)
#     counts = np.zeros((4, 256), np.int32)
#     individual_bits = np.zeros((4, 8), np.int32)
#     res = []
#     for i in range(N):
#         for (j,gen) in enumerate([gen_exact, gen_diag, gen_tridiag, gen_dense]):
#             b = gen.__next__()
#             res += [b]
#             counts[j,b] += 1
#             for k in range(8):
#                 individual_bits[j,k] += (b // 2**k) % 2
            
#         if i % update_period == 0:
#             print('\r%3d%% complete' % (i // update_period), end="")
# #            print([ss.kstest(c, ss.randint.cdf, args=(0,256)).pvalue for c in counts])
#     print("\r100% complete")


