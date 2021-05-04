#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 10:48:04 2021

Compile the c code with
cc -fPIC -shared -o lib.so function.c

Or, later on
cc -fPIC -shared -o lib.so ./src/*

@author: green
"""

import ctypes
import multiprocessing as mp
import numpy as np
import time

# ################ PART 1: DATATYPES #####################

# NUM = 16

# function = ctypes.CDLL("./lib.so")

# function.myFunction.argtypes = [ctypes.c_int]

# returnValue = function.myFunction(NUM)


# arr = ctypes.pointer((ctypes.c_int * NUM)())
# function.linarray.argtypes = [type(arr), ctypes.c_int]

# function.linarray(arr, NUM)

# print(arr.contents[:])

# ################ PART 2: MULTITHREADING #####################

# function.setnum.argtypes = [ctypes.c_int]

# def read_or_write(n):
#     if n == 0:
#         # read
#         function.setnum(1000)
#         for i in range(5):
#             print("Reading %d:" % function.getnum())
#             time.sleep(1)
#     else:
#         # write
#         for i in range(5):
#             n = np.random.randint(100)
#             print("Writing %d:" % n)
#             function.setnum(n)
#             time.sleep(1)
        

# pool = mp.Pool(mp.cpu_count())
# pool.map(read_or_write, [0,1])
# pool.close()

################ PART 3: THE TESTS #####################

NUMOFPVALS = 188

f = ctypes.CDLL("./lib.so")
res = ctypes.pointer((ctypes.c_double * NUMOFPVALS)())
f.main_func.argtypes = [type(res), ctypes.c_char_p]

s = "NameHere"
buf = ctypes.create_string_buffer(len(s))
buf.value = s.encode("utf-8")
f.main_func(res, buf)

print(res.contents[:])













