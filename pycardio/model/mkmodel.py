import ctypes as ct
import numpy as np
import os
from . import parameters

def setupModel():
    src_dir = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]) # The path to pycardio directory
    rsp = ct.CDLL(os.path.join(src_dir, 'c_model/resp_odes.so'))
    rsp.integrate.restype = None
    rsp.integrate.argtypes = [
        ct.c_int, # length of vector
        ct.c_int, # number of dimensions of solution array
        np.ctypeslib.ndpointer(ct.c_double, flags="C_CONTIGUOUS"), # solution array
        ct.c_double, # end of model time as double
        ct.POINTER( parameters.Parameters) ] # parameters struct
    return( rsp )
