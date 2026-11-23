import ctypes as ct
import os
import numpy as np

# Robert Capps 2018

class Utils_ct(object):
    """
    Class for connecting the c_utils module to the Optimizer class.
    """
    def __init__(self):
        self.src_dir = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]) # The path to pycardio directory
        self.setupC()
        self.data = {}

    def setupC(self):
        self.dll = ct.CDLL(os.path.join(self.src_dir, 'c_utils/utils.so')) # Load the shared library

        self.dll.downsample.argtypes = [ np.ctypeslib.ndpointer(ct.c_double, flags="C_CONTIGUOUS"), # x1
                                         ct.c_int, # N1
                                         np.ctypeslib.ndpointer(ct.c_double, flags="C_CONTIGUOUS"), # x2
                                         ct.c_int, # N2
                                         ct.c_int, # N3
                                         np.ctypeslib.ndpointer(ct.c_double, ndim=2, flags="C_CONTIGUOUS") # xout
                                        ]
        self.dll.downsample.restype = None

    def storeData(self, data_key, data_in):
        self.data.update({ data_key : data_in })


class CMAES_ct(object):
    """
    Class for instantiating a new location in memory for the shared object containing
    the C implementation of CMAES.
    """
    def __init__(self):
        self.src_dir = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]) # The path to pycardio directory
        self.setupC()

    def setupC(self):
        self.cmaes = ct.CDLL(os.path.join(self.src_dir, 'c_fit-model/cmaes.so')) # Load the shared library
        
        # Generation loop function
        self.cmaes.generation_loop.argtypes = [ct.py_object, ct.POINTER( ResultCMAES ) ]
        self.cmaes.generation_loop.restype = None

        # Setup function
        self.cmaes.setup.argtypes = [ct.c_int, # dimension of sample x
                                     np.ctypeslib.ndpointer(ct.c_double, flags="C_CONTIGUOUS"), # initial sample
                                     ct.c_double, # initial stepsize
                                     ct.c_double, # tolerance for convergence
                                     ct.c_int, # max number of function evals
        ]
        self.cmaes.setup.restype = None
            
class ResultCMAES(ct.Structure):
    """
    Results class for CMAES
    """
    _fields_ = [
        ("x", np.ctypeslib.ndpointer(ct.c_double, flags="C_CONTIGUOUS")), # optimal sample
        ("C", np.ctypeslib.ndpointer(ct.c_double, ndim=2, flags="C_CONTIGUOUS")), # covariance matrix estimate
        ("samples", np.ctypeslib.ndpointer(ct.c_double, ndim=2, flags="C_CONTIGUOUS")), # accepted samples
    ]