import ctypes as ct
import os
import numpy as np

### Load the shared library objects through ctypes

src_dir = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]) # The path to pycardio directory

calc_rsaC = ct.CDLL(os.path.join(src_dir, 'c_process-data/calc_rsa.so'))
calc_cvcC = ct.CDLL(os.path.join(src_dir, 'c_process-data/calc_cvc.so'))
calc_dfaC = ct.CDLL(os.path.join(src_dir, 'c_process-data/calc_dfa.so'))

def setupC():
    calc_rsaC.calc_rsa.argtypes = [
        np.ctypeslib.ndpointer(ct.c_double, flags="C_CONTIGUOUS"), # time
        np.ctypeslib.ndpointer(ct.c_int, flags="C_CONTIGUOUS"), # Beat annotations
        ct.c_int, # len of beat annotations
        np.ctypeslib.ndpointer(ct.c_int, flags="C_CONTIGUOUS"), # Indices of peaks from tidal volume
        ct.c_int, # len of resp_peaks
        np.ctypeslib.ndpointer(ct.c_double, flags="C_CONTIGUOUS") # [phase, RRI]
    ]
    calc_rsaC.calc_rsa.restype = None

    calc_cvcC.calc_cvc.argtypes = [
        np.ctypeslib.ndpointer(ct.c_double, flags="C_CONTIGUOUS"), # time
        np.ctypeslib.ndpointer(ct.c_int, flags="C_CONTIGUOUS"), # Beat annotations
        ct.c_int, # len of beat annotations
        np.ctypeslib.ndpointer(ct.c_int, flags="C_CONTIGUOUS"), # Indices of peaks from tidal volume
        ct.c_int, # len of resp_peaks
        np.ctypeslib.ndpointer(ct.c_double, flags="C_CONTIGUOUS") # beat latencies
    ]
    calc_cvcC.calc_cvc.restype = None

    calc_dfaC.calc_dfa.argtypes = [
        np.ctypeslib.ndpointer(ct.c_float, flags="C_CONTIGUOUS"), # x_signal (time)
        np.ctypeslib.ndpointer(ct.c_float, flags="C_CONTIGUOUS"), # y_detrended
        ct.c_uint32, # length
        np.ctypeslib.ndpointer(ct.c_float, flags="C_CONTIGUOUS"), # Fluctuation (N)
        ct.c_uint32, ct.c_uint32 # min_N, max_N        
    ]
    calc_dfaC.calc_dfa.restype = None
    
    return calc_rsaC.calc_rsa, calc_cvcC.calc_cvc, calc_dfaC.calc_dfa

calc_rsa, calc_cvc, calc_dfa = setupC()
