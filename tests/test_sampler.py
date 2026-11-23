import sys, os

sys.path.append(os.path.expanduser("~/Git/nemati-collab")) # Path to library

import pycardio

import ctypes as ct
import numpy as np

if __name__ == '__main__':

    subject = "f2y03"
    opt_feature = "rsa"

    if len(sys.argv) > 2:
        subject, opt_feature = sys.argv[1:]

    if subject == 'f2y03':
        params = {
            "baro": 0.05833,
            "taup": 1367,
            "po": 0.15,
        }
        dparams = {
            "rs1": 0.00001,
            "d2": 0.642,
            "d4": 0.527,
            "hr": 0.00083,
            "rsa1": 0.000168,
            "rsa2": 0.00057
        }
    elif subject == 'f1o01':
        params = {
            "baro": 0.0,
            "po": 0.01,
            "taup": 1000
        }
        dparams = {
            "rs1": 0.00001,
            "d2": 0.616,
            "d4": 0.587,
            "hr": 0.000937,
            "rsa1": 0.000129,
            "rsa2": 0.000349
        }

    bounds = [
        (0.0, 0.5),
        (0.0, 1.0),
        (0.0, 3000),
    ]

    step_size = np.ones((len(bounds),len(bounds)))*np.array([(b[1]-b[0])*0.01 for b in bounds])
    
    sampler = pycardio.SamplerMH(subject,
                                 params=params,
                                 defaultParams=dparams,
                                 step_size=step_size,
                                 bounds=bounds,
                                 time=1000,
                                 miniter=50, maxiter=100000,
                                 njobs=8, nsamples=16,
                                 tol=3e-5)

    sampler.sample(opt_feature)
    sampler.save()
