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
            "baro": 0.075,
            "d2": 0.642,
        }
        dparams = {
            "po": 0.01,
            "taup": 1367,
            "rs1": 0.00001,
            "hr": 0.0008303,
            "d4": 0.527,
            "rsa1": 0.000197,
            "rsa2": 0.0006
        }
    elif subject == 'f1o01':
        params = {
            "baro": 0.0,
            "d2": 0.65
        }
        dparams = {
            "hr": 0.000937,
            "po": 0.01,
            "taup": 1000,
            "rs1": 0.00001,
            "d4": 0.587,
            "rsa1": 0.000129,
            "rsa2": 0.000349
        }

    bounds = [
        (0.0, 0.25),
        (0.0, 1.0)
    ]

    step_size = np.ones((len(bounds),len(bounds)))*np.array([(b[1]-b[0])*0.0005 for b in bounds])
    
    sampler = pycardio.SamplerMH(subject,
                                 params=params,
                                 defaultParams=dparams,
                                 step_size=step_size,
                                 bounds=bounds,
                                 time=700,
                                 miniter=700, maxiter=100000,
                                 njobs=56, nsamples=4,
                                 minsamples=500,
                                 tol=5e-4)

    sampler.sample(opt_feature)
    sampler.save()
