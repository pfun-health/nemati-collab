import sys, os, copy

sys.path.append(os.path.expanduser("~/Git/nemati-collab")) # Path to library

import pycardio

import numpy as np
import matplotlib.pyplot as plt

def run_process(*args, **kwargs):
    """ run a single subject for the given parameters """
    subject_name, database, path, defaultParams, opt_feature, simtime, params, bounds, alg_args = args
    defaults=dict(method='basinhopping', tol=None)
    for dk in defaults.keys():
        if dk not in kwargs.keys():
            kwargs[dk] = defaults[dk]
    opt = pycardio.Optimizer(subject_name,
                             database=kwargs['database'],
                             path=kwargs['path'],
                             params=kwargs['params'],
                             defaultParams=kwargs['defaultParams'])

    result = opt.optimize(opt_feature,
                          method=kwargs['method'],
                          params=params, alg_args=alg_args,
                          sim_time=simtime, bounds=bounds,
                          tol=kwargs['tol'])
    
    if 'lowest_optimization_result' in result.keys():
        print("\nlowest_optimization_result: ", result.lowest_optimization_result)
    else:
        print(result.x)
        print(result.message)


if __name__ == "__main__":

    database = 'fantasia' # Name of database
    ftype = 'ecg' # Type of ECG data
    path = os.path.expanduser('~/Public') # Path to database

    subject_name = "f1y02" # name of the subject
    defaultParams = copy.copy(pycardio.defaultParams)
    defaultParams['hr'] = 0.00082 # f1y02
    defaultParams['d4'] = 0.436
    defaultParams['d5'] = 0.702
    defaultParams['rsa1'] = 0.00055
    defaultParams['rsa2'] = 0.00080
    simtime = 120 # seconds of model simulation time
    
    opt = pycardio.Optimizer(subject_name, database=database, path=path,
                             params=defaultParams,
                             defaultParams=defaultParams)

    params = {
        'hr': defaultParams['hr'],
        'rsa1': defaultParams['rsa1'],
        'rsa2': defaultParams['rsa2']
    }
    
    bounds = {
        'hr': (0.0007, 0.0009),
        'rsa1': (0.00045, 0.0007),
        'rsa2': (0.0007, 0.0009)
    }
        
    # ga_args = dict(disp=1, popsize=100,
    #                seed=123000,
    #                mutation=(0.1, 1.5)
    # )
    # result = opt.optimize('ave_resp', method='genetic_alg',
                          # params=params,
                          # alg_args=ga_args, sim_time=simtime, bounds=bounds)

    # min_args = dict(options={'maxiter': 10000, 'disp': True})
    # result = opt.optimize('rsa', method='minimize',
    #                       params=params,
    #                       alg_args=min_args,
    #                       sim_time=simtime,
    #                       bounds=bounds,
    #                       tol=1e-6)

    basin_args = dict(niter=100, disp=True,
                      T=0.5, stepsize=0.00001,
                      niter_success=20, interval=10)
    result = opt.optimize('rsa', method='basinhopping',
                          params=params,
                          alg_args=basin_args,
                          sim_time=simtime,
                          bounds=bounds,
                          tol=1e-6)

    if 'lowest_optimization_result' in result.keys():
        print("\nlowest_optimization_result: ", result.lowest_optimization_result)
    else:
        print(result.x)
        print(result.message)
