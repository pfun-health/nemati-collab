import sys, os, copy, json

# Path to pycardio
LIB_PATH = os.path.expanduser("~/Git/nemati-collab")

sys.path.append(LIB_PATH) 

import pycardio

import numpy as np

def print_help(adict):
    help_txt = "Usage:\n\n  python3 tests/test_optimize_mp.py [args]\n\nArgs:\n\n"
    args_list = ["  {} : {}".format(k,v) for k, v in adict.items()]
    help_txt = help_txt + "\n".join(args_list)
    print(help_txt)

def parseArgs(args):
    args = args[1:]
    adict = {"subject_name": "f1y02",
             "opt_feature": "rsa",
             "path": "~/Public",
             "database": "fantasia",
             "method": "basinhopping",
             "tol": 1e-5,
             "simtime": 60,
             "nr_p": 2,
             "T": 0.5,
             "stepsize": 0.000001,
             "niter": 10000,
             "niter_success": 1500,
             "interval": 10}
    if 'help' in args:
        print_help(adict)
        sys.exit()
    for i, akey in enumerate(adict.keys()):
        if akey in args:
            adict[akey] = eval(args[args.index(akey)+1])
    return adict

def createConfigDict(adict, defaultParams,
                     params, bounds,
                     basin_args=None,
                     ga_args=None,
                     min_args=None):
    configDict = {}

    configDict['LIB_PATH'] = LIB_PATH # path to pycardio module

    configDict['path'] = adict['path'] # path to the database (e.g. '/home/USER/Public')
    configDict['database'] = adict['database'] # name of database (e.g. 'fantasia')
    
    configDict['subject_name'] = adict['subject_name'] # name of the subject (e.g. 'f1y02')
    configDict['opt_feature'] = adict['opt_feature'] # optimization feature (e.g. 'rsa')
    
    configDict['method'] = adict['method'] # optimization routine to use (e.g. 'basinhopping')

    configDict['defaultParams'] = defaultParams # default parameters (all parameters, not just hot params)
    configDict['freeParams'] = params # these are "hot"/"free" parameters that will be used for optimization
    configDict['bounds'] = bounds # these are the bounds (min, max) for each free parameter
    
    configDict['tol'] = adict['tol']  # tolerance for cost function convergence criteria (e.g. 1e-3)

    configDict['simtime'] = adict['simtime'] # time to run model simulation in seconds (e.g. 60)

    configDict['nr_p'] = adict['nr_p'] # number of processes to use for multiprocessing (e.g. 2)

    if not basin_args:
        basin_args = dict(niter=adict['niter'], disp=True,
                          T=adict['T'], stepsize=adict['stepsize'],
                          niter_success=adict['niter_success'],
                          interval=adict['interval'])

    if not ga_args:
        ga_args = dict(disp=1, popsize=64,
                       seed=None,
                       mutation=(0.1, 1.25),
                       recombination=0.25)

    if not min_args:
        min_args = dict(options={'maxiter': 10000,
                                 'disp': True})

    configDict.update(
        dict(basin_args=basin_args,
             ga_args=ga_args,
             min_args=min_args))
                           
    return configDict

def saveConfigDict(configDict, fp):
    with open(fp, 'w') as jf:
        json.dump(configDict, jf, indent=5)
    print('saved config to {}'.format(fp))


if __name__ == '__main__':
    
    adict = parseArgs(sys.argv) # parse command line args

    # Default parameters (not necessarily free parameters)
    defaultParams = copy.copy(pycardio.defaultParams)

    defaultParams['d4'] = 0.436
    defaultParams['d5'] = 0.702

    defaultParams['hr'] = 0.00099637
    defaultParams['rsa1'] = 0.000700
    defaultParams['rsa2'] = 0.000999
    
    params = {
        'hr': defaultParams['hr'],
        'rsa1': defaultParams['rsa1'],
        'rsa2': defaultParams['rsa2']
    }
    
    bounds = {
        'hr': (0.0003, 0.001),
        'rsa1': (0.0003, 0.001),
        'rsa2': (0.0003, 0.001)
    }

    configDict = createConfigDict(adict, defaultParams, params, bounds)
    saveConfigDict(configDict, "fantasia_f1y02_rsa.json")
