import sys, os, copy, json, tempfile
import multiprocessing as mp

sys.path.append(os.path.expanduser("~/Git/nemati-collab")) # Path to library

try:
    import pycardio
except:
    pass

import numpy as np
import matplotlib.pyplot as plt

DONE = 0
TEMP_FILES = []

def proc_helper(arg_tuple):
    args, kwargs = arg_tuple
    return run_process(*args, **kwargs)

def run_process(*args, **kwargs):
    """ run a single subject in a separate process for the given parameters """

    subject_name, database, path, defaultParams, opt_feature, simtime, params, bounds, alg_args = args
    
    defaults=dict(method='basinhopping',
                  tol=None,
                  maxiter=None,
                  miniter=None)

    for dk in defaults.keys():
        if dk not in kwargs.keys():
            kwargs[dk] = defaults[dk]

    opt = pycardio.Optimizer(subject_name,
                             database=database,
                             path=path,
                             params=params,
                             defaultParams=defaultParams)

    opt.optimize(opt_feature,
                 **kwargs,
                 params=params, alg_args=alg_args,
                 sim_time=simtime, bounds=bounds,
                 mp=True)
    
    return opt.fnps, opt.paramSpace, opt.params_init

def proc_callback(p_result):
    global DONE
    DONE += 1
    fnps, paramSpace, params_init = p_result
    print('\n\nprocess {:03d} done.'.format(DONE,))
    database_dir = os.path.dirname(fnps)
    tempf = tempfile.NamedTemporaryFile(prefix="TEMP{:03d}".format(DONE,),
                                        suffix='.npy',
                                        dir=database_dir)
    np.save(tempf, paramSpace)
    TEMP_FILES.append(tempf.name)
    print('saved intermediate paramSpace to tempfile: {}\n\n'.format(tempf.name,))

def choose_optimal(opt_list):
    min_ix = np.argmin([o[1]['result'].fun for o in opt_list])
    chosen_pre = opt_list[min_ix]
    chosen_opt = tuple(copy.copy(chosen_pre[i]) for i in range(len(chosen_pre)))
    chosen_opt[1]['opt_label'] = min_ix # label optimal params
    chosen_opt[1]['labels'] = [min_ix,]*len(chosen_opt[1]['cost']) # labels in paramSpace dict
    for li, opti in enumerate(opt_list):
        if li != min_ix:
            label = [li,]*len(opti[1]['cost'])
            chosen_opt[1]['labels'].extend(label)
            chosen_opt[1]['cost'].extend(opti[1]['cost']) # append the cost
            chosen_opt[1]['covar'].extend(opti[1]['covar']) # append the covariance
            for pn in opti[2].keys(): # append the parameters
                chosen_opt[1][pn].extend(opti[1][pn])
    return chosen_opt

def save_chosen(chosen_opt):
    fnps, paramSpace, params_init = chosen_opt
    np.save(fnps, paramSpace)
    print("\nSaved the parameter space corresponding to the global minimum.")
    for tf in TEMP_FILES:
        tf.close()

def print_help(adict):
    help_txt = "Usage:\n\n  python3 tests/test_optimize_mp.py [args]\n\nArgs:\n\n"
    args_list = ["  {} : {}".format(k,v) for k, v in adict.items()]
    help_txt = help_txt + "\n".join(args_list)
    print(help_txt)
    
def parseArgs(args):
    def eval_arg(a):
        try:
            aout = eval(a)
        except:
            aout = str(a)
        return aout
    args = args[1:]
    adict = {"config": "config/fantasia_f1y02_rsa.json",}
    if len(args) > 0:
        adict.update(dict([(args[i], eval_arg(args[i+1]))\
                           for i in range(len(args)-1)]))
        if 'help' in args:
            print_help(adict)
            sys.exit()
        for i, akey in enumerate(adict.keys()):
            if akey in args:
                adict[akey] = eval_arg(args[args.index(akey)+1])
    return adict

def loadConfigDict(adict):
    fp = adict['config']
    with open(fp, 'r') as jf:
        configDict = json.load(jf)
    if any(k in configDict.keys() for k in adict):
        configDict.update(adict)
    return configDict

if __name__ == "__main__":

    adict = parseArgs(sys.argv) # parse command line args
    configDict = loadConfigDict(adict) # load config dictionary

    # print('method', configDict['method'])
    
    # Append the path to pycardio and import pycardio
    sys.path.append(os.path.expanduser(configDict['LIB_PATH']))
    import pycardio

    subject_name = configDict['subject_name'] # Name of subject (e.g. 'f1y02')
    database = configDict['database'] # Name of database
    path = os.path.expanduser(configDict['path']) # Path to database
    
    # Default parameters (not necessarily free parameters)
    defaultParams = configDict['defaultParams']

    print(defaultParams)
    
    params = configDict['freeParams'] # free parameters

    print('\n free params: ', params.keys(), '\n')
    
    bounds = configDict['bounds'] # bounds for free parameters

    methods_args = {
        'basinhopping': 'basin_args',
        'genetic_alg': 'ga_args',
        "minimize": "min_args",
        "cmaes": "cmaes_args"
    }
    
    # Choice of method, alg_args dict given to routine
    arg_dict = {}
    for mk, ma in methods_args.items():
        if ma in configDict.keys():
            arg_dict.update({mk: configDict[ma]})
    
    alg_args = arg_dict[configDict['method']]

    args = (subject_name, database, path, defaultParams,
            configDict['opt_feature'], configDict['simtime'],
            params, bounds, alg_args)
    
    kwargs = dict(method=configDict['method'],
                  tol=configDict['tol'],
                  maxiter=configDict['maxiter'],
                  miniter=configDict['miniter'])

    # Multiprocessing
    nr_p = configDict['nr_p']
    arg_list = [(args, kwargs) for i in range(nr_p)]
    with mp.Pool(processes=nr_p) as pool:
        mp_res = pool.map_async(proc_helper, arg_list)
        optimal_opt = choose_optimal(mp_res.get())
        save_chosen(optimal_opt)

    result = optimal_opt[1]['result']
    if 'lowest_optimization_result' in result.keys():
        print("\nlowest_optimization_result: ", result.lowest_optimization_result)
    else:
        print(result.x)
        print(result.message)
