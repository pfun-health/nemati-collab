import numpy as np
from scipy import optimize
import os, copy
import ctypes as ct
import gc
from ..model import defaultParams as defParams
from ..model import setupModel, setParams, ModelOutput
from ..load_data import load_stats
from .cost_functions import CF_simple, CF_rsa, CF_resp, CF_cvc
from .adaptive_stepsize import AdaptiveStepsize, RandomDisplacement
from ._fit_model_ct import CMAES_ct, ResultCMAES, Utils_ct

class Optimizer(object):
    """
    Optimizer is a class for easily configuring and optimizing the cardiovascular control model,
    given a specific subject and some model specifications.
    """
    def __init__(self, subject,
                 database='fantasia', path="~/Public",
                 params={"hr": 0.001,
                         "rsa1": 0.001,
                         "rsa2": 0.0011},
                 defaultParams=None,
                 sampling_freq=250,
                 model_freq=1000):

        self.subject = subject
        self.database = database
        self.path = os.path.expanduser(path)

        if defaultParams: self.defaultParams = defaultParams
        elif not defaultParams: self.defaultParams = copy.copy(defParams) # if not provided, use default parameters given in parameters.py

        self.params_init = params # this is a dictionary of initial parameters (free params)
        self.paramSpace = dict([(pk, []) for pk in params.keys()]) # dictionary that defines the parameter space
        self.paramSpace.update( { "cost" : [] } ) # put the cost from each function eval in the param space
        self.paramSpace.update( { "covar" : [] } ) # covariance at each function eval
        self.paramSpace.update( { "cdf" : [] } ) # cumulative distr. function for CVC

        self.tot_evals = 0 # total number of function evals
        self.nr_accepted = 0 # number of accepted samples
        self.arate = 1.0 # acceptance rate
        
        self.tol = None # tolerance, if set
        self.fun = None # current cost
        self.miniter = None # minimum number of accepted samples, if set
        self.var = None # variance from the data

        self.time = None
        self.time_vec = None

        self.f_model = None
        self.f_data = None
        
        self.sampling_freq = sampling_freq
        self.model_freq = model_freq

        self.optMethod = None # method of optimization/sampling
        self.optFeature = None # feature being optimized

        self.cost_args = tuple() # tuple of constant arguments for calculating cost

        self.utils = Utils_ct() # Load utils from c_utils

        self._setupModel()
        self._loadData()

    def _setTime(self, time):
        self.time = time
        self.time_vec = np.linspace(0, time, int(self.model_freq*time))

    @property
    def _getx0(self): # Get the initial sample as a numpy array
        return np.array([pi for pk, pi in sorted(list(self.params_init.items()))], dtype=np.float)
        
    def _getData(self, name):
        """ quickly get any optimization feature by its name, e.g. "rri"
        """
        name = name.lower() # lowercase
        if '_ave' in name: name = name.replace('_ave','')
        s = self.s
        if name in s.keys():
            return s[name]
        if name in s['rsa'].keys():
            return s['rsa'][name]
        if name in s['cvc'].keys():
            return s['cvc'][name]
        if name in s['descriptive'].keys():
            return s['descriptive'][name]
        if name in s['signals'].keys():
            return s['signals'][name]

    def _loadData(self):
        """ Load the stats array from the chosen subject, database """
        self.s = load_stats(self.subject, self.database, path=self.path)

    def _setupModel(self, params=None):
        """ Initial model setup with initial params """
        if not params: params = self.params_init
        self.model = setupModel()
        self.params = setParams(dParams=self.defaultParams, **params)

    def _runModel(self, time, params=None, ndim=12):
        """ run model once with chosen parameters """
        if params and isinstance(params, dict):
            self.params = setParams(dParams=self.defaultParams, **params) # params is a dict {'param' : value}
        vect_len = time * self.model_freq
        soln = np.zeros((vect_len, ndim+1))
        self.model.integrate(ct.c_int(vect_len), ct.c_int(ndim), soln,
                             ct.c_double(time), ct.byref(self.params))
        return soln

    def _getModelOutput(self, time, soln, output_key=None):
        model_output = ModelOutput(time, soln,
                                   sampling_freq=self.sampling_freq,
                                   model_freq=self.model_freq)(output_key)
        return model_output

    @staticmethod
    def _chooseCostFunc(opt_feature):
        """ choose the cost function (e.g. simple) based on the selected optimization feature """
        simple_features = ['exp_time', 'insp_time', 'rri']
        simple_features.extend([sf+'_ave' for sf in simple_features])
        if opt_feature in simple_features: return CF_simple
        elif opt_feature == 'rsa': return CF_rsa
        elif opt_feature == 'ave_resp': return CF_resp
        elif opt_feature == 'cvc': return CF_cvc
        else: raise Exception("Cost Function Type NotImplementedYet")

    def _generateCost(self, x, *args):
        """ Called by the optimization routine
        x (list/array) : contains the parameters that will be changed
          (corresponds to params argument for class initialization)
        args : tuple that contains the stuff listed below"""
        if not args: args = self.cost_args
        cost_func = args[0]
        f_data = args[1]
        var_data = args[2]
        time = args[3]
        param_names = args[4]
        opt_feature = args[5]
        
        model_output = self._runModel(self.time, params=dict(zip(param_names, x)))

        f_model = self._getModelOutput(self.time_vec, model_output, output_key=opt_feature)
        self.f_model = f_model

        if opt_feature == 'cvc':
            cost = cost_func(f_data, f_model, var_data=var_data)
        else:
            cost = cost_func(f_data, f_model, var_data=var_data)

        if not self.fun or cost < self.fun:
            self.fun = cost
            print('min cost:', self.fun)

        if self.optMethod == 'cmaes':
            self.paramSpace['cost'].append(self.fun)
            for pi, pk in enumerate(sorted(self.params_init.keys())):
                self.paramSpace[pk].append(x[pi])
        return cost

    def optimize(self, opt_feature, method='genetic_alg', alg_args={},
                 sim_time=None, params={}, bounds=None, tol=None,
                 options=None, mp=False, maxiter=None, seed=None,
                 miniter=None):

        """ Method for optimizing the model for a particular
        optimization feature, model output.
        input arguments:
        opt_feature (str) : feature to extract from the data, which is used to optimize the model (e.g. 'rri')
        method [optional] (str) : the optimization method to use (e.g. 'genetic_alg')
        sim_time [optional] (int) : the number of seconds to run the model
        params [optional] (dict) : this is a dictionary of parameters and values
          (overrides the ones given for initialization of this class!)
        bounds [optional] (dict) : this is a dictionary of bounds,
          e.g. { 'hr' : (0, 0.001) }
        """

        print("\nsubject: {}".format(self.subject))
        print("optimization feature: {}\n".format(opt_feature))

        self.optMethod = method
        self.optFeature = opt_feature
        
        if tol: self.tol = tol
        if miniter: self.miniter = miniter
        
        if params: # overrides self.params_init!
            self.params_init = params
            self._setupModel(params=params)

        param_names = sorted(self.params_init.keys()) # parameter names in alphanumeric order
        x_init = [v for k, v in sorted(self.params_init.items())] # parameter values in corresponding order

        if not bounds:
            bounds = [(pv-0.5*pv, pv+0.5*pv) for pk, pv in sorted(self.params_init.items())]
        elif bounds is not None:
            for pn in self.params_init.keys():
                if pn not in bounds.keys():
                    pval = self.params_init[pn]
                    bounds[pn] = (pval-0.5*pval, pval+0.5*pval)
            bounds = [bv for bk, bv in sorted(bounds.items())]

        f_data = self._getData(opt_feature).copy()
        var_data = None

        if sim_time is None:
            temp_time = self._getData('time')
            sim_time = int((temp_time[-1] - temp_time[0])/10)
            print("sim_time:", sim_time)

        self._setTime(sim_time)
        
        if isinstance(f_data, np.ndarray):
            f_data = f_data[:int(self.sampling_freq*sim_time)]
            var_data = f_data.var()
            self.var = var_data
            if 'ave' in opt_feature: f_data = f_data.mean()

        if isinstance(f_data, dict):
            if opt_feature == 'cvc':
                f_data.update(self._getData('rsa'))
                f_data.update(self._getData('ave_resp'))

            if opt_feature in ['ave_resp', 'cvc']:
                f_data['insp_var'] = np.var(f_data['insp_times'])
                f_data['exp_var'] = np.var(f_data['exp_times'])
                self.var = f_data['insp_var'] + f_data['exp_var']
                print('\ninsp_time: {}'.format(f_data['insp_time']))
                print('exp_time: {}\n'.format(f_data['exp_time']))

            if opt_feature in ['rsa', 'cvc']:
                f_data['rri_ave'] = f_data['rri'].mean()
                self.var = f_data['rsa_var']


        cost_func = Optimizer._chooseCostFunc(opt_feature) # Choose cost function based on opt_feature

        args = [cost_func, f_data, var_data, # args for self._generateCost(...)
                sim_time, param_names, opt_feature]

        # Select optimization function
        methods = {"genetic_alg": optimize.differential_evolution,
                   "basinhopping": optimize.basinhopping,
                   "minimize": optimize.minimize,
        }
        opt_method = methods[method]

        print("starting optimization procedure...\n")

        self.f_data = f_data

        if method == 'genetic_alg':
            alg_args.update({'args': tuple(args),
                             'callback': self.ga_callback})

            if maxiter: alg_args.update({'maxiter': maxiter})
            if tol: alg_args.update({'tol': tol})
            if seed: alg_args.update({'seed': seed})
            elif not seed: alg_args.update( { 'seed': np.random.RandomState() } )

            alg_args.update({ 'polish': False })

            psize = alg_args['popsize']
            ga_init = np.array([np.linspace(x_init[xi]-0.1*x_init[xi], x_init[xi]+0.1*x_init[xi], psize)\
                                for xi in range(len(x_init))],dtype=np.float).reshape(-1,len(x_init))
            alg_args.update({ 'init': ga_init })
            
            result = opt_method(self._generateCost, bounds, **alg_args)

        elif method in ['basinhopping',]:
            alg_args.update( { 'minimizer_kwargs': {'args': tuple(args), 'bounds': bounds} } )
            alg_args.update( { 'callback': self.print_callback} )
            alg_args.update( { 'seed': np.random.RandomState() } )

            # Custom adaptive stepsize routine
            displace = RandomDisplacement(alg_args['stepsize'],
                                          random_state=alg_args['seed'])
            take_step_wrapped = AdaptiveStepsize(displace,
                                                 interval=alg_args['interval'],
                                                 verbose=alg_args['disp'],
                                                 accept_rate=alg_args['accept_rate'])
            alg_args.update( { 'take_step': take_step_wrapped })

            if tol: alg_args['minimizer_kwargs']['tol'] = tol
            if maxiter:
                alg_args['minimizer_kwargs']['options'] = {'maxiter': maxiter}

            result = opt_method(self._generateCost, x_init, **alg_args)

        elif method in ['minimize',]:
            alg_args.update( { 'args': tuple(args), 'bounds': bounds} )
            if tol: alg_args['tol'] = tol
            if options: alg_args['options'] = options
            
            result = opt_method(self._generateCost, x_init, **alg_args)

        elif method in ['cmaes',]:
            self.cost_args = (cost_func, f_data, var_data,
                              time, param_names, opt_feature)

            # CMAES instance (per Optimizer instance)
            self.cmaes = CMAES_ct()
            self.cmaes_result = ResultCMAES()

            x0 = self._getx0
            N = len(x0)
            self.cmaes.setup(ct.c_int(N), x0, ct.c_double(alg_args['stepsize']),
                             ct.c_double(tol), ct.c_int(maxiter))

            opt_method = self.cmaes.cmaes.generation_loop,
            opt_method(self, self.cmaes_result)
                

        print("\ndone with optimization.")
        
        # Save result object and full parameter space
        
        self.paramSpace['result'] = result
        self.paramSpace['defaultParams'] = self.defaultParams

        fnp = os.path.join(self.path, self.database)
        fnps = os.path.join(fnp, 'paramSpace_{}_{}.npy'.format(opt_feature, self.subject))
        self.fnps = fnps
        if not mp:
            np.save(fnps, self.paramSpace)
            return self.paramSpace
        elif mp:
            return self
    
    @property
    def covar(self):
        if self.optMethod != 'cmaes':
            ps = copy.copy(self.paramSpace)
            params = sorted(list(self.params_init.keys()))
            p = np.array([ps[pn] for pn in params]).reshape((len(params),-1))
            p = p.T
            np.random.shuffle(p)
            C = np.cov(p.T)
        u, s, v = np.linalg.svd(C)
        evecs = v.T
        evals = np.power(s, 2)
        return C, evals, evecs

    def pdf(self, fun=None, var=None):
        if fun is None: fun = self.fun
        if var is None: var = self.var
        try:
            likelihood = ( 1.0 / np.sqrt(2*np.pi*var) ) * np.exp(-fun)
            return likelihood.sum()
        except Exception as E:
            print("RuntimeWarning: PDF evaluated to be nan!\n",E)
            return np.nan

    def log_likelihood(self, fun=None, var=None):
        fx = self.pdf(fun, var)
        return -np.log(fx)
    
    def ga_acceptance(self, xk, convergence=0.0):
        """ Metropolis Hastings for genetic_alg """
        
        # return True

        def getPrevX():
            pnames = sorted(list(self.params_init.keys()))
            plist = [self.paramSpace[pk][-1] for pk in pnames]
            return np.array(plist, dtype=np.float)

        if len(self.paramSpace['cost']) < 2: return True
        
        x0 = getPrevX()
        fun0 = self.paramSpace['cost'][-1]

        print("fun0:",fun0,"fun1:",self.fun)

        if self.fun < fun0 and self.nr_accepted < self.miniter: return True

        if self.tot_evals < self.miniter: return False
        
        p0 = self.log_likelihood(fun=fun0, var=self.var)

        p1 = self.log_likelihood(fun=self.fun, var=self.var)

        if np.isinf(p0): p0 = -1.0

        print("p0:",p0,"p1:",p1)

        alpha = p0 / p1
        U = np.random.random()

        print("alpha:",alpha,"U:",U)
        
        if not np.isnan(alpha) and alpha > U:
            return True
        return False
    
    def ga_callback(self, xk, convergence=0.0):
        self.tot_evals += 1 # update total number of function evals
        print(['{} : {:.7f}'.format(str(pk),xv) for pk, xv in zip(sorted(list(self.params_init.keys())), xk)])
        print("convergence : {:f}".format(convergence,))
        accepted = self.ga_acceptance(xk, convergence=convergence)
        print("Accepted: {}\n".format(repr(accepted),))
        if np.isinf(self.fun) or np.isnan(self.fun): self.fun = 1e4
        if accepted:
            self.nr_accepted += 1 # update number of accepted samples
            self.arate = self.nr_accepted / self.tot_evals # update acceptance rate
            print("Acceptance: {:.01f}\n\n".format(self.arate,))
            self.paramSpace['cost'].append(self.fun)
            if len(xk) > 1 and self.nr_accepted > 2:
                self.paramSpace['covar'].append(self.covar)
            for pi, pk in enumerate(sorted(self.params_init.keys())):
                self.paramSpace[pk].append(xk[pi])
        gc.collect()

    def print_callback(self, x, f, accepted):
        print("params: {}".format(repr(x)))
        print("f: {}".format(repr(f)))
        print("accepted: {}\n".format(repr(accepted)))
        if accepted: # append accepted samples and corresponding costs
            for pi, pk in enumerate(sorted(self.params_init.keys())):
                self.paramSpace[pk].append(x[pi])
            self.paramSpace['cdf'].append(copy.copy(self.utils.data['cdf']))
            self.paramSpace['cost'].append(f)
