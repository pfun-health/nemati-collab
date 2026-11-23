import numpy as np

# supr = np.testing.suppress_warnings()
# supr.filter(module=np.random)

import ctypes as ct
import os, copy, time
import multiprocessing as mp
from queue import Empty
import gc

from ..fit_model import cost_functions as cf, Optimizer
from ..model import defaultParams as defParams, setupModel, setParams, ModelOutput
from ..load_data import load_stats

model = setupModel()

def yield_from_process(q, p): # yield new results as soon as they are produced
    while p.is_alive():
        p.join(timeout=10)
        while True:
            try:
                yield q.get(block=False)
            except Empty:
                break

def mp_helper(args):
            vlen, ndim, time, params_ = args
            soln = np.zeros((vlen, ndim+1))
            model.integrate(ct.c_int(vlen), ct.c_int(ndim), soln,
                            ct.c_double(time), ct.byref(params_))
            return soln


class Sampler(object):
    
    def __init__(self, subject,
                 database='fantasia',
                 path="~/Public",
                 params={"hr": 0.001,
                         "rsa1": 0.001,
                         "rsa2": 0.0011},
                 defaultParams=None,
                 miniter=100,
                 maxiter=1e4,
                 tol=1e-4,
                 atarget=0.23,
                 sampling_freq=250,
                 model_freq=1000,
                 time=120,
                 njobs=16,
                 nsamples=16,
                 minsamples=100,
                 step_size=[[0.00001, 0.00001, 0.00001]*3],
                 bounds=[(0, 0.002), (0, 0.0012), (0, 0.0012)]):

        self.subject = subject
        self.database = database
        self.path = os.path.expanduser(path)

        if defaultParams: self.defaultParams = defaultParams
        elif not defaultParams: self.defaultParams = copy.copy(defParams) # if not provided, use default parameters given in parameters.py
        
        self.params_init = params # this is a dictionary of initial parameters (free params)
        self.paramSpace = dict([(pk, []) for pk in params.keys()]) # dictionary that defines the parameter space
        self.paramSpace.update( { "L" : [] } ) # Likelihood from each sample
        self.paramSpace.update( { "accept": [] } ) # acceptance rate

        self.tot_evals = 0 # total number of function evals
        self.nr_accepted = 0 # number of accepted samples
        self.arate = 0.5 # acceptance rate
        self.atarget = atarget # acceptance rate target

        self.tol = tol # tolerance, if set
        self.fun = None # current cost
        self.miniter = miniter # minimum number of accepted samples, if set
        self.maxiter = maxiter # max number of acc samples
        self.var = None # variance from the data
        self.bounds = bounds # bounds of each parameter
        self.nsamples = nsamples # number of samples to produce per process
        self.minsamples = minsamples # minimum number of samples per iteration
        
        self.cov = np.array(step_size, dtype=float).reshape(
            (len(params.keys()),len(params.keys()))) # multivariate step size
        self.pre_cov = [] # previous covariance matrices
        
        self.njobs = int(njobs) # number of processes to use
        
        self.sampling_freq = sampling_freq
        self.model_freq = model_freq

        self.time = time
        self.time_vec = np.linspace(0, time, int(self.model_freq*time))

        self.f_model = None
        self.f_data = None
        
        self.optMethod = None # Method for sampling
        self.optFeature = None # Feature to optimize (e.g. RSA, CVC)
        
        self.cost_args = tuple()

        self._loadData()

    @property
    def _getx0(self): # Get the initial sample as a numpy array
        return np.array([pi for pk, pi in sorted(list(self.params_init.items()))], dtype=float)

    @property
    def _getx1(self): # Get the last sample in the paramSpace
        return np.array([self.paramSpace[pk][-1] for\
                         pk in sorted(self.params_init.keys())],
                        dtype=float)

    @property
    def _getx(self): # Get the paramSpace as an array of all samples
        xp = []
        for pix in range(len(self.paramSpace['L'])):
            prow = [self.paramSpace[pk][pix] for pk in sorted(list(self.params_init.keys()))]
            xp.append(prow)
        return np.array(xp, dtype=float).reshape(-1,len(self.params_init.keys()))

    def _updateParamSpace(self, x): # Append the sample to the list in paramSpace
        for px, pk in enumerate(sorted(list(self.params_init.keys()))):
            self.paramSpace[pk].append(x[px])

    def _calcL(self, x): # Calculate likelihood
        return 1.0 / self._generateCost(x)
        
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
    
    def _runModel(self, time, params=None, ndim=12):
        """ run model once with chosen parameters """

        njobs = 4 #int(mp.cpu_count() / 2)
        
        vect_len = int(time * self.model_freq / njobs)

        params_ = setParams(dParams=self.defaultParams, **params)

        outs = []
        args = ( vect_len, ndim, time, params_ )
        args_list = [args for i in range(njobs)]
        with mp.Pool(processes=int(njobs)) as pool:
            res = pool.map_async(mp_helper, args_list, callback=outs.append)
            res.wait()
            
        outs_array = np.array(outs).reshape(-1,ndim+1)
        return outs_array

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

    def _generateCost(self, x):
        """ Called by the optimization routine
        x (list/array) : contains the parameters that will be changed
          (corresponds to params argument for class initialization)
        args : tuple that contains the stuff listed below"""
        args = self.cost_args
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

        return cost

    def _genSamp(self, Q, x0):
        """ generate a set of samples, using one process """
        
        xs = np.sqrt(np.diag(self.cov)) * np.random.randn(self.nsamples, x0.size) + x0

        for xii in xs:
            test = []
            for i in range(len(self.bounds)):
                if all([xii[i] >= self.bounds[i][0], xii[i] <= self.bounds[i][1]]):
                    test.append(True)
                else:
                    test.append(False)
            if all(test):
                Qtup = (xii.tolist(), self._calcL(xii))
                Q.put(Qtup, block=True)

    def _genSamples(self, x0):
        """ generate many sets of samples using njobs processes. """
        
        Q = mp.Queue()
        jobs = []
        outs = []
        for j in range(self.njobs):
            xj = np.sqrt(np.diag(self.cov)) * np.random.randn(x0.size) + x0
            pargs = (Q, xj)
            proc = mp.Process(target=self._genSamp, args=pargs)
            jobs.append(proc)
            proc.start()

        live = list(jobs)
        nr_out = 0
        while all([live, nr_out < self.minsamples]):
            for pro in jobs:
                for pout in yield_from_process(Q, pro):
                    if pout in outs:
                        print("\nWARNING: Trying to append an old sample from the same queue.\n")
                    elif pout not in outs:
                        outs.append(pout)
                        nr_out += 1
            live = [p for p in live if p.is_alive()]
        
        try:
            if len(outs) == 0: raise(ValueError("len(outs) = 0.\n"))
            return outs
        except ValueError as e:
            pass

    def _acceptance(self, candidates, x, L):
        """ determine candidates to be accepted """
        nr_accepted = 0
        for c in candidates:
            print(c)
            if c[1] >= L:
                self._updateParamSpace(c[0])
                self.paramSpace['L'].append(c[1])
                nr_accepted += 1
        self.arate += (nr_accepted / len(candidates))
        self.arate /= 2.0

    def _burnConditions(self, ccheck):
        conditions = [
            self.tot_evals > self.miniter*2,
            not ccheck
        ]
        return all(conditions)

    def _genLoop(self):
        """ Sample generation loop; includes covariance matrix update """

        self._updateParamSpace(self._getx0) # put x0 in paramspace
        L0 = self._calcL(self._getx0) # calculate initial likelihood
        self.paramSpace['L'].append(L0) # put in paramspace
        
        wcheck = False # distribution convergence
        ccheck = False # covariance convergence

        print("\nSampling loop started...\n")

        try:
            
            while wcheck is False:
                
                x = self._getx1 # Last sample in paramspace
                L = self.paramSpace['L'][-1] # Last likelihood
                
                candidates = self._genSamples(x) # generate nsamples candidates

                self.cov *= (self.atarget + self.arate) # scale covariance by normalized acceptance rate

                if candidates:
                    self.tot_evals += len(candidates) # update total number of function evals
                    self._acceptance(candidates, x, L) # determine which samples are accepted
                    self.paramSpace['accept'].append(self.arate)
                    
                    if self._burnConditions(ccheck): # update covariance matrix after burn-in period
                        self.pre_cov.append(self.cov)
                        ixs = np.random.randint(0, int(len(self.paramSpace['L'])/2),
                                                int(len(self.paramSpace['L'])/4))
                        csamples = self._getx[int(len(self.paramSpace['L'])/2):,:]
                        chosen = csamples[ixs,:]
                        self.cov = np.cov(chosen, rowvar=False)
                        mean_diff = np.mean([np.abs(self.pre_cov[pi] - self.cov)\
                                             for pi in range(len(self.pre_cov))])
                        if mean_diff < self.tol:
                            ccheck = True # covariance convergence
                            print("Covariance matrix converged.")

                    print(
                        "{:d} : a={:.3f}, x={}, L={:.3e}  ;\n  step: {}, cov_conv: {}\n".format(
                            self.tot_evals,
                            self.arate,
                            str(self._getx1),
                            self.paramSpace['L'][-1],
                            str(np.diag(self.cov)),
                            str(ccheck))
                    )

                    wcheck = all([self.atarget > self.arate, ccheck])
                    if self.maxiter < self.tot_evals: break

        except KeyboardInterrupt:
            
            print("\nInterrupted.\n")
            time.sleep(3)

        finally:
            
            print("\nSampling finished.\n")

    def sample(self, opt_feature, tol=None, miniter=None):
        """ Perform the sampling procedure defined in the self.genSamples(...) method """

        print("\nsubject: {}".format(self.subject))
        print("optimization feature: {}\n".format(opt_feature))

        self.optFeature = opt_feature
        
        if tol: self.tol = tol
        if miniter: self.miniter = miniter

        param_names = sorted(self.params_init.keys()) # parameter names in alphanumeric order
        x_init = [v for k, v in sorted(self.params_init.items())] # parameter values in corresponding order

        f_data = self._getData(opt_feature).copy()
        var_data = None
                
        if isinstance(f_data, np.ndarray):
            f_data = f_data[:int(self.sampling_freq*self.time)]
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

            if opt_feature in ['rsa', 'cvc']:
                f_data['rri_ave'] = f_data['rri'].mean()
                self.var = f_data['rsa_var']
        
        cost_func = Optimizer._chooseCostFunc(opt_feature) # Choose cost function based on opt_feature

        args = [cost_func, f_data, var_data, # args for self._generateCost(...)
                self.time, param_names, opt_feature]
        args = tuple(args)
        self.cost_args = args

        self._genLoop() # Generate samples

        self.paramSpace['defaultParams'] = self.defaultParams # store default parameters
        self.paramSpace['cov'] = self.cov # store covariance matrix
        self.paramSpace['std'] = np.std(self._getx,0) # store standard error of the distribution
        self.paramSpace['samples'] = self._getx # store the samples as a numpy array

        fnp = os.path.join(self.path, self.database)
        fnps = os.path.join(fnp, 'sampleSpace_{}_{}.npy'.format(opt_feature, self.subject))
        self.fnps = fnps
        
        return self

    def save(self, fname=None):
        if not fname: fname = self.fnps
        np.save(fname, self.paramSpace)
        print("\nSaved paramSpace to {} .".format(fname))
