import numpy as np
from scipy._lib._util import check_random_state

"""

Robert A. Capps
April 25, 2018

Adapted from scipy v1.0.0
optimize/_basinhopping.py

"""

class AdaptiveStepsize(object):
    """
    Class to implement adaptive stepsize.
    This class wraps the step taking class and modifies the stepsize to
    ensure the true acceptance rate is as close as possible to the target.
    Parameters
    ----------
    takestep : callable
        The step taking routine.  Must contain modifiable attribute
        takestep.stepsize
    accept_rate : float, optional
        The target step acceptance rate
    interval : int, optional
        Interval for how often to update the stepsize
    factor : float, optional
        The step size is multiplied or divided by this factor upon each
        update.
    verbose : bool, optional
        Print information about each update
    """
    def __init__(self, takestep, accept_rate=0.5, interval=50, factor=0.95,
                 verbose=True):
        self.takestep = takestep
        self.target_accept_rate = accept_rate
        self.interval = interval
        self.factor = factor
        self.verbose = verbose

        self.nstep = 0
        self.nstep_tot = 0
        self.naccept = 0

    def __call__(self, x):
        if self.nstep_tot == 0:
            self.original_stepsize = np.array(x,dtype='float')*0.1
        return self.take_step(x)

    def _adjust_step_size(self):
        old_stepsize = self.takestep.stepsize.copy()
        accept_rate = float(self.naccept) / self.nstep
        if self.nstep_tot >= self.interval*5 and accept_rate <= 0.1*self.target_accept_rate:
            self.takestep.stepsize = self.original_stepsize
        if accept_rate > self.target_accept_rate:
            #We're accepting too many steps.  This generally means we're
            #trapped in a basin.  Take bigger steps
            self.takestep.stepsize /= self.factor
        else:
            #We're not accepting enough steps.  Take smaller steps
            self.takestep.stepsize *= self.factor
        if self.verbose:
            print( "adaptive stepsize:\n acceptance rate: %f ; target: %f\n new "
                   "stepsize: %s ; old stepsize: %s\n factor: %f\n" % (accept_rate,
                                                                       self.target_accept_rate,
                                                                       repr(self.takestep.stepsize),
                                                                       repr(old_stepsize),
                                                                       self.factor,) )

    def take_step(self, x):
        self.nstep += 1
        self.nstep_tot += 1
        constraints = [self.nstep % self.interval == 0,
                       self.nstep_tot >= self.interval * 2 or\
                       self.naccept >= self.interval\
                       or ( float(self.naccept/self.nstep) <= self.target_accept_rate*0.3\
                            and float(self.naccept/self.nstep) > 0.0 )
        ]
        if all(constraints): # constraints must be satisfied to start adapting step
            self._adjust_step_size()
        return self.takestep(x)

    def report(self, accept, **kwargs):
        "called by basinhopping to report the result of the step"
        if accept:
            self.naccept += 1


class RandomDisplacement(object):
    """
    Add a random displacement of maximum size, stepsize, to the coordinates
    update x inplace
    Parameters
    ----------
    stepsize : float, optional
        stepsize
    random_state : None or `np.random.RandomState` instance, optional
        The random number generator that generates the displacements
    """
    def __init__(self, stepsize, random_state=None):
        self.stepsize = np.array(stepsize, dtype='float')
        self.random_state = check_random_state(random_state)

    def __call__(self, x):
        x += self.random_state.normal(0.0, self.stepsize, size=np.shape(x))
        return x
