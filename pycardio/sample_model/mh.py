from .sample import *

class SamplerMH(Sampler):
    
    def __init__(self, *args, **kwargs):
        Sampler.__init__(self, *args, **kwargs)

    def _acceptance(self, candidates, x, L):
        """ Metropolis Hastings acceptance criteria """
        nr_accepted = 0
        U = np.random.random()
        keepers = [c for c in candidates if (c[1] / L) >= U]
        for cx, cl in keepers:
            print(cx, cl)
            self._updateParamSpace(cx)
            self.paramSpace['L'].append(cl)
            nr_accepted += 1
        self.arate += (nr_accepted / len(candidates))
        self.arate /= 2.0

    
