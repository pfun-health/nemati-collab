from .sample import *

class SamplerMH(Sampler):
    
    def __init__(self, *args, **kwargs):
        Sampler.__init__(self, *args, **kwargs)

    def _acceptance(self, candidates, x, L):
        """ Metropolis Hastings acceptance criteria """
        cx, cl = candidates[np.random.randint(len(candidates))]
        accepted = cl >= L or (L > 0 and np.random.random() < cl / L)
        if not accepted:
            cx, cl = x, L
        print(cx, cl)
        self._updateParamSpace(cx)
        self.paramSpace['L'].append(cl)
        self.arate = (self.arate + float(accepted)) / 2.0

    
