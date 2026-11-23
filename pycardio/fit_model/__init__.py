try:
    from cost_functions import *
    from optimize import Optimizer
except (ModuleNotFoundError, ImportError) as e:
    from .cost_functions import *
    from .optimize import Optimizer
    
