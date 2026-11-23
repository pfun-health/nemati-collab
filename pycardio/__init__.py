try:
    from process_data import process_data
    from load_data import load_data
    import model
    from fit_model import *
    from sample_model import *
    from utils import *
except (ModuleNotFoundError, ImportError):
    from .process_data import process_data
    from .load_data import load_data
    from .fit_model import *
    from .sample_model import *
    from .model import *
    from .utils import *
    
