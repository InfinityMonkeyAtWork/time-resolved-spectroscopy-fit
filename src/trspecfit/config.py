#
# Configuration and constants for trspecfit
#
from trspecfit.functions import energy as fcts_energy
from trspecfit.functions import time as fcts_time
#from trspecfit.functions import depth as fcts_depth

#
def prefix_exceptions():
    """
    <prefix_exceptions> are function names that will not be added as prefix, i.e. 
    functions that have only one instance in <comps> and only have one parameter
    """
    return background_functions() + convolution_functions()

#
def background_functions():
    """
    Hardcode all background functions here
    Location: src/trspecfit/functions/energy.py
    """
    return ('Offset', 'Shirley', 'ExpBack', 'LinBack')

#
def convolution_functions():
    """
    Get all convolution functions from /src/trspecfit/functions/time.py
    It does not seem to me that it ever makes sense to allow convolution with
    the same function more than once.
    Convoluting with two Gaussians/ Cauchys/ other "closed" functions should be 
    represented with one kernel instead of multiple kernels.
    Other kernels do not preserve the shape but there can be parameter confusion
    and convoluting with e.g. two boxes or two triangles should be represented 
    by the resulting shape instead.
    """
    # Get all function names ending in 'CONV' from time functions
    return tuple(name for name in dir(fcts_time) 
                 if callable(getattr(fcts_time, name)) 
                 and name.endswith('CONV') 
                 and not name.startswith('_'))

#
def energy_functions():
    """Get all energy functions from /src/trspecfit/functions/energy.py"""
    return tuple(name for name in dir(fcts_energy)
                 if callable(getattr(fcts_energy, name))
                 and not name.startswith('_'))
