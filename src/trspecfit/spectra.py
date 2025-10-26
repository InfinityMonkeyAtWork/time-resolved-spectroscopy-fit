"""
Generate 1D and 2D (time-resolved) spectra
["x, par, plot_ind, args" is the typical fit function structure]
"""
from trspecfit.mcp import Model
from IPython.display import display

# INFORMATION
# This package contains any functions that create spectra by
# - the model/component/parameter approach (see mcp.py)
# - custom code enabling other fit routines 
#   (add function here and pass function name as Project.spec_fct_str)  

#
def fit_model_mcp(x, par, plot_ind, model, dim, DEBUG):
    """
    <model> is a mcp.Model()
    <dim> =1: 1D spectra, =2: 2D data
    <plot_ind> =False: return the sum, =True: or individual component spectra
    [meaningless for dim=2 where the entire 2D dataset is returned]
    """
    model.update_value(new_par_values=par) # update lmfit parameters
    
    if DEBUG: 
        display(model.lmfit_pars)
        model.print_all_pars(detail=1)
    
    # create energy- (and time-)resolved spectrum/ data
    if dim == 1: # 1D
        if plot_ind: # return individual components
            model.create_value1D(store1D=1)
            return model.component_spectra
        else: # return sum of all components
            model.create_value1D()
            return model.value1D
        
    elif dim == 2: # 2D
        model.create_value2D()
        return model.value2D