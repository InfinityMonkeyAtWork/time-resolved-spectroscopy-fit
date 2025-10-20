#
# generate 1D and 2D (time-resolved) spectra
# ["x, par, plt_ind, args" is the typical fit function structure]
#
from trspecfit.mcp import Model
from IPython.display import display

# INFORMATION
# This package contains any functions that create spectra by
# - the model/component/parameter approach (see mcp.py)
# - custom code enabling other fit routines (pass function added here in  

#
def fit_model_mcp(x, par, plot_ind, model, dim, DEBUG):
    """
    <model> is a mcp.Model()
    <dim> =1: 1D spectra, =2: 2D data
    <plt_ind> =0: return the sum, =1: or individual component spectra
    [meaningless for dim=2 where the entire 2D dataset is returned]
    """
    model.update_value(new_par_values=par) # update lmfit parameters
    
    if DEBUG == True: 
        display(model.lmfit_pars)
        model.print_all_pars(detail=1)
    
    # create energy- (and time-)resolved spectrum/ data
    if dim == 1: # 1D
        if plot_ind == 0: # return sum of all components
            model.create_value1D()
            return model.value1D
        elif plot_ind == 1: # return individual components
            model.create_value1D(store1D=1)
            return model.component_spectra
        
    elif dim == 2: # 2D
        model.create_value2D()
        return model.value2D