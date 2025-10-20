#
# description
#
import math
from decimal import Decimal
import pandas as pd
import numpy as np
import random as rndm
from scipy.signal import convolve
#from scipy.ndimage import uniform_filter1d

#
# formatting
#

#
def format_float_scientific(number, exp_digits=4, precision=14):
    """
    Similar to numpy.format_float_scientific, but has consitent length
    even if input number does not have "enough trailing zeros"
    [trim='k' in numpy method sounds like it might fix this but does not]
    """
    exp_digits_str = str("%02d"%float(exp_digits))
    precision_str = str("%02d"%float(precision))
    
    num, exp_raw = ("{:." +precision_str +"E}").format(Decimal(number)).split('E')
    sign = exp_raw[0]
    # create exponent number (leaving out sign "[1:]")
    exp = str(f"%{exp_digits_str}d"%float(exp_raw[1:]))
    #
    return num +'E' +sign +exp

#
# math
#

#
def OoM(x):
    """
    Get order of magnitude (OoM) of a number
    """
    return int(math.floor(math.log10(x)))

#
# pandas
#

#
def get_item(df, row, col, astype='series'):
    """
    <df> is a pandas dataframe that you would like to get an item from
    <row> and <col> reference the item and possible options are:
    
    <row>: integer of target row that item will be selected from or 
    list containing names of [0]column(str) and [1]item(s) defining
    target row(s)
    [single items need to be passed as lists: "row=[col, [item,]]"]
    
    <col>: integer or name (str) or column to be selected
    
    return first find, all finds, last find, index?
    
    <astype>: return the item as is, i.e. a pandas 'series' object or
              as 'float', or 'bool' (autodetect and unify these cases)
    """
    # check for empty dataframe
    if df.empty:
        item = -1
        
    else:
        # row
        if isinstance(row, int):
            series = df.iloc[row]
        elif isinstance(row, list):
            series = df.loc[df[row[0]].isin(row[1]) ]

        # column
        if isinstance(col, str):
            item = series[col]
        elif isinstance(col, int):
            item = series[ series.columns[col] ]

    # type
    if astype == 'series':
        return item
    elif astype == 'float':
        return float(item)
    elif astype == 'bool':
        return bool(item)

#
def set_item():
    """
    to do (necessary? probably need to understand pandas better)
    """
    
    #
    return None

#
# numpy / scipy
#

#
def sign_change(array, ignore_zeros=True):
    """
    Detect sign changes in an array
    (zero has its own sign in "np.sign()")
    To ingore zero-induced sign changes, set <ignore_zeros>=True
    """
    asign = np.sign(array)

    if ignore_zeros == True:
        sz = asign == 0
        while sz.any():
            asign[sz] = np.roll(asign, 1)[sz]
            sz = asign == 0

    sign_change = ((np.roll(asign, 1) -asign) != 0).astype(int)
    sign_change[0] = 0
    #
    return sign_change

#
def add_white_noise(x, f, seed_int, threshold):
    """
    Adds white noise to an input function f(x), where <threshold>
    determines the level of "dark count" noise, i.e. noise present
    independent of the signal level
    Whereever the signal level (function value) is above <threshold>
    the white noise will be proportional to the signal level
    [laplacian statistics]
    <seed_int> (int) is the seed for the pseudo-random number generator
    
    Returns 
    function f(x) + white noise, and white noise itself
    """
    # seed for pseudorandom number generator
    rndm.seed(seed_int)
    # generate white noise
    white_noise = []
    for i in range(np.shape(x)[0]):
        if np.abs(f[i]) > np.abs(threshold):
            temp = rndm.random()*np.sqrt(np.abs(f[i]))
        else:
            temp = rndm.random()*np.sqrt(np.abs(threshold))
        white_noise.append(temp)
    # add white noise to input function 
    function_plus_white_noise = white_noise + f
    #
    return function_plus_white_noise, white_noise

#
def pad_x_y(x, y, x_step, pad_size):
    """
    add padding according to padding size (pad_size)
    copy first and last y value for padding
    continue x axis on grid defined by x_step. for this to 
    properly work x must be a linearly interpolated array
    """
    # make sure pad_size is an integer
    pad_size = int(pad_size)
    # generate padding for y and add to y
    y_pad_l = y[0]*np.ones(pad_size)
    y_pad_r = y[-1]*np.ones(pad_size)
    y_pad = np.concatenate((y_pad_l, y, y_pad_r), axis=0)
    # generate padding for x and add to x
    x_pad_l = np.linspace(x[0]-x_step*pad_size, x[0], pad_size, 
                          endpoint=False)
    x_pad_r = np.linspace(x[-1]+x_step, x[-1]+pad_size*x_step, pad_size,
                          endpoint=True)
    x_pad = np.concatenate((x_pad_l, x, x_pad_r), axis=0)
    # return padded x and y values
    return x_pad, y_pad

#
def my_conv(x, y, kernel, method='scipy'):
    """
    convolution function, mode='same' in numpy/scipy has weird edges
    try alternatively mode='full' and figure out how to remove padding
    fft stuff doesn't work, should be faster [might not matter though]
    """
    kernel_div = 2
    # determine padding size from kernel chosen for convolution
    pad_size = int(kernel.size /kernel_div)
    # add padding
    x_pad, y_pad = pad_x_y(x, y, x[1] -x[0], pad_size)
    # compute convolution
    if method == 'scipy':
        y_conv_pad = convolve(y_pad, kernel, mode='same') /np.sum(kernel)
    elif method == 'numpy':
        y_conv_pad = np.convolve(y_pad, kernel, mode='same') /np.sum(kernel)
    # remove padding and return
    return y_conv_pad[pad_size:-pad_size]

#
def phi_norm(phi, norm=2*np.pi):
    """
    Normalize an angle <phi> to be within 0 and 2*pi (norm=2*pi, allows
    sine or cosine character) or pi (norm=pi, forces sine character)
    works for postive and negative values (in the 2*pi sense)
    """
    return phi -norm*math.floor(phi/norm)

#
def running_mean(x, y, N):
    """
    https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    
    - numpy.convolve slowest
    - "cumsum" has a floating point error starts to become an issue at N > 1E5
    - scipy best solution as the size gets preserved and it is the fastest
    """
    # numpy
    #return np.convolve(x, np.ones(N) / float(N), 'valid')

    # cumsum
    #cumsum = np.cumsum(np.insert(x, 0, 0)) 
    #return (cumsum[N:] - cumsum[:-N]) / float(N)
    
    # scipy (does not preserve size)
    # [uncomment import at the top of this file to use!]
    #return uniform_filter1d(x, N, mode='constant', origin=-(N//2))[:-(N-1)]
    
    # probably slow but takes care of proper padding
    return my_conv(x, y, np.ones(N))

#
#
#