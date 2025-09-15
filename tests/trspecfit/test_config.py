# 
# test retrieval of all functions that will not be numbered in a model
#

import pytest
# local imports
from trspecfit.config import prefix_exceptions, background_functions, convolution_functions

class TestPrefixExceptions:
    
    #
    def test_background_functions(self):
            """Test to see which functions are background functions"""
            
            background_fcts = background_functions()
            assert background_fcts == ('Offset', 'Shirley', 'ExpBack', 'LinBack')
            
    #
    def test_convolution_functions(self):
        """Test to see which functions are registered as convolution functions"""
        
        conv_fcts = set(convolution_functions())
        conv_fcts_local = {'gaussCONV', 'lorentzCONV', 'voigtCONV', 'expSymCONV', 'expDecayCONV', 'expRiseCONV', 'boxCONV'}
        
        # Check that the sets are equal
        assert conv_fcts == conv_fcts_local, f"Expected {conv_fcts_local}, got {conv_fcts}"

