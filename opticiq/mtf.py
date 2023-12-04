#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 20:10:29 2023

@author: chris
"""


class MTF_Analysis():
    '''
    MTF_Analysis can find optical Modulation Transfer Function (MTF) using the
    edges of slant-rectangle test images (recommended) or using the
    line-spread-function of cross-hair test images.

    Example::
        
        import numpy as np
        im = np.asarray(...)
        mtfa = MTF_Analysis(im)
        mtfa.edge2mtf()
    '''
    def __init__(self):
        raise NotImplementedError

    def lsf(self):
        pass

    def lsf2mtf(self):
        pass

    def edge(self):
        pass

    def edge2mtf(self):
        pass
