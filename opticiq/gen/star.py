#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 20:09:37 2023

@author: chris
"""

import numpy as _np
from scipy import ndimage as _ndimage


def genstarfield(res, nstar, sigma, magrange=[0, 4], E=1000, dark=1):
    '''
    Generate a blurry star field in the raw (array of float, without Poisson noise)

    Parameters
    ----------
    res : int or [int, int]
        resolution of image panel
    nstar : int
        number of stars in field
    sigma : float or [float, float]
        radius of Gaussian blur (see scipy.ndimage.gaussian_filter)
    magrange : [float, float], optional
        Magnitude range of stars in the field, where 1 is defined by E.
        Each star will have a randomly selected energy in this range.
        The default is [0, 4].
    E : float, optional
        Signal energy level, in counts, of Mag 0 star. The default is 1000.
    dark : float, optional
        Bias level, in counts, of darkness. The default is 1.

    Returns
    -------
    image_raw : 2d array of float
        Image of star field in floating point array format
    '''
    res *= _np.array([1, 1], dtype=int)
    hres = res[0]
    vres = res[1]
    #x, y = _np.meshgrid(_np.arange(res[0]), _np.arange(res[1]))
    canvas = dark*_np.ones((vres, hres))
    for i in range(nstar):
        x = _np.random.randint(0, hres)
        y = _np.random.randint(0, vres)
        mag = _np.random.randint(magrange[0], magrange[1]+1)
        canvas[y, x] = E * 100**(mag/5)
    image_raw = _ndimage.gaussian_filter(canvas, sigma)
    return image_raw
