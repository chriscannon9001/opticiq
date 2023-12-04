#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:44:55 2023

@author: chris
"""

import numpy as _np
from ..math import _rotate_2d


def gaussian(res, A0, sigma, dark=0, center=None, order=2, theta=0):
    '''
    Generate an image containing a Gaussian beam, e.g. in the simplest case
    image = A0 * exp(-2*(r/2sigma)^2)

    Parameters
    ----------
    res : int or [int, int]
        resolution of image panel
    A0 : float
        peak amplitude
    sigma : float or [float, float]
        width of Gaussian w=2sigma
    dark : float
        dark level. The default is 0.
    center : (x, y), optional
        pixel of center of Gaussian. The default is None, meaning res/2.
    order : int
        exponent order of Gaussian function. The default is 2.
    theta : float, optional
        if sigma is paired, this is the rotation of sigma, so that an
        assymmetric Gaussian can be at any angle. Default is 0

    Returns
    -------
    image : 2d array of float
        image of Gaussian
    '''
    res *= _np.array([1, 1], dtype=int)
    if center is None:
        center = res / 2
    cx, cy = tuple(center)
    #x, y = _np.meshgrid(_np.arange(res[0]), _np.arange(res[1]))
    x, y = _np.meshgrid(_np.arange(res[0]) - cx, _np.arange(res[1]) - cy)
    if not _np.isscalar(sigma) and not _np.isclose(theta, 0):
        x, y = _rotate_2d(x, y, theta)
    w = sigma * _np.array([2, 2], dtype=int)
    image = dark + A0 * _np.exp(
        -2*((x/w[0])**order + (y/w[1])**order))
    return image
