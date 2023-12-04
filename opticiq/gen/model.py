#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:07:24 2023

@author: chris
"""

import numpy as _np
from scipy import ndimage as _ndimage


def image_transfer_simple(im0, upsample=None, distortion=None, sigma=None,
                          poisson=None, darklevel=0):
    '''
    Similar to modeling an image transfer from a source, through optics, to a
    sensor, except each step is given by the simplest possible transform.

    Parameters
    ----------
    im0 : 2d array
        input image, normally a flawless generated image
    upsample : float, optional
        Upsample using scipy.ndimage.zoom. The default is None.
    distortion : TYPE, optional
        DESCRIPTION. The default is None.
    sigma : float or [float, float]
        radius of Gaussian blur (see scipy.ndimage.gaussian_filter). The
        default is None.
    poisson : float, optional
        The gain of the sensor, i.e. count/photo-electrons. The default is
        None. If used, poisson statistics create grainy noise as seen on
        a camera.
    darklevel : float, optional
        Add a dark level, in counts

    Returns
    -------
    im : 2d array
        image at sensor after transfer
    '''
    if upsample is None and distortion is None:
        im1 = im0
    else:
        if distortion is None:
            im1 = _ndimage.zoom(im0, upsample, order=0)
        else:
            raise NotImplementedError
    if sigma is None:
        im2 = im1
    else:
        im2 = _ndimage.gaussian_filter(im1, sigma)
    if not _np.isclose(darklevel, 0):
        im2 += darklevel
    if poisson is None:
        im3 = im2
    else:
        gain = poisson
        rgen = _np.random.default_rng()
        im3 = rgen.poisson(im2 / gain) * gain
    return im3


def image_transfer_model(source, sensor=None, optics=None):
    '''
    NOT IMPLEMENTED

    Simulate a model for image transfer from a source, through optics, to a
    sensor.

    Parameters
    ----------
    im0 : TYPE
        DESCRIPTION.
    sensor : TYPE, optional
        DESCRIPTION. The default is None.
    optics : TYPE, optional
        DESCRIPTION. The default is None.
    source : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''
    raise NotImplementedError


class ModelSensor():
    def __init__(self):
        raise NotImplementedError


class ModelObject():
    def __init__(self):
        raise NotImplementedError


class ModelImage():
    def __init__(self):
        raise NotImplementedError
