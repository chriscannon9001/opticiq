#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 11:42:43 2025

@author: chris
"""

import numpy as _np


def fft_Efield(EFp):
    '''
    Use FFT to compute a Point Spread Function given complex valued Efield.
    See also fft_coords

    Parameters
    ----------
    EFp : 2d array
        Complex valued Efield.

    Returns
    -------
    PSF : 2d array
        Diffracted Point Spread Function
    '''
    EFinf = _np.fft.fftshift(_np.fft.fft2(EFp))
    PSF = _np.abs(EFinf * _np.conjugate(EFinf))
    return PSF


def fft_coords(x, wavelength):
    '''
    Formula to convert linear pupil axis to frequency FFT axis output.

    Parameters
    ----------
    x : 1d array
        Pupil plane axis
    wavelength : float
        Same units as x (e.g. mm)

    Returns
    -------
    u : 1d array
        Frequency space coordinates of after fft_Efield (radians)

    Example::
        # assume EFp has 128 samples at 0.1mm pitch and wavelength is 0.5um;
        x = numpy.arange(-64, 65) * .1
        u = fft_coords(x, 0.5/1000)
    '''
    assert _np.ndim(x) == 1, 'expected x to be 1d array'
    nx = len(x)
    xrange = _np.abs(x[-1] - x[0])    # fixme; these should work a little different for even nx or ny
    u = _np.arange(-nx / 2 + .5, nx / 2 + .5) * wavelength / xrange
    return u


def aperture_Efield(x, y, EF, ODx, ODy, cx=0, cy=0):
    '''
    Zero Efield outside of an elliptical aperture

    Parameters
    ----------
    x, y : 2d array
        Lateral coordinates of EF
    EF : 2d array
        Complex valued Efield
    ODx, ODy : float
        DESCRIPTION.
    cx, cy : float, optional
        x, y centroid location of aperture. The default is 0.

    Returns
    -------
    EF_aprt : 2d array
        Efield after aperture
    '''
    radx = ODx/2
    rady = ODy/2
    x2 = ((x - cx) / radx)**2
    y2 = ((y - cy) / rady)**2
    mask = (x2 + y2) < 1
    EF_aprt = EF * mask
    return EF_aprt


def obscure_Efield(x, y, EF, IDx, IDy, cx=0, cy=0):
    '''
    Zero Efield within an elliptical obscuration

    Parameters
    ----------
    x, y : 2d array
        Lateral coordinates of EF
    EF : 2d array
        Complex valued Efield
    IDx, IDy : float
        DESCRIPTION.
    cx, cy : float, optional
        x, y centroid location of aperture. The default is 0.

    Returns
    -------
    EF_obs : 2d array
        Efield after obscuration
    '''
    radx = IDx/2
    rady = IDy/2
    x2 = ((x - cx) / radx)**2
    y2 = ((y - cy) / rady)**2
    mask = (x2 + y2) >= 1
    EF_obs = EF * mask
    return EF_obs


def illum_2_PSFcam(OD, ID, Fnumber, wavelength, pixel, n=None):
    PSF = 0
    return PSF
