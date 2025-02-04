#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 11:42:43 2025

@author: chris
"""

import numpy as _np
from matplotlib import pyplot as _plt


def plot_Efield(EF, *args):
    '''
    Simple plot of Intensity and Phase of an Efield.
    Returns pyplot figure and subplot objects for further customization.

    Parameters
    ----------
    EF : 2d Array
        Efield
    x, y : 2d Array, optional
        Provide x,y ordinates if you want them plotted instead of
        row,column

    Returns
    -------
    fig : 
        figure
    ax0, ax1 : 
        subplots
    '''
    fig, (ax0, ax1) = _plt.subplots(1, 2)
    I = _np.abs(EF * _np.conjugate(EF))
    phase = _np.arctan2(_np.real(EF), _np.imag(EF))
    # plot intensity
    if len(args) == 2:
        ax0.contourf(*args, I)
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
    else:
        ax0.contourf(I)
    ax0.set_title('Intensity')
    # plot phase
    if len(args) == 2:
        ax1.contourf(*args, phase)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
    else:
        ax1.contourf(phase)
    ax1.set_title('Phase')
    return fig, ax0, ax1


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


def fftfreq(n, d, wavelength):
    '''
    Warp around np.fft.fftfreq to consider wavelength, and throw in fftshift.
    f = fftshift(0, 1, ..., n/2-1, ..., -1) * wavelength/(d*n) if n is even
    f = fftshift(0, 1, ..., (n-1)/2, ..., -1) * wavelength/(d*n) if n is odd

    Parameters
    ----------
    n : int
        Window length
    d : float
        Sample spacing
    wavelength: float
        Same units as d

    Returns
    -------
    f : 1d array
        Frequency space coordinates of after fft_Efield (radians)

    Example::
        
        # assume EFp has 128 samples at 0.1mm pitch and wavelength is 0.5um;
        f = fftfreq(128, .1, 0.5/1000)
    '''
    f = _np.fft.fftshift(_np.fft.fftfreq(n, d) * wavelength)
    return f


def init_coords(apitch, xpitch, wavelength):
    '''
    Helper to initialize pupil and frequency coordinates while meeting expected
    pitch in each.

    Parameters
    ----------
    apitch : float
        Desired angular pitch in angular (image) space. (radians).
        Negative indicates a min apitch rather than exact (will round down).
    xpitch : float
        Desired linear pitch in pupil space. (same units as wavelength).
        Negative indicates a min xpitch rather than exact (will round down).
        One of xpitch or apitch needs to be negative.
    wavelength : float
        Wavelength of light in units of choice

    Returns
    -------
    apitch : float
    xpitch : float
    npix : int
    '''
    if apitch < 0:
        assert xpitch>0, 'Expected one of apitch or xpitch to be >0'
        pupilspan0 = wavelength / -apitch
        npix = int(_np.ceil(pupilspan0 / xpitch)) + 1
        # after rounding, recalc pupilspan and apitch
        pupilspan = (npix - 1) * xpitch
        apitch = wavelength / pupilspan
    elif xpitch < 0:
        assert apitch>0, 'Expected one of apitch or xpitch to be >0'
        pupilspan = wavelength / apitch
        npix = int(_np.ceil(pupilspan / -xpitch))
        # after rounding, recalc xpitch
        xpitch = pupilspan / npix
    else:
        raise(ValueError, 'Expected one of apitch or xpitch to be <0')
    x_ax = _np.linspace(-pupilspan/2, pupilspan/2, npix)
    a_ax = fftfreq(npix, xpitch, wavelength)
    return apitch, xpitch, npix, x_ax, a_ax


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
