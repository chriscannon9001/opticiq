#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 09:37:43 2024

@author: chris
"""

from matplotlib import pyplot as plt
import numpy as _np
from scipy import interpolate as _interp


def edgeV_Analysis(imG_k, upsample=1, cutoffs=(85, .85, 2)):
    '''
    Parameters
    ----------
    imG_k : dict of 2d arrays
        Contains 'I0', 'I_y', 'ones'. Has been trimmed to an ROI.
    upsample : float, optional.
        Increase sampling by this factor, e.g. 2x. The default is 1x.
    cutoffs : (float, float, int), optional
        (percentile0-100, threshold0-1, min_offset).
        The default is (85, .85, 2).
        Cutoffs are used to select rows that have enough good samples.
        e.g. (100, .5, 0) means FWHM (of samples per row).
        e.g. (50, 1, 0) means >median (of samples per row).

    Returns
    -------
    ESF : 1d array
        Edge Spread Function
    dx : 1d array
        Sample column w.r.t. center of edge. Spacing is 1/upsample
    edgeV_detail : TYPE
        dict of various intermediates.

    Pseudo-code:
        1. sum(I_x), cx, n for each row
        2. mask: select rows above a cutoff criterea for n
        3. line fit cx[mask] vs y[mask]
        4. dx = x - cx
        5. flatten I0 vs dx
        6. resample on new x
    '''
    I_x = imG_k['I_x']
    ones = imG_k['ones']
    nx = I_x.shape[1]
    x_ax = _np.arange(nx)
    # stats per row based on I_x
    n = ones.sum(1)
    Ix = I_x.sum(1)
    cx_raw = (I_x*x_ax).sum(1) / Ix
    # mask off rows that don't have enough samples
    # maybe think this out more later
    percentile, threshold, min_offset = cutoffs
    high = _np.percentile(n, percentile)
    mask = n > min((high - min_offset), (threshold * high))
    good, = _np.nonzero(mask)
    # fit y, cx to a line, ignoring masked off data
    y_ax = _np.arange(I_x.shape[0])
    y2 = y_ax[mask]
    cx2 = cx_raw[mask]
    c, _, _, _ = _np.linalg.lstsq(_np.vstack([y2, _np.ones(len(y2))]).T,
                                  cx2, rcond=-1)
    m = c[0]
    b = c[1]
    cx_fit = b + m*y_ax
    # need 2d arrays: ones; mask, I0, dx; x position w.r.t. the edge
    x, y = _np.meshgrid(x_ax, y_ax)
    dx = x - cx_fit.reshape(-1, 1)
    ones = _np.array(ones * mask.reshape(-1, 1), dtype=bool)
    I0 = imG_k['I0']
    # resample irregular (I0 vs dx) onto a new grid
    ones_flat = ones.flatten()
    dx_flat = dx.flatten()[ones_flat]
    y_flat = y.flatten()[ones_flat]
    I0_flat = I0.flatten()[ones_flat]
    nx2 = int(_np.median(n[mask]))
    mnmx = (nx2-1/upsample)/2
    dx2, y2 = _np.meshgrid(_np.linspace(-mnmx, mnmx, nx2*upsample),
                           good)
    #dx2, y2 = _np.meshgrid(_np.arange(-(nx2-1)/2, nx2-(nx2-1)/2),
    #                       good)
    ESF2 = _interp.griddata((dx_flat, y_flat), I0_flat, (dx2, y2), 'cubic',
                            fill_value=0)
    ones2 = _interp.griddata((dx.flatten(), y.flatten()), ones.flatten(),
                             (dx2, y2), 'cubic', fill_value=0)
    #rbf = _interp.Rbf(dx_flat, y_flat, I0_flat)
    #ESF2 = rbf(dx2, y2)
    #rbf = _interp.Rbf(dx.flatten(), y.flatten(), ones.flatten())
    #ones2 = rbf(dx2, y2)
    # project onto dx axis and take average
    nc = ones2.sum(0)
    ESF = (ESF2*ones2).sum(0) / nc
    # intermediate details
    edgeV_detail = dict(cx_raw=cx_raw, cx_fit=cx_fit, Ix=Ix, n=n, mask=mask,
                        y_ax=y_ax, dx=dx, I0=I0*ones, ones=ones,
                        dx_flat=dx_flat, I0_flat=I0_flat, dx2=dx2, ESF2=ESF2,
                        ones2=ones2)
    return ESF, dx2[0, :], edgeV_detail


def plot_edgeV_data(ESF, dx, edgeV_details):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(edgeV_details['I0'], cmap='gray')
    mask = edgeV_details['mask']
    cx_raw = edgeV_details['cx_raw'][mask]
    cx_fit = edgeV_details['cx_fit'][mask]
    y_ax = edgeV_details['y_ax'][mask]
    plt.plot(cx_raw, y_ax, 'r+')
    plt.plot(cx_fit, y_ax)
    plt.subplot(1, 2, 2)
    dx_flat = edgeV_details['dx_flat']
    I0_flat = edgeV_details['I0_flat']
    plt.plot(dx_flat, I0_flat, '.')
    plt.plot(dx, ESF)


def edgeH_Analysis(imG_k, upsample=1, cutoffs=(85, .85, 2)):
    '''
    Parameters
    ----------
    imG_k : dict of 2d arrays
        Contains 'I0', 'I_x', 'ones'.
    upsample : float, optional.
        Increase sampling by this factor, e.g. 2x. The default is 1x.
    cutoffs : (float, float, int), optional
        (percentile0-100, threshold0-1, min_offset).
        The default is (85, .85, 2).
        Cutoffs are used to select column that have enough good samples.
        e.g. (100, .5, 0) means FWHM (of samples per column).
        e.g. (50, 1, 0) means >median (of samples per column).

    Returns
    -------
    ESF : 1d array
        Edge Spread Function
    dy : 1d array
        Sample column w.r.t. center of edge. Spacing is 1/upsample
    edgeV_detail : dict
        dict of various intermediates.

    Pseudo-code:
        1. sum(I_y), cy, n for each row
        2. mask: select rows above a cutoff criterea for n
        3. line fit cy[mask] vs x[mask]
        4. dx = x - cx
        5. flatten I0 vs dy
        6. resample on new y
    '''
    I_y = imG_k['I_y']
    ones = imG_k['ones']
    ny = I_y.shape[0]
    y_ax = _np.arange(ny)
    # stats per row based on I_y
    n = ones.sum(0)
    Iy = I_y.sum(0)
    cy_raw = (I_y*y_ax.reshape(-1, 1)).sum(0) / Iy
    # mask off rows that don't have enough samples
    # maybe think this out more later
    percentile, threshold, min_offset = cutoffs
    high = _np.percentile(n, percentile)
    mask = n > min((high - min_offset), (threshold * high))
    good, = _np.nonzero(mask)
    # fit y, cx to a line, ignoring masked off data
    x_ax = _np.arange(I_y.shape[1])
    x2 = x_ax[mask]
    cy2 = cy_raw[mask]
    c, _, _, _ = _np.linalg.lstsq(_np.vstack([x2, _np.ones(len(cy2))]).T,
                                  cy2, rcond=-1)
    m = c[0]
    b = c[1]
    cy_fit = b + m*x_ax
    # need 2d arrays: ones; mask, I0, dy; y position w.r.t. the edge
    x, y = _np.meshgrid(x_ax, y_ax)
    dy = y - cy_fit
    ones = _np.array(ones * mask, dtype=bool)
    I0 = imG_k['I0']
    # resample irregular (I0 vs dx) onto a new grid
    ones_flat = ones.flatten()
    dy_flat = dy.flatten()[ones_flat]
    x_flat = x.flatten()[ones_flat]
    I0_flat = I0.flatten()[ones_flat]
    ny2 = int(_np.median(n[mask]))
    mnmx = (ny2-1/upsample)/2
    x2, dy2 = _np.meshgrid(good,
                           _np.linspace(-mnmx, mnmx, ny2*upsample))
    #dx2, y2 = _np.meshgrid(_np.arange(-(nx2-1)/2, nx2-(nx2-1)/2),
    #                       good)
    ESF2 = _interp.griddata((x_flat, dy_flat), I0_flat, (x2, dy2), 'cubic',
                            fill_value=0)
    ones2 = _interp.griddata((x.flatten(), dy.flatten()), ones.flatten(),
                             (x2, dy2), 'cubic', fill_value=0)
    #rbf = _interp.Rbf(dx_flat, y_flat, I0_flat)
    #ESF2 = rbf(dx2, y2)
    #rbf = _interp.Rbf(dx.flatten(), y.flatten(), ones.flatten())
    #ones2 = rbf(dx2, y2)
    # project onto dx axis and take average
    nc = ones2.sum(1)
    ESF = (ESF2*ones2).sum(1) / nc
    # intermediate details
    edgeV_detail = dict(cy_raw=cy_raw, cy_fit=cy_fit, Iy=Iy, n=n, mask=mask,
                        x_ax=x_ax, dyx=dy, I0=I0*ones, ones=ones,
                        dy_flat=dy_flat, I0_flat=I0_flat, dy2=dy2, ESF2=ESF2,
                        ones2=ones2)
    return ESF, dy2[:, 0], edgeV_detail


def plot_edgeH_data(ESF, dy, edgeV_details):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(edgeV_details['I0'], cmap='gray')
    mask = edgeV_details['mask']
    cy_raw = edgeV_details['cy_raw'][mask]
    cy_fit = edgeV_details['cy_fit'][mask]
    x_ax = edgeV_details['x_ax'][mask]
    plt.plot(x_ax, cy_raw, 'r+')
    plt.plot(x_ax, cy_fit)
    plt.subplot(2, 1, 2)
    dy_flat = edgeV_details['dy_flat']
    I0_flat = edgeV_details['I0_flat']
    plt.plot(dy_flat, I0_flat, '.')
    plt.plot(dy, ESF)


def esf2mtf(ESF, dx=None):
    '''
    Parameters
    ----------
    ESF : 1d array
        Edge-Spread-Function of length n
    dx : 1d array, optional
        x coordinates.

    Returns
    -------
    LSF : 1d array
        Line-Spread-Function, of length n-1, normalized so integral is 1.
    MTF : 1d-array
        Modulation-Transfer-Function = abs(fft(LSF))[:(n-1)//2]
    '''
    pkpk = _np.max(ESF) - _np.min(ESF)
    #delta = dx[1] - dx[0]
    #delta = _np.median(_np.diff(dx))
    LSF = _np.diff(ESF/pkpk)
    n = len(LSF)
    MTF = _np.abs(_np.fft.fft(LSF))[:n//2]
    return LSF, MTF
