#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 18:59:25 2023

@author: chris
"""

import numpy as _np
from scipy import ndimage as _ndimage
from skimage import morphology as _morph


def mask(f, threshold=0.1, f0=0):
    '''mask = (f - f0) / np.max(f) > threshold'''
    return (f - f0) / _np.max(f) > threshold


def maskedges(imgrad, threshold):
    imgrad['maskedges'] = mask(
        imgrad['I_r'], threshold)


def maskdark(imgrad):
    mid = _np.median(imgrad['I1'].flatten()[imgrad['maskedges'].flatten()])
    maskdark0 = imgrad['I1'] < mid
    imgrad['maskdark'] = _np.logical_and(
        maskdark0,
        _np.logical_not(imgrad['maskedges']))


def maskedges_hvx(imgrad, threshold):
    mask_v_abs = mask(_np.abs(imgrad['I_x']), threshold)
    mask_h_abs = mask(_np.abs(imgrad['I_y']), threshold)
    mask_v_pos0 = mask(imgrad['I_x'], threshold)
    mask_v_neg0 = mask(-imgrad['I_x'], threshold)
    mask_h_pos0 = mask(imgrad['I_y'], threshold)
    mask_h_neg0 = mask(-imgrad['I_y'], threshold)
    masks = [_np.logical_and(mask_v_pos0, _np.logical_not(mask_h_abs)),
             _np.logical_and(mask_v_neg0, _np.logical_not(mask_h_abs)),
             _np.logical_and(mask_h_pos0, _np.logical_not(mask_v_abs)),
             _np.logical_and(mask_h_neg0, _np.logical_not(mask_v_abs))]
    for i in range(4):
        # erosion of dilation can cleanup if there's noise in a mask
        masks[i] = _morph.erosion(_morph.dilation(masks[i]))
    imgrad['mask_v_pos'] = masks[0]
    imgrad['mask_v_neg'] = masks[1]
    imgrad['mask_h_pos'] = masks[2]
    imgrad['mask_h_neg'] = masks[3]


def maskridges_hvx(imgrad, threshold):
    mask_v0 = mask(-imgrad['I_xx'], threshold)
    mask_h0 = mask(-imgrad['I_yy'], threshold)
    imgrad['mask_v'] = _np.logical_and(mask_v0, _np.logical_not(mask_h0))
    imgrad['mask_h'] = _np.logical_and(mask_h0, _np.logical_not(mask_v0))


def imageGradients(I0, sigma, require=[], threshold=None):
    '''
    Preprocessing steps potentially applicable to every image quality test.

    Given a raw image, apply Gaussian blur, make derivatives, package results
    as a dict of 2d arrays. The keys of the dict follow a naming convention,
    see below.

    Parameters
    ----------
    I0 : 2d array
        Original image
    sigma : float
        Sigma (pixels) to blur image before derivatives
    require : list of str
        Each str in list should be a standard name of an array (above).
        Dependencies can be resolved automatically, so ['D_hessian'] will
        also call all the first and second order derivatives.
        The default is [], which will only return the blurred image.
    threshold : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    imgrad : dict of 2d arrays
        All results required or implied.

    Names of Arrays
    ---------------
    | x, y : global coordinates (column, row) of pixels
    | ones : 1s for each pixel, initially. Masks leave a footprint
    | I0 : original image
    | I1 : blurred image
    | I_x : dI1/dx
    | I_y : dI1/dy
    | I_r : sqrt(I_x**2 + I_y**2) aka magnitude of slope
    | I_xx : d2I1/dxdx
    | I_xy : d2I1/dxdy
    | I_yx : d2I1/dydx
    | I_yy : d2I1/dydy
    | curve : I_xx + I_yy aka curvature, which goes to 0 at saddle point
    | D_hessian : I_xx * I_yy - I_xy * I_yx
    | maskedges : I_r / max(I_r) < threshold
    | mask_h_abs : abs(I_y)/max(abs(I_y)) < threshold
    | mask_v_abs : ... similar to mask_h_abs
    | mask_v_pos : I_x / max(I_x) < threshold and not mask_h_abs 
    | mask_v_neg, mask_h_pos, mask_h_get : .., similar to mask_v_pos
    '''
    # some logic finds implied parameters
    do_dhesse = 'D_hessian' in require
    do_curve = 'curve' in require
    do_d2 = (do_dhesse or do_curve or 'I_xx' in require)
    do_Ir = 'I_r' in require
    do_d1 = (do_d2 or do_Ir or 'I_x' in require or 'I_y' in require)
    # xframe and yframe indicate global coordinates of pixels
    ny, nx = I0.shape
    x, y = _np.meshgrid(range(nx), range(ny))
    # initial dataset is xframe, yframe, and original image
    imG = dict(I0=I0, x=x, y=y, ones=_np.ones(I0.shape))
    # add blurred image
    imG['I1'] = _ndimage.gaussian_filter(I0, sigma)
    if do_d1:
        # first derivatives
        imG['I_y'], imG['I_x'] = _np.gradient(imG['I1'])
    if do_Ir:
        imG['I_r'] = _np.sqrt(imG['I_y']**2 + imG['I_x']**2)
    if do_d2:
        # second derivatives
        imG['I_xy'], imG['I_xx'] = _np.gradient(imG['I_x'])
        imG['I_yy'], imG['I_yx'] = _np.gradient(imG['I_y'])
    if do_curve:
        imG['curve'] = imG['I_xx'] + imG['I_yy'] #+ self.I_xy + self.I_yx
    if do_dhesse:
        # Hessian determinant
        imG['D_hessian'] = imG['I_xx'] * imG['I_yy'] - imG['I_xy'] * imG['I_yx']
    return imG
