#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 21:39:35 2023

@author: chris
"""

import numpy as _np
from skimage.feature import peak_local_max as _peak

from .edge import edgeH_Analysis as _edgeH_Analysis
from .edge import edgeV_Analysis as _edgeV_Analysis
from .edge import esf2mtf as _esf2mtf
from .grad import imageGradients as _imageGrad
from .grad import maskedges_hvx as _mask_ehvx
from .peak import peaksAnalysis as _peaksAnalysis
from .roi import Regions as _Regions


def unit_norm(a, ref=None):
    '''
    Normalizes a to 0-1.

    unit_norm = (a - max(a, ref)) / (max(a, ref) - min(a, ref))
    '''
    if ref is None:
        mx = _np.max(a)
        mn = _np.min(a)
    else:
        mx = max(_np.max(a), _np.max(ref))
        mn = min(_np.min(a), _np.min(ref))
    return (a - mn) / (mx - mn)


def recipe_checkerboard1(I0, sigma, threshold=0.2):
    '''
    UNFINISHED

    Use the D_hessian to set ROI around saddle points.

    Parameters
    ----------
    I0 : 2d array
        original image
    sigma : float
        sigma for Gaussian blur (pixels)
    threshold : float, optional
        threshold used for mask. The default is 0.2.

    Returns
    -------
    imG : dict of 2d arrays
        See imageGradients, also will have 'mask'
    roi : Regions object
        Regions of Interest
    peaks : dict of 1d arrays
    '''
    imG = _imageGrad(I0, sigma, ['D_hessian'])
    D_h = imG['D_hessian']
    mask = -D_h / _np.max(-D_h) > threshold
    imG['mask'] = mask
    imG['f_saddle'] = -D_h
    #poi = _peak(-D_h, min_distance=sigma)
    #roi = _Regions.from_poi(poi, sigma*2, sigma*2, imG)
    roi = _Regions.from_mask(mask, imG)
    peaks = _peaksAnalysis(roi, imG, 'f_saddle')
    return imG, roi, peaks


def recipe_checkerboard2(I0, sigma, threshold=0.1):
    '''
    UNFINISHED

    Use D_hessian combined with I_r to find saddle points. This is more
    advanced than D_hessian alone, it rejects corner features where the slope
    magnitude is not 0.

    Parameters
    ----------
    I0 : 2d array
        original image
    sigma : float
        sigma for Gaussian blur (pixels)
    threshold : float, optional
        threshold of mask. The default is 0.1.

    Returns
    -------
    imG : dict of 2d arrays
        See imageGradients, also will have 'mask'
    roi : Regions object
        Regions of Interest
    peaks : dict of 1d arrays
    '''
    imG = _imageGrad(I0, sigma, ['D_hessian', 'I_r'])
    # upside-down, 0-1 norm, I_r (slope magnitude) makes a filter function
    # that will reject corners that aren't actual saddle-points
    f_minslope = unit_norm(-imG['I_r'])
    D_h = imG['D_hessian']
    # f_saddle is max where D_hessian is min and where I_r is min
    f_saddle = _np.maximum(0, -D_h * f_minslope)
    mask = f_saddle / _np.max(f_saddle) > threshold
    imG['mask'] = mask
    imG['f_saddle'] = f_saddle
    poi = _peak(f_saddle * mask, min_distance=sigma)
    roi = _Regions.from_poi(poi, sigma*2, sigma*2, imG)
    peaks = _peaksAnalysis(roi, imG, 'f_saddle')
    return imG, roi, peaks


def recipe_slantedge(I0, sigma, threshold=0.1, min_area=20):
    '''
    UNFINISHED

    Slantedge analysis of PSF and MTF.

    Parameters
    ----------
    I0 : 2d array
        original image
    sigma : float
        sigma for Gaussian blur (pixels)
    threshold : float, optional
        threshold of mask. The default is 0.1.

    Returns
    -------
    imG : TYPE
        DESCRIPTION.
    roi_v_pos : Regions
        DESCRIPTION.
    roi_v_neg : Regions
        DESCRIPTION.
    roi_h_pos : Regions
        DESCRIPTION.
    roi_h_neg : Regions
        DESCRIPTION.
    '''
    imG = _imageGrad(I0, sigma, ['I_x', 'I_y'])
    _mask_ehvx(imG, threshold)
    roi_vp = _Regions.from_mask(imG['mask_v_pos'], imG, min_area=min_area)
    roi_vn = _Regions.from_mask(imG['mask_v_neg'], imG, min_area=min_area)
    roi_hp = _Regions.from_mask(imG['mask_h_pos'], imG, min_area=min_area)
    roi_hn = _Regions.from_mask(imG['mask_h_neg'], imG, min_area=min_area)
    rval = dict(imG=imG, roi_vp=roi_vp, roi_vn=roi_vn, roi_hp=roi_hp,
                roi_hn=roi_hn)
    esf_vp = []
    rval['esf_vp'] = esf_vp
    for k in range(len(roi_vp.rslices)):
        imG_k = roi_vp.region_imG(k, imG)
        ESF, dx, details = _edgeV_Analysis(imG_k)
        LSF, MTF = _esf2mtf(ESF, dx)
        esf_vp.append((k, ESF, dx, LSF, MTF, details))
    esf_vn = []
    rval['esf_vn'] = esf_vn
    for k in range(len(roi_vn.rslices)):
        imG_k = roi_vn.region_imG(k, imG)
        ESF, dx, details = _edgeV_Analysis(imG_k)
        LSF, MTF = _esf2mtf(ESF, dx)
        esf_vn.append((k, ESF, dx, LSF, MTF, details))
    esf_hp = []
    rval['esf_hp'] = esf_hp
    for k in range(len(roi_hp.rslices)):
        imG_k = roi_hp.region_imG(k, imG)
        ESF, dy, details = _edgeH_Analysis(imG_k)
        LSF, MTF = _esf2mtf(ESF, dy)
        esf_hp.append((k, ESF, dy, LSF, MTF, details))
    esf_hn =[]
    rval['esf_hn'] = esf_hn
    for k in range(len(roi_hn.rslices)):
        imG_k = roi_hn.region_imG(k, imG)
        ESF, dy, details = _edgeH_Analysis(imG_k)
        LSF, MTF = _esf2mtf(ESF, dy)
        esf_hn.append((k, ESF, dy, LSF, MTF, details))
    return rval


'''def recipe_starfield(I, sigma, threshold=0.05):
    grad = _imageGrad(I, sigma, ['curve'])'''
