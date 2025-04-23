#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 20:24:29 2025

@author: chris
"""

import numpy as _np
from skimage import morphology as _morph

from .roi import Regions


def ringRegions(roi, pad, *args, exclusion=None):
    '''
    For any Regions object, expand forming a ring around the orginal.

    Parameters
    ----------
    roi : Regions
        Original ROI
    pad : int
        Number of pixels spanning around the original.
    Ishape : (nj, ni)
        shape of image containing all POI, i.e. I.shape
    x, y : 2d array
        global pixel coordinates
    exclusion : 2d array of bool, optional
        If given, needs to match shape of I. Any area that matches this mask
        will be excluded from the rings. The default is None.

    Returns
    -------
    roi_BG : Regions
        New ROI's each forming a ring around the original.
    '''
    if len(args) == 1:
        if type(args[0]) is tuple:
            Ishape = args[0]
            #raise NotImplementedError()
        else:
            x, y = args[0]['x'], args[0]['y']
            Ishape = x.shape
    elif len(args) == 2:
        x, y = args
        Ishape = x.shape
    ny, nx = Ishape
    rslices = []
    rmasks = None if roi.rmasks is None else []
    for k in range(len(roi)):
        # need to expand each ROI to fit a ring
        jslice, islice = roi.rslices[k]
        j0, j1 = jslice.start, jslice.stop
        i0, i1 = islice.start, islice.stop
        # but need to limit expanded ROI so it stays in the frame
        j0_new, j1_new = max(0, j0 - pad), min(ny - 1, j1 + pad)
        i0_new, i1_new = max(0, i0 - pad), min(nx - 1, i1 + pad)
        rslice = (slice(j0_new, j1_new), slice(i0_new, i1_new))
        rslices.append(rslice)
        if roi.rmasks is not None:
            mask0 = roi.rmasks[k]
            mask1 = _np.pad(mask0, pad)
            mask2 = _morph.binary_dilation(mask1)
            for i in range(1, pad):
                mask2 = _morph.binary_dilation(mask2)
            mask3 = mask2 & _np.logical_not(mask1)
            # need to be sure the padding doesn't fall outside the original shape
            jslice_int = slice(j0_new + pad - j0, j1_new - j0 + pad)
            islice_int = slice(i0_new + pad - i0, i1_new - i0 + pad)
            #print(jslice_int, islice_int)
            # if an exclusion mask was passed, need to remove it from mask
            mask4 = mask3[jslice_int, islice_int]
            if exclusion is not None:
                print('scaffold!')
                exclusion_k = exclusion[rslice]
                mask4 = mask4 & _np.logical_not(exclusion_k)
            rmasks.append(mask4)
    return Regions(rslices, rmasks)


def background_fromroiBG(roi_BG, I0):
    bg = _np.zeros(len(roi_BG))
    for k in range(len(roi_BG)):
        I0_k = roi_BG.region_I(k, I0).flatten()
        mask = roi_BG.rmasks[k].flatten()
        bg[k] = _np.median(I0_k[mask])
    return bg
