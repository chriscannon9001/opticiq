#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI has a lot of variability to bear in mind.
1. Number of ROI per image (1 to 1000s)
2. Need for ROI boundaries to evolve as analysis progresses from quick moments
    to fitting or other. Or not evolve.
3. Need to downselect POI/ROI at some point in an analysis. Or not.
4. Need to setup fixed ROI for production testing where expected
    result is fixed and auto is not wanted.
5. Must assume ROI are ragged (3d array is bad, also slower)
6. Need to avoid interference between ROI_i mask and any other ROI, or not

Created on Wed Mar 29 09:56:52 2023

@author: chris
"""

import numpy as _np
from skimage.measure import label as _label


class Regions():
    '''
    RegionsOI is a class useful for slicing a frame (Region Of Interest) out of
    an imageGradient (dict of 2d arrays)

    Properties
    ----------
    j1, j2, i1, i2 : array of int
        For each POI, these define the slices, e.g.
        frame0 = I[j1[0]:j2[0], i1[0]:i2[0]]. Relatedly, note that
        slices[0] = (slice(j1[0], j2[0]), slice(i1[0], i2[0]))
    slices : [(jslice0, islice0), (jslice1, islice1), ...]
        The same slices that would get frames from the original image.
    '''
    def __init__(self, j1, j2, i1, i2):
        self.j1 = j1
        self.j2 = j2
        self.i1 = i1
        self.i2 = i2
        Npoi = len(j1)
        slices = [None] * Npoi
        for k in range(Npoi):
            jslice = slice(self.j1[k], self.j2[k])
            islice = slice(self.i1[k], self.i2[k])
            slices[k] = (jslice, islice)
        self.slices = slices

    @classmethod
    def from_POI_width(cls, Ishape, POI, swROI, shROI):
        '''
        Get frame boundaries (e.g. for slicing frames at POI) using semi-width
        and semi-height, but enforce boundaries around Ishape.
    
        Parameters
        ----------
        Ishape : (nj, ni)
            shape of image containing all POI, i.e. I.shape
        POI : array([[j0, i0], [j1, i1], ...])
            Points-of-Interest (same as peaks from
            skimage.feature.peak_local_max)
        swROI : array-like (often int type)
            semi-width of ROI
        shROI : array-like (often int type)
            semi-height of ROI
        '''
        ny, nx = Ishape
        j = POI[:, 0]
        i = POI[:, 1]
        j1 = _np.maximum(0, j - shROI)
        j2 = _np.minimum(ny - 1, j + shROI + 1)
        i1 = _np.maximum(0, i - swROI)
        i2 = _np.minimum(nx - 1, i + swROI + 1)
        return cls(j1, j2, i1, i2)

    @classmethod
    def from_mask(cls, imG, mask, min_area=1):
        '''
        Parameters
        ----------
        imgrad : dict of arrays
            requires imgrad['xframe'] and imgrad['yframe']
        mask : dict of array of bool
            mask to be factored into ROI
        min_area : int, optional
            Any ROI of area < min_area will be excluded. The default is 1.
        '''
        label, n = _label(mask, return_num=True)
        x = imG['x']
        y = imG['y']
        j1 = []
        j2 = []
        i1 = []
        i2 = []
        for k in range(1, n+1):
            mask_k = label==k
            area = _np.sum(mask_k)
            if area >= min_area:
                x_k = x[mask_k]
                y_k = y[mask_k]
                i1.append(_np.min(x_k))
                i2.append(_np.max(x_k))
                j1.append(_np.min(y_k))
                j2.append(_np.max(y_k))
        return cls(_np.array(j1), _np.array(j2), _np.array(i1), _np.array(i2))

    def frame_images(self, k, imgrad):
        '''
        Parameters
        ----------
        k : int
            which ROI
        imgrad : dict of 2d arrays
            Normally, the original image array and its gradients, possibly
            including relevant masks. All 2d arrays are the same dimension as
            the original image

        Returns
        -------
        imgrad_frame : dict of 2d arrays
            Each key, val is sliced from imgrad using self.slices[k]
        '''
        jslice, islice = self.slices[k]
        imgrad_frame = {}
        for key, val in imgrad.items():
            imgrad_frame[key] = val[jslice, islice]
        return imgrad_frame

    def downselect(self, isvalid):
        '''
        Parameters
        ----------
        isvalid : array of bool
            Determines whether each ROI is included in the downselect

        Returns
        -------
        downselect : ROI
            A new ROI object
        '''
        j1 = self.j1[isvalid]
        j2 = self.j2[isvalid]
        i1 = self.i1[isvalid]
        i2 = self.i2[isvalid]
        return Regions(j1, j2, i1, i2)
