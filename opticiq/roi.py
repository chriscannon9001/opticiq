#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI has a lot of variability to bear in mind.
1. Number of ROI per image (1 to 1000s)
2. Need for ROI boundaries to evolve as analysis progresses from quick moments
    to fitting or other. Or not evolve.
3. Need to downselect POI/ROI at some point in an analysis. Or not.
4. Need to setup fixed ROI for production testing where expected
    result is fixed.
5. Must assume ROI are ragged (don't bother with a 3d array)

Created on Wed Mar 29 09:56:52 2023

@author: chris
"""


from abc import ABC as _ABC, abstractmethod as _abmeth
import numpy as _np


def oi_ABC(_ABC):
    @_abmeth
    def __init__(self):
        pass

    def __getitem__(self, idx):
        # this part might be pointless
        pass

    def items():
        # yield item(s)
        pass


def _get_subframe(I, peak, swROI, shROI):
    '''
    Parameters
    ----------
    I : 2d array
        image
    peak : int, int
        y, x coordinates of a peak
    swROI, shROI : int, int or float, float
        semi-width and semi-height of ROI

    Returns
    -------
    frame : 2d array
        subframe of image local to peak
    xframe, yframe : 2d arrays
        same frame slices applied to x, y
    dx, dy: int, int
        offset of frame corner w.r.t. I
    slice2, slice1 : slices
        the slices that were used to get the frame
    '''
    ny, nx = I.shape
    peak = _np.array(peak)
    y1 = max(0, peak[0] - shROI)
    y2 = min(ny - 1, peak[0] + shROI + 1)
    x1 = max(0, peak[1] - swROI)
    x2 = min(nx - 1, peak[1] + swROI + 1)
    slice1 = slice(x1, x2)
    slice2 = slice(y1, y2)
    frame = I[slice2, slice1] 
    return frame, x1, y1, slice2, slice1


def _frameboundaries_wlimits(Ishape, POI, swROI, shROI):
    '''
    Get frame boundaries (e.g. for slicing frames at POI), but enforce
    boundaries around Ishape.

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

    Returns
    -------
    j1 : array of int
        for each POI, min row of frame (or slice)
    j2 : array of int
        for each POI, max row +1 of frame (or slice)
    i1 : array of int
        for each POI, min column of frame (or slice)
    i2 : array of int
        for each POI, max column +1 of frame (or slice)
    '''
    ny, nx = Ishape
    j = POI[:, 0]
    i = POI[:, 1]
    j1 = _np.maximum(0, j - shROI)
    j2 = _np.minimum(ny - 1, j + shROI + 1)
    i1 = _np.maximum(0, i - swROI)
    i2 = _np.minimum(nx - 1, i + swROI + 1)
    return j1, j2, i1, i2


class ROI_ABC_old(_ABC):
    '''
    ROI_ABC is an ABC for all classes that provide a series of ROI
    (Region-of-Interest).

    Properties
    ----------
    frames : list of 2d arrays, or 3d array
        for each peak, a frame containing an ROI sliced from I
    xframes, yframes : list of 2d arrays, or 3d arrays
        for each peak, the x, y coordinates (column, row) of the pixels
        contained in the frame, in the global coordinates of I
        (local frame coordinates are xframes - dx, yframes - dy, but using
        local coordinates is discouraged because peaks and centroids should be
        in global coordinates)
    dx, dy : 1d array
        offset of the corner of each from, i.e. local frame coordinates
        relative to I.
    masks : list of 2d arrays of bool
        for each frame, a mask with the same shape as the frame with bool to
        include/exclude each pixel in the frame
    slices : [(slice_j0, slice_i0), (slice_j1, slice_i1), ...]
        The same slices that would get frames from the original image.
    '''
    @_abmeth
    def __init__(self):
        self.frames = None
        self.xframes = None
        self.yframes = None
        self.dx = None
        self.dy = None
        pass

    @property
    @_abmeth
    def masks(self):
        pass

    def _subframeROI(self, I, POI, semiROIw, semiROIh):
        '''
        A default method that may work for most ROI sub-classes.

        Parameters
        ----------
        I : 2d array
            image to slice into frames
        POI : array([[j0, i0], [j1, i1], ...])
            Points-of-Interest (same as peaks from
            skimage.feature.peak_local_max)
        semiROIw : int or array
            half-width of ROI frames, constant if uniform, or an array for each
            peak
        semiROIh : int or array
            half-height of ROI frames, constant if uniform, or an array for each
            peak

        Returns
        -------
        self.frames self.xframes, yframes dx, dy 
        '''
        ny, nx = I.shape
        x, y = _np.meshgrid(range(nx), range(ny))
        Npoi = len(POI)
        dx, dy = _np.zeros(Npoi, dtype=int), _np.zeros(Npoi, dtype=int)
        semiROIh = _np.ones(Npoi, dtype=int) * semiROIh
        semiROIw = _np.ones(Npoi, dtype=int) * semiROIw
        frames = [None] * Npoi
        xframes = [None] * Npoi
        yframes = [None] * Npoi
        slices = [None] * Npoi
        for i in range(Npoi):
            point = POI[i]
            frame_i, dx_i, dy_i, slice2, slice1 = _get_subframe(
                I, point, int(semiROIw[i]), int(semiROIh[i]))
            xframe_i = x[slice2, slice1]
            yframe_i = y[slice2, slice1]
            frames[i] = (frame_i)
            xframes[i] = (xframe_i)
            yframes[i] = (yframe_i)
            slices[i] = (slice2, slice1)
            dx[i] = dx_i
            dy[i] = dy_i
        self.frames = frames
        self.xframes = xframes
        self.yframes = yframes
        self.slices = slices
        self.dx = dx
        self.dy = dy


class ROI_ABC(_ABC):
    '''
    ROI_ABC is an ABC for all classes that provide a series of ROI
    (Region-of-Interest).

    Properties
    ----------
    j1, j2, i1, i2 : array of int
        For each POI, these define the slices, e.g.
        frame0 = I[j1[0]:j2[0], i1[0]:i2[0]]. Relatedly, note that
        slices[0] = (slice(j1[0], j2[0]), slice(i1[0], i2[0]))
    slices : [(jslice0, islice0), (jslice1, islice1), ...]
        The same slices that would get frames from the original image.
    frames : 3d array (array of 2d frames)
        for each point, a frame containing an ROI sliced from I
    xframes, yframes : 3d arrays
        for each peak, the x, y coordinates (column, row) of the pixels
        contained in the frame, in the global coordinates of I
        So, local coordinates frame i is (xframes[i] - i1[i], yframes - j1).
        Note that local coordinates are discouraged, e.g. centroids should be
        in global coordinates.
    masks : list of 2d arrays of bool
        for each frame, a mask with the same shape as the frame with bool to
        include/exclude each pixel in the frame
    '''
    @_abmeth
    def __init__(self):
        self.I = None
        self.j1 = None
        self.j2 = None
        self.i1 = None
        self.i2 = None
        pass

    @property
    def slices(self):
        if not hasattr(self, '_slices'):
            n = len(self.j1)
            slices = []
            for k in range(n):
                slices.append((slice(self.j1[k], self.j2[k]), 
                               slice(self.i1[k], self.i2[k])))
            self._slices = slices
        return self._slices

    @property
    def frames(self):
        pass

    @property
    @_abmeth
    def masks(self):
        pass

    def _set_frameboundaries(self, Ishape, POI, swROI, shROI):
        self.j1, self.j2, self.i1, self.i2 = _frameboundaries_wlimits(
            Ishape, POI, swROI, shROI)


class ROI_fixedellipse(ROI_ABC):
    def __init__(self, I, POI, xrad_ROI, yrad_ROI):
        '''
        Parameters
        ----------
        I : 2d array
            image
        POI : array([[j0, i0], [j1, i1], ...])
            Points-of-Interest (same as peaks from
            skimage.feature.peak_local_max)
        xrad_ROI : float
            xradius of ROI
        yrad_ROI : float
            yradius of ROI
        '''
        self._args = POI, xrad_ROI, yrad_ROI
        self._subframeROI(I, POI, xrad_ROI, yrad_ROI)

    @property
    def masks(self):
        if not hasattr(self, '_masks'):
            POI, xrad_ROI, yrad_ROI = self._args
            Npoi = len(POI)
            masks = [None] * Npoi
            for i in range(Npoi):
                urad2 = (((self.xframes[i] - POI[i][1])/xrad_ROI)**2 
                         + ((self.yframes[i] - POI[i][0])/yrad_ROI)**2)
                mask = urad2 <= 1
                masks[i] = mask
            self._masks = masks
        return self._masks


class ROI_xysigma2ellipse(ROI_fixedellipse):
    def __init__(self, I, POI, xsigma, ysigma, NsigmaROI=4,
                 mnmx_semiROIw=(2,1e5), mnmx_semiROIh=(2,1e5)):
        '''
        Parameters
        ----------
        I : 2d array
            image
        POI : array([[j0, i0], [j1, i1], ...])
            Points-of-Interest (same as peaks from
            skimage.feature.peak_local_max)
        xsigma : array, same length as PO
            xsigma values
        ysigma : array, same length as PO
            ysigma values
        NsigmaROI : float, optional
            N*xsigma and N*ysigma will be used to define ROI ellipse.
            The default is 4.
        mnmx_semiROIw : (float, float), optional
            Min and max semi-ROI-width. The default is (2,1e5).
        mnmx_semiROIh : (float, float), optional
            Min and max semi-ROI-width. The default is (2,1e5).
        '''
        semiROIw = xsigma * NsigmaROI
        semiROIh = ysigma * NsigmaROI
        mn_sROIw, mx_sROIw = mnmx_semiROIw
        mn_sROIh, mx_sROIh = mnmx_semiROIh
        # apply min, max on semi-ROI values - also filter nan to max
        if mx_sROIw is not None:
            semiROIw = _np.minimum(mx_sROIw, _np.nan_to_num(semiROIw, nan=mn_sROIw))
        if mn_sROIw is not None:
            semiROIw = _np.maximum(mn_sROIw, semiROIw)
        if mx_sROIh is not None:
            semiROIh = _np.minimum(mx_sROIh, _np.nan_to_num(semiROIh, nan=mn_sROIh))
        if mn_sROIh is not None:
            semiROIh = _np.maximum(mn_sROIh, semiROIh)
        self._args = POI, semiROIw, semiROIh
        self._subframeROI(I, POI, semiROIw, semiROIh)

    @property
    def masks(self):
        if not hasattr(self, '_masks'):
            POI, xrad_ROI, yrad_ROI = self._args
            Npoi = len(POI)
            masks = [None] * Npoi
            for i in range(Npoi):
                urad2 = (((self.xframes[i] - POI[i][1])/xrad_ROI[i])**2 
                         + ((self.yframes[i] - POI[i][0])/yrad_ROI[i])**2)
                mask = urad2 <= 1
                masks[i] = mask
            self._masks = masks
        return self._masks
