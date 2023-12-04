#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 18:59:25 2023

@author: chris
"""

from matplotlib import pyplot as _plt
import numpy as _np
from scipy import ndimage as _ndimage
from skimage import morphology as _morph


def mask(f, threshold=0.1):
    '''
    Parameters
    ----------
    f : 2d array
        DESCRIPTION.
    threshold : float
        DESCRIPTION.

    Returns
    -------
    mask : 2d array (bool)
        Description
    '''
    mx = _np.max(f)
    f_norm = f / mx
    mask = f_norm > threshold
    return mask


class ImGrad():
    '''
    ImGrad is a low-level collection of gradient calcs commonly used in image
    post-processsing.

    One-time getters produce calcs and then are shorted by storage
    subsequently, and so redundant calls may be streamlined despite.

    Properties
    ----------
    sigma : 0 or float
        Gaussian sigma for blur which is applied before anything else. If 0, no
        blur is used.
    I1 : 2d array
        Image after Gaussian blur. All calculations use this as input.
    I_x, I_y : 2d array
        first derivatives of image with respect to x and y
    I_r : 2d array
        the magnitude of the linear gradient = sqrt(I_x^2 + I_y^2)
    I_xx, I_xy, I_yy, I_yx : 2d array
        second derivatives
    curve : 2d array
        curvature, or a mix of curvature terms (can be 0 for a saddle point)
        = I_xx + I_yy + I_xy
    D_hessian : 2d array
        determinant of Hessian matrix ( = I_xx * I_yy - I_xy * I_yx)
    '''
    def __init__(self, I0, sigma):
        '''
        Parameters
        ----------
        I0 : 2d array
            original image
        sigma : float
            Gaussian sigma for blur which is applied before anything else.
        '''
        I1 = (I0 if (_np.isscalar(sigma) and _np.isclose(sigma, 0))
              else _ndimage.gaussian_filter(I0, sigma))
        self.I0 = I0
        self.sigma = sigma
        self._min_d = int(sigma + .5)
        self.I1 = I1

    @property
    def I_x(self):
        if not hasattr(self, '_I_x'):
            # say, can images vary between row-major and column-major?
            self._I_y, self._I_x = _np.gradient(self.I1)
        return self._I_x

    @property
    def I_y(self):
        if not hasattr(self, '_I_y'):
            self._I_y, self._I_x = _np.gradient(self.I1)
        return self._I_y

    @property
    def I_r(self):
        if not hasattr(self, '_I_r'):
            self._I_r = _np.sqrt(self.I_x**2 + self.I_y**2)
        return self._I_r

    @property
    def I_xx(self):
        if not hasattr(self, '_I_xx'):
            self._I_xy, self._I_xx = _np.gradient(self.I_x)
        return self._I_xx

    @property
    def I_xy(self):
        if not hasattr(self, '_I_xy'):
            self._I_xy, self._I_xx = _np.gradient(self.I_x)
        return self._I_xy

    @property
    def I_yx(self):
        if not hasattr(self, '_I_yx'):
            self._I_yy, self._I_yx = _np.gradient(self.I_y)
        return self._I_yx

    @property
    def I_yy(self):
        if not hasattr(self, '_I_yy'):
            self._I_yy, self._I_yx = _np.gradient(self.I_y)
        return self._I_yy

    @property
    def curve(self):
        if not hasattr(self, '_curve'):
            #self._curve = _np.sqrt(self.I_xx**2 + self.I_yy**2 + self.I_xy * self.I_yx)
            self._curve = self.I_xx + self.I_yy #+ self.I_xy + self.I_yx
        return self._curve

    @property
    def D_hessian(self):
        if not hasattr(self, '_D_hessian'):
            self._D_hessian = self.I_xx * self.I_yy - self.I_xy * self.I_yx
        return self._D_hessian


class ImMask():
    '''
    MAYBE
    do I want to redo this as functional programming?

    maskedges_hvx : tuple(mask_v_pos, mask_v_neg, mask_h_pos, mask_h_neg)
        Exclusive hor, ver masks delineating edges. Note that edges have a
        sign, so positive and negative edges are separate. Exclusive means
        ignoring corners areas, and this helps clean downstream edge-function.

        mask_v_pos : 2d bool array
            ROI mask; ver edge (step function along x), positive-going edge
        mask_v_neg : 2d bool array
            ROI mask; ver edge (step function along x), negative-going edge
        mask_h_pos : 2d bool array
            ROI mask; hor edge (step function along y), positive-going edge
        mask_h_neg : 2d bool array
            ROI mask; hor edge (step function along y), negative-going edge
    maskridges_hvx : tuple(mask_v, mask_h)
        Exclusive hor, ver masks delineating ridges. Exclusive means ignoring
        crossings, and this helps clean downstream line-spread-function.

        mask_v : 2d bool array
            ROI mask; ver line (line-function along x)
        mask_h : 2d bool array
            ROI mask; hor line (line-function along y)
    '''
    def __init__(self, imgrad, threshold):
        '''
        imgrad : ImGrad
            blurred image with gradients
        threshold : float 0-1, optional
            Threshold used to get masks from derivatives.
        '''
        self.imgrad = imgrad
        self._threshold = threshold

    @property
    def maskedges(self):
        if not hasattr(self, '_maskedges'):
            self._maskedges = mask(
                self.imgrad.I_r, threshold=self._threshold)
        return self._maskedges

    @property
    def maskdark(self):
        if not hasattr(self, '_maskdark'):
            mid = _np.median(self.imgrad.I1.flatten()[self.maskedges.flatten()])
            maskdark0 = self.imgrad.I1 < mid
            self._maskdark = _np.logical_and(
                maskdark0,
                _np.logical_not(self.maskedges))
        return self._maskdark

    @property
    def dark(self):
        if not hasattr(self, '_dark'):
            dark = self.imgrad.I0.flatten()[self.maskdark.flatten()]
            avg = _np.mean(dark)
            med = _np.median(dark)
            rms = _np.std(dark)
            self._dark = avg, med, rms
        return self._dark

    @property
    def maskedges_hvx(self):
        if not hasattr(self, '_maskedges_hvx'):
            mask_v_abs = mask(_np.abs(self.imgrad.I_x), threshold=self._threshold)
            mask_h_abs = mask(_np.abs(self.imgrad.I_y), threshold=self._threshold)
            mask_v_pos0 = mask(self.imgrad.I_x, threshold=self._threshold)
            mask_v_neg0 = mask(-self.imgrad.I_x, threshold=self._threshold)
            mask_h_pos0 = mask(self.imgrad.I_y, threshold=self._threshold)
            mask_h_neg0 = mask(-self.imgrad.I_y, threshold=self._threshold)
            masks = [_np.logical_and(mask_v_pos0, _np.logical_not(mask_h_abs)),
                     _np.logical_and(mask_v_neg0, _np.logical_not(mask_h_abs)),
                     _np.logical_and(mask_h_pos0, _np.logical_not(mask_v_abs)),
                     _np.logical_and(mask_h_neg0, _np.logical_not(mask_v_abs))]
            for i in range(4):
                # dilation of erosion can cleanup if there's noise in a mask
                masks[i] = _morph.dilation(_morph.erosion(masks[i]))
            self._maskedges_hvx = tuple(masks)
        return self._maskedges_hvx

    @property
    def maskridges_hvx(self):
        if not hasattr(self, '_maskridges_hvx'):
            mask_v0 = mask(-self.imgrad.I_xx, self._threshold)
            mask_h0 = mask(-self.imgrad.I_yy, self._threshold)
            mask_v = _np.logical_and(mask_v0, _np.logical_not(mask_h0))
            mask_h = _np.logical_and(mask_h0, _np.logical_not(mask_v0))
            self._maskridges_hvx = mask_v, mask_h
        return self._maskridges_hvx

    def plot_dark(self):
        dark = self.imgrad.I0 * self.maskdark
        _plt.figure()
        _plt.imshow(dark, cmap='gray')
        title = 'Masked for dark level\n'
        avg, med, rms = self.dark
        if med > 1:
            title += '(avg %0.3f med %0.3f rms %0.3f)' % (avg, med, rms)
        else:
            title += '(avg %0.3e med %0.3e rms %0.3e)' % (avg, med, rms)
        _plt.title(title)
