#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:56:18 2023

@author: chris
"""


from matplotlib import pyplot as _plt
import numpy as _np
from skimage.feature import peak_local_max as _peak
from scipy import optimize as _opt

from .roi import ROI_xysigma2ellipse
from .grad import ImGrad
from .math import _rotate_2d


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


def eval_at_POI(a, POI):
    '''
    Parameters
    ----------
    a : 2d array
        Possibly image used in the peak finder, or any other 2d array to eval
        at peaks.
    POI : array([[j0, i0], [j1, i1], ...])
        Points-of-Interest (same as peaks from
        skimage.feature.peak_local_max)

    Returns
    -------
    ev : array
        evaluation of image at each point
    '''
    # flatten the peak indices
    i = POI[:, 0]
    j = POI[:, 1]
    idx = _np.ravel_multi_index((i, j), a.shape)
    ev = a.flatten()[idx]
    return ev


def peaks2sigma0(imgrad, peaks):
    '''
    Parameters
    ----------
    imgrad : ImGrad
        Image and gradient data.
    peaks : array([[j0, i0], [j1, i1], ...])
        array of peaks (same format as skimage.feature.peak_local_max)

    Returns
    -------
    xsigma0, ysigma0 : 1d arrays of float
        Generally sigma is a second moment with respect to each peak. Sigma0
        is a fast approximation that doesn't require any ROI.
        e.g. xsigma0(peak)=sqrt(-2*I(peak)/I_xx(peak)). If the profile is
        Gaussian form, this formula yields the second moment which is also the
        same sigma in I=exp(-2*(r/sigma)^2).
    '''
    xcurves = eval_at_POI(imgrad.I_xx, peaks)
    ycurves = eval_at_POI(imgrad.I_yy, peaks)
    amplitudes = eval_at_POI(imgrad.I1, peaks)
    xsigma = 2*_np.sqrt(-amplitudes / xcurves)
    ysigma = 2*_np.sqrt(-amplitudes / ycurves)
    # lastly, assume we need to make a correction using rss:
    s = imgrad.sigma
    xsigma_blur = s if _np.isscalar(s) else s[0]
    ysigma_blur = s if _np.isscalar(s) else s[1]
    xsigma = _np.sqrt(xsigma**2 - xsigma_blur**2)
    ysigma = _np.sqrt(ysigma**2 - ysigma_blur**2)
    return xsigma, ysigma


def find_major_sigma(x, y, Inorm):
    '''
    Uses second moments and rotation matrices to find the major axis of an
    beam or peak image.

    Parameters
    ----------
    x, y : 2d arrays
        x, y with respect to the centroid
    Inorm : 2d array
        normalized image

    Returns
    -------
    maj_sigma, min_sigma : float
        sigma (second-moment) along minor, major axes
    theta : float
        rotation angle (degrees) from y axis to the major axis. I.e.
        if you rotate x,y,I by -theta then you'll see the major axis along
        y, and minor axis along x.
    '''
    #print('xshape %s yshape %s' % (x.shape, y.shape))
    def _rotated_xysigma(theta):
        x2, y2 = _rotate_2d(x, y, theta)
        xsigma = _np.sqrt(_np.sum(x2**2 * Inorm))
        ysigma = _np.sqrt(_np.sum(y2**2 * Inorm))
        return xsigma, ysigma
    def _rotation_merit(x):
        theta = x[0]
        #print('rotating', theta)
        xsigma, ysigma = _rotated_xysigma(theta)
        #print('xsigma: %0.1f' % xsigma)
        return xsigma
    res =_opt.minimize(_rotation_merit, [0], bounds=[(-180, 180)])
    #assert res.success, ''
    theta = res.x[0]
    min_sigma, maj_sigma = _rotated_xysigma(theta)
    return maj_sigma, min_sigma, -theta


class Peaks_basic():
    '''
    Properties
    ----------
    energy : array
        volume (energy) under each peak.
    centroid : arrays ([cy], [cx])
        centroid coordinates of each peak. cy means fractional row,
        cx means fractional column
    xy_sigma : arrays ([xsigma], [ysigma])
        Sigma is a second moment. xsigma, ysigma are oriented to x, y axes.
    major_sigma : arrays ([maj_sigma], [min_sigma], [theta])
        Sigma is a second moment. maj_sigma, min_sigma are major, minor sigmas
        which are oriented to theta
    d4sigma : d4s_x, d4s_y, d4s_maj, d4s_min, theta
        Actually it's just the same as sigma, but 4x, i.e. 4*(second-moment).
        And that makes it a well used measurement of diameter.
    '''
    def __init__(self, I, POI, ROI, weight=None, darklevel=0):
        '''
        Parameters
        ----------
        I : 2d array
            image
        POI : array([[j0, i0], [j1, i1], ...])
            Points-of-Interest (same as peaks from
            skimage.feature.peak_local_max)
        ROI : child of ROI_ABC
            defines Region of Interest for each peak
        weight : array of weights, optional
            The default is None, which triggers weight =
            eval_at_peak(I, POI). This can be overrided if it ought
            to mean something else.
        darklevel : float
            background dark level
        '''
        self.I = I
        self.POI = POI
        self.Npoi = len(POI)
        self.weight = (eval_at_POI(I, POI)
                       if weight is None else weight)
        self.darklevel = darklevel * _np.ones(len(POI))
        self.ROI = ROI
        self.simplified = True
        light = [None] * self.Npoi
        for i in range(self.Npoi):
            mask = self.ROI.masks[i]
            frame = self.ROI.frames[i]
            dark = self.darklevel[i]
            light[i] = (frame - dark) * mask
        self._light = light

    @property
    def energy(self):
        if not hasattr(self, '_energy'):
            energy = _np.zeros(self.Npoi)
            for i in range(self.Npoi):
                energy[i] = _np.sum(self._light[i])
            self._energy = energy
        return self._energy

    @property
    def centroid(self):
        if not hasattr(self, '_centroid'):
            norm = self.energy
            cy, cx = _np.zeros(self.Npoi), _np.zeros(self.Npoi)
            for i in range(self.Npoi):
                xframe = self.ROI.xframes[i]
                yframe = self.ROI.yframes[i]
                cy[i] = _np.sum(yframe * self._light[i]) / norm[i]
                cx[i] = _np.sum(xframe * self._light[i]) / norm[i]
            self._centroid = cy, cx
        return self._centroid

    @property
    def xy_sigma(self):
        if not hasattr(self, '_xy_sigma'):
            cy, cx = self.centroid
            xframes = self.ROI.xframes
            yframes = self.ROI.yframes
            norm = self.energy
            xsigma = _np.zeros(self.Npoi)
            ysigma = _np.zeros(self.Npoi)
            for i in range(self.Npoi):
                xsigma[i] = _np.sqrt(_np.sum(
                    (xframes[i] - cx[i])**2 * self._light[i] / norm[i]))
                ysigma[i] = _np.sqrt(_np.sum(
                    (yframes[i] - cy[i])**2 * self._light[i] / norm[i]))
            self._xy_sigma = xsigma, ysigma
        return self._xy_sigma

    @property
    def major_sigma(self, i):
        if not hasattr(self, '_major_sigma'):
            cy, cx = self.centroid
            xframes = self.ROI.xframes
            yframes = self.ROI.yframes
            norm = self.energy
            maj_sigma = _np.zeros(self.Npoi)
            min_sigma = _np.zeros(self.Npoi)
            thetas = _np.zeros(self.Npoi)
            for i in range(self.Npoi):
                x = xframes[i] - cx[i]
                y = yframes[i] - cy[i]
                maj_sigma, min_sigma, theta = find_major_sigma(
                    x, y, self._light[i]/norm[i])
                maj_sigma[i] = maj_sigma
                min_sigma[i] = min_sigma
                thetas[i] = theta
            self._major_sigma = maj_sigma, min_sigma, thetas
        return self._major_sigma

    def encircled(self, level):
        raise NotImplementedError

    def iteratePeaks_xysigma2ROI(self):
        '''
        NOT IMPLEMENTED

        Use self.xy_sigma to get new ROI and return an iteration of self.
        '''
        raise NotImplementedError

    def plot_singlepeak(self, i):
        fig = _plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        e = self.energy
        cy, cx = self.centroid
        dx = self.ROI.dx
        dy = self.ROI.dy
        ax1.imshow(self._light[i], cmap='gray')
        ax1.plot(cx[i] - dx[i], cy[i] - dy[i], '.r')
        msg = 'E %0.2e at (%0.2f, %0.2f)' % (e[i], cx[i], cy[i])
        _plt.title(msg)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(self.ROI.masks[i], cmap='gray')
        _plt.title('Mask')


class Peaks_findMaxima(Peaks_basic):
    def __init__(self, immask, masks_pos=(), masks_neg=(),
                 peak_kwargs={'threshold_rel':.05}, NsigmaROI=2, ROI=None):
        imgrad = immask.imgrad
        mask = _np.ones(imgrad.I0.shape, dtype=bool)
        for pmask in masks_pos:
            mask = _np.logical_and(mask, pmask)
        for nmask in masks_neg:
            mask = _np.logical_and(mask, _np.logical_not(nmask))
        peaks = _peak(imgrad.I1*mask, **peak_kwargs)
        xsigma, ysigma = peaks2sigma0(imgrad, peaks)
        if ROI is None:
            # then we get a default ROI
            ROI = ROI_xysigma2ellipse(imgrad.I0, peaks, xsigma, ysigma,
                                      NsigmaROI=NsigmaROI)
        dark_avg, _, _ = immask.dark
        super().__init__(imgrad.I0, peaks, ROI, darklevel=dark_avg)

    def plot_all(self):
        fig = _plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(self.I, cmap='gray')
        _plt.title('Maxima - Original')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(self.imgrad.I1, cmap='gray')
        _plt.title('w Gaussian Blur')
        cy, cx = self.centroid
        for ax in [ax1, ax2]:
            for i in range(len(self.POI)):
                y, x = tuple(self.POI[i])
                #wt = weights[i]
                ax.plot(cx[i], y[i], '.r')


class Peaks_findStars(Peaks_basic):
    def __init__(self, immask, masks_pos=(), masks_neg=(),
                 peak_kwargs={'threshold_rel':.05}, NsigmaROI=2, ROI=None):
        '''
        Parameters
        ----------
        immask : ImMask
            image, gradient, and mask object
        masks_pos : tuple of 2d array of bool, same dim as image
            Positive sense masks, selecting a section of interest.
            The default is ().
        masks_neg : tuple of 2d array of bool, same dim as image
            Negative sense masks, deselecting a section. The default is ().
        peak_kwargs : dict, optional
            Arguments to control the finder (skimage.feature.peak_local_max).
            The default is {'threshold_rel':.05}.
        NsigmaROI : TYPE, optional
            DESCRIPTION. The default is 2.
        ROI : TYPE, optional
            DESCRIPTION. The default is None.
        '''
        imgrad = immask.imgrad
        mask = _np.ones(imgrad.I0.shape, dtype=bool)
        for pmask in masks_pos:
            mask = _np.logical_and(mask, pmask)
        for nmask in masks_neg:
            mask = _np.logical_and(mask, _np.logical_not(nmask))
        stariness = -imgrad.curve * unit_norm(-imgrad.I_r)
        self._stariness = stariness
        POI = _peak(stariness*mask, **peak_kwargs)
        weights = eval_at_POI(stariness, POI)
        xsigma, ysigma = peaks2sigma0(imgrad, POI)
        if ROI is None:
            # then we get a default ROI
            ROI = ROI_xysigma2ellipse(imgrad.I0, POI, xsigma, ysigma,
                                      NsigmaROI=NsigmaROI)
        dark_avg, _, _ = immask.dark
        super().__init__(imgrad.I0, POI, ROI, weight=weights,
                         darklevel=dark_avg)

    def plot_all(self):
        fig = _plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(self.I, cmap='gray')
        _plt.title('Stars - Original')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(self._stariness, cmap='gray')
        _plt.title('stariness (neg curvature & 0 slope)')
        cy, cx = self.centroid
        for ax in [ax1, ax2]:
            for i in range(len(self.POI)):
                y, x = tuple(self.POI[i])
                #wt = weights[i]
                ax.plot(cx[i], cy[i], '.r')


class Peaks_findCheckerPoints(Peaks_basic):
    def __init__(self, imgrad, peak_kwargs={'threshold_rel':.1},
                 NsigmaROI=1.5, ROI=None):
        self._I0 = imgrad.I0
        D = -imgrad.D_hessian
        POI = _peak(D, **peak_kwargs)
        img_D = ImGrad(D, 0)
        xsigma, ysigma = peaks2sigma0(img_D, POI)
        D_filter = _np.maximum(D, 0)
        if ROI is None:
            # then we get a default ROI
            ROI = ROI_xysigma2ellipse(D_filter, POI, xsigma, ysigma,
                                      NsigmaROI=NsigmaROI)
        super().__init__(D_filter, POI, ROI)

    def plot_singlepeak(self, i):
        fig = _plt.figure()
        cy, cx = self.centroid
        dx = self.ROI.dx
        dy = self.ROI.dy
        slice2, slice1 = self.ROI.slices[i]
        I_frame = self._I0[slice2, slice1]
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(I_frame, cmap='gray')
        ax1.plot(cx[i] - dx[i], cy[i] - dy[i], '.r')
        msg = 'Saddle at (%0.2f, %0.2f)' % (cx[i], cy[i])
        _plt.title(msg)
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(self.ROI.masks[i], cmap='gray')
        _plt.title('Mask')
        ax1 = fig.add_subplot(1, 3, 3)
        ax1.imshow(self._light[i], cmap='gray')
        ax1.plot(cx[i] - dx[i], cy[i] - dy[i], '.r')
        _plt.title('-D_hessian')

    def plot_all(self):
        fig = _plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(self._I0, cmap='gray')
        _plt.title('Checkers - Original')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(self.I, cmap='gray')
        _plt.title('saddliness (-D_hessian)')
        cy, cx = self.centroid
        for ax in [ax1, ax2]:
            for i in range(len(self.POI)):
                y, x = tuple(self.POI[i])
                #wt = weights[i]
                ax.plot(cx[i], cy[i], '.r')
