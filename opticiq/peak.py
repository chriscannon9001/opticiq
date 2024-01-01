#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:56:18 2023

@author: chris
"""

from matplotlib import pyplot as _plt
import numpy as _np
from scipy import optimize as _opt

from .math import _rotate_2d


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


class PeaksAnalysis():
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
    def __init__(self, I, ROI, weight=None, darklevel=0):
        '''
        Parameters
        ----------
        I : 2d array
            image
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
        self.Npoi = len(ROI.slices)
        self.ROI = ROI
        self.simplified = True
        light = [None] * self.Npoi
        for i in range(self.Npoi):
            # yeah, this is probably unnecessary
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

    def plot_singlepeak(self, k):
        # FIXME
        fig = _plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        e = self.energy
        cy, cx = self.centroid
        dx = self.ROI.dx
        dy = self.ROI.dy
        ax1.imshow(self._light[k], cmap='gray')
        ax1.plot(cx[k] - dx[k], cy[k] - dy[k], '.r')
        msg = 'E %0.2e at (%0.2f, %0.2f)' % (e[k], cx[k], cy[k])
        _plt.title(msg)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(self.ROI.masks[k], cmap='gray')
        _plt.title('Mask')
