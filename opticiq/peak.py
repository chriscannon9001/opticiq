#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:56:18 2023

@author: chris
"""

from matplotlib import pyplot as _plt
import numpy as _np
from scipy import optimize as _opt
import warnings

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


def peaksAnalysis(roi, imG, key, require=['cx', 'cy'], bg=None):
    '''
    For each region in roi, analyze properties of a peak in that region.

    Parameters
    ----------
    roi : Regions
        Regions of interest
    imG : dict of 2d arrays
        Normally, the original image array I0 and its gradients. Needs to have
        'x', 'y', and key at least.
    key : str
        The image in imG to be analyzed
    require : list of str, optional
        Specify min set of analyses that will run. The default is ['cx', 'cy'].
        Dependencies left unspecified will be auto resolved. For example,
        'xsigma' alone will imply 'cx'.

    Returns
    -------
    peaks : dict of arrays
        Table of results.

    Names of Arrays
    ---------------
    | E : integral
    | cx, cy : fractional centroids in global coordinates
    | xsigma, ysigma : sigma is a second moment, oriented to x, y axes.
    '''
    peaks = {}
    do_xysigma = 'xsigma' in require or 'ysigma' in require
    do_majsigma = 'maj_sigma' in require or 'min_sigma' in require
    do_cent = do_xysigma or do_majsigma or ('cx' in require or 'cy' in require)
    do_E = do_cent or 'E' in require
    if do_E:
        E = _np.zeros(len(roi))
    if do_cent:
        cx, cy = _np.zeros(len(roi)), _np.zeros(len(roi))
    if do_xysigma:
        xsigma, ysigma = _np.zeros(len(roi)), _np.zeros(len(roi))
    if do_majsigma:
        maj_sigma = _np.zeros(len(roi))
        min_sigma = _np.zeros(len(roi))
        theta = _np.zeros(len(roi))
    for k in range(len(roi)):
        imG_k = roi.region_imG(k, imG, keys=['x', 'y', key])
        xk = imG_k['x']
        yk = imG_k['y']
        I = imG_k[key] if bg is None else _np.maximum(0, imG_k[key] - bg[k])
        if do_E:
            E[k] = _np.sum(I)
        if do_cent:
            cy[k] = _np.sum(yk * I) / E[k]
            cx[k] = _np.sum(xk * I) / E[k]
        if do_xysigma:
            xsigma[k] = _np.sqrt(_np.sum(
                (xk - cx[k])**2 * I) / E[k])
            ysigma[k] = _np.sqrt(_np.sum(
                (yk - cy[k])**2 * I) / E[k])
        if do_majsigma:
            maj_sigma[k], min_sigma[k], theta[k] = find_major_sigma(
                xk - cx[k], yk - cy[k], I/E[k])
    if do_E:
        peaks['E'] = E
    if do_cent:
        peaks['cx'] = cx
        peaks['cy'] = cy
    if do_xysigma:
        peaks['xsigma'] = xsigma
        peaks['ysigma'] = ysigma
    if do_majsigma:
        peaks['maj_sigma'] = maj_sigma
        peaks['min_sigma'] = min_sigma
        peaks['theta'] = theta
    return peaks


def peaksPlotter(roi, I0, peaks, tag='k+', cmap='jet', header=None):
    _, ax = _plt.subplots(1, 1, layout='tight')
    ax.imshow(I0, cmap=cmap)
    if header is None:
        header = list(peaks.keys())
    Npeaks = len(peaks[header[0]])
    #print('Npeaks', Npeaks)
    for i in range(Npeaks):
        cx, cy = peaks['cx'][i], peaks['cy'][i]
        key = header[0]
        text = '%s %0.3e' % (key, peaks[key][i])
        for key in header[1:]:
            text += '\n%s %0.3e' % (key, peaks[key][i])
        #print(text)
        ax.plot(cx, cy, tag)
        ax.text(cx+10, cy, text,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=.6))
    return ax


def peaksDownselect(peaks, isvalid):
    '''
    Parameters
    ----------
    peaks : dict of uniform arrays
        Typical form of peaks is e.g. {'E':[E0, E1, ...], 'cx':..}
        Where all arrays are the same length.
    isvalid : array of bool
        Length needs to match the length of arrays in peaks.

    Returns
    -------
    rpeaks : dict of uniform (shorter) arrays
        Downselect all the peaks that aren't valid.
    '''
    rpeaks = {}
    for key, val in peaks.items():
        rpeaks[key] = val[isvalid]
    return rpeaks


@warnings.deprecated("Use peaksAnalysis function instead of PeaksAnalysis class")
class PeaksAnalysis():
    '''
    DEPRECIATED

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
    def __init__(self, roi, imG, key):
        '''
        Parameters
        ----------
        roi : Regions object
            defines Region of Interest for each peak
        imG : dict of 2d arrays
            must contain key and 'x', 'y' at a minimum
        '''
        self.imG = imG
        self.key = key
        self.N = len(roi)
        self.roi = roi

    @property
    def energy(self):
        if not hasattr(self, '_energy'):
            energy = _np.zeros(self.N)
            for k in range(self.N):
                
                energy[k] = _np.sum(self.roi.region_I(k, self.I))
            self._energy = energy
        return self._energy

    @property
    def centroid(self):
        if not hasattr(self, '_centroid'):
            norm = self.energy
            cy, cx = _np.zeros(self.N), _np.zeros(self.N)
            for k in range(self.N):
                # FIXME
                xframe = self.roi.xframes[k]
                yframe = self.roi.yframes[k]
                cy[k] = _np.sum(yframe * self._light[k]) / norm[k]
                cx[k] = _np.sum(xframe * self._light[k]) / norm[k]
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
