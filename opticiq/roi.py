#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regions has a lot of variability to bear in mind.
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

from abc import ABC as _ABC, abstractmethod as _abmeth
from matplotlib import pyplot as _plt
import numpy as _np
from skimage.measure import label as _label


def get_rslices(j1, j2, i1, i2):
    '''
    Parameters
    ----------
    j1, j2, i1, i2 : list or array of int
        Indices that define slices

    Returns
    -------
    rslices :  list of (jslice, islice)
        This is the format that rslices needs to follow.
        [(slice(j1[0]:j2[0]), slice(i1[0]:i2[0])),
         (slice(j1[0]:j2[1]), slice(i1[0]:i2[1]))...]
    '''
    Npoi = len(j1)
    rslices = [None] * Npoi
    for k in range(Npoi):
        jslice = slice(j1[k], j2[k])
        islice = slice(i1[k], i2[k])
        rslices[k] = (jslice, islice)
    return rslices


class quasiList_ABC(_ABC):
    '''
    Inheritable ABC with the purpose of making callables appear and behave
    like a list. Requires a child class to define ._select(k) and
    ._downselect(bool0, bool1, etc)

    The purpose of this is to delay execution, esp. for multi-threading when
    the number of ROI exceeds the number of cores.
    '''
    def __getitem__(self, indices):
        if _np.isscalar(indices):
            return self._select(indices)
        elif type(indices) is tuple:
            raise ValueError()
        else:
            return self._downselect(indices)

    def aslist(self):
        '''
        Executes _select(k) for each k, returning an actual list. This may
        save time at the expense of memory, if a given [k] will be retrieved
        multiple times.
        '''
        return [self._select(k) for k in range(len(self))]

    def __iter__(self):
        for k in range(len(self)):
            yield self._select(k)

    @_abmeth
    def __len__(self):
        pass

    @_abmeth
    def _select(self, k):
        pass

    @_abmeth
    def _downselect(self, idx):
        pass


class rmasks_fromlabels(quasiList_ABC):
    def __init__(self, rslices, labels, idx):
        '''
        Creates a list-like (quasiList_ABC) set of masks, where each
        mask corresponds to exactly one label, even if the labels are close
        together.

        Parameters
        ----------
        rslices : list of (jslice, islice)
            See get_rslices() for formatting.
        labels : 2d array of int
            See return value of skimage.measure.label
        idx : int, list, or array
            
        Returns
        -------
        rmasks : list-like
            Appears to be a list of 2d masks, but actually makes internal
            calls during member access. See quasiList_ABC
        '''
        self._idx = list(range(1, idx+1)) if type(idx) is int else idx
        assert len(self._idx) == len(rslices), 'idx is inconsistent with len(rslices)'
        self._labels = labels
        self._rslices = rslices

    def __len__(self):
        return len(self._idx)

    def _select(self, k):
        label = self._idx[k]
        jslice, islice = self._rslices[k]
        # note that the slice executes before the equality
        mask = self._labels[jslice, islice] == label
        return mask

    def _downselect(self, isvalid):
        idx = self._idx[isvalid]
        rslices = self._rslices[isvalid]
        return rmasks_fromlabels(rslices, self._labels, idx)


class rmasks_xyellipse(quasiList_ABC):
    def __init__(self, rslices, poi, xrad, yrad, x, y):
        '''
        Creates a list-like (quasiList_ABC) set of masks, where each
        mask is defined by elliptical radii xrad, yrad.

        Parameters
        ----------
        rslices : list of (jslice, islice)
            See get_rslices() for formatting.
        poi : array([[j0, i0], [j1, i1], ...])
            Points-of-Interest (same as peaks from
            skimage.feature.peak_local_max)
        xrad : array-like (often int type)
            semi-width of ROI
        yrad : array-like (often int type)
            semi-height of ROI
        x, y : 2d array
            global pixel coordinates

        Returns
        -------
        rmasks : list-like
            Appears to be a list of 2d masks, but actually makes internal
            calls during member access. See quasiList_ABC
        '''
        # also need x, y as inputs, though almost silly
        self._rslices = rslices
        self._poi = poi
        n = len(poi)
        self._xrad = xrad * _np.ones(n)
        self._yrad = yrad * _np.ones(n)
        self._x = x
        self._y = y

    def __len__(self):
        return len(self._poi)

    def _select(self, k):
        point = self._poi[k]
        cy, cx = tuple(point)
        jslice, islice = self._rslices[k]
        x = self._x[jslice, islice]
        y = self._y[jslice, islice]
        mask = (((x - cx)/self._xrad[k])**2
                + ((y - cy)/self._yrad[k])**2) <= 1
        return mask

    def _downselect(self, isvalid):
        poi = self._poi[isvalid]
        xrad = self._xrad[isvalid]
        yrad = self._yrad[isvalid]
        rslices = self._rslices[isvalid]
        return rmasks_xyellipse(
            rslices, poi, xrad, yrad, self._x, self._y)


class Regions():
    def __init__(self, rslices, rmasks=None):
        if rmasks is not None:
            assert len(rslices) == len(rmasks), 'mismatch length'
        self.rslices = rslices
        self.rmasks = rmasks

    def __len__(self):
        return len(self.rslices)

    def region_I(self, k, I):
        jslice, islice = self.rslices[k]
        val = I[jslice, islice]
        return val if (self.rmasks is None) else val * self.rmasks[k]

    def region_imG(self, k, imG, keys=None):
        '''
        Parameters
        ----------
        k : int
            which region of interest to select
        imG : dict of 2d arrays
            Normally, the original image array and its gradients, possibly
            including relevant masks. All 2d arrays are the same dimension as
            the original image
        keys : None or list of str, optional
            Override keys of imG. If None (default), all keys of imG are used.

        Returns
        -------
        imG_k : dict of 2d arrays
            Each key, val is sliced from imgrad using self.slices[k]    
        '''
        jslice, islice = self.rslices[k]
        imG_k = {}
        if keys is None:
            keys = imG.keys()
        for key in keys:
            imG_k[key] = self.region_I(k, imG[key])
        return imG_k

    def downselect(self, isvalid):
        rslices = self.rslices[isvalid]
        rmasks = None if (self.rmasks is None) else self.rmasks[isvalid]
        return Regions(rslices, rmasks)

    def plot_k(self, k, imG, keys=['ones', 'I0', 'I1'], axes=None):
        '''
        Make a subplot series for a single region.

        Parameters
        ----------
        k : int
            Select which region to plot/
        imG : dict of 2d arrays
            Needs to contain keys. 2d arrays of the same size as the original
            image, will be sliced by self.region_k(...)
        keys : list of str, optional
            Select the keys in imG to be plotted.
            The default is ['ones', 'I0', 'I1'].
        axes : tuple of pyplot Axes
            Optional. Provide axes (matching length of keys), i.e. in order to
            customize panel construction.

        Returns
        -------
        axes : tuple of pyplot Axes
            Useful for customizing axes
        '''
        Nplot = len(keys)
        if Nplot < 1: return
        if axes is None:
            _, axes = _plt.subplots(1, Nplot, layout='tight')
        _plt.figure()
        imG_k = self.region_imG(k, imG, keys=keys)
        for i in range(Nplot):
            key = keys[i]
            ax = axes[i]
            ax.imshow(imG_k[key], cmap='gray')
            ax.set_title(key)
        return axes

    @classmethod
    def from_mask(cls, mask, *args, min_area=1):
        '''
        Signatures::
    
            regions, rmasks = Regions.from_mask(mask, imG)
            regions, rmasks = Regions.from_mask(mask, x, y)
            regions, rmasks = Regions.from_mask(mask)

        Parameters
        ----------
        mask : 2d array of bool
            mask (same dimensions as original image) to be factored into Regions
        x, y : array of int, optional
            global pixel coordinates
        imG : dict of arrays, optional
            requires imG['x'] and imG['y']
        min_area : int, optional
            Any ROI of area < min_area will be excluded. The default is 1.
        '''
        if len(args) == 0:
            raise NotImplementedError()
        elif len(args) == 1:
            x = args[0]['x']
            y = args[0]['y']
        elif len(args) == 2:
            x, y = args
        else:
            raise ValueError('wrong number of args')
        label, n = _label(mask, return_num=True)
        j1 = []
        j2 = []
        i1 = []
        i2 = []
        ind = []
        for k in range(1, n+1):
            mask_k = label==k
            area = _np.sum(mask_k)
            if area >= min_area:
                ind.append(k)
                x_k = x[mask_k]
                y_k = y[mask_k]
                i1.append(_np.min(x_k))
                i2.append(_np.max(x_k))
                j1.append(_np.min(y_k))
                j2.append(_np.max(y_k))
        rslices = get_rslices(_np.array(j1), _np.array(j2),
                              _np.array(i1), _np.array(i2))
        rmasks = rmasks_fromlabels(rslices, label, ind)
        return cls(rslices, rmasks)

    @classmethod
    def from_poi(cls, poi, xrad, yrad, *args):
        '''
        Signatures::
            
            Regions.from_poi(poi, xrad, yrad, Ishape)
            Regions.from_poi(poi, xrad, yrad, x, y)
            Regions.from_poi(poi, xrad, yrad, imG)

        Get frame boundaries (e.g. for slicing frames at POI) using semi-width
        and semi-height, but enforce boundaries around Ishape.

        Parameters
        ----------
        poi : array([[j0, i0], [j1, i1], ...])
            Points-of-Interest (same as peaks from
            skimage.feature.peak_local_max)
        xrad : array-like (often int type)
            semi-width of ROI
        yrad : array-like (often int type)
            semi-height of ROI
        Ishape : (nj, ni)
            shape of image containing all POI, i.e. I.shape
        x, y : 2d array
            global pixel coordinates
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
        j = poi[:, 0]
        i = poi[:, 1]
        j1 = _np.maximum(0, j - yrad + .5).astype(int)
        j2 = _np.minimum(ny - 1, j + yrad + 1.5).astype(int)
        i1 = _np.maximum(0, i - xrad + .5).astype(int)
        i2 = _np.minimum(nx - 1, i + xrad + 1.5).astype(int)
        rslices = get_rslices(_np.array(j1), _np.array(j2),
                              _np.array(i1), _np.array(i2))
        rmasks = rmasks_xyellipse(rslices, poi, xrad, yrad, x, y)
        return cls(rslices, rmasks)

