#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 18:25:02 2023

@author: chris
"""

import numpy as _np

from . import gen


class DemoData():
    def __init__(self, res=[6000, 5000], tile=300, cr=16, sigma=2, poisson=1):
        self._res = res
        self._tile = tile
        self._sigma = sigma
        self._poisson = poisson
        self._imgen = gen.ChartGen_img(res, cr=cr)

    @property
    def slant(self):
        im, _ = self._imgen.slant_rectangles(self._tile, tile=True, slant=4)
        return gen.image_transfer_simple(
            _np.array(im, dtype='int'),
            sigma=self._sigma, poisson=self._poisson)

    @property
    def cb(self):
        im, _ = self._imgen.checkerboard(self._tile, tile=True)
        return gen.image_transfer_simple(
            _np.array(im, dtype='int'),
            sigma=self._sigma, poisson=self._poisson)

    @property
    def ptgrid(self):
        im, _ = self._imgen.pointgrid(self._tile, tile=True)
        return gen.image_transfer_simple(
            _np.array(im, dtype='int'),
            sigma=self._sigma, poisson=self._poisson)

    '''@property
    def starfield(self):
        im = gen.genstarfield(self._res, nstar, sigma)'''


demoL = DemoData([6000, 5000], 300)
demoM = DemoData([1800, 1500], 300)
demoS = DemoData([600, 500], 150)
