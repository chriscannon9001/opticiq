#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 14:31:36 2023

@author: chris
"""

from itertools import pairwise as _pairwise
import numpy as _np
from PIL import Image as _Image, ImageDraw as _ImageDraw
from sympy import Point as _Point, Polygon as _Polygon
from sympy.geometry.util import convex_hull as _convex_hull

from ..math import _rotate_vertices, _offset_vertices


def _naming_grid(frame, p, tile=True, unit=1):
    '''
    Naming string to describe pattern tiling
    '''
    p = _np.ones(2, dtype=int) * p
    if tile:
        d = p
    else:
        d = frame / p
    d = d / unit
    if _np.allclose(0, _np.mod(p, 1)):
        naming = ('%d' % d[0]) if _np.isclose(d[0], d[1]) else ('%dx%d' % (d[0], d[1]))
    else:
        naming = ('%0.2f' % d[0]) if _np.isclose(d[0], d[1]) else ('%0.2fx%0.2f' % (d[0], d[1]))
    return naming


def pattern_grid(frame, p, tile=True):
    '''
    Low-level utility common to all pattern generators, to control how
    patterns are tiled within a frame. Provides a tile mode and a stretch mode.

    E.g. pattern gen application may iterate pairwise over the grid::
        
        from itertools import pairwise
        xp, yp, _, _ = pattern_grid(frame, p, tile=tile)
        for x1, x2 in pairwise(xp):
            for y1, y2 in pairwise(yp):
                # x1, x2, y1, y2 are corners of the tile to draw a pattern

    Parameters
    ----------
    frame : [width, height]
        dimensions of the frame
    p : float or [float, float]
        (tile) - interval between patterns, (stretch) - number of patterns
    tile : bool, optional
        (True) - tile at even interval ; (False)- stretch pattern to fit the frame

    Returns
    -------
    xp, yp : 1d arrays
        Coordinates of points which define the corners of pattern tiles
        (not the centers of pattern tiles). Tiling always starts from the
        center of the frame, i.e. so that corners may clip non-integer tiles.
        ... And that means that non-integer framing can lead to corner points
        outside of the frame.
    d : [float, float]
        interval spacing between tiles
    N : [int, int]
        number of tiles
    '''
    p = _np.ones(2, dtype=int) * p
    frame = _np.ones(2, dtype=int) * frame
    centers = frame / 2
    if tile:
        d = p
        N = _np.ceil(frame / p)
    else:
        d = frame / p
        N = _np.ceil(p)
    corner1 = centers - (d * N) / 2
    corner2 = centers + (d * N) / 2
    xp = _np.linspace(corner1[0], corner2[0], int(N[0] + 1))
    yp = _np.linspace(corner1[1], corner2[1], int(N[1] + 1))
    return xp, yp, d, N


def limit_grid2frame(xp, yp, frame):
    xp = _np.minimum(xp, frame[0])
    xp = _np.maximum(xp, 0)
    yp = _np.minimum(yp, frame[1])
    yp = _np.maximum(yp, 0)
    return xp, yp


def limit_polygon2frame(frame, vertices):
    '''
    Some pattern generators may want to use a filter to clip polygons if they
    fall outside the frame.
    '''
    pgon_points = [_Point(v) for v in vertices]
    pgon_poly = _Polygon(*tuple(pgon_points))
    if pgon_poly.area < 0:
        # reverse polygons if the path area is - (CW order of vertices)
        pgon_poly = _Polygon(*tuple(reversed(pgon_points)))
        print('after reversal ', pgon_poly)
    w, h = tuple(frame)
    frame_points = [_Point(v) for v in [(0,0), (w, 0), (w, h), (0, h)]]
    frame_poly = _Polygon(*tuple(frame_points))
    # get all points enclosed in each other
    enclosed = []
    for point in pgon_points:
        if frame_poly.encloses(point):
            enclosed.append(point)
    for point in frame_points:
        if pgon_poly.encloses(point):
            enclosed.append(point)
    # intersect may be empty, a point, or a line (it's not a polygon)
    intersect = list(frame_poly.intersection(pgon_poly))
    # the answer is the convex hull of all enclosed points plus the intersect
    hull = _convex_hull(*tuple(enclosed + intersect))
    # we do want to turn it back into a list of tuples though
    return [tuple(p) for p in hull.vertices]
    #return hull


def itr_pointgrid(frame, p, **kwargs):
    '''
    Parameters
    ----------
    see pattern_grid

    Yields
    ------
    x, y : float, float
        x, y point to feed to a plotter

    Suppresses points outside the frame.
    '''
    xp, yp, _, _ = pattern_grid(frame, p, **kwargs)
    for x1, x2 in _pairwise(xp):
        for y1, y2 in _pairwise(yp):
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
            if x >= 0 and x < frame[0] and y >= 0 and y < frame[1]:
                yield x, y


def itr_checkerboard(frame, p, tile=True, even=True):
    '''
    Parameters
    ----------
    see pattern_grid

    Yields
    ------
    x1, y1, x2, y2 : 
        upper-left and lower-right rectangle corners to feed to plotter

    Clips checkers to frame if needed.
    '''
    xp, yp, _, _ = pattern_grid(frame, p, tile=tile)
    xp, yp = limit_grid2frame(xp, yp, frame)
    Nh = len(xp) - 1
    Nv = len(yp) - 1
    mod = 0 if even else 1
    for col in range(Nh):
        for row in range(Nv):
            if _np.mod(row + col, 2) == mod:
                # yield alternating patterns
                yield xp[col], yp[row], xp[col + 1], yp[row + 1]


def itr_slant_rectangles(frame, p, tile=True, slant=5, fill_factor=.5):
    '''
    Parameters
    ----------
    frame, p, tile :
        see pattern_grid
    slant : float, optional
        angle in degrees to slant the squares (default = 5 degrees)
    fill_factor : float 0 to ~.95 (default = 0.5)
        low fill factor means more space between slanted squares

    Yields
    ------
    vertices :
        vertices [(x0, y0), (x1, y1), ...] to feed to a polygon plotter

    Does not clip polygon vertices to frame (see limit_polygon2frame).
    '''
    xp, yp, d, _ = pattern_grid(frame, p, tile=tile)
    half_w, half_h = tuple(d * fill_factor / 2)
    vertices0 = [(-half_w, -half_h), (half_w, -half_h),
                 (half_w, half_h), (-half_w, half_h)]
    vertices0 = _rotate_vertices(vertices0, slant)
    for x1, x2 in _pairwise(xp):
        for y1, y2 in _pairwise(yp):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            vertices = _offset_vertices(cx, cy, vertices0)
            yield vertices


class ChartGen_img():
    '''
    Raster image chart generator for optical testing.

    Example::
        
        igen = ChartGetn_img([1600, 1200], cr=12)
        im, name = igen.checkerboard(20)
        im.save(name + '.png')
    '''
    def __init__(self, res, mode='W', cr=None, fg=255, bg=0):
        '''
        Parameters
        ----------
        res : [hres, vres]
            resolution in pixels
        [mode] : str
            color mode (default='W') - 'W', 'R', 'G', 'B'
        [contrast] : float, None means ignore
            contrast ratio; bg=round(fg/(1 + contrast))
        [fg] : int 0-255
            foreground (default 255)
        [bg] : int 0-255
            background (default 0)
        '''
        mode = mode.upper()
        if cr is not None:
            bg = int(fg / (1 + cr) + .5)
        self._basename = '%dx%d' % tuple(res) + '%s_%don%d_' % (mode, fg, bg)
        if mode == 'W':
            imgmode = 'L'
            imgfg = (fg,)
            imgbg = (bg,)
        elif mode == 'R':
            imgmode = 'RGB'
            imgfg = (fg, 0, 0)
            imgbg = (bg, 0, 0)
        elif mode.lower() == 'G':
            imgmode = 'RGB'
            imgfg = (0, fg, 0)
            imgbg = (0, bg, 0)
        elif mode.lower() == 'B':
            imgmode = 'RGB'
            imgfg = (0, 0, fg)
            imgbg = (0, 0, bg)
        else:
            raise ValueError('unknown mode "%s"' % mode)
        self.res = res
        self.imgfg = imgfg
        self.imgbg = imgbg
        self.imgmode = imgmode

    def _get_blank(self):
        return _Image.new(self.imgmode, tuple(self.res), self.imgbg)

    def pointgrid(self, p, **kwargs):
        '''
        Parameters
        ----------
        p : float or [float, float]
            (tile) - interval between patterns in unit ;
            (stretch) - number of patterns
        tile : bool, optional
            (True) - tile at even interval ; (False) - stretch pattern to fit the frame

        Returns
        -------
        img : PIL.Image
            test chart; grid of points
        name : str
            Name (recommended filename without extension)
        '''
        img = self._get_blank()
        draw = _ImageDraw.Draw(img)
        for xy in itr_pointgrid(self.res, p, **kwargs):
            draw.point(xy, fill=self.imgfg)
        name = self._basename + 'pointgrid_'
        name += _naming_grid(self.res, p, **kwargs) + 'pix'
        return img, name

    def checkerboard(self, p, **kwargs):
        '''
        Parameters
        ----------
        p : float or [float, float]
            (tile) - interval between patterns in unit ;
            (stretch) - number of patterns
        tile : bool, optional
            (True) - tile at even interval ; (False) - stretch pattern to fit the frame

        Returns
        -------
        img : PIL.Image
            test chart; checkerboard
        name : str
            Name (recommended filename without extension)
        '''
        img = self._get_blank()
        draw = _ImageDraw.Draw(img)
        for x1, y1, x2, y2 in itr_checkerboard(self.res, p, **kwargs):
            draw.rectangle((x1, y1, x2 - 1, y2 - 1),
                           fill=self.imgfg, outline=self.imgfg)
        name = self._basename + 'checker_'
        name += _naming_grid(self.res, p, **kwargs) + 'pix'
        return img, name

    def slant_rectangles(self, p, tile=True, slant=5, fill_factor=0.5):
        '''
        Parameters
        ----------
        p : float or [float, float]
            (tile) - interval between patterns in unit,
            or (stretch) - number of patterns
        tile : bool, optional
            (True) - tile at even interval ; (False) - stretch pattern to fit the frame
        slant : float, optional
            angle in degrees to slant the squares (default = 5 degrees)
        fill_factor : float 0 to ~.95 (default = 0.5)
            low fill factor means more space between slanted squares

        Returns
        -------
        img : PIL.Image
            test chart; slanted rectangles, use for slant-edge SFR
        name : str
            Name (recommended filename without extension)
        '''
        img = self._get_blank()
        draw = _ImageDraw.Draw(img)
        for vertices in itr_slant_rectangles(
                self.res, p, tile=tile, slant=slant, fill_factor=fill_factor):
            draw.polygon(vertices, fill=self.imgfg, outline=self.imgfg)
        name = self._basename + 'slantrec_'
        name += _naming_grid(self.res, p, tile=tile) + 'pix'
        return img, name

    def checkers_and_points(self, p, tile=True):
        '''
        '''
        img = self._get_blank()
        draw = _ImageDraw.Draw(img)
        for x1, y1, x2, y2 in itr_checkerboard(self.res, p, tile=tile,
                                               even=True):
            draw.rectangle((x1, y1, x2 - 1, y2 - 1),
                           fill=self.imgfg, outline=self.imgfg)
        for x1, y1, x2, y2 in itr_checkerboard(self.res, p, tile=tile,
                                               even=False):
            draw.point(((x1 + x2)/2, (y1 + y2)/2), fill=self.imgfg)
        name = self._basename + 'checkerpoints_'
        name += _naming_grid(self.res, p, tile=tile) + 'pix'
        return img, name
