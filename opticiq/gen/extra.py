#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 14:49:13 2023

@author: chris
"""

import numpy as _np

from .chart import itr_checkerboard, itr_slant_rectangles, _naming_grid, limit_polygon2frame

# try imports for extras_require
try:
    import gdstk as _gdstk
    _GDS_isavailable = True
except ImportError:
    _GDS_isavailable = False
try:
    from reportlab.pdfgen.canvas import Canvas as _Canvas
    from reportlab.lib import units as _units
    from reportlab.lib import pagesizes as _ps
    from reportlab.graphics.shapes import Polygon, Drawing, Rect
    from reportlab.lib.colors import Color as _Color
    _PDF_isavailable = True
except ImportError:
    _PDF_isavailable = False


if _GDS_isavailable:
    class ChartGen_gds():
        def __init__(self):
            raise NotImplementedError
else:
    class ChartGen_gds():
        def __init__(*args, **kwargs):
            raise ImportError('missing optional gdstk, try "pip install opticiq[extras]"')


if _PDF_isavailable:
    _P_IND = dict(
        letter=_ps.letter, legal=_ps.legal, TABLOID=_ps.TABLOID,
        tabloid=_ps.TABLOID, A0=_ps.A0, A1=_ps.A1, A2=_ps.A2, A3=_ps.A3, A4=_ps.A4,
        A5=_ps.A5, A6=_ps.A6, A7=_ps.A7, A8=_ps.A8, A9=_ps.A9, B0=_ps.B0, B1=_ps.B1,
        B2=_ps.B2, B3=_ps.B3, B4=_ps.B4, B5=_ps.B5, B6=_ps.B6, B7=_ps.B7, B8=_ps.B8,
        B9=_ps.B9, B10=_ps.B10, C0=_ps.C0, C1=_ps.C1, C2=_ps.C2, C3=_ps.C3, C4=_ps.C4,
        C5=_ps.C5, C6=_ps.C6, C7=_ps.C7, C8=_ps.C8, C9=_ps.C9, C10=_ps.C10)
    _UNIT_IND = dict(
        inch=_units.inch, mm=_units.mm, cm=_units.cm)

    class ChartGen_pdf():
        '''
        Vector pdf chart generators for optical testing.

        Example::
            
            g = ChartGen_pdf(6, 'mm', 'A4')
            c = g.checkerboard(10, tile=True)  # 10x10mm checkers
            c.save()
        '''
        def __init__(self, border, unit, pagesize, cr=None, light=1, invert=False):
            '''
            Parameters
            ----------
            border : float
                size of the page border, in unit
            unit : str
                one of 'inch', 'mm', 'cm'
            pagesize : str
                one of 'letter', 'legal', 'tabloid', 'A0', 'A1', 'A2', 'A3',
                'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'B0', 'B1', 'B2', 'B3',
                'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'C0', 'C1', 'C2',
                'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'
            cr : float >1, optional
                Contrast ratio, with None meaning inf. The default is None.
            light : float 0-1, optional
            invert : bool, optional
                Inverts the pattern (white on black instead of black on white).
                The default is False, because paper is nominally white and
                paper is probably more uniform than ink or toner.
            '''
            self._basename = pagesize + '_'
            dark = 0 if cr is None else light/(1 + cr)
            if invert:
                self._fill = _Color(light, light, light)
                self._bg = _Color(dark, dark, dark)
            else:
                self._fill = _Color(dark, dark, dark)
                self._bg = None if _np.isclose(light, 1) else _Color(light, light, light)
            self._unit = unit
            self._pagesize = pagesize
            unit = _UNIT_IND[unit]
            pagesize = _P_IND[pagesize]
            self._border = border*unit
            width = pagesize[0] - 2*border*unit
            height = pagesize[1] - 2*border*unit
            assert (width > 0 and height > 0), 'Borders exceeded canvas size'
            self.frame = [width, height]

        def _get_canvas(self, fname):
            '''
            Returns new blank canvas and drawing
            '''
            pagesize = _P_IND[self._pagesize]
            c = _Canvas(fname, pagesize=pagesize)
            if self._bg is not None:
                #r = Rect(0, 0, self.frame[0], self.frame[1])
                #d.setProperties()
                c.setFillColor(self._bg)
                c.setStrokeColor(self._bg)
                b = self._border
                c.rect(b, b, self.frame[0], self.frame[1], fill=1)
            c.setFillColor(self._fill)
            c.setStrokeColor(self._fill)
            d = Drawing(*tuple(self.frame))
            return c, d

        def checkerboard(self, p, tile=True):
            '''
            Parameters
            ----------
            p : float or [float, float]
                (tile) - interval between patterns in unit,
                or (stretch) - number of patterns
            tile : bool, optional
                (True) - tile at even interval ; (False)- stretch pattern to fit the frame

            Returns
            -------
            canvas : reporltab Canvas
                test chart; checkerboard
            '''
            unit = _UNIT_IND[self._unit]
            if tile:
                p = _np.array(p) * unit
            b = self._border
            naming = _naming_grid(self.frame, p, tile=tile, unit=unit)
            naming += self._unit
            fname = self._basename + 'checker' + naming
            c, d = self._get_canvas(fname)
            for x1, y1, x2, y2 in itr_checkerboard(self.frame, p, tile=tile):
                r = Rect(x1, y1, x2 - x1, y2 - y1)
                #c.rect(x1 + b, y1 + b, x2 - x1, y2 - y1, fill=1, stroke=0)
                r.fillColor = self._fill
                r.strokeWidth = 0
                r.strokeColor = self._fill
                d.add(r)
            d.drawOn(c, b, b)
            return c

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
            canvas : reporltab Canvas
                test chart; checkerboard
            '''
            unit = _UNIT_IND[self._unit]
            if tile:
                p = _np.array(p) * unit
            naming = _naming_grid(self.frame, p, tile=tile, unit=unit)
            naming += self._unit
            fname = (self._basename + 'slantrect' + naming
                     + '_fill%0.2fslant%0.1fdeg' % (fill_factor, slant))
            c, d = self._get_canvas(fname)
            for vertices in itr_slant_rectangles(
                    self.frame, p, tile=tile, slant=slant, fill_factor=fill_factor):
                vertices = limit_polygon2frame(self.frame, vertices)
                # vertices are a zipped list, need to flatten it for Polygon
                vertices = [el for xy in vertices for el in xy]
                pg = Polygon(vertices)
                pg.fillColor = self._fill
                pg.strokeWidth = 0
                pg.strokeColor = self._fill
                d.add(pg)
            b = self._border
            d.drawOn(c, b, b)        
            return c

else:
    # if reportlab was not available, define a dummy
    class ChartGen_pdf():
        def __init__(*args, **kwargs):
            '''
            ChartGen_pdf is an extra that requires reportlab to work, but
            reportlab was not installed during import. So this is a dummy that
            will only raise an ImportError

            Try ::
                
                pip install opticiq[extras]
            '''
            raise ImportError('missing optional reportlab, try "pip install opticiq[extras]"')
