#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 08:20:56 2023

@author: chris
"""


import numpy as _np


def _rot_mtrx2d(theta):
    '''Rotation matrix for xy vectors (theta is degrees)'''
    theta = _np.deg2rad(theta)
    A = _np.array([[_np.cos(theta), -_np.sin(theta)],
                  [_np.sin(theta), _np.cos(theta)]])
    return A


def _vert2vect(vertices):
    """For vertices=[(x0, y0), (x1, y1), ...] return 2d vector"""
    vector = _np.array(vertices)
    return vector


def _vect2vert(vector):
    """Return vertices=[(x0, y0), (x1, y1), ...] made from 2d vector"""
    vertices = []
    for vi in vector:
        vertices.append(tuple(vi))
    return vertices


def _offset_vertices(dx, dy, vertices):
    """Apply offsets dx, dy to vertices=[(x0, y0), (x1, y1), ...]"""
    vect = _vert2vect(vertices)
    vect += _np.array([dx, dy])
    vert = _vect2vert(vect)
    return vert


def _rotate_vertices(vertices, theta):
    '''Apply theta(deg) rotation to vertices'''
    vector = _vert2vect(vertices)
    #print(theta)
    A = _rot_mtrx2d(theta)
    #print(A)
    vector = _np.matmul(vector, A)
    vertices = _vect2vert(vector)
    return vertices


def _rotate_2d(xarray, yarray, theta):
    '''Apply theta(deg) rotation to 1d or 2d xarray, yarray'''
    A = _rot_mtrx2d(theta)
    #print(A)
    shape0 = _np.shape(xarray)
    #print('shape0', shape0)
    assert shape0 == _np.shape(yarray), 'shapes disagree'
    # get a vector form of x,y
    vec1 = _np.stack((_np.ravel(xarray), _np.ravel(yarray)), axis=1)
    # rotate by matrix multiplication
    vec2 = _np.matmul(vec1, A)
    # return x, y to original array shape
    x2 = _np.reshape(vec2[:, 0], shape0)
    y2 = _np.reshape(vec2[:, 1], shape0)
    return x2, y2
