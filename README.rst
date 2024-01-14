Introduction
============
opticiq should one day become a collection of flexible tools for quantitative optical testing.

The goal is to bring experts together to collaborate and codify image quality testing, beam quality testing, and some related metrics, into an open source lib. And by doing so, to elevate the field, and minimize wasted redundant code.

Aspiring Feature Set
--------------------
* IQ test chart generation, for display, print, or lithography (partially complete)
* Various MTF test methods, using point, line, or edge function tests
* Grid and distortion testing
* Beam diameter (or star diameter), and other beam metrics
* Low-level functions that support the above features, e.g. gradient tests and auto-ROI (partially complete)

Look in the docs folder
-----------------------
Some jupyter-notebook sessions have been written to demonstrate emerging features.

Reusability Goals
-----------------

A key aim for opticiq is quality and reuse. To illustrate, here are some goals:

    * Library functions emphasizing generalization before automation
    * Docstrings
    * Documentation
    * Examples

Status
======
early prototype

Some ideas have been represented. Most of the functionality is not setup.


TO DO list
==========
    1. [x] basic charts (cb, pointgrid, slant-edge)
       [] misc charts (ghost-hunters)
       [x] filenames, x pdf-inversion
       [x] stretch v tile
       [] raster; save or display
       [] vector: x pdf, lithography
    2. grid intel (cb or pointgrid)
        see opencv findChessboardCorners and calibrateCamera
        (did opencv use a good method?)
        (does that work for projection too?)
    3. x imageGradient
    4. slant-edge, lsf
    5. licensing, etc
    6. x beam generator
    7. beam metrics
        auto-aperture d4s, encircled
    8. M^2
    9. Gerchberg-Saxton
    10. x Simple image transfer (blur, noise)
    11. Model image transfer (object/image, distortion, blur, noise)
