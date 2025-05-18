Introduction
============
The optical engineer's library for quantitative image quality testing and analysis of laser beams.

Installation::

    pip install opticiq
    pip install opticiq[extras]

From the docs folder
--------------------
A few jupyter-notebook sessions demonstrate working features:

* `demoCheckerboard.ipynb <https://github.com/chriscannon9001/opticiq/blob/main/docs/demoCheckerBoard.ipynb>`_ Saddle-point detection in recipe_checkerboard2 maps a checkerboard image.
* `demoSlantEdgeIQ.ipynb <https://github.com/chriscannon9001/opticiq/blob/main/docs/demoSlantEdgeIQ.ipynb>`_ Demonstrates recipe_slantedge which auto finds regions of interest (ROI) and analyzes (slanted) edges to determine Line Spread Function LSF and Modulation Transfer Function MTF.
* `demoStarsBeams.ipynb <https://github.com/chriscannon9001/opticiq/blob/main/docs/demoStarsBeams.ipynb>`_ Demonstrates recipe_star2 (which is applicable to beams as well as stars) to find and analyze local maxima.

2d Array of float
-----------------

Note that opticiq always expects an image to be 2d array of type float. Recipes use imageGradient and (as of 0.0.5) imageGradient doesn't verify or auto-convert to float. Even after blur, a gradient of integers will have "spikes" at discreet steps, and those artifacts may ruin feature detection.

Library Coding
--------------

A goal of opticiq is reusability before automation, so recipe_xx functions are high-level front-ends but you can always copy and paste a recipe to hack at the low-level components if needed. (BSD license keeps the door open.)

Take a moment to look at a recipe and see how the most common low-level components are used:

    * The first step is imageGradients() which blurs the image first and then returns gradients.
    * The second step is usually use some combo of gradients to get a Regions object. The structure of Regions may seem counterintuitive, but the hidden motive is to defer region-specific computations until after instantiation. Someday it may be possible for this API to multi-thread when the number of regions is high (esp. star fields), although of course Python itself can't do that.
    * It depends on the recipe's purpose, but a third step may be peaksAnalysis.

Aspiring Feature Set
--------------------
* IQ test chart generation, for display, print, or lithography (partially complete)
* Various MTF test methods, using point, line, or edge function tests
* Grid and distortion testing
* Beam diameter (or star diameter), and other beam metrics
* Low-level functions that support the above features, e.g. gradient tests and auto-ROI (partially complete)

Status
======
Alpha

A chunk of the planned functionality is working now.

TO DO - Planned features
========================
    1. Chart generators:
        a. More misc charts (ghost-hunters)
        b. Vector chart gen: pdf and lithography
        c. Model image transfer (object/image, distortion, blur, noise)
        d. In addition to ChartGen_img (raster): allow enslaving a projector device so that camera and projector can be scripted to execute multiple image quality tests in sequence, whether the two reside on the same machine or networked.
    2. Checkerboard:
        a. Ordering the grid
        b. Another recipe for point grid - Checkerboard is usually better but why not?
        c. Are there use cases for a line grid as well?
        d. Compare to opencv findChessboardCorners and calibrateCamera. Does my method have any advantage? Does opencv support projection cases or only camera cases?
    3. Sharpness:
        a. (Do I want to add an analysis for explicit measurement from Line Spread chart or only use Edge Spread?)
    4. Beam metrics:
        a. Gaussian fit
        b. Encircled energy
        c. Ensquared energy
        d. Off-axis metrics
    5. M^2
