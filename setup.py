#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 14:11:31 2023

@author: chris
"""

import setuptools

# this is the right way to import version.py here:
main_ns = {}
with open('opticiq/version.py') as vf:
    exec(vf.read(), main_ns)
#fname_readme = setuptools.convert_path("README.rst")
#with open(fname_readme) as rf:
#    README = rf.read()

setuptools.setup(
    name = 'opticiq',
    version = main_ns['__version__'],
    author = "Chris Cannon",
    author_email = 'chris.cannon.9001@gmail.com',
    description = 'Optical Image Quality and Beam Tests routines',
    #long_description=README,
    #long_description_content_type='text/x-rst',
    #license = 'BSD',
    #url = 'https://github.com/chriscannon9001/tablarray',
    #packages=setuptools.find_packages(include=[
    #    'tablarray', 'tablarray.kwtools', 'tablarray.linalg',
    #    'tablarray.np2ta', 'tablarray.tests']),
    python_requires='>=3.2',
    install_requires=[
        'numpy',
        'matplotlib',
        'PIllow',
        'scikit-image'],
    tests_require=['pytest'],
    extras_require={
        'extras' : ['gdstk', 'reportlab']})
