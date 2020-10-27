#! /usr/bin/python3
# -*- coding: utf-8 -*-


from setuptools import setup, find_packages
import nenupy


setup(
    name='nenupy',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'astropy',
        'matplotlib', 
        'healpy', #==1.13.0
        'reproject',
        'numba',
        'numexpr',
        'pyproj',
        'dask[array]',
        'sqlalchemy'
    ],
    extras_require={
        #'astroplan': 'astroplan',
        'mocpy': 'mocpy'
    },
    python_requires='>=3.6',
    scripts=[
        'bin/nenupy_vcr_coordinates'
    ],
    version=nenupy.__version__,
    description='NenuFAR Python package',
    url='https://github.com/AlanLoh/nenupy.git',
    author=nenupy.__author__,
    author_email=nenupy.__email__,
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research"
    ],
    zip_safe=False
)

# make the package:
# python3 setup.py sdist bdist_wheel
# upload it:
# python3 -m twine upload dist/*version*

# Release:
# git tag -a v*version* -m "annotation for this release"
# git push origin --tags

# Documentation
# sphinx-build -b html docs/ docs/_build/

# Update on nancep:
# /usr/local/bin/pip3.5 install nenupy --install-option=--prefix=/cep/lofar/nenupy3 --upgrade
