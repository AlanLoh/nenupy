#! /usr/bin/python3
# -*- coding: utf-8 -*-


from setuptools import setup, find_packages
import nenupy


setup(
    name='nenupy',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy', # 1.19.4
        'scipy', #1.5.2
        'astropy', #4.1
        'matplotlib', # 3.3.3
        'reproject', # 0.5.1
        'numba', # 0.48.0
        'numexpr', # 2.7.1
        'pyproj', # 2.6.0
        'dask[array]', # 2020.12.0
        'sqlalchemy', # 1.3.18
        'healpy'
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
