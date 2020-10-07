#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""
    .. table:: Summary of :mod:`nenupy.astro` classes and functions
        :widths: auto

        +--------------------------------------+------------------------------------------------+------------------------------------------------------+
        | Class                                | Method/Attribute                               | Description                                          |
        +======================================+================================================+======================================================+
        |                                      | :func:`~nenupy.astro.astro.lst`                | Sidereal time at NenuFAR                             |
        +--------------------------------------+------------------------------------------------+------------------------------------------------------+
        |                                      | :func:`~nenupy.astro.astro.lha`                | Hour angle at NenuFAR                                |
        +--------------------------------------+------------------------------------------------+------------------------------------------------------+
        |                                      | :func:`~nenupy.astro.astro.wavelength`         | Frequency to wavelength conversion                   |
        +--------------------------------------+------------------------------------------------+------------------------------------------------------+
        |                                      | :func:`~nenupy.astro.astro.ho_coord`           | Horizontal sky coordinates definition wrapper        |
        +--------------------------------------+------------------------------------------------+------------------------------------------------------+
        |                                      | :func:`~nenupy.astro.astro.eq_coord`           | Equatorial sky coordinates definition wrapper        |
        +--------------------------------------+------------------------------------------------+------------------------------------------------------+
        |                                      | :func:`~nenupy.astro.astro.to_radec`           | From horizontal to equatorial coordinates conversion |
        +--------------------------------------+------------------------------------------------+------------------------------------------------------+
        |                                      | :func:`~nenupy.astro.astro.toAltaz`            | From equatorial to horizontal coordinates conversion |
        +--------------------------------------+------------------------------------------------+------------------------------------------------------+
        |                                      | :func:`~nenupy.astro.astro.ho_zenith`          | NenuFAR's zenith in horizontal coordinates           |
        +--------------------------------------+------------------------------------------------+------------------------------------------------------+
        |                                      | :func:`~nenupy.astro.astro.eq_zenith`          | NenuFAR's zenith in equatorial coordinates           |
        +--------------------------------------+------------------------------------------------+------------------------------------------------------+
        |                                      | :func:`~nenupy.astro.astro.meridianTransit`    | Source meridian transit time                         |
        +--------------------------------------+------------------------------------------------+------------------------------------------------------+
        |                                      | :func:`~nenupy.astro.astro.radio_sources`      | Main radio source positions                          |
        +--------------------------------------+------------------------------------------------+------------------------------------------------------+
        |                                      | :func:`~nenupy.astro.astro.getSource`          | Get a particular source coordinates                  |
        +--------------------------------------+------------------------------------------------+------------------------------------------------------+
        |                                      | :func:`~nenupy.astro.astro.altazProfile`       | Retrieve horizontal coordinates versus time          |
        +--------------------------------------+------------------------------------------------+------------------------------------------------------+
        |                                      | :func:`~nenupy.astro.astro.dispersion_delay`   | Dispersion delay from electron plasma propagation    |
        +--------------------------------------+------------------------------------------------+------------------------------------------------------+
        | :class:`~nenupy.astro.hpxsky.HpxSky` |                                 **HEALPix sky representation wrapper**                                |
        |                                      +------------------------------------------------+------------------------------------------------------+
        |                                      | :attr:`~nenupy.astro.hpxsky.HpxSky.resolution` | Angular resolution of the HEALPix grid               |
        |                                      +------------------------------------------------+------------------------------------------------------+
        |                                      | :attr:`~nenupy.astro.hpxsky.HpxSky.time`       | Observation time to compute horizontal coordinates   |
        |                                      +------------------------------------------------+------------------------------------------------------+
        |                                      | :attr:`~nenupy.astro.hpxsky.HpxSky.skymap`     | HEALPix array containing sky pixel values            | 
        |                                      +------------------------------------------------+------------------------------------------------------+
        |                                      | :attr:`~nenupy.astro.hpxsky.HpxSky.visible_sky`| Hide or not the sky below NenuFAR's local horizon    |
        |                                      +------------------------------------------------+------------------------------------------------------+
        |                                      | :attr:`~nenupy.astro.hpxsky.HpxSky.eq_coords`  | HEALPIx grid's equatorial coordinates                |
        |                                      +------------------------------------------------+------------------------------------------------------+
        |                                      | :attr:`~nenupy.astro.hpxsky.HpxSky.ho_coords`  | HEALPIx grid's horizontal coordinates                |
        |                                      +------------------------------------------------+------------------------------------------------------+
        |                                      | :meth:`~nenupy.astro.hpxsky.HpxSky.lmn`        | Convert equatorial coordinates to (l, m, n)          |
        |                                      +------------------------------------------------+------------------------------------------------------+
        |                                      | :meth:`~nenupy.astro.hpxsky.HpxSky.radec_value`| Retrieve `skymap` values based on EQ coordinates     |
        |                                      +------------------------------------------------+------------------------------------------------------+
        |                                      | :meth:`~nenupy.astro.hpxsky.HpxSky.azel_value` | Retrieve `skymap` values based on HO coordinates     |
        |                                      +------------------------------------------------+------------------------------------------------------+
        |                                      | :meth:`~nenupy.astro.hpxsky.HpxSky.plot`       | Display `skymap` on an equatorial grid               |
        |                                      +------------------------------------------------+------------------------------------------------------+
        |                                      | :meth:`~nenupy.astro.hpxsky.HpxSky.save`       | Save `skymap` as a HEALPix FITS file                 |
        +--------------------------------------+------------------------------------------------+------------------------------------------------------+
    
"""

from .hpxsky import HpxSky
from .astro import *

