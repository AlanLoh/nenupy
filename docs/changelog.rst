Changelog
=========

1.x.x (unreleased)
^^^^^^^^^^^^^^^^^^

* Extrapolation of NenuFAR antenna model (:func:`~nenupy.instru.instru.nenufar_ant_gain`) above 80 MHz [`#22 <https://github.com/AlanLoh/nenupy/issues/22>`_].
* Implementation of :class:`~nenupy.observation.database.ObsDatabase` class, which enables queries over the NenuFAR BST database in order to seach for particular past observations within a given :attr:`~nenupy.observation.database.ObsDatabase.time_range` and/or :attr:`~nenupy.observation.database.ObsDatabase.freq_range` and/or around a given sky position :attr:`~nenupy.observation.database.ObsDatabase.fov_center` with a given search radius :attr:`~nenupy.observation.database.ObsDatabase.fov_radius`.
* Addition of a ``text`` option to :meth:`~nenupy.astro.hpxsky.HpxSky.plot` aiming at overplotting text (such as source names) at some given equatorial positions.
* UVW computation corrected (sign convention in order to call imaging TF as :math:`\int V e^{-2\pi i (ul + vm)}\, du\, dv`) [`#23 <https://github.com/AlanLoh/nenupy/issues/23>`_].
* NenuFAR data rate estimation implemented (:func:`~nenupy.instru.instru.data_rate`).
* Implementation of :class:`~nenupy.undysputed.dynspec.Dynspec` to read/de-disperse/rebin (in time and/or frequency) high-rate UnDySPuTeD time-frequency data (or `DynSpec data <https://nenufar.obs-nancay.fr/en/astronomer/#data-products>`_) [`#30 <https://github.com/AlanLoh/nenupy/issues/30>`_].


1.0.0 (2020-04-29)
^^^^^^^^^^^^^^^^^^

Major refactoring of the original `nenupy` package.