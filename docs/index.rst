.. nenupytf documentation master file, created by
   sphinx-quickstart on Wed Nov 27 11:46:36 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to nenupy's documentation!
====================================

`nenupy` stands for Python package for `NenuFAR <https://nenufar.obs-nancay.fr/en/astronomer/>`_ (New Extension in Nançay Upgrading `LOFAR <http://www.lofar.org/>`_), a low-frequency radio-telescope located in Nançay, France. 

It enables reading of the so-called statistics data (or low-rate data) ouptput from the *LANewBa* (LOFAR-like Advanced New Backend) backend, namely *SST*, *BST* and *XST*. Further analysis can be performed depending on the considered dataset (see the tutorial section). 

`nenupy` also allows for *NenuFAR* beam simulation in the radio phased-array frame on a `HEALPix <https://healpix.jpl.nasa.gov/>`_ grid, while taking most of instrumental effects and properties into account.

Finally, observations can then be simulated using the simulated beam and a sky model.

.. note::
   By default, `logging <https://docs.python.org/3/library/logging.html>`_ is set to ``WARNING`` level. However, this can be changed dynamically by the user, for e.g.:

   >>> import nenupy
   >>> import logging
   >>> logging.getLogger('nenupy').setLevel(logging.INFO)

.. note::
   Users are most welcome to signal bugs or ask for complementary functionalities via sending a 
   `Github issue <https://github.com/AlanLoh/nenupy/issues/new/>`_.

.. note::
   DOI for the `nenupy` package: `10.5281/zenodo.3775196 <https://zenodo.org/record/3775196/>`_.

   BibTeX citation:

   .. code-block:: bash

      @software{alan_loh_2020_4279405,
         author       = {Alan Loh and the NenuFAR team},
         title        = {nenupy: a Python package for the low-frequency radio telescope NenuFAR},
         month        = nov,
         year         = 2020,
         publisher    = {Zenodo},
         version      = {v1.1.0},
         doi          = {10.5281/zenodo.3667815},
         url          = {https://doi.org/10.5281/zenodo.3667815}
      }
   
      @software{alan_loh_2020_3775196,
         author       = {Alan Loh and the NenuFAR team},
         title        = {nenupy: a Python package for the low-frequency radio telescope NenuFAR},
         month        = apr,
         year         = 2020,
         publisher    = {Zenodo},
         version      = {v1.0.0},
         doi          = {10.5281/zenodo.3775196},
         url          = {https://doi.org/10.5281/zenodo.3775196}
      }


.. _getting-started:

.. toctree::
   :caption: Getting Started
   :maxdepth: 1

   install

.. _tutorials:

.. toctree::
   :caption: Tutorials
   :maxdepth: 1
   
   tuto_xst2bst
   tuto_tv

.. toctree::
   :caption: Instrument
   :maxdepth: 1

   instru/array_configuration
   instru/instrument_properties
   instru/tools

.. toctree::
   :caption: Astronomy
   :maxdepth: 1

   astro/target
   astro/sky
   astro/skymodel
   astro/pointing
   astro/tools

.. toctree::
   :caption: Simulation
   :maxdepth: 1

   simu/beam_simulation
   simu/obs_simulation

.. toctree::
   :caption: Observation management
   :maxdepth: 1

   obs/obs_scheduling
   obs/src_in_lobes
   obs/obs_configuration

.. _data-reading:

.. toctree::
   :caption: Data reading
   :maxdepth: 2

   nenupy.beamlet
   nenupy.undysputed

.. _data-analysis:

.. .. toctree::
..    :caption: Simulation
..    :maxdepth: 2

..    nenupy.beam
..    nenupy.skymodel
..    nenupy.simulation

.. _tools:

.. toctree::
   :caption: Tools
   :maxdepth: 2

   nenupy.astro
   nenupy.instru
   nenupy.observation
   nenupy.schedule


.. toctree::
   :caption: Changelog
   :maxdepth: 1

   changelog

*****
Index
*****

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
