.. _bst_reading_doc:

Beamlet STatistics (BST)
========================

`NenuFAR <https://nenufar.obs-nancay.fr/en/astronomer>`_ delivers BST statistical data from the *LaNewBA* receiver.
Beamlet STatistics (BST) are the beamlet outputs per polarization averaged at 1 s resolution


BST reading
-----------

:class:`~nenupy.io.bst.BST`

.. code-block:: python

    from nenupy.io.bst import BST

    bst = BST("/path/to/BST.fits")


BST data selection
------------------

.. code-block:: python

    data = bst.get(
        frequency_selection="<=52MHz",
        time_selection=">=2022-01-24T11:08:10 & <2022-01-24T11:14:08",
        polarization="NW",
        beam=8
    )

``data`` is a :class:`~nenupy.io.io_tools.ST_Slice` object.
See ... for this class usage.

