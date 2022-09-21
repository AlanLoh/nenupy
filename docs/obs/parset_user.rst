.. _parset_user_doc:

Parset_user Editor
==================

`NenuFAR <https://nenufar.obs-nancay.fr/en/astronomer/>`_ observations are configured thanks to so-called *parset* files.
The latter are generated through the web platform known as the `Virtual Control Room <https://gui-nenufar.obs-nancay.fr/>`_.
It is achieved either by setting up the configuration using the graphic interface or by uploading *parset_user* files, which basically are a user-defined light-weight version of the *parset* files.

`nenupy` provides a way for editing such *parset_user* files, with an interface suitable for scripting and automating observation programmation.
The following describes how to use the :class:`~nenupy.observation.parset.ParsetUser` class.

.. note::
    User-written *parset_user* files obey to a specific syntax.
    Before editing, the user needs to be aware of the various available keywords.
    They are listed, along with their description in the NenuFAR `Parset_User guide <https://doc-nenufar.obs-nancay.fr/UsersGuide/parsetFileuserparset_user.html>`_.


.. code-block:: python

    >>> from nenupy.observation import ParsetUser


Configure observation and output
--------------------------------

After instantiation of a :class:`~nenupy.observation.parset.ParsetUser` object, a `print()` of the corresponding variable displays the current status of the *parset_user* being defined.
No verification is made at this stage, therefore careful knowledge of expected field syntax is required.

.. code-block:: python

    >>> p = ParsetUser()
    >>> print(p)

    Observation.contactName=contact_name
    Observation.name=observation_name
    Observation.contactEmail=contact_email
    Observation.nrAnabeams=0
    Observation.nrBeams=0
    Observation.topic=key_project_name

The various blocks of the *parset_user* file can be filled using the dedicated attributes.
For instance, the *observation* block could be updated as a dictionnary (`p.observation["contactName"] = "Alan"`).
The most convenient way lies in using the dedicated methods, e.g. :meth:`~nenupy.observation.parset.ParsetUser.set_observation_config`:

.. code-block:: python

    >>> p.set_observation_config(
    >>>     contactName="Alan",
    >>>     contactEmail="alan.loh@obspm.fr",
    >>>     name="My_observation",
    >>>     topic="DEBUG"
    >>> )
    >>> print(p)

    Observation.contactName=Alan
    Observation.name=My_observation
    Observation.contactEmail=alan.loh@obspm.fr
    Observation.nrAnabeams=0
    Observation.nrBeams=0
    Observation.topic=DEBUG

The *output* block can be equivalently modified using the :meth:`~nenupy.observation.parset.ParsetUser.set_output_config` method:

.. code-block:: python

    >>> p.set_output_config(
    >>>     hd_bitMode=16,
    >>>     hd_receivers="[undysputed]"
    >>> )
    >>> print(p)

    Observation.contactName=Alan
    Observation.name=My observation
    Observation.contactEmail=alan.loh@obspm.fr
    Observation.nrAnabeams=0
    Observation.nrBeams=0
    Observation.topic=DEBUG

    Output.hd_bitMode=16
    Output.hd_receivers=[undysputed]


.. seealso::
    The attributes :attr:`~nenupy.observation.parset.ParsetUser.observation_fields` and :attr:`~nenupy.observation.parset.ParsetUser.output_fields` return the list of valid keyword names for the respective blocks.


Configure analog beams
----------------------

`NenuFAR <https://nenufar.obs-nancay.fr/en/astronomer/>`_ *analog beams* define the state of the antennas, and could be understood as the array *field of view*.
Setting up the associated parameters is essential.
An observation can contain as many *analog beams* as desired.
They could either be simultaneous (if each simultaneous *analog beam* includes unique *Mini-Array* indices) or not (then their start and stop times must not intersect and similar *Mini-Arrays* can be selected multiple times).

Adding an *analog beam* to the observation is simply done via the method :meth:`~nenupy.observation.parset.ParsetUser.add_analog_beam`, which takes keyword arguments as parameters.
As an example, an *analog beam* is added while setting up two fields.
The ``target`` field exists and can be modified.
However, ``wrong_key`` is not expected from the `list of available keys <https://doc-nenufar.obs-nancay.fr/UsersGuide/parsetFileuserparset_user.html>`_ and a `warning` is raised telling the user which field name are valid.

.. code-block:: python

    >>> p.add_analog_beam(wrong_key="test", target="My fav target")

    WARNING: Key 'wrong_key' is invalid. Available keys are: dict_keys(['target', 'simbadSearch',
    'psrcatSearch', 'trackingType', 'directionType', 'transitDate', 'decal_transit', 'azimuth',
    'elevation', 'ra', 'dec', 'startHA', 'stopHA', 'startTime', 'duration', 'antState', 'antList',
    'maList', 'attList', 'filterStart', 'filter', 'filterTime', 'beamSquint', 'optFrq']).

.. warning::
    Even if all entered keys are invalid, an analog beam is still added to the list :attr:`~nenupy.observation.parset.ParsetUser.analog_beams` with default parameters.

.. note::
    If the user needs a quick reminder of the available fields associated to any *parset_user* block, the lists could be accessed through any of the corresponding attributes:
    
    * :attr:`~nenupy.observation.parset.ParsetUser.analog_beam_fields`
    * :attr:`~nenupy.observation.parset.ParsetUser.numerical_beam_fields`
    * :attr:`~nenupy.observation.parset.ParsetUser.observation_fields`
    * :attr:`~nenupy.observation.parset.ParsetUser.output_fields`

    .. code-block:: python

        >>> p.analog_beam_fields

        ['target',
        'simbadSearch',
        'psrcatSearch',
        'trackingType',
        'directionType',
        'transitDate',
        'decal_transit',
        'azimuth',
        'elevation',
        'ra',
        'dec',
        'startHA',
        'stopHA',
        'startTime',
        'duration',
        'antState',
        'antList',
        'maList',
        'attList',
        'filterStart',
        'filter',
        'filterTime',
        'beamSquint',
        'optFrq']


Once an *analog beam* is added, the corresponding object is stored in the :attr:`~nenupy.observation.parset.ParsetUser.analog_beams` attribute (which is a `list`).
Its parameters could be modified using the `list` and `dict` combination, e.g. `p.analog_beams[0]["target"] = "My target"`.
However, for convenience purpose, it could be easier to use the :meth:`~nenupy.observation.parset.ParsetUser.modify_analog_beam` method:

.. code-block:: python

    >>> from astropy.time import Time, TimeDelta

    >>> p.modify_analog_beam(
    >>>     anabeam_index=0,
    >>>     target="My fav target",
    >>>     simbadSearch="Cygnus X-3",
    >>>     trackingType="tracking",
    >>>     duration=TimeDelta(3600, format="sec"),
    >>>     startTime=Time("2022-01-01 12:00:00")
    >>> )
    >>> print(p)

    Observation.contactName=Alan
    Observation.name=My_observation
    Observation.contactEmail=alan.loh@obspm.fr
    Observation.nrAnabeams=1
    Observation.nrBeams=0
    Observation.topic=DEBUG

    Anabeam[0].target=My fav target
    Anabeam[0].simbadSearch=Cygnus X-3
    Anabeam[0].trackingType=tracking
    Anabeam[0].startTime=2022-01-01T12:00:00Z
    Anabeam[0].duration=3600s


While adding multiple *analog beams*, one could lose track of them.
There is a way to quickly display a short summary of the already registered *analog beams*:

.. code-block:: python

    >>> p.add_analog_beam()
    >>> p.analog_beams

    [<AnalogBeam(target=My fav target, index=0)>, <AnalogBeam(target=analog_beam_name, index=1)>]

Only the ``target`` value is printed as well as the ``index``, particularly critical when modifying, removing the *analog beam*, and when adding associated *numerical beams*.

Removing *analog beams* is achieved using the :meth:`~nenupy.observation.parset.ParsetUser.remove_analog_beam` method.
One can also quickly check that the corresponding object has properly been removed afterward:

.. code-block:: python

    >>> p.remove_analog_beam(anabeam_index=1)
    >>> p.analog_beams

    [<AnalogBeam(target=My fav target, index=0)>]



Configure numerical beams
-------------------------

:meth:`~nenupy.observation.parset.ParsetUser.add_numerical_beam`

.. code-block:: python

    >>> p.add_numerical_beam(
    >>>     anabeam_index=0,
    >>>     target="My fav target",
    >>>     useParentPointing=True,
    >>>     subbandList="[200..300]"
    >>> )
    >>> p.add_numerical_beam(
    >>>     anabeam_index=0,
    >>>     target="Away from target",
    >>>     useParentPointing=True,
    >>>     subbandList="[200..300]",
    >>>     decal_el=3
    >>> )

:meth:`~nenupy.observation.parset.ParsetUser.modify_numerical_beam`

:meth:`~nenupy.observation.parset.ParsetUser.remove_numerical_beam`


Syntax validation
-----------------

.. code-block:: python

    p.validate()


.. note::

    if

    .. code-block:: python

        >>> p.observation["contactEmail"] = "alan.loh&obspm.fr"
        >>> p.analog_beams[0].numerical_beams[0]["subbandList"] = "[1200..1300]"
        >>> p.validate()

        ERROR: Syntax error on 'alan.loh&obspm.fr' (key 'contactEmail').
        ERROR: Syntax error on '[1200..1300]' (key 'subbandList').


Parset_user file writing
------------------------

.. code-block:: python

    >>> p.write("my_obs.parset_user")


``Import files`` ``Stairway to Heaven`` tab in the `Virtual Control Room <https://gui-nenufar.obs-nancay.fr/>`_ 
