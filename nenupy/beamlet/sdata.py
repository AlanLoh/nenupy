#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
    *****
    SData
    *****

    :class:`~nenupy.beamlet.sdata.SData` is a class designed for
    embedding dynamic spectrum data sets. Many methods from the
    `nenupy` library return :class:`~nenupy.beamlet.sdata.SData`
    objects such as:

    * :meth:`~nenupy.beamlet.bstdata.BST_Data.select`
    * :meth:`~nenupy.crosslet.crosslet.Crosslet.beamform`
    * :meth:`~nenupy.simulation.hpxsimu.HpxSimu.time_profile`
    * :meth:`~nenupy.simulation.hpxsimu.HpxSimu.azel_transit`
    * :meth:`~nenupy.simulation.hpxsimu.HpxSimu.radec_transit`
    * :meth:`~nenupy.simulation.hpxsimu.HpxSimu.radec_tracking`

    It therefore provides a unified way of accessing, comparing
    and eventually plotting dynamic spectrum data sets throughout
    every `nenupy` manipulations. 

    :class:`~nenupy.beamlet.sdata.SData` are unit sensitive
    objets, therefore in addition it is worth importing
    :mod:`astropy.time` and :mod:`astropy.units` modules:

    >>> from nenupy.beamlet import SData
    >>> import numpy as np
    >>> from astropy.time import Time, TimeDelta
    >>> import astropy.units as u

    **Operations**

    All basic mathematical operations are enabled between
    :class:`~nenupy.beamlet.sdata.SData` instances (addition,
    subtraction, multiplication and division).

    In the following example, ``s1`` and ``s2`` are two identical
    :class:`~nenupy.beamlet.sdata.SData` instances with every
    data set to ``1.``. They define data taken at the same 
    times, frequencies and polarizations (``time``, ``freq`` and
    ``polar`` respectively):

    >>> dts = TimeDelta(np.arange(3), format='sec')
    >>> s1 = SData(
            data=np.ones((3, 5, 1)),
            time=Time('2020-04-01 12:00:00') + dts,
            freq=np.ones(5) * u.MHz,
            polar=['NE']
        )
    >>> s2 = SData(
            data=np.ones((3, 5, 1)),
            time=Time('2020-04-01 12:00:00') + dts,
            freq=np.ones(5) * u.MHz,
            polar=['NE']
        )

    Addition:

    >>> s_add = s1 + s2
    >>> s_add.amp
    array([[2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 2.]])

    Subtraction:

    >>> s_sub = s1 - s2
    >>> s_add.amp
    array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]])

    Multiplication:

    >>> s_mul = s1 * s2
    >>> s_add.amp
    array([[1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.]])

    Division:

    >>> s_div = s1 / s2
    >>> s_add.amp
    array([[1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.]])

    
    **Concatenation**
    
    Besides mathematical operations,
    :class:`~nenupy.beamlet.sdata.SData` objects may also be
    concatenated along the time or frequency axis for building up
    dataset purposes.
    
    In the following example, ``s1`` is the main instance, ``s2``
    has similar times but different frequencies and ``s3`` has
    similar frequencies but different times than ``s1``.

    >>> dts = TimeDelta(np.arange(2), format='sec')
    >>> s1 = SData(
            data=np.ones((2, 3, 1)),
            time=Time('2020-04-01 12:00:00') + dts,
            freq=np.arange(3) * u.MHz,
            polar=['NE']
        )
    >>> s2 = SData(
            data=np.ones((2, 3, 1)),
            time=Time('2020-04-01 12:00:00') + dts,
            freq=(10+np.arange(3)) * u.MHz,
            polar=['NE']
        )
    >>> s2 = SData(
            data=np.ones((2, 3, 1)),
            time=Time('2020-04-01 13:00:00') + dts,
            freq=np.arange(3) * u.MHz,
            polar=['NE']
        )

    Concatenation in frequency:

    >>> s_freq_concat = s1 & s2
    >>> s_freq_concat.shape
    (2, 6, 1)

    Concatenation in time:

    >>> s_time_concat = s1 | s3
    >>> s_time_concat.shape
    (4, 3, 1)

"""


__author__ = 'Alan Loh'
__copyright__ = 'Copyright 2020, nenupy'
__credits__ = ['Alan Loh']
__maintainer__ = 'Alan'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'Production'
__all__ = [
    'SData'
]


from astropy.time import Time
import astropy.units as u
import numpy as np
import logging
log = logging.getLogger(__name__)


# ============================================================= #
# --------------------------- SData --------------------------- #
# ============================================================= #
class SData(object):
    """ A class to handle dynamic spectrum data.

        :param data: Dynamic spectrum data (time, freq, polar)
        :type data: :class:`~numpy.ndarray`
        :param freq: Times
        :type time: :class:`~astropy.time.Time`
        :param time: Frequencies
        :type freq: :class:`~astropy.units.Quantity`
        :param polar: Polarizations
        :type polar: `str`, `list` or :class:`~numpy.ndarray`
    """

    def __init__(self, data, time, freq, polar):
        self.time = time
        self.freq = freq
        self.polar = polar
        self.data = data


    def __repr__(self):
        return self.data.__repr__()


    def __and__(self, other):
        """ Concatenate two SData object in frequency
        """
        if not isinstance(other, SData):
            raise TypeError(
                'Trying to concatenate something else than SData'
                )
        if list(map(str, self.polar)) != list(map(str, other.polar)):
            raise ValueError(
                'Inconsistent polar parameters'
                )
        if not all(self.time.isclose(other.time, self.time[1] - self.time[0])):
            # Assumes times are the same if their shift is less than dt
            raise ValueError(
                'Inconsistent time parameters'
                )

        if self.freq.max() < other.freq.min():
            new_data = np.hstack((self.data, other.data))
            new_time = self.time
            new_freq = np.concatenate((self.freq, other.freq))
        else:
            new_data = np.hstack((other.data, self.data))
            new_time = self.time
            new_freq = np.concatenate((other.freq, self.freq))

        return SData(
            data=new_data,
            time=new_time,
            freq=new_freq,
            polar=self.polar
        )


    def __or__(self, other):
        """ Concatenate two SData in time
        """
        if not isinstance(other, SData):
            raise TypeError(
                'Trying to concatenate something else than SData'
                )
        if list(map(str, self.polar)) != list(map(str, other.polar)):
            raise ValueError(
                'Inconsistent polar parameters'
                )
        if not all(self.freq == other.freq):
            raise ValueError(
                'Inconsistent freq parameters'
                )

        if self.time.max() < other.time.min():
            new_data = np.vstack((self.data, other.data))
            new_time = Time(np.concatenate((self.time, other.time)))
            new_freq = self.freq
        else:
            new_data = np.vstack((other.data, self.data))
            new_time = Time(np.concatenate((self.time, other.time)))
            new_freq = self.freq

        return SData(
            data=new_data,
            time=new_time,
            freq=new_freq,
            polar=self.polar
        )


    def __add__(self, other):
        """ Add two SData
        """
        if isinstance(other, SData):
            self._check_conformity(other)
            add = other.data
            if list(map(str, self.polar)) != list(map(str, other.polar)):
                raise ValueError(f"Polarization inconsistent {self.polar} vs {other.polar}")
        else:
            self._check_value(other)
            add = other

        return SData(
            data=self.data + add,
            time=self.time,
            freq=self.freq,
            polar=self.polar
        )


    def __sub__(self, other):
        """ Subtract two SData
        """
        if isinstance(other, SData):
            self._check_conformity(other)
            sub = other.data
            if list(map(str, self.polar)) != list(map(str, other.polar)):
                raise ValueError(f"Polarization inconsistent {self.polar} vs {other.polar}")
        else:
            self._check_value(other)
            sub = other

        return SData(
            data=self.data - sub,
            time=self.time,
            freq=self.freq,
            polar=self.polar
        )


    def __mul__(self, other):
        """ Multiply two SData
        """
        if isinstance(other, SData):
            self._check_conformity(other)
            mul = other.data
            new_polar = self.polar * other.polar
        else:
            self._check_value(other)
            mul = other
            new_polar = self.polar

        return SData(
            data=self.data * mul,
            time=self.time,
            freq=self.freq,
            polar=new_polar
        )


    def __truediv__(self, other):
        """ Divide two SData
        """
        if isinstance(other, SData):
            self._check_conformity(other)
            div = other.data
            new_polar = self.polar / other.polar
        else:
            self._check_value(other)
            div = other
            new_polar = self.polar

        return SData(
            data=self.data / div,
            time=self.time,
            freq=self.freq,
            polar=new_polar
        )

    def __getitem__(self, slice_tuple):
        """ (time, frequency) """
        # Expects an explicit tuple of length 2
        if not (isinstance(slice_tuple, tuple) and\
                (len(slice_tuple) == 3) and\
                all([isinstance(s, slice) for s in slice_tuple])
            ):
            raise IndexError("Only tuple of 3 slices allowed.")
        return SData(
            data=self.data[slice_tuple],
            time=self.time[slice_tuple[0]],
            freq=self.freq[slice_tuple[1]],
            polar=self.polar[slice_tuple[2]]
        )

    # --------------------------------------------------------- #
    # --------------------- Getter/Setter --------------------- #
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, d):
        ts, fs, ps = d.shape
        assert self.time.size == ts,\
            f'time axis inconsistent ({self.time.size} instead of {ts})'
        assert self.freq.size == fs,\
            f'frequency axis inconsistent ({self.freq.size} instead of {fs})'
        assert self.polar.size == ps,\
            f'polar axis inconsistent ({self.polar.size} instead of {ps})'
        self._data = d
        return


    @property
    def time(self):
        return self._time
    @time.setter
    def time(self, t):
        if not isinstance(t, Time):
            raise TypeError('Time object expected')
        self._time = t
        return


    @property
    def freq(self):
        return self._freq
    @freq.setter
    def freq(self, f):
        if not isinstance(f, u.Quantity):
            raise TypeError('Quantity object expected')
        self._freq = f
        return


    @property
    def polar(self):
        """Returns a list of polarizations"""
        return self._polar
    @polar.setter
    def polar(self, p) -> None:
        """Transforms the polarization in units."""
        # if isinstance(p, list):
        #     p = np.array(p)
        # if np.isscalar(p):
        #     p = np.array([p])
        if isinstance(p, (np.ndarray, list)):
            p = np.array([u.def_unit(pol) if (not isinstance(pol, u.UnitBase)) else pol for pol in p])
        elif np.isscalar(p):
            if isinstance(p, u.UnitBase):
                p = np.array([p])
            else:
                p = np.array([u.def_unit(p)])            
        else:
            raise TypeError("Unexpected polar type.")
        self._polar = p


    @property
    def shape(self):
        """
        """
        return self.data.shape
    

    @property
    def mjd(self):
        """ Return MJD dates

            :getter: Times in Modified Julian Days
            
            :type: :class:`~numpy.ndarray`
        """
        return self.time.mjd


    @property
    def jd(self):
        """ Return JD dates

            :getter: Times in Julian days
            
            :type: :class:`~numpy.ndarray`
        """
        return self.time.jd


    @property
    def datetime(self):
        """ Return datetime dates

            :getter: Times in datetime format
            
            :type: :class:`~numpy.ndarray`
        """
        return self.time.datetime


    @property
    def amp(self):
        """ Linear amplitude of the data

            :getter: Data in linear scaling
            
            :type: :class:`~numpy.ndarray`
        """
        return self.data.squeeze()

    
    @property
    def db(self):
        """ Convert the amplitude in decibels

            :getter: Data in decibels
            
            :type: :class:`~numpy.ndarray`
        """
        return 10 * np.log10(self.data.squeeze())


    @property
    def medianFProfile(self):
        """
            .. versionadded:: 1.1.0

        """
        return  SData(
            data=np.expand_dims(np.nanmedian(self.data, axis=0), 0),
            time=Time([(self.time[0] + (self.time[-1] - self.time[0])/2).isot]),
            freq=self.freq,
            polar=self.polar
        )


    @property
    def medianTProfile(self):
        """
            .. versionadded:: 1.1.0

        """
        return  SData(
            data=np.expand_dims(np.nanmedian(self.data, axis=1), 1),
            time=self.time,
            freq=[np.nanmean(self.freq.to(u.MHz).value)]*u.MHz,
            polar=self.polar
        )
    


    @property
    def background(self):
        """
        """
        med_spec = np.nanmedian(self.data, axis=0)
        med_prof = np.nanmedian(self.data, axis=1)
        
        # med_prof is affected by nanmedian while dedispersing
        # because signal and dispersion delay are frequency-dependent
        # the median is thus artificially decreasing with time
        nans = np.isnan(self.data[:, 0, 0]) # lowest frequency is the most affected
        if any(nans):
            prof_nans = np.isnan(med_prof[:, 0])
            x_tot = np.arange(med_prof.size)
            max_id = np.argmax(nans)
            fit = np.polyfit(
                x_tot[~prof_nans][max_id:],
                med_prof[~prof_nans][max_id:, 0],
                3
            )
            lastpart = np.poly1d(fit)
            med_prof[max_id:, :] /= lastpart(x_tot[max_id:])[:, np.newaxis]
            med_prof[nans, :] *= np.median(med_prof[~nans])

        bg = np.ones_like(self.data)
        bg *= med_spec[np.newaxis, :, :]
        bg *= med_prof[:, np.newaxis, :] / np.nanmax(med_prof)
        return SData(
            data=bg,
            time=self.time,
            freq=self.freq,
            polar=self.polar
        )


    @property
    def fbackground(self):
        """
        """
        med_spec = np.nanmedian(self.data, axis=0)
        bg = np.ones_like(self.data)
        bg *= med_spec[np.newaxis, :, :]
        return SData(
            data=bg,
            time=self.time,
            freq=self.freq,
            polar=self.polar
        )


    # --------------------------------------------------------- #
    # ------------------------ Methods ------------------------ #
    def plot(self, fig=None, ax=None, polarization=None, figname=None, db=True, **kwargs):
        """
            kwargs keys: cmap, title, show_colorbar, cblabel, figsize, altaza, vmin, vmax
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if polarization is None:
            pol_idx = 0
        else:
            try:
                pol_idx = np.argwhere(np.array(list(map(str, self.polar))) == polarization)[0, 0]
            except IndexError:
                print(f"Warning, no '{polarization}' polarization recorded (select from {self.polar}).")
                pol_idx = 0

        data = self.data[..., pol_idx]
        if np.iscomplexobj(data):
            log.warning("Complex array found, only the real part is plotted.")
            data = data.real

        dynspec = 10*np.log10(data).T if db else data.T

        # Make sure everything is correctly set up
        if 'cmap' not in kwargs.keys():
            kwargs['cmap'] = 'YlGnBu_r'
        if 'title' not in kwargs.keys():
            kwargs['title'] = str(self.polar[pol_idx])
        if 'cblabel' not in kwargs.keys():
            kwargs['cblabel'] = 'dB' if db else 'Amplitude' 
        if 'figsize' not in kwargs.keys():
            kwargs['figsize'] = (15, 10)
        
        if ax is None:
            # Create a full figure from the start
            fig = plt.figure(figsize=kwargs['figsize'])
            ax = fig.add_subplot()
        else:
            # Fill up the input ax with the plot
            pass

        if len(dynspec.shape) == 1:
            if dynspec.size == self.datetime.size:
                ax.plot(
                    self.datetime,
                    dynspec
                )
                if 'altaza' in kwargs.keys():
                    ptimes = kwargs['altaza'].pointingTimes
                    for ptime in ptimes:
                        if (ptime < self.datetime[0]) or (ptime > self.datetime[-1]):
                            continue
                        ax.axvline(ptime.datetime, linestyle='-.', color='black')
                ax.ylim((kwargs.get('vmin', None), kwargs.get('vmax', None)))
                ax.set_xlabel(
                    f'Time (UTC from {self.time[0].isot})'
                )
                ax.set_ylabel(kwargs['cblabel'])
            elif dynspec.size == self.freq.size:
                ax.plot(
                    self.freq.to(u.MHz).value,
                    dynspec
                )
                ax.set_ylim((kwargs.get('vmin', None), kwargs.get('vmax', None)))
                ax.set_xlabel('Frequency (MHz)')
                ax.set_ylabel(kwargs['cblabel'])
        else:
            if 'vmin' not in kwargs.keys():
                kwargs['vmin'] = np.nanpercentile(dynspec, 5)
            elif kwargs['vmin'] is None:
                kwargs['vmin'] = np.nanpercentile(dynspec, 5)
            else:
                pass
            if 'vmax' not in kwargs.keys():
                kwargs['vmax'] = np.nanpercentile(dynspec, 95)
            elif kwargs['vmax'] is None:
                kwargs['vmax'] = np.nanpercentile(dynspec, 95)
            else:
                pass
            im = ax.pcolormesh(
                self.datetime,
                self.freq.to(u.MHz).value,
                dynspec,
                shading='auto',
                cmap=kwargs['cmap'],
                vmin=kwargs['vmin'],
                vmax=kwargs['vmax']
            )
            if 'altaza' in kwargs.keys():
                    ptimes = kwargs['altaza'].pointingTimes
                    for ptime in ptimes:
                        if (ptime < self.datetime[0]) or (ptime > self.datetime[-1]):
                            continue
                        ax.axvline(ptime.datetime, linestyle='-.', color='black')
            if kwargs.get("show_colorbar", True):
                cbar = plt.colorbar(im, pad=0.03)#format='%.1e')
                cbar.set_label(kwargs['cblabel'])

            if kwargs.get("overlay", None) is not None:
                #ax = plt.gca()
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                overlay_time, overlay_frequency, overlay_values = kwargs["overlay"]
                ax.pcolor(
                    overlay_time.datetime,
                    overlay_frequency.to(u.MHz).value,
                    overlay_values,
                    #color="tab:red",
                    #edgecolor="tab:red",
                    #linewidth=0.5,
                    hatch='/',
                    alpha=0.
                )
                # plt.pcolormesh(
                #     overlay_time.datetime,
                #     overlay_frequency.to(u.MHz).value,
                #     overlay_values,
                #     cmap='Reds',
                #     alpha=0.3
                # )
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

            ax.set_xlabel(
                f'Time (UTC from {self.time[0].isot})'
            )
            ax.set_ylabel('Frequency (MHz)')
        ax.set_title(kwargs['title'])

        # Set x axis labels        
        locator = ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(locator, show_offset=False)
        )

        # Save or show
        if not (ax is None):
            return
        elif (figname is None) or (figname == ""):
            plt.show()
        else:
            fig.savefig(
                figname,
                dpi=300,
                transparent=True,
                bbox_inches='tight'
            )
        plt.close('all')


    # --------------------------------------------------------- #
    # ----------------------- Internal ------------------------ #
    def _check_conformity(self, other):
        """ Checks that other if of same type, same time, 
            frequency ans Stokes parameters than self
        """
        # if self.polar != other.polar:
        #     raise ValueError(
        #         'Different polarization parameters'
        #         )

        if self.data.shape != other.data.shape:
            raise ValueError(
                'SData objects do not have the same dimensions'
                )

        if all(self.time != other.time):
            raise ValueError(
                'Not the same times'
                )

        if all(self.freq != other.freq):
            raise ValueError(
                'Not the same frequencies'
                )

        return


    def _check_value(self, other):
        """ Checks that other can be operated with self if
            other is not a SData object
        """
        if isinstance(other, np.ndarray):
            if other.shape != self.data.shape:
                raise ValueError(
                    'Shape mismatch'
                )
        elif isinstance(other, (int, float)):
            pass
        else:
            raise Exception(
                'Operation unknown with {}'.format(type(other))
            )
        return
# ============================================================= #


