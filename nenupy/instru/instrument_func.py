#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

"""
"""

__author__ = ['Alan Loh']
__copyright__ = 'Copyright 2019, nenupy'
__credits__ = ['Alan Loh']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Alan Loh'
__email__ = 'alan.loh@obspm.fr'
__status__ = 'WIP'
__all__ = ['cor_azel',
           'ant_eff_area',
           'ma_eff_area',
           'nenufar_eff_area',
           'sky_temperature',
           'inst_temperature',
           'nenufar_sefd',
           'nenufar_plot_sefd',
           'nenufar_bf_sensitivity',
           'nenufar_im_sensitivity',
           'ma_resol',
           'nenufar_fov',
           'nenufar_resol',
           'nenufar_im_rate',
           'nenufar_bf_rate',
           'nenufar_baselines',
           'nenufar_uvw_snap',
           'nenufar_uvw_track',
           'uvw_profile',
           'nenufar_psf']


import astropy.units as u
from astropy import constants as const
from nenupy.beam.antenna import miniarrays
import numpy as np


# =================================================================================== #
# ------------------------------------ cor_azel ------------------------------------- #
# =================================================================================== #
def cor_azel(az, el):
    """
    """
    from scipy.io import readsav
    from scipy.interpolate import interp1d
    aa = readsav('/Users/aloh/Desktop/cor_azel.sav')
    daz_interp = interp1d(aa['x_cor'], aa['daz_cor'])
    del_interp = interp1d(aa['x_cor'], aa['del_cor'])

    az_cor = az + daz_interp(az)/60.
    el_cor = el + del_interp(az)/60.
    return az_cor, el_cor


# =================================================================================== #
# ---------------------------------- ant_eff_area ----------------------------------- #
# =================================================================================== #
def ant_eff_area(freq):
    """ Effective area of a NenuFAR single antenna

        Parameters
        ----------
        - freq : float
            Frequency in MHz

        Returns
        -------
        - eff_area : float
            Effective area in m^2
    """
    k = 3 # NenuFAR
    freq *= u.MHz
    wavelength = const.c.to(u.m/u.s).value / freq.to(u.Hz).value
    eff_area = wavelength**2 / k 
    return eff_area * (u.m**2)


# =================================================================================== #
# ----------------------------------- mr_eff_area ----------------------------------- #
# =================================================================================== #
def ma_eff_area(freq, ant=None, plot=False):
    """ Effective area of a NenuFAR mini-array

        Parameters
        ----------
        - freq : float
            Frequency in MHz
        - ant : list, np.ndarray
            Antenna indices to use for the calculation (None=all antennas)
        - plot : bool
            Plot the mini-array effective area

        Returns
        -------
        - eff_area : float
            Effective area in m^2
    """
    antpos = miniarrays.antpos
    if ant is not None:
        ant = np.array(ant, dtype=np.int32)
        antpos = antpos[ant].reshape((ant.size, 3))
    grid_elements = 1000 
    grid = np.zeros((grid_elements, grid_elements), dtype=np.int32)

    ant_eff_area_radius = np.sqrt(ant_eff_area(freq=freq).value / np.pi)

    x_grid = np.linspace(antpos[:, 0].min() - ant_eff_area_radius,
        antpos[:, 0].max() + ant_eff_area_radius,
        grid_elements)
    dx = x_grid[1] - x_grid[0]
    y_grid = np.linspace(antpos[:, 1].min() - ant_eff_area_radius,
        antpos[:, 1].max() + ant_eff_area_radius,
        grid_elements)
    dy = y_grid[1] - y_grid[0]
    xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)

    for xi, yi, zi in antpos:
        dist = np.sqrt((xx_grid - xi)**2. + (yy_grid - yi)**2.)
        grid[dist <= ant_eff_area_radius] += 1

    if plot:
        import pylab as plt
        plt.imshow(grid, origin='lower', cmap='Blues')
        plt.show()

    grid[grid != 0] = 1
    return (grid * dx * dy).sum() * (u.m**2)


# =================================================================================== #
# -------------------------------- nenufar_eff_area --------------------------------- #
# =================================================================================== #
def nenufar_eff_area(freq, ant=None, ma=None):
    """ Effective area of NenuFAR

        Parameters
        ----------
        - freq : float
            Frequency in MHz
        - ant : list, np.ndarray
            Antenna indices to use for the calculation (None=all antennas)
        - ma : list, np.ndarray
            Mini-arrays indices to use for the calculation (None=all mini-arrays)

        Returns
        -------
        - eff_area : float
            Effective area in m^2

        Examples
        --------
        - Whole array, all antennas within each MA:
            `>>> nenufar_eff_area(freq=50, ant=None, ma=None)`
        - Whole array, some antennas within each MA:
            `>>> nenufar_eff_area(freq=50, ant=[2, 4, 10], ma=None)`
        - Some mini arrays selected, all antennas within each MA:
            `>>> nenufar_eff_area(freq=50, ant=None, ma=[0, 1, 26])`
        - Some mini arrays selected, some antennas within each MA:
            `>>> nenufar_eff_area(freq=50, ant=[None, [1,2,3], 9], ma=[0, 1, 26])`
    """
    if ma is not None:
        ma = np.array(ma, dtype=np.int32)
    else:
        mapos = miniarrays.ma[:, 2:5]
        ma = np.arange(mapos.shape[0], dtype=np.int32)
    
    if ant is not None:
        ant = np.array(ant)
        if (ant.size == 1) or (ma is None):
            eff_area = ma_eff_area(freq=freq, ant=ant).value * ma.size
        elif ant.size == ma.size:
            eff_area = 0
            for a in ant:
                eff_area += ma_eff_area(freq=freq, ant=a).value * ma.size 
        else:
            print('Didnt understand ant / ma combinations...')
    else:
        eff_area = ma_eff_area(freq=freq, ant=ant).value * ma.size

    return eff_area * (u.m**2)


# =================================================================================== #
# -------------------------------- sky_temperature ---------------------------------- #
# =================================================================================== #
def sky_temperature(freq):
    """ Sky noise temperature / sky brightness temperature
        https://www.astron.nl/radio-observatory/astronomers/lofar-imaging-capabilities-sensitivity/sensitivity-lofar-array/sensiti

        Parameters
        ----------
        - freq : float
            Frequency in MHz

        Returns
        -------
        - t_sky : float
            Sky temperature in K
    """
    freq *= u.MHz
    wavelength = const.c.to(u.m/u.s).value / freq.to(u.Hz).value
    t0 = 60. * u.K # +/- 20 K for Galactic latitudes between 10 and 90 degrees
    tsky = t0 * wavelength**2.55
    return tsky


# =================================================================================== #
# -------------------------------- inst_temperature --------------------------------- #
# =================================================================================== #
def inst_temperature(freq):
    """ Instrumental noise temperature

        Parameters
        ----------
        - freq : float
            Frequency in MHz

        Returns
        -------
        - t_sky : float
            Sky temperature in K
    """
    lna_sky = np.array([5.0965,2.3284,1.0268,0.4399,0.2113,0.1190,0.0822,0.0686,0.0656,0.0683,
                        0.0728,0.0770,0.0795,0.0799,0.0783,0.0751,0.0710,0.0667,0.0629,0.0610,
                        0.0614,0.0630,0.0651,0.0672,0.0694,0.0714,0.0728,0.0739,0.0751,0.0769,
                        0.0797,0.0837,0.0889,0.0952,0.1027,0.1114,0.1212,0.1318,0.1434,0.1562,
                        0.1700,0.1841,0.1971,0.2072,0.2135,0.2168,0.2175,0.2159,0.2121,0.2070,
                        0.2022,0.1985,0.1974,0.2001,0.2063,0.2148,0.2246,0.2348,0.2462,0.2600,
                        0.2783,0.3040,0.3390,0.3846,0.4425,0.5167,0.6183,0.7689,1.0086,1.4042,2.0732])
    freqs = np.arange(71) + 15 # MHz
    tsky = sky_temperature(freq=freq)
    tinst = tsky * lna_sky[ np.abs(freqs - freq).argmin() ]
    return tinst


# =================================================================================== #
# ---------------------------------- nenufar_sefd ----------------------------------- #
# =================================================================================== #
def nenufar_sefd(freq, ant=None, ma=None):
    """ System Equivalent Flux Density

        Parameters
        ----------
        - freq : float
            Frequency in MHz
        - ant : list, np.ndarray
            Antenna indices to use for the calculation (None=all antennas)
        - ma : list, np.ndarray
            Mini-arrays indices to use for the calculation (None=all mini-arrays)

        Returns
        -------
        - sefd : float
            SEFD in Jy
    """
    efficiency = 1.
    aeff = nenufar_eff_area(freq=freq, ant=ant, ma=ma)
    tsky = sky_temperature(freq=freq)
    tinst = inst_temperature(freq=freq)
    tsys = tsky + tinst
    sefd = 2 * efficiency * const.k_B * tsys / aeff
    return sefd.to(u.Jy)


# =================================================================================== #
# -------------------------------- nenufar_plot_sefd -------------------------------- #
# =================================================================================== #
def nenufar_plot_sefd(ant=None, ma=None):
    """ Plot the System Equivalent Flux Density
        over NenuFAR ferquency range

        Parameters
        ----------
        - ant : list, np.ndarray
            Antenna indices to use for the calculation (None=all antennas)
        - ma : list, np.ndarray
            Mini-arrays indices to use for the calculation (None=all mini-arrays)
    """
    import pylab as plt
    freqs = np.linspace(20, 90, 50)
    sefds = np.zeros(freqs.size)
    for i, freq in enumerate(freqs):
        sefds[i] = nenufar_sefd(freq=freq, ant=ant, ma=ma).value
    plt.plot(freqs, sefds)
    plt.show()
    return


# =================================================================================== #
# ------------------------------ nenufar_bf_sensitivity ----------------------------- #
# =================================================================================== #
def nenufar_bf_sensitivity(freq, ant=None, ma=None, dt=1, df=1):
    """ Compute the sensitivity equation / radiometer equation 
        Thermal noise

        Parameters
        ----------
        - freq : float
            Frequency in MHz
        - ant : list, np.ndarray
            Antenna indices to use for the calculation (None=all antennas)
        - ma : list, np.ndarray
            Mini-arrays indices to use for the calculation (None=all mini-arrays)
        - dt : float
            Integration time in sec
        - df : float 
            Bandwidth in MHz

        Returns
        -------
        - sensitivity : float
            Sensitivity in Jy
    """
    dt *= u.s

    df *= u.MHz

    mapos = miniarrays.ma[:, 2:5]
    if ma is not None:
        ma = np.array(ma, dtype=np.int32)
        mapos = mapos[ma].reshape((ma.size, 3))
    nant = mapos.shape[0]

    sefd = nenufar_sefd(freq=freq, ant=ant, ma=ma)
    sensitivity = sefd / np.sqrt(dt.value * df.to(u.Hz).value)
    return sensitivity


# =================================================================================== #
# ------------------------------ nenufar_im_sensitivity ----------------------------- #
# =================================================================================== #
def nenufar_im_sensitivity(freq, ant=None, ma=None, dt=1, df=1):
    """ Compute the sensitivity equation / radiometer equation 

        Parameters
        ----------
        - freq : float
            Frequency in MHz
        - ant : list, np.ndarray
            Antenna indices to use for the calculation (None=all antennas)
        - ma : list, np.ndarray
            Mini-arrays indices to use for the calculation (None=all mini-arrays)
        - dt : float
            Integration time in sec
        - df : float 
            Bandwidth in MHz

        Returns
        -------
        - sensitivity : float
            Sensitivity in Jy
    """
    dt *= u.s

    df *= u.MHz

    mapos = miniarrays.ma[:, 2:5]
    if ma is not None:
        ma = np.array(ma, dtype=np.int32)
        mapos = mapos[ma].reshape((ma.size, 3))
    nant = mapos.shape[0]

    sefd = nenufar_sefd(freq=freq, ant=ant, ma=ma)
    sensitivity = sefd / np.sqrt(nant*(nant-1) * 2 * dt.value * df.to(u.Hz).value)
    return sensitivity


# =================================================================================== #
# ------------------------------------ ma_resol ------------------------------------- #
# =================================================================================== #
def ma_resol(freq):
    """ Mini-Array PSF

        Parameters
        ----------
        - freq : float
            Frequency in MHz
        
        Returns
        -------
        - psf : float
            Point Spread Function in degrees
    """
    freq *= u.MHz
    wavelength = const.c.to(u.m/u.s) / freq.to(u.Hz)

    ma_diameter = 25 * u.m
    psf = wavelength / ma_diameter
    return (psf.value * u.rad).to(u.deg)


# =================================================================================== #
# ----------------------------------- nenufar_fov ----------------------------------- #
# =================================================================================== #
def nenufar_fov(freq):
    """ NenuFAR field of view

        Parameters
        ----------
        - freq : float
            Frequency in MHz

        Returns
        -------
        - fov : float
            Field of view in deg^2
    """
    fwhm = ma_resol(freq=freq)
    radius = fwhm / 2 
    fov = np.pi * radius**2
    return fov


# =================================================================================== #
# ---------------------------------- nenufar_resol ---------------------------------- #
# =================================================================================== #
def nenufar_resol(freq, ma=None):
    """ NenuFAR PSF

        Parameters
        ----------
        - freq : float
            Frequency in MHz
        - ma : list, np.ndarray
            Mini-arrays indices to use for the calculation (None=all mini-arrays)
        
        Returns
        -------
        - psf : float
            Point Spread Function in degrees
    """
    freq *= u.MHz
    wavelength = const.c.to(u.m/u.s) / freq.to(u.Hz)

    mapos = miniarrays.ma[:, 2:5]
    if ma is not None:
        ma = np.array(ma, dtype=np.int32)
        mapos = mapos[ma].reshape((ma.size, 3))
    
    dist = 0
    for i in range(mapos.shape[0]):
        for j in range(i, mapos.shape[0]-1):
            dd = np.sqrt((mapos[i, 0]-mapos[j, 0])**2 + (mapos[i, 1]-mapos[j, 1])**2)
            dist = dd if dd > dist else dist

    psf = wavelength / (dist * u.m)
    return (psf.value * u.rad).to(u.deg)


# =================================================================================== #
# --------------------------------- nenufar_im_rate --------------------------------- #
# =================================================================================== #
def nenufar_im_rate(ma=96, nch=64, dt=1, bwidth=75, tobs=3600):
    """ Imaging data rate
        
        Parameters
        ----------
        - ma
    """
    bwidth *= u.MHz
    bw_sb = 0.1953125 * u.MHz
    n_subband = (bwidth / bw_sb).value
    n_subband = np.round(n_subband) if np.round(n_subband) < 768 else 768

    sb_rate = 8. * 4. * nch * (ma*(ma - 1)/2. + ma)/dt
    rate = sb_rate * n_subband
    volume = rate * tobs
    return volume


# =================================================================================== #
# --------------------------------- nenufar_bf_rate --------------------------------- #
# =================================================================================== #
def nenufar_bf_rate():
    return


# =================================================================================== #
# -------------------------------- nenufar_baselines -------------------------------- #
# =================================================================================== #
def nenufar_baselines(ma=None, auto=False):
    """ Get all the NenuFAR baselines

        Parameters
        ----------
        - ma : list, np.ndarray, str
            Mini-arrays indices to use for the calculation (None=all 96 mini-arrays)
            'core': 96 core mini-arrays
            'remote': core + 6 remote
        - auto : bool
            Take into account auto-correlations or not

        Returns
        -------
        - baselines
    """
    mapos = miniarrays.all_ma
    mapos2 = miniarrays.all_remote
    if isinstance(ma, str):
        if ma.lower() == 'core':
            pass
        elif ma.lower() == 'remote':
            mapos = np.vstack((mapos, mapos2))
        else:
            pass
    elif ma is None:
        pass
    else:
        ma = np.array(ma, dtype=np.int32) 
        mapos = mapos[ma].reshape((ma.size, 3))
    n_ant = mapos.shape[0]
    baselines = []
    for i in range(0, n_ant-int(not auto)):
        for j in range(i+int(not auto), n_ant):
            ant1 = mapos[i]
            ant2 = mapos[j]
            baselines.append( (ant1, ant2) )
    return np.array(baselines)


# =================================================================================== #
# ------------------------------------- plot_uv ------------------------------------- #
# =================================================================================== #
def plot_uv(uvw, title='', unit='lambda'):
    import pylab as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(figsize=(7, 7))
    u = np.hstack((-uvw[:, 0], uvw[:, 0]))
    v = np.hstack((-uvw[:, 1], uvw[:, 1]))
    hbins = ax.hexbin(x=u,
                      y=v,
                      C=None,
                      cmap='inferno',#'bone_r',
                      mincnt=1,
                      bins=None,
                      gridsize=200,
                      vmin=0.1,
                      xscale='linear',
                      yscale='linear',
                      edgecolors='face',
                      linewidths=0,
                      vmax=None,)#,
                      # C=None,
                      # gridsize=200,
                      # mincnt=1,
                      # bins='log',#None,
                      # xscale='linear',
                      # yscale='linear',
                      # cmap='bone_r',
                      # edgecolors='face',
                      # vmin=0.1,#0,
                      # linewidths=0,
                      # vmax=None)
    ax.margins(0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    cb = fig.colorbar(hbins, cax=cax)
    cb.set_label('Histogram')
    ax.set_title(title)
    if unit.lower == 'lambda':
        ax.set_xlabel('u ($\\lambda$)')
        ax.set_ylabel('v ($\\lambda$)')
    else:
        ax.set_xlabel('u (m)')
        ax.set_ylabel('v (m)')
    lim = np.max((np.abs(ax.get_xlim()).max(), np.abs(ax.get_ylim()).max()))

    # Show circles of resolutions:
    lambdas = np.linspace(0, lim, 4)
    for i, lamb in enumerate(lambdas[1:]):
        col = '0.7'# 'white' if i<2 else 'black'
        resol = np.degrees(1. / lamb) #* 3600
        circle = plt.Circle((0, 0), lamb, color=col, linestyle='-', fill=False, linewidth=1)#, alpha=0.5)
        ax.add_artist(circle)
        # ax.text(0, -lamb, '{:.2f} deg'.format(resol), horizontalalignment='center', verticalalignment='bottom', color='black')
        ax.text(lamb*np.cos(-np.pi/4), lamb*np.sin(-np.pi/4), '{:.2f} deg'.format(resol),
            horizontalalignment='right',
            verticalalignment='bottom',
            color='black',
            bbox=dict(boxstyle="round",
                   edgecolor='none',
                   facecolor='white',
                   alpha = 0.8,
                   )
            )

        # print(arcesc_resol / 60)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    plt.show()
    plt.close('all')

# =================================================================================== #
# --------------------------------- nenufar_uvw_snap -------------------------------- #
# =================================================================================== #
def nenufar_uvw_snap(time='now', freq=50, source='North Celestial Pole', ma=None, auto=False, plot=False, unit='lambda'):
    """ Compute the snapshot (u,v,w) plane at time `time

        Parameters
        ----------
        - time : str or `astropy.Time`
            UTC time of the snapshot
            e.g.: `time='2019-03-15 12:30:45'` or `time='now'` 
        - freq : float
            Frequency in MHz
        - source : str or tuple
            RA/Dec coordinates of the pointing
            e.g.: `source='Cyg A'` or `source=(299.86815263, 40.73391583)`
        - ma : list, np.ndarray
            Mini-arrays indices to use for the calculation (None=all 96 mini-arrays)
        - auto : bool
            Take into account auto-correlations or not
        - plot : bool
            Plot the (u, v) distribution

        Returns
        -------
        - baselines
    """
    from nenupy.astro import getSrc, getTime

    nenufar_lat = miniarrays.nenufarloc.lat.deg
    nenufar_lon = miniarrays.nenufarloc.lon.deg
    radec_src = getSrc(source)
    ra, dec = radec_src.ra.deg, radec_src.dec.deg 
    time = getTime(time)
    ha = np.radians(time.sidereal_time('apparent', nenufar_lon).deg - ra)

    baselines = nenufar_baselines(ma=ma, auto=auto)
    n_baselines = len(baselines)
    # uvw = np.zeros( (n_baselines, 3) )
    
    nenufar_lat = np.radians(nenufar_lat)
    # E, N, U coordinates to x, y, z (https://web.njit.edu/~gary/728/Lecture6.html)
    trans1 = np.matrix([ [0, -np.sin(nenufar_lat), np.cos(nenufar_lat)],
                         [1,                    0,                   0],
                         [0,  np.cos(nenufar_lat), np.sin(nenufar_lat)]])
    trans2 = np.matrix([[             np.sin(ha),              np.cos(ha),           0],
                        [-np.sin(dec)*np.cos(ha),  np.sin(dec)*np.sin(ha), np.cos(dec)],
                        [ np.cos(dec)*np.cos(ha), -np.cos(dec)*np.sin(ha), np.sin(dec)]])
    
    # for k in range( n_baselines ):
    #     xyz_prime = baselines[k][1] - baselines[k][0]
    #     xyz = trans1 * np.matrix( [ [xyz_prime[0]], [xyz_prime[1]], [xyz_prime[2]] ] )
    #     # Go from CE, east, NCP to u, v, w
    #     temp = trans2 * xyz
    #     uvw[k, :] = np.squeeze(temp) * freq*1.e6 / const.c.value 

    xyz = np.dot(baselines[:, 1, :] - baselines[:, 0, :], trans1)
    uvw = np.dot(xyz, trans2) * freq*1.e6 / const.c.value if unit.lower()=='lambda' else np.dot(xyz, trans2)

    if plot:
        plot_uv(uvw, title='Instantaneous UV coverage ({} MHz)'.format(freq), unit=unit)
        return
    else:
        return np.array(uvw)


# =================================================================================== #
# -------------------------------- nenufar_uvw_track -------------------------------- #
# =================================================================================== #
def nenufar_uvw_track(start, stop, dt=1, freq=50, source='North Celestial Pole', ma=None, auto=False, plot=False, unit='lambda'):
    """
    """
    from nenupy.astro import getSrc, getTime
    from astropy.time import TimeDelta
    
    nenufar_lat = miniarrays.nenufarloc.lat.deg
    nenufar_lon = miniarrays.nenufarloc.lon.deg
    radec_src = getSrc(source)
    ra, dec = radec_src.ra.deg, radec_src.dec.deg 

    start = getTime(start)
    stop = getTime(stop)
    dt = TimeDelta(dt, format='sec')
    ntimes = int(np.floor((stop-start)/dt))

    baselines = nenufar_baselines(ma=ma, auto=auto)
    n_baselines = len(baselines)
    nenufar_lat = np.radians(nenufar_lat)
    trans1 = np.matrix([ [0, -np.sin(nenufar_lat), np.cos(nenufar_lat)],
                         [1,                    0,                   0],
                         [0,  np.cos(nenufar_lat), np.sin(nenufar_lat)]])
    

    for i in range(ntimes):
        time = getTime(start+i*dt)
        ha = np.radians(time.sidereal_time('apparent', nenufar_lon).deg - ra)
        trans2 = np.matrix([[             np.sin(ha),              np.cos(ha),           0],
                            [-np.sin(dec)*np.cos(ha),  np.sin(dec)*np.sin(ha), np.cos(dec)],
                            [ np.cos(dec)*np.cos(ha), -np.cos(dec)*np.sin(ha), np.sin(dec)]])
        xyz = np.dot(baselines[:, 1, :] - baselines[:, 0, :], trans1)
        # tmp_uvw = np.dot(xyz, trans2) * freq*1.e6 / const.c.value
        tmp_uvw = np.dot(xyz, trans2) * freq*1.e6 / const.c.value if unit.lower()=='lambda' else np.dot(xyz, trans2)

        # tmp_uvw = nenufar_uvw_snap(time=start+i*dt, freq=freq, source=source, ma=ma, auto=auto)
        
        if not 'uvw' in locals():
            uvw = tmp_uvw.copy()
        else:
            uvw = np.vstack((uvw, tmp_uvw))

    if plot:
        plot_uv(uvw, title='UV coverage ({0:.1f} sec, {1:} MHz)'.format((i+1)*dt.value, freq), unit=unit)
        return
    else:
        return uvw


# =================================================================================== #
# ----------------------------------- uvw_profile ----------------------------------- #
# =================================================================================== #
def uvw_profile(uvw):
    """
    """
    import pylab as plt
    from astropy.modeling import models, fitting

    u = np.hstack((-uvw[:, 0], uvw[:, 0]))
    v = np.hstack((-uvw[:, 1], uvw[:, 1]))
    pos_v = v > 0.
    u = u[pos_v]
    v = v[pos_v]
    uv_dist = np.sqrt(u**2 + v**2)
    uv_ang = np.degrees( np.arccos(u/uv_dist) )

    # Radial profile
    dist = []
    density = []
    ddist = 3 # lambda
    min_dists = np.arange(uv_dist.min(), uv_dist.max(), ddist)
    max_dists = np.arange(uv_dist.min()+ddist, uv_dist.max()+ddist, ddist)
    for min_dist, max_dist in zip(min_dists, max_dists):
        mask = (uv_dist>=min_dist) & (uv_dist<=max_dist)
        dist.append( np.mean([min_dist, max_dist]) )
        density.append( u[mask].size )
    density = np.array(density)/np.max(density)

    plt.bar(dist, height=density, width=ddist, edgecolor='black', linewidth=0.5)
    try:
        gaussian_init = models.Gaussian1D(amplitude=1., mean=0, stddev=0.68 * max(dist),
            bounds={"mean": (0., 0.)})
        fit_gaussian = fitting.LevMarLSQFitter()
        gaussian = fit_gaussian(gaussian_init, dist, density)
        x = np.linspace(min(dist), max(dist), 100)
        plt.plot(x, gaussian(x), linestyle=':', color='black', linewidth=1, label='HWHM = {:.2f} $\\lambda$'.format(gaussian.stddev.value))
        plt.legend()
    except:
        pass
    plt.title('Radial profile')
    plt.xlabel('UV distance ($\\lambda$)')
    plt.ylabel('Density') 
    plt.show()

    # Azimuthal profile
    ang = []
    density = []
    dang = 5
    min_angs = np.arange(0, 180, dang)
    max_angs = np.arange(0+dang, 180+dang, dang)
    for min_ang, max_ang in zip(min_angs, max_angs):
        mask = (uv_ang>=min_ang) & (uv_ang<=max_ang)
        ang.append( np.mean([min_ang, max_ang]) )
        density.append( u[mask].size )
    plt.bar(ang, height=np.array(density)/np.max(density), width=dang, edgecolor='black', linewidth=0.5)
    plt.title('Azimuthal profile')
    plt.xlabel('Azimuth (deg)')
    plt.ylabel('Density') 
    plt.show()

    return


# =================================================================================== #
# -------------------------------- nenufar_psf -------------------------------- #
# =================================================================================== #
# def nenufar_psf(uvw, plot=True):
#     """
#     """
#     u = np.hstack((-uvw[:, 0], uvw[:, 0]))
#     v = np.hstack((-uvw[:, 1], uvw[:, 1]))
    
#     nbins = 1000
#     size = nbins
#     uv_grid = np.zeros( (size, size) )
#     for i in range( u.size ):
#         uID = int( np.round( (u[i]+size/2)*1 ) )
#         vID = int( np.round( (v[i]+size/2)*1 ) )
#         try:
#             uv_grid[uID, vID] += 1
#         except IndexError:
#             pass   

#     # nbins = 500
#     # uv_grid, hx, hy = np.histogram2d(x=u, y=v, bins=nbins)

#     psf = np.fft.fft2(uv_grid)
#     psf = np.fft.fftshift(psf)
#     psf = np.abs(psf)#**2
#     # psf /= psf.max()
#     psf = np.log10(psf)*10.0
    
#     if plot:
#         import pylab as plt
#         ext = [nbins/2-100, nbins/2+100, nbins/2-100, nbins/2+100]
#         plt.imshow(psf, origin='lower', cmap='bone_r', vmin=-20, vmax=psf.max(), extent=ext)
#         plt.show()
#         plt.close('all')
#     else:
#         return psf


class AA_filter:
    """
    Anti-Aliasing filter
    
    Keyword arguments for __init__:
    filter_half_support --- Half support (N) of the filter; the filter has a full support of N*2 + 1 taps
    filter_oversampling_factor --- Number of spaces in-between grid-steps (improves gridding/degridding accuracy)
    filter_type --- box (nearest-neighbour), sinc or gaussian_sinc
    """
    half_sup = 0
    oversample = 0
    full_sup_wo_padding = 0
    full_sup = 0
    no_taps = 0
    filter_taps = None
    def __init__(self, filter_half_support, filter_oversampling_factor, filter_type):
        self.half_sup = filter_half_support
        self.oversample = filter_oversampling_factor
        self.full_sup_wo_padding = (filter_half_support * 2 + 1)
        self.full_sup = self.full_sup_wo_padding + 2 #+ padding
        self.no_taps = self.full_sup + (self.full_sup - 1) * (filter_oversampling_factor - 1)
        taps = np.arange(self.no_taps)/float(filter_oversampling_factor) - self.full_sup / 2
        if filter_type == "box":
            self.filter_taps = np.where((taps >= -0.5) & (taps <= 0.5),
                                        np.ones([len(taps)]),np.zeros([len(taps)]))
        elif filter_type == "sinc":
            self.filter_taps = np.sinc(taps)
        elif filter_type == "gaussian_sinc":
            alpha_1=1.55
            alpha_2=2.52
            self.filter_taps = np.sin(np.pi/alpha_1*(taps+0.00000000001))/(np.pi*(taps+0.00000000001))*np.exp(-(taps/alpha_2)**2)
        else:
            raise ValueError("Expected one of 'box','sinc' or 'gausian_sinc'")

def nenufar_psf(uvw):
    u = np.hstack((-uvw[:, 0], uvw[:, 0]))
    v = np.hstack((-uvw[:, 1], uvw[:, 1]))

    Nx = 256#model_sky.shape[0]
    Ny = 256#model_sky.shape[1]
    cell_size_u = 2 * np.max(np.abs(u)) / (Nx)
    cell_size_v =  2 * np.max(np.abs(v)) / (Ny)

    scaled_uv = np.copy(uvw[:, 0:2])
    scaled_uv[:, 0] /= cell_size_u
    scaled_uv[:, 1] /= cell_size_v
    scaled_uv[:, 0] += Nx/2
    scaled_uv[:, 1] += Ny/2

    convolution_filter = AA_filter(3, 63, "sinc")

    # measurement_regular = np.zeros([Nx,Ny],dtype=np.complex)
    sampling_regular = np.zeros([2*Nx, 2*Ny], dtype=np.complex)
    for r in range(0, scaled_uv.shape[0]):
        disc_u = int(round(scaled_uv[r, 0]))
        disc_v = int(round(scaled_uv[r, 1]))
        frac_u_offset = int((1 - scaled_uv[r, 0] + disc_u) * convolution_filter.oversample)
        frac_v_offset = int((1 - scaled_uv[r, 1] + disc_v) * convolution_filter.oversample)
        if (disc_v + convolution_filter.full_sup_wo_padding  >= Ny or 
            disc_u + convolution_filter.full_sup_wo_padding >= Nx or
            disc_v < 0 or disc_u < 0): 
            continue
        for conv_v in range(0,convolution_filter.full_sup_wo_padding):
            v_tap = convolution_filter.filter_taps[conv_v * convolution_filter.oversample + frac_v_offset]  
            for conv_u in range(0,convolution_filter.full_sup_wo_padding):
                u_tap = convolution_filter.filter_taps[conv_u * convolution_filter.oversample + frac_u_offset]
                conv_weight = v_tap * u_tap
                # measurement_regular[disc_u - convolution_filter.half_sup + conv_u, disc_v - \
                #                     convolution_filter.half_sup + conv_v] += vis[r] * conv_weight
                sampling_regular[disc_u - convolution_filter.half_sup + conv_u, disc_v - \
                                 convolution_filter.half_sup + conv_v] += (1+0.0j) * conv_weight
    # dirty_sky = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(measurement_regular)))
    psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(sampling_regular)))

    import pylab as plt
    plt.imshow(np.log10(np.abs(psf)), origin='lower', cmap='bone_r', vmin=np.log10(np.abs(psf)).max() - 3)
    # plt.imshow(np.abs(psf), origin='lower', cmap='bone_r')
    plt.show()
    plt.close('all')



