from nenupy.hpx import Digibeam, Skymodel
import healpy as hp
import matplotlib as mpl
import pylab as plt
import matplotlib.ticker as mtick
import numpy as np


db = Digibeam(azana=180,
            azdig=180,
            elana=75,
            eldig=75,
            miniarrays=[10, 20, 30, 40],
            freq=60,
            polar='NW',
            resol=0.2)

beam = db.get_digibeam()

print(db._nside)

sm = Skymodel(nside=db._nside,
            freq=60)

skybeam = beam * sm.get_skymodel(time='2019-03-15 12:00:00', model='GSM')


hp.mollview(skybeam, flip='astro', title=None, max=skybeam.max(), min=skybeam.max()*1e-5, norm='log')
hp.graticule()
plt.show()

