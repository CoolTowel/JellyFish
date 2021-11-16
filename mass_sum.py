
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import numpy as np
from marvin.tools.cube import Cube
import pandas as pd

with fits.open('./data2/manga_firefly-v2_4_3-STELLARPOP.fits')as fin:
    firefly_plateifus = fin['GALAXY_INFO'].data['PLATEIFU']
    spaxel_binid = fin['SPATIAL_BINID'].data
    mstar_all = fin['STELLAR_MASS_VORONOI'].data
    spa_info = fin['SPATIAL_INFO'].data

def get_mass(gal):
    # Select galaxy and binids
    ind1 = np.where(firefly_plateifus == gal)[0][0]
    mass_cell = mstar_all[ind1, :, 0]
    mass_cell[mass_cell<0]=0.0
    mass_list = np.ma.array(data=mass_cell, mask = mass_cell==0.0)
    mass = np.log10(np.ma.sum(np.power(mass_list,10)))
    return mass