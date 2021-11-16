import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import numpy as np
from marvin.tools.cube import Cube
import pandas as pd

with fits.open('./data2/manga_firefly-v2_4_3-STELLARPOP.fits')as fin:
    firefly_plateifus = fin['GALAXY_INFO'].data['PLATEIFU']
    spaxel_binid = fin['SPATIAL_BINID'].data
    mstar_all = fin['SURFACE_MASS_DENSITY_VORONOI'].data
    spa_info = fin['SPATIAL_INFO'].data

def get_massmap(gal):
    # Select galaxy and binids
    ind1 = np.where(firefly_plateifus == gal)[0][0]
    ind_binid = spaxel_binid[ind1, :, :].astype(int)
    cells_binid = spa_info[ind1,:,0]

    # Create 2D stellar mass array
    mstar = np.ones(ind_binid.shape) * np.nan
    for row, inds in enumerate(ind_binid):
        inds[inds==-1]=-9999
        ind_nans = np.where(np.logical_or(inds==-1,inds==-9999))
        # print(inds)
        cells=[]
        for i in inds:
            cells.append(np.where(cells_binid==(i+0.0))[0][0])
        mstar[row] = mstar_all[ind1, cells, 0]
        mstar[row][ind_nans] = 0.0
        
    # trim mstar to match size of DAP maps 
    cube = Cube(gal)
    len_x = int(cube.header['NAXIS1'])
    mdens = mstar[:len_x, :len_x]
    mdens_ma = np.ma.array(data=mdens, mask=mdens==0.0)
    return mdens_ma

if __name__ == '__main__':
    massmap = get_massmap('8932-9102')
    plt.imshow(massmap)
    plt.colorbar()
    plt.show()
    