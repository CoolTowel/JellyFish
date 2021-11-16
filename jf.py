from marvin import config
import math
from marvin.tools.maps import Maps
from marvin.tools.cube import Cube

import numpy as np
import matplotlib.pyplot as plt

from astropy import units
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.convolution import convolve
from astropy.io import fits

import mass_dens_mapext
import mass_mapext
import mass_sum

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)


config.setRelease('DR16')
config.setDR('DR16')

# with fits.open('./data2/manga_firefly-v2_1_2-STELLARPOP.fits')as fin:
#     firefly_plateifus = fin['GALAXY_INFO'].data['PLATEIFU']
#     spaxel_binid = fin['SPAXEL_BINID'].data
#     mstar_all = fin['SURFACE_MASS_DENSITY_VORONOI'].data

class Mymaps(Maps):
    def __init__(self, min_snr=3, max_radii=1.5, interp=False,
                 input=None, filename=None, mangaid=None, plateifu=None,
                 mode=None, data=None, release=None,
                 drpall=None, download=None, nsa_source='auto',
                 bintype=None, template=None, template_kin=None):
        """inhreriting all from marvin.tools.maps.Maps"""
        Maps.__init__(self, input=input, filename=filename, mangaid=mangaid, plateifu=plateifu,
                      mode=mode, data=data, release=release,
                      drpall=drpall, download=download, nsa_source=nsa_source,
                      bintype=bintype, template=template, template_kin=template_kin)
        """plateifu"""
        self.id = plateifu

        """effective radius"""
        self.radii_mask = (self.spx_ellcoo_r_re.value > max_radii)

        """the luminosity distance with a given redshift. In cm"""
        self.lum_dis = cosmo.luminosity_distance(
            self.nsa['z']).to_value(unit='cm')

        """the angular diameter distance with a given redshift. In kpc/arcsec"""
        self.ang_dis = cosmo.kpc_proper_per_arcmin(
            self.nsa['z']).to_value(unit='kpc/arcsec')

        """Ha emission flux in 1e-17 erg/s/cm^2/spaxel"""
        ha = self.emline_gflux_ha_6564
        self.ha_raw = ha
        ha_drpall_mask = ha.pixmask.get_mask(
            ['DONOTUSE', 'UNRELIABLE', 'NOCOV'],
            dtype=bool
        )
        ha_snr_mask = (ha.snr < min_snr) + ha_drpall_mask
        self.ha = np.ma.array(ha.value, mask=ha_snr_mask+self.radii_mask)

        """Hb emission in 1e-17 erg/s/cm^2/spaxel"""
        hb = self.emline_gflux_hb_4862
        self.hb_raw = hb
        hb_drpall_mask = hb.pixmask.get_mask(
            ['DONOTUSE', 'UNRELIABLE', 'NOCOV'],
            dtype=bool
        )
        hb_snr_mask = (hb.snr < min_snr) + hb_drpall_mask
        self.hb = np.ma.array(hb.value, mask=hb_snr_mask+self.radii_mask)

        # star formation rate

        self.ext_factor = ((self.ha/self.hb)/2.8)**2.36  # extiction factor

        """correct Ha emission flux 
        then convert into erg/s/spaxel
        then convert to SFR in M_Sun/year/spaxel
        """
        self.sfr_map = (self.ha * 10**(-17) * self.ext_factor) * \
            (4 * math.pi * (self.lum_dis**2)) / (10**41.1)

        self.ha_corr = self.ha * self.ext_factor

        """interpolate the missing data"""
        if interp is True:
            self.interp = True
            holes = (self.radii_mask*1) + \
                (self.sfr_map.filled(fill_value=-9999.0) != -9999.0)
            self.sfr_noholes = np.copy(self.sfr_map)
            self.sfr_noholes[holes == 0] = np.nan
            self.sfr_noholes = np.ma.array(
                self.sfr_noholes, mask=self.radii_mask)
            # # We smooth with a Gaussian kernel with x_stddev=1 (and y_stddev=1)
            # # It is a 9x9 array
            kernel = Gaussian2DKernel(x_stddev=1)
            # # create a "fixed" image with NaNs replaced by interpolated values
            self.sfr_noholes = interpolate_replace_nans(
                self.sfr_noholes, kernel)

    def _get_lead_trail_asy(self, image, angle):
        '''
        Divide galaxy into leading and trailing part.
        Image should be a numpy.ma.array instance.
        Angle should be in degree unit.
        '''
        whole_mask = image.mask
        center = np.unravel_index(
            np.argmin(self.spx_ellcoo_r_re.value),
            self.spx_ellcoo_r_re.value.shape
        )  # get index of the center of the galaxy
        w = image.shape[0]
        h = image.shape[1]
        x = math.cos(math.radians(angle))
        y = math.sin(math.radians(angle))
        u = center[0]  # row index of center
        v = center[1]  # col index of center
        index = np.indices((w, h))
        vactor_row = index[0] - u
        vactor_col = index[1] - v
        # the inner product of direction vactor and pixel location vactor
        inner_product = vactor_row * y + vactor_col * x
        lead_mask = image.mask + (inner_product >= 0)
        trail_mask = image.mask + (inner_product <= 0)
        leading = np.ma.array(image.data, mask=lead_mask)
        trailing = np.ma.array(image.data, mask=trail_mask)
        '''
        asymmetry part
        '''
        if w / u == 2 and h / v == 2:  # 对于中心在矩阵中间的图简单处理
            new_image = image[1:, 1:]  # 删除index为0的列和行
            new_image_rot = new_image[::-1, ::-1]
            dif = new_image - new_image_rot
            A = np.ma.sum(abs(dif)) / (np.ma.sum(new_image + new_image_rot))
        else:
            new_image = image
            if w / u > 2:
                new_image = new_image[0:2 * u + 1, :]
            else:
                new_image = new_image[2 * u - w + 1:w, :]
            if h / v > 2:
                new_image = new_image[:, 0:2 * v + 1]
            else:
                new_image = new_image[:, 2 * v - h + 1:h]
            new_image_rot = new_image[::-1, ::-1]
            dif = new_image - new_image_rot
            A = np.ma.sum(abs(dif)) / (np.ma.sum(new_image + new_image_rot))
        return leading, trailing, A

    def gal_div(self, image=None, angle=1):
        """
        return leading and trailing and asymmetry of a galaxy
        sfr map by default
        """
        if image is None:
            return self._get_lead_trail_asy(self.sfr_map, angle)
        else:
            return self._get_lead_trail_asy(image, angle)
        
    def get_mass_dens_map(self):
        '''
        get mass density map in mass/kpc^2
        '''
        massmap_all_data = mass_dens_mapext.get_massmap(self.id)
        massmap = np.ma.array(data = massmap_all_data.data, mask = np.logical_or(massmap_all_data.mask, self.radii_mask))
        return massmap
    
    # def get_mass_map(self):
    #     massmap_all_data = mass_mapext.get_massmap(self.id)
    #     massmap = np.ma.array(data = massmap_all_data.data, mask = np.logical_or(massmap_all_data.mask, self.radii_mask))
    #     # massmap = massmap * (0.5 * self.ang_dis)**2
    #     return massmap
    # 
    # def get_mass_map_fromdens(self):
    #     '''
    #     get mass density map in mass/spexal
    #     '''
    #     massmap = self.get_mass_dens_map()
    #     massmap = np.log10(10**massmap * (0.5 * self.ang_dis)**2)
    #     return massmap
    
    def get_mass(self):
        massmap = self.get_mass_dens_map()
        massmap = 10**massmap * (0.5 * self.ang_dis)**2
        mass = np.log10(np.ma.sum(massmap))
        return mass

    def mass_dens_map_asy(self, angle=1):
        return self._get_lead_trail_asy(self.get_mass_dens_map(), angle)
        
    def _get_stat(self, image, angle):
        '''
        get statistic data of the leading and trailing side.
        '''
        dic = {}
        l, t, A = self._get_lead_trail_asy(image, angle)
        l_mass, t_mass, A_mass = self.mass_dens_map_asy(angle)
        dic['plateifu'] = self.id
        dic['angle'] = angle
        dic['lead_area'] = np.sum(1-l.mask)+0.0
        dic['trail_area'] = np.sum(1-t.mask)+0.0
        dic['lead_mean'] = np.ma.sum(l) / dic['lead_area']
        dic['trail_mean'] = np.ma.sum(t) / dic['trail_area']
        # dic['lead_error'] = np.ma.sqrt(np.ma.sum(np.square(
        #     l / snr_map))) / dic['lead_area']
        # dic['trail_error'] = np.ma.sqrt(np.ma.sum(np.square(
        #     t / snr_map))) / dic['trail_area']
        dic['lead_median'] = np.ma.median(l)
        dic['trail_median'] = np.ma.median(t)
        dic['sfr'] = np.ma.sum(self.sfr_map)
        dic['used_area'] = np.sum(1-image.mask)+0.0
        dic['total_area'] = np.sum(1-self.radii_mask)+0.0
        dic['pix_used_ratio'] = dic['used_area']/dic['total_area']
        dic['asymmetry'] = A
        dic['lead_mass_area'] = np.sum(1-l_mass.mask)+0.0
        dic['trail_mass_area'] = np.sum(1-t_mass.mask)+0.0
        dic['lead_mass_mean'] = np.ma.sum(l_mass) / dic['lead_mass_area']
        dic['trail_mass_mean'] = np.ma.sum(t_mass) / dic['trail_mass_area']
        dic['lead_mass_median'] = np.ma.median(l_mass)
        dic['trail_mass_median'] = np.ma.median(t_mass)
        dic['asymmetry_mass'] = A_mass
        dic['stellar_mass'] = self.get_mass()
        
        
        return dic

    def stat(self, image=None, angle=1):
        if image is None:
            return self._get_stat(self.sfr_map, angle)
        elif image == 'ha':
            return self._get_stat(self.ha, angle)
        elif image == 'hb':
            return self._get_stat(self.hb, angle)
        else:
            return self._get_stat(image, angle)

    def psf_m(self, fwhm=6):
        # gaussian fwhm = 2 sqrt(2 ln 2) * sigma(stddev)
        k = 2*np.sqrt(2*np.log(2))
        sig = np.sqrt((fwhm/k/0.5)**2-(2.5/k/0.5)**2)
        kernel = Gaussian2DKernel(x_stddev=sig)
        sfr_match = self.ha_corr.data.copy()
        sfr_match[self.ha_corr.mask] = np.nan
        sfr_match = convolve(sfr_match, kernel)
        sfr_match[self.ha_corr.mask] = np.nan
        return sfr_match


def stat_list(plateifu_list, angle_list=None, min_snr=3, max_radii=1.5):
    table = []
    plateifu_list = np.char.strip(plateifu_list)
    if angle_list is None:
        for plateifu in plateifu_list:
            table.append(Mymaps(plateifu=plateifu,
                         min_snr=min_snr, max_radii=max_radii).stat())
    else:
        for plateifu, angle in zip(plateifu_list, angle_list):
            table.append(Mymaps(plateifu=plateifu, min_snr=min_snr,
                         max_radii=max_radii).stat(angle=angle))
    return Table(table)
# def _sfr_noholes(self):


def radibin(plateifu, min_snr=3, max_radii=1.5, interp=False):
    gal = Mymaps(plateifu=plateifu,
                 min_snr=min_snr, max_radii=max_radii, interp=interp)
    if interp is True:
        whole_map = gal.sfr_noholes.filled(
            fill_value=0.0)/(0.5 * gal.ang_dis)**2  # transfer to sfr/kpc^2
    else:
        whole_map = gal.sfr_map.filled(
            fill_value=0.0)/(0.5 * gal.ang_dis)**2  # transfer to sfr/kpc^2
    mask = gal.sfr_map.mask
    max_effradi = np.ma.max(np.ma.array(data=gal.spx_ellcoo_r_re.value, mask=mask), fill_value=0.0)
    # s = np.sum((1.0-gal.radii_mask))
    # r = np.sqrt(s/2/math.pi)
    # bin_width = 1.5/(r-2)
    i = 0.0
    index = 0 
    bin_r = []
    radbin_med = []
    radbin = []
    indices = []
    """center sfr"""
    bin_r.append(0.0) 
    center = np.unravel_index(
        np.argmin(gal.spx_ellcoo_r_re.value), gal.spx_ellcoo_r_re.value.shape)
    radbin_med.append(whole_map[center[0],center[1]])
    indices.append(index)
    """radial profile"""
    bin_width = 0.2
    while i < max_effradi:
        index += 1
        i += bin_width
        mask = np.logical_and(gal.spx_ellcoo_r_re.value <= i,
                              gal.spx_ellcoo_r_re.value > (i-bin_width))
        bin_spaxels = whole_map[mask]
        bin_spaxels = bin_spaxels[bin_spaxels != 0]
        if bin_spaxels.size != 0:
            med = np.median(bin_spaxels)
            radbin_med.append(med)
            indices.append(index)
            
    return radbin_med, indices
