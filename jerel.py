import os
import math
import numpy as np
import matplotlib.pyplot as plt

from marvin import config

config.setRelease('DR16')
config.setDR('DR16')

from marvin.tools.image import Image
from marvin.tools import Maps
from marvin.tools import Cube

from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
drpall_path = config.drpall


def get_dl(red_shift):
    '''
    Calculate the luminosity distance with a given redshift. In cm unit.
    '''
    dl = cosmo.luminosity_distance(red_shift).to_value(unit='cm')
    return dl


def my_ha(plateifu, min_snr=3, max_radii=1.5):
    '''
    get my own Ha emission map in numpy.ma.array instance within minimum SNR and maximum effective readius. 
    Then convert it to Luminosity (erg /s /arcsec^2)
    '''
    maps = Maps(plateifu=plateifu)
    ha = maps.emline_gflux_ha_6564
    er = maps.spx_ellcoo_r_re
    z = maps.nsa['z']
    drpall_mask = ha.pixmask.get_mask(['DONOTUSE', 'UNRELIABLE', 'NOCOV'],
                                      dtype=bool)
    snr_mask = (ha.snr < min_snr) + drpall_mask
    radii_mask = (er.value > max_radii) + drpall_mask
    my_ha = np.ma.array(ha.data, mask=snr_mask + radii_mask)
    dl = get_dl(z)
    masked_lumi = my_ha * 1e-17 * 4 * math.pi * (dl**2) / 0.25
    snr_map = ha.snr
    return masked_lumi, snr_map, er.value

def my_hb(plateifu, min_snr=3, max_radii=1.5):
    '''
    get my own H Beta emission map in numpy.ma.array instance within minimum SNR and maximum effective readius. 
    Then convert it to Luminosity (erg /s /arcsec^2)
    '''
    maps = Maps(plateifu=plateifu)
    hb = maps.emline_gflux_hb_4862
    er = maps.spx_ellcoo_r_re
    z = maps.nsa['z']
    drpall_mask = hb.pixmask.get_mask(['DONOTUSE', 'UNRELIABLE', 'NOCOV'],
                                      dtype=bool)
    snr_mask = (hb.snr < min_snr) + drpall_mask
    radii_mask = (er.value > max_radii) + drpall_mask
    my_hb = np.ma.array(hb.data, mask=snr_mask + radii_mask)
    dl = get_dl(z)
    masked_lumi = my_hb * 1e-17 * 4 * math.pi * (dl**2) / 0.25
    snr_map = hb.snr
    return masked_lumi, snr_map, er.value

def sfr(plateifu, min_snr=3, max_radii=1.5):
    ha, snr_map_ha, er_map = my_ha(plateifu,min_snr,max_radii)
    hb, snr_map_hb, er_map = my_hb(plateifu,min_snr,max_radii)
    ext_factor=((ha/hb)/2.8)**2.36
    ha_cor = ha*ext_factor
    sfr = ha_cor/(10**41.1)
    return sfr , snr_map_ha, er_map

def asy(plateifu, min_snr=3, max_radii=1.5):
    image, snr_map, er_map = sfr(plateifu, min_snr, max_radii)
    center = np.unravel_index(np.argmin(er_map), er_map.shape)
    w = er_map.shape[0]  # width of image
    h = er_map.shape[1]  # hieght of image
    u = center[0]  # row index of center
    v = center[1]  # col index of center
    print(str(u) + ',' + str(v))
    if w / u == 2 and h / v == 2:  # 对于中心在矩阵中间的图简单处理
        new_image = image[1:, 1:]  # 删除index为0的列和行
        new_image_rot = new_image[::-1, ::-1]
        dif = new_image - new_image_rot
        A = np.ma.sum(abs(dif)) / (2 * np.ma.sum(dif + new_image_rot))
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
        A = np.ma.sum(abs(dif)) / (2 * np.ma.sum(dif + new_image_rot))
    return A

def get_lead_trail(plateifu, angle, min_snr=3, max_radii=1.5):
    '''
    Divide galaxy into leading and trailing part.
    Image should be a numpy.ma.array intance.
    Angle should be in degree unit.
    Center is the center of galaxy, should be a numpy.array
    can be get by (np.unravel_index(np.argmin(effective_radius), er.shape))
    '''
    image, snr_map, er_map = sfr(plateifu, min_snr, max_radii)
    whole_mask = image.mask
    center = np.unravel_index(
        np.argmin(er_map),
        er_map.shape)  #get index of the center of the galaxy
    w = image.shape[0]
    h = image.shape[1]
    x = math.cos(math.radians(angle))
    y = math.sin(math.radians(angle))
    u = center[0]  # row index of center
    v = center[1]  # col index of center
    index = np.indices((w, h))
    vactor_row = index[0] - u
    vactor_col = index[1] - v
    inner_product = vactor_row * y + vactor_col * x  # the inner product of direction vactor and pixel location vactor
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
        A = np.ma.sum(abs(dif)) / (2 * np.ma.sum(dif + new_image_rot))
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
        A = np.ma.sum(abs(dif)) / (2 * np.ma.sum(dif + new_image_rot))

    return leading, trailing, snr_map, A





def stat(plateifu, angle, min_snr=3, max_radii=1.5):
    '''
    get statistic data of the leading and trailing side.
    '''
    dic = {}
    l, t, snr_map, A = get_lead_trail(plateifu, angle, min_snr, max_radii)
    dic['lead_area'] = sum(sum(l.mask == False))
    dic['trail_area'] = sum(sum(t.mask == False))
    dic['lead_sum'] = np.ma.sum(l) / dic['lead_area']
    dic['trail_sum'] = np.ma.sum(t) / dic['trail_area']
    dic['lead_error'] = np.ma.sqrt(np.ma.sum(np.square(
        l / snr_map))) / dic['lead_area']
    dic['trail_error'] = np.ma.sqrt(np.ma.sum(np.square(
        t / snr_map))) / dic['trail_area']
    dic['lead_median'] = np.ma.median(l)
    dic['trail_median'] = np.ma.median(t)
    dic['asymmetry'] = A
    return dic


def stat_list(plateifu_list, angle_list, min_snr=3, max_radii=1.5):
    '''
    platifu_list should be a list of the platifu.
    get the stat list of all of these galaxies.
    '''
    num = plateifu_list.shape[0]  # get the number of the galaxies
    num_list = np.arange(num)
    l_a = np.zeros(shape=num,
                   dtype='int')  # list of the leading side area (spaxels)
    t_a = np.zeros(shape=num,
                   dtype='int')  # list of the trailing side area (spaxels)
    l_s = np.zeros(
        shape=num,
        dtype='float64')  # list of the leading side sum flux (erg/s)
    t_s = np.zeros(
        shape=num,
        dtype='float64')  # list of the trailing side sum flux (erg/s)
    l_se = np.zeros(
        shape=num,
        dtype='float64')  # list of the error of the leading side sum flux
    t_se = np.zeros(
        shape=num,
        dtype='float64')  # list of the error of the trailing side sum flux
    l_m = np.zeros(shape=num,
                   dtype='float64')  # list of the leading side median flux
    t_m = np.zeros(shape=num,
                   dtype='float64')  # list of the trailing side median flux
    a = np.zeros(shape=num,
                 dtype='float64')  # list of the trailing side median flux
    for i in range(num):
        l, t, snr_map, asy_value = get_lead_trail(plateifu_list[i],
                                                  angle_list[i], min_snr,
                                                  max_radii)
        l_a[i] = sum(sum(l.mask == False))
        t_a[i] = sum(sum(t.mask == False))
        l_s[i] = np.ma.sum(l) / l_a[i]
        t_s[i] = np.ma.sum(t) / t_a[i]
        l_se[i] = np.ma.sqrt(np.ma.sum(np.square(l / snr_map))) / l_a[i]
        t_se[i] = np.ma.sqrt(np.ma.sum(np.square(t / snr_map))) / t_a[i]
        l_m[i] = np.ma.median(l)
        t_m[i] = np.ma.median(t)
        a[i] = asy_value
    t = Table([
        num_list, plateifu_list, angle_list, l_a, t_a, l_s, t_s, l_se, t_se,
        l_m, t_m, a
    ],
              names=('num', 'ifu_id', 'd_angle', 'l_area', 't_area', 'l_sum',
                     't_sum', 'l_error', 't_error', 'l_median', 't_median',
                     'asy'))
    return t