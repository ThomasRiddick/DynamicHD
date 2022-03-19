'''
Utilities for scaling coordinates
Created on May 9, 2020

@author: thomasriddick
'''
import numpy as np
from warnings import warn

def guess_bound(coord,tolerance=1.0):
    if 90.0 - tolerance < coord <= 90.0 + tolerance:
        return 90.0
    elif -90.0 - tolerance < coord <= -90.0 + tolerance:
        return -90.0
    elif 180.0 - tolerance < coord <= 180.0 + tolerance:
        return 180.0
    elif -180.0 - tolerance < coord <= -180.0 + tolerance:
        return -180.0
    elif 0.0 - tolerance < coord <= 0.0 + tolerance:
        return 0.0
    elif 360.0 - tolerance < coord <= 360.0 + tolerance:
        return 360.0
    else:
        warn("Bounds of input data can't be inferred - is this a regional grid?")
        warn("Setting bounds to nearest degree")
        return float(round(coord))

def generate_coarse_coords(nlat_fine,nlon_fine,
                           lat_pts_fine,lon_pts_fine,
                           scaling_factor):
    nlat_coarse = nlat_fine//scaling_factor
    nlon_coarse = nlon_fine//scaling_factor
    lat_pts_coarse,lon_pts_coarse = generate_coarse_pts(nlat_fine,nlon_fine,
                                                        lat_pts_fine,lon_pts_fine,
                                                        nlat_coarse,nlon_coarse)
    return nlat_coarse,nlon_coarse,lat_pts_coarse,lon_pts_coarse

def generate_coarse_pts(nlat_fine,nlon_fine,
                        lat_pts_fine,lon_pts_fine,
                        nlat_coarse,nlon_coarse):
    lat_min_bound_fine = guess_bound(lat_pts_fine[0])
    lat_max_bound_fine = guess_bound(lat_pts_fine[-1])
    lon_min_bound_fine = guess_bound(lon_pts_fine[0])
    lon_max_bound_fine = guess_bound(lon_pts_fine[-1])
    lat_step_coarse = (lat_max_bound_fine - lat_min_bound_fine)/nlat_coarse
    lon_step_coarse = (lon_max_bound_fine - lon_min_bound_fine)/nlon_coarse
    lat_pts_coarse = np.linspace(lat_min_bound_fine+0.5*lat_step_coarse,
                                 lat_max_bound_fine-0.5*lat_step_coarse,
                                     num=nlat_coarse,endpoint=True)
    lon_pts_coarse = np.linspace(lon_min_bound_fine+0.5*lon_step_coarse,
                                 lon_max_bound_fine-0.5*lon_step_coarse,
                                 num=nlon_coarse,endpoint=True)
    return lat_pts_coarse,lon_pts_coarse
