'''
Utilities for scaling coordinates
Created on May 9, 2020

@author: thomasriddick
'''
import numpy as np

def guess_bound(coord,tolerance=5.0):
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
        raise RuntimeError("Bounds of input data can't be inferred")

def generate_course_coords(nlat_fine,nlon_fine,
                           lat_pts_fine,lon_pts_fine,
                           scaling_factor):
    nlat_course = nlat_fine/scaling_factor
    nlon_course = nlon_fine/scaling_factor
    lat_pts_course,lon_pts_course = generate_course_pts(nlat_fine,nlon_fine,
                                                        lat_pts_fine,lon_pts_fine,
                                                        nlat_course,nlon_course)
    return nlat_course,nlon_course,lat_pts_course,lon_pts_course

def generate_course_pts(nlat_fine,nlon_fine,
                        lat_pts_fine,lon_pts_fine,
                        nlat_course,nlon_course):
    lat_step_course = 180.0/nlat_course
    lon_step_course = 360.0/nlon_course
    lat_min_bound_fine = guess_bound(lat_pts_fine[0])
    lat_max_bound_fine = guess_bound(lat_pts_fine[-1])
    lon_min_bound_fine = guess_bound(lon_pts_fine[0])
    lon_max_bound_fine = guess_bound(lon_pts_fine[-1])
    if lat_min_bound_fine > 0:
        lat_pts_course = np.linspace(lat_min_bound_fine-0.5*lat_step_course,
                                     lat_max_bound_fine+0.5*lat_step_course,
                                     num=nlat_course,endpoint=True)
    else:
        lat_pts_course = np.linspace(lat_min_bound_fine+0.5*lat_step_course,
                                     lat_max_bound_fine-0.5*lat_step_course,
                                     num=nlat_course,endpoint=True)
    lon_pts_course = np.linspace(lon_min_bound_fine+0.5*lon_step_course,
                                 lon_max_bound_fine-0.5*lon_step_course,
                                 num=nlon_course,endpoint=True)
    return lat_pts_course,lon_pts_course
