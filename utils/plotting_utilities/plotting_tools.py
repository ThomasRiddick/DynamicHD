'''
A library of tools to help with the creation of HD plots

Created on Feb 5, 2016

@author: thomasriddick
'''

import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
from Dynamic_HD_Scripts.base import grid
from Dynamic_HD_Scripts.base import field
from plotting_utilities import river_names


class OrogCoordFormatter:
    """Class that creates an object to call to give cursor position)"""

    def __init__(self,xoffset,yoffset,add_latlon=False,
                 scale_factor=1):
        self.xoffset = xoffset
        self.yoffset = yoffset
        self.add_latlon = add_latlon
        self.scale_factor = scale_factor

    def __call__(self,xpos,ypos):
        if not self.add_latlon:
            return "Array Indices: x= {0} y= {1}".format(int(round(xpos+self.xoffset)),
                                                         int(round(ypos+self.yoffset)))
        else:
            return "Array Indices: x= {0} y= {1}, {2} {3}".\
                format(int(round(xpos+self.xoffset)),
                       int(round(ypos+self.yoffset)),
                       calculate_lon_label(xpos,self.xoffset/self.scale_factor,
                                           scale_factor=self.scale_factor,
                                           precision=2),
                       calculate_lat_label(ypos,self.yoffset/self.scale_factor,
                                           scale_factor=self.scale_factor,
                                           precision=2))

class LonAxisFormatter:
    """Class that creates an object to call to give longitude axis tick labels"""

    def __init__(self,xoffset,scale_factor=1,precision=0):
        self.xoffset = xoffset
        self.scale_factor = scale_factor
        self.precision = precision

    def __call__(self,x,pos):
        return calculate_lon_label(x,self.xoffset,self.scale_factor,
                                   precision=self.precision)

class LatAxisFormatter:
    """Class that creates an object to call to give latitude axis tick labels"""

    def __init__(self,yoffset,scale_factor=1,precision=0):
        self.yoffset = yoffset
        self.scale_factor = scale_factor
        self.precision = precision

    def __call__(self,y,pos):
        return calculate_lat_label(y,self.yoffset,self.scale_factor,
                                   precision=self.precision)

def set_ticks_to_zero(ax=None):
    """Remove all the ticks a set of axes

    The ax argument is optional; if not specified then get the current axes.
    """

    params = {'axis'       :'both',
              'which'      :'both',
              'bottom'     : False,
              'top'        : False,
              'left'       : False,
              'right'      : False,
              'labelleft'  : False,
              'labelbottom': False}
    if ax:
        ax.tick_params(**params)
    else:
        plt.tick_params(**params)

def invert_y_axis(ax=None):
    """Invert the y-axis of a plot

    The ax argument is optional; if not specified then get the current axes.
    """

    if not ax:
        ax = plt.gca()
    ax.invert_yaxis()

def plot_catchments(catchments):
    """Plot the 1000 largest catchments of field of labelled catchments."""
    levels = np.arange(0,1001) + 0.5
    catchments = np.flipud(catchments)
    plt.contour(catchments,levels=levels,colors='black',linestypes='-')
    plt.show()

def print_nums(ax,section,x_lims,y_lims):
    for lat in range(int(math.ceil(y_lims[1]))+1,int(math.ceil(y_lims[0]))):
        for lon in range(int(math.ceil(x_lims[0]))+1,int(math.ceil(x_lims[1]))):
            if isinstance(section[lat,lon],str):
                continue
            try:
                ax.text(lon,lat,"{:.0f}".format(section[lat,lon]),
                        horizontalalignment='center',verticalalignment='center')
            except ValueError:
                #sometimes get value errors - this is because section changes between
                #the check and the next statement - if this occurs then ignore the
                #error
                pass

def apply_ocean_mask_to_catchments(catchments,field):
    """Applies a crude ocean mask to a field

    This routine simply masks all points with elevations less than zero
    """

    masked_field = np.ma.masked_less_equal(field,0)
    catchments = np.ma.array(catchments,
                             mask=np.ma.getmaskarray(masked_field))
    return np.ma.filled(catchments,0)

def relabel_catchment(catchments,old_catchment_num,new_catchment_num):
    """Renumber a catchment"""
    catchments[catchments == old_catchment_num] = new_catchment_num
    return catchments

def relabel_outflow_catchment(catchments,original_outflow_coords,relabel_as_outflow_coords,
                              data_catchment_field_original_scale=None,
                              data_original_scale_flowtocellfield=None,
                              catchment_grid_changed=False,original_coords_not_outflow=False,
                              data_original_scale_grid_type=None,data_original_scale_grid_kwargs={},
                              grid_type=None,**grid_kwargs):
    """Relabel the catchment of an input outflow with the catchment number of a second outflow"""
    if data_catchment_field_original_scale is None:
        old_catchment_num = catchments[original_outflow_coords]
        new_catchment_num = catchments[relabel_as_outflow_coords]
    else:
        if original_coords_not_outflow:
            old_catchment_num = catchments[original_outflow_coords]
        else:
            old_catchment_num = find_data_catchment_number(data_catchment_field=catchments,
                                                           data_catchment_field_original_scale=\
                                                           data_catchment_field_original_scale,
                                                           data_original_scale_flowtocellfield=\
                                                           data_original_scale_flowtocellfield,
                                                           data_coarse_coords=original_outflow_coords,
                                                           catchment_grid_changed=catchment_grid_changed,
                                                           data_original_scale_grid_type=\
                                                           data_original_scale_grid_type,
                                                           data_original_scale_grid_kwargs=\
                                                           data_original_scale_grid_kwargs,
                                                           grid_type=grid_type,**grid_kwargs)[0]
        new_catchment_num = find_data_catchment_number(data_catchment_field=catchments,
                                                       data_catchment_field_original_scale=\
                                                       data_catchment_field_original_scale,
                                                       data_original_scale_flowtocellfield=\
                                                       data_original_scale_flowtocellfield,
                                                       data_coarse_coords=relabel_as_outflow_coords,
                                                       catchment_grid_changed=catchment_grid_changed,
                                                       data_original_scale_grid_type=\
                                                       data_original_scale_grid_type,
                                                       data_original_scale_grid_kwargs=\
                                                       data_original_scale_grid_kwargs,
                                                       grid_type=grid_type,**grid_kwargs)[0]
    return relabel_catchment(catchments, old_catchment_num, new_catchment_num)

def find_data_catchment_number(data_catchment_field,
                               data_catchment_field_original_scale,
                               data_original_scale_flowtocellfield,data_coarse_coords,
                               catchment_grid_changed,data_original_scale_grid_type,
                               data_original_scale_grid_kwargs,grid_type,**grid_kwargs):
    if not catchment_grid_changed:
        data_catchment_num = data_catchment_field[data_coarse_coords]
        scale_factor = 1
    else:
        scale_factor = calculate_scale_factor(coarse_grid_type=grid_type,
                                              coarse_grid_kwargs = grid_kwargs,
                                              fine_grid_type=data_original_scale_grid_type,
                                              fine_grid_kwargs=data_original_scale_grid_kwargs)
        data_catchment_num = data_catchment_field_original_scale[\
            find_downscaled_outflow_coords(data_original_scale_flowtocellfield,
                                           coarse_coords=data_coarse_coords,
                                           scale_factor=scale_factor)]
    return data_catchment_num,scale_factor

def find_downscaled_outflow_coords(original_scale_field,coarse_coords,scale_factor):
    """Look for a river mouth in a fine scale field given coarse scale coordinates"""
    search_area = original_scale_field[coarse_coords[0]*scale_factor:(coarse_coords[0]+1)*scale_factor,
                                       coarse_coords[1]*scale_factor:(coarse_coords[1]+1)*scale_factor]
    internal_coords = np.unravel_index(search_area.argmax(),(scale_factor,scale_factor))
    mouth_coords = internal_coords[0]+coarse_coords[0]*scale_factor,internal_coords[1]+coarse_coords[1]*scale_factor
    mouth_neighbors = np.copy(original_scale_field[mouth_coords[0]-1:mouth_coords[0]+2,
                                                   mouth_coords[1]-1:mouth_coords[1]+2])
    mouth_neighbors[1,1] = 0
    mouth_neighbor_coords = np.unravel_index(mouth_neighbors.argmax(),(3,3))
    return (mouth_neighbor_coords[0] + mouth_coords[0] -  1, mouth_neighbor_coords[1] + mouth_coords[1] - 1)

def move_outflow(outflows,original_outflow_coords,new_outflow_coords,
                 flowmap=None,rdirs=None):
    """Move the outflow from one point to another"""
    downstream_cell_map = [(1,-1),(1,0),(1,1),
                           (0,-1),(0,0),(0,1),
                           (-1,-1),(-1,0),(-1,1)]
    if original_outflow_coords != new_outflow_coords:
        if rdirs is not None:
            original_outflow_coords = tuple(map(sum,
                                                list(zip(downstream_cell_map[int(rdirs[original_outflow_coords])
                                                                            - 1],
                                                    original_outflow_coords))))
        print('Old outflow at second point {0}'.format(outflows[new_outflow_coords]))
        if flowmap is not None:
            print('Old outflow at first point {0}'.\
                format(flowmap[original_outflow_coords]))
            outflows[new_outflow_coords] += flowmap[original_outflow_coords]
        else:
            print('Old outflow at first point {0}'.\
                format(outflows[original_outflow_coords]))
            outflows[new_outflow_coords] += outflows[original_outflow_coords]
            outflows[original_outflow_coords] = 0
        print('New outflow at second point {0}'.format(outflows[new_outflow_coords]))
    return outflows

def calculate_lat_label(y_index,offset,scale_factor=1,precision=1):
    """

    A scale factor of 1 is for the half degree grid. Notice offsets are
    assumed to be prescaled to the 1/2 degree scale - be careful this may be unexpected behaviour.
    This is a historical artifact. Using unicode for the degree symbol is more portable
    than latex.
    """
    return (lambda y: "{:.{prec}f}".format((0.5*y - 90)*(-1 if y < 180 else 1),prec=precision)
                            + u'\u00B0 ' + (('N' if y < 180 else '') if y <=180 else 'S'))((y_index/scale_factor)+offset)

def calculate_lon_label(x_index,offset,scale_factor=1,precision=1):
    """

    A scale factor of 1 is for the half degree grid.  Notice
    offsets are assumed to be prescaled to the 1/2 degree scale - be careful this may be unexpected
    behaviour. This is a historical artifact. Using unicode for the degree symbol is more portable
    than latex.
    """

    return (lambda x: "{:.{prec}f}".format((0.5*x-180)*(-1 if x < 360 else 1),prec=precision)
                            + u'\u00B0 ' + (('W' if ((x > 0) and (x < 360)) else '') if (x <= 360 or x >= 720) else 'E'))((x_index/scale_factor)+
                                offset)

def calc_displayed_plot_size(xlim,ylim):
    return (xlim[1] - xlim[0])*(ylim[0]-ylim[1])

def calculate_scale_factor(coarse_grid_type,coarse_grid_kwargs,fine_grid_type,fine_grid_kwargs):
    fine_grid_nlat = grid.makeGrid(fine_grid_type,**fine_grid_kwargs).get_grid_dimensions()[0]
    coarse_grid_nlat = grid.makeGrid(coarse_grid_type,**coarse_grid_kwargs).get_grid_dimensions()[0]
    return fine_grid_nlat//coarse_grid_nlat

def find_ocean_basin_catchments(rdirs,catchments,areas=[]):
  ocean_catchments = catchments.copy()
  ocean_basin_catchments = field.makeEmptyField("Generic",grid_type = catchments.get_grid(),dtype=np.int64)
  ocean_catchments.get_data()[np.logical_not(np.logical_and(rdirs.get_data() == 0,
                                                            catchments.get_data() != 0))] = 0
  for i,area in enumerate(areas,1):
    ocean_catchments_area =  ocean_catchments.get_data()[area['min_lat']:area['max_lat']+1,
                                                         area['min_lon']:area['max_lon']+1]
    ocean_catchment_list = np.unique(ocean_catchments_area)
    for catchment in ocean_catchment_list:
      if catchment == 0:
        continue
      ocean_basin_catchments.get_data()[catchments.get_data() == catchment] = i
  return ocean_basin_catchments

def guess_river_name(outflow_lat,outflow_lon,search_radius=1.0):
    """Guess the name of a river from the position of its mouth"""
    potential_names = []
    for name,coords in river_names.river_names_and_mouth_positions.items():
        if ((coords[0] - (outflow_lat -90))**2)+((coords[1] - (outflow_lon - 180))**2) <= search_radius**2:
            potential_names.append(name)
    if not potential_names:
        potential_names.append("Unknown")
    return " or ".join(potential_names)

def generate_label_for_river(pair,scale_factor,river_names,search_radius=1.0):
    river_name = guess_river_name(pair[0].get_lat()/scale_factor,
                                  pair[0].get_lon()/scale_factor)
    formatted_river_name = river_name
    for i in itertools.count(start=1):
        if not formatted_river_name in river_names:
            river_names.append(formatted_river_name)
            break
        else:
            formatted_river_name = river_name + " {}".format(i)
    return formatted_river_name
