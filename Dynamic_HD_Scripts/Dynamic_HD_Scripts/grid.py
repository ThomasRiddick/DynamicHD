'''
Contains an abstract class that acts as a template for classes that stored and manipulate 
specific grid types and classes derived from this for specific grid types
Created on Jan 13, 2016

@author: thomasriddick
'''
import numpy as np
import scipy.ndimage as ndi
from abc import ABCMeta, abstractmethod
import f2py_manager as f2py_mg
import warnings
import os.path as path
from context import fortran_source_path

class Grid(object):
    """Parent class for classes that store and manipulate specific grid types
    
    Public methods (all abstract):
    extend_mask_to_neighbours
    compute_flow_directions
    get_grid_dimensions
    mask_insignificant_gradient_changes
    calculate_gradients
    flip_ud
    one_hundred_eighty_degree_longitude_translation
    find_area_minima
    get_flagged_points_coords
    flag_point
    create_empty_field
    mask_outside_region
    replace_zeros_with_highest_valued_neighbor
    get_scale_factor_for_geographic_coords
    """
    
    __metaclass__ = ABCMeta
    nlat = 1
    nlong = 1

    @abstractmethod
    def extend_mask_to_neighbours(self,changes_mask):
        """Extend a mask on this grid to include neighbouring points
        
        Arguments:
        changes_mask: the mask to be extended
        
        Implementations should return the extended mask
        """
        
        pass
    
    @abstractmethod
    def compute_flow_directions(self,data):
        """Compute the flow direction from a given orography
        
        Arguments:
        data: the orography to compute the flow direction from
        
        Implementations should return the flow directions
        """
        
        pass
    
    @abstractmethod
    def get_grid_dimensions(self):
        """Get the dimension of the grid"""
        pass
   
    @abstractmethod
    def mask_insignificant_gradient_changes(self,gradient_changes,old_gradients,
                                            gc_method,**kwargs):
        """Mask a field of gradient changes where the changes are insignicant
        
        Arguments:
        gradient_changes: the gradient changes to consider
        old_gradients: the old gradients, possibly used in some gc_methods
        gc_method: the method/criteria used to decide which gradient changes are not significant
        **kwargs: parameters of the method/criteria used to decide which gradient changes are not
            significant
            
        Implementations should return the masked field of gradient changes
        """
            
        pass
       
    @abstractmethod
    def calculate_gradients(self,orography):
        """Calculate gradients from an orography
        
        Arguments:
        orography: the orography to use
        
        Implementations should return the field of gradients calculated
        """

        pass
    
    @abstractmethod
    def mark_river_mouths(self,flow_direction_data):
        """Mark all sea points that a river flows into as river mouth points
        
        Arguments:
        flow_direction_data: the flow direction data to work on
        
        Implementations should return the flow direction data with river-mouth sea 
        points marked with the specified river mouth point flow direction value 
        and other sea points left with the (general) sea point flow direction value 
        """
        
        pass
    
    @abstractmethod
    def flip_ud(self,data):
        """Flips input data in the up down (north south) direction
        
        Arguments:
        data: the data object to be flipped
        
        Implementations should return the flipped data object in the same
        format as the input data object
        """
        
        pass
    
    @abstractmethod
    def one_hundred_eighty_degree_longitude_translation(self,data):
        """Translate the longitude of a grid by 180 degrees
        
        Arguments:
        data: the data object to be translated 
        
        Implementations should return the translated data object in the same
        format as the input data object. This method effectively translate 
        between a using the Greenwich Meridan as the origin of the x axis 
        and placing it in the centre of the x axis.
        """

        pass 
    
    @abstractmethod
    def find_area_minima(self,data,area_corner_coords_list,area_size):
        """Find the minimum for each of a set of areas of a regular size
        
        Arguments:
        data: the data object to find minima in
        area_corner_coords_list: a list of the coordinates of the reference corner of each
            area
        area_size: the size of the areas specified as a tuple of the revelant lengths
        
        Implementations should return the minima marked as True in a field that is otherwise
        false given in the same format as the input data object. This method is intended to
        position true sinks points derived from a courser grid at the minima of the areas on 
        the fine grid corresponding to the position/cell of the points on the course grid.
        """

        pass
    
    @abstractmethod
    def get_flagged_points_coords(self,data):
        """Get the coordinates of points that are flagged true in a boolean field
       
        Arguments: the data object (to be interpreted as a boolean field) to find flagged points in
        
        Implementations should return the coordinates of the flagged points as list of objects of a type 
        appropriate to represent objects on the grid type of the implementation. 
        """

        pass
    
    @abstractmethod
    def flag_point(self,data,coords):
        """Flag the point at the supplied coordinates as True (or equivalently 1)
       
        Arguments:
        data: the data object (to be interpreted as a boolean field) to flag the point in
        coords: the coordinates of the point to flag
        """
    
    pass

    @abstractmethod
    def create_empty_field(self,dtype):
        """Creates a field full of zeros/False for this grid type
        
        Arguments:
        dtype: the data type of the field to create
        
        Implementations should return an empty field of zeros with the size and shape
        of this grid
        """
        
        pass
    
    @abstractmethod
    def mask_outside_region(self,data,region):
        """Mask a supplied field outside a supplied region and return it
        
        Arguments:
        data: the data object to be masked
        region: dictionary; the appropriate set of coordinates required
                to specify a region on the grid child class in question.
        Returns:
        The data object with a masked added
        """
        
        pass
    
    @abstractmethod
    def replace_zeros_with_highest_valued_neighbor(self,data):
        """Replace any zeros with the value of the highest value out of any (non negative) neighboring cell
        
        Arguments: 
        data: the data object to replace the zeros in
        Returns:
        The data object with the zeros replaced by the highest (non negative) neighboring cells value
        
        If all neighboring cell are negative or zero then a zero valued cell is left unchanged
        """ 

        pass
    
    @abstractmethod
    def get_scale_factor_for_geographic_coords(self): 
        """Get the scale factor for this size of grid used to convert to geographical coordinates
        
        Arguments: none
        Returns:
        The scale factor used to convert coordinates values for this grid to geographical coordinates
        (actual latitude and longitude)
        """
        
        pass
        
class LatLongGrid(Grid):
    """Class that stores information on and functions to work with a Latitude-Longitude grid.
    
    Public methods:
    As for parent class and in addition
    get_sea_point_flow_direction_value
    get_longitude_offset_adjustment
    
    This class should work on any latitude-longitude grid that stores data in 2D array-like objects. Note
    iternally all the functions within this class will work even if the size of the array(s) given is not
    the same as the number of latitude and longitude points set. This allows for easy testing but means that
    there is no check that the size of arrays given as arguments to methods of this class are correct.""" 
  
    flow_direction_labels=np.arange(1,10,dtype=np.float64)
    pole_boundary_condition = 1.0e+7
    sea_point_flow_direction_value = -1
    default_gc_method = 'all_neighbours'
    
    def __init__(self,nlat=360,nlong=720,longitude_offset_adjustment=0):
        """Class constructor. Set the grid size.
        
        Arguments:
        nlat (optional): integer; the number of latitude points in the grid
        nlong (optional): integer, the number of longitude points in the grid
        """

        self.gc_methods = {'all_neighbours':
                           LatLongGridGradientChangeMaskingHelper.all_neighbours_method}
        self.nlat = nlat
        self.nlong = nlong 
        self.longitude_offset_adjustment = longitude_offset_adjustment
        
    def get_longitude_offset_adjustment(self):
        """Return the factor need to adjust the longitude offset compared to that of 1/2 degree grid
        
        Arguments: None
        Returns: the required longitude offset adjustment compared to that of the half degree grid 
        required to give the correct longitude labels. This adjustment itself should of been set when
        the class was initialized pre-scaled such that it an offset on a 1/2 degree grid scale.
        """

        return self.longitude_offset_adjustment
    
    def extend_mask_to_neighbours(self,changes_mask):
        """Extend a mask on this grid to include neighbouring points
        
        Arguments:
        changes_mask: numpy 2d array of boolean value; the mask to be extended
        Return:
        The extended mask as a numpy 2d array of boolean values
        
        Extends the mask using the binary erosion method from the morophology class of
        numpy's image processing module. The copies of the first and last column are
        inserted into the position after the last column and before the first column
        respectively to wrap the calculation around the globe. The borders at the poles
        assume the mask to be true and any cross pole neighbours are ignore in this 
        calculation. 
        """
        #For an  unknown reason insert only works as required if the array axis are 
        #swapped so axis 0 is being inserted into.
        changes_mask = changes_mask.swapaxes(0,1)
        changes_mask = np.insert(changes_mask,obj=(0,np.size(changes_mask,axis=0)),
                                 values=(changes_mask[-1,:],changes_mask[0,:]),
                                 axis=0)
        changes_mask = changes_mask.swapaxes(0,1)
        #binary_erosion has a keyword mask that is nothing to do with the mask we are 
        #manipulating; its value defaults to None but set this explicitly for clarity
        return ndi.morphology.binary_erosion(changes_mask,
                                             structure=ndi.generate_binary_structure(2,2),
                                             iterations=1,
                                             mask=None,
                                             border_value=1)[:,1:-1]
       
    def flow_direction_kernel(self,orog_section):
        """Find flow direction a single grid point
        
        Arguements:
        orog_section: a flatten 1d array-like object; a 3 by 3 square of orography centre on 
            the grid point under consideration flattened into a 1d array
        Returns:
        A flow direction (which will take an integer value) as a numpy float 64 (for technical 
            reasons)
            
        Takes coordinate of the minimum of the orography section (the first minimum if several
        point equal minima exist) and uses it to look up a flow direction. If this grid cell itself
        is also a minima then assign the flow to it (diretion 5; min_coord=4) instead of the first
        minimum found.
        """
        
        #Only the first minimum is found by argmin. Thus it will be this that
        #is assigned as minima if two points have the same height
        min_coord = np.argmin(orog_section)
        if orog_section[4] == np.amin(orog_section):
            min_coord = 4
        return self.flow_direction_labels[min_coord] 
        
    def compute_flow_directions(self,data,use_fortran_kernel=True):
        """Compute the flow direction for a given orograpy
        
        Arguments:
        data: numpy array-like object; the orography to compute the flow direction for
        use_fortran_kernel: boolean: a flag to choose between using a python kernel (False)
            or using a Fortran kernel (True)
        Returns:
        A numpy array of flow directions as integers
        
        Inserts extra rows to ensure there is no cross pole flow; then flip the field to get the same first
        minima as Stefan uses for easy comparison. Setup the kernel function to use depending on the input
        flag. Then run a generic filter from the numpy image processing module that considers the 3 by 3
        area around each grid cell; wrapping around the globe in the east-west direction. Return the output
        as an integer after flipping it back the right way up and removing the extra rows added at the top
        and bottom
        """
        
        #insert extra rows across the top and bottom of the grid to ensure correct
        #treatmen of boundaries
        data = np.insert(data,obj=(0,np.size(data,axis=0)),values=self.pole_boundary_condition, axis=0)
        #processing the data upside down to ensure that the handling of the case where two neighbours
        #have exactly the same height matches that of Stefan's scripts
        data = np.flipud(data)
        
        if use_fortran_kernel:
            f2py_mngr = f2py_mg.f2py_manager(path.join(fortran_source_path,
                                                       'mod_grid_flow_direction_kernels.f90'), 
                                             func_name='HDgrid_fdir_kernel')
            flow_direction_kernel = f2py_mngr.run_current_function_or_subroutine
        else:
            flow_direction_kernel = self.flow_direction_kernel
            
        flow_directions_as_float = ndi.generic_filter(data,
                                                      flow_direction_kernel,
                                                      size=(3,3),
                                                      mode = 'wrap') 
        flow_directions_as_float = np.flipud(flow_directions_as_float)
        
        return flow_directions_as_float[1:-1].astype(int)
    
    def get_grid_dimensions(self):
        """Get the dimension of the grid and return them as a tuple"""
        return (self.nlat,self.nlong)
    
    def get_sea_point_flow_direction_value(self):
        """Get the sea point flow direction value"""
        return self.sea_point_flow_direction_value
    
    def mask_insignificant_gradient_changes(self,gradient_changes,old_gradients,
                                            gc_method=default_gc_method,**kwargs):
        """Mask insignicant gradient changes using a keyword selected method
        
        Arguments:
        gradient_changes: numpy array; a numpy array with gradient changes in 8 direction from 
            each grid cell plus flow to self gradient changes set to zero to make 9 elements 
            per grid cell in an order matching the river flow direction 1-9 compass rose keypad
            labelling scheme. Thus this should be 9 by nlat by nlong sized numpy array. 
        old_gradients: numpy array; the gradients from the base orography in the same format as
        the gradient changes
        gc_method: str; a keyword specifying which method to use to mask insignificant gradient 
        changes
        **kwargs: keyword dictionary; keyword parameter to pass to the selected gradient change
            masking method
        Returns: the masked gradient changes returned by the selected method
       
        The methods selected should be from a helper class called 
        LatLongGridGradientChangeMaskingHelper where all such methods should be stored in order
        to keep the main LatLongGrid class from becoming too cluttered. The gradient changes,
        old_gradient and **kwargs (except for gc_method) will be passed onto the selected 
        method.
        """

        return self.gc_methods[gc_method](gradient_changes,old_gradients,**kwargs)
   
    def calculate_gradients(self,orography):
        """Calculate the (pseudo) gradients between neighbouring cells from an orography
        
        Arguments:
        orography: numpy array; an array contain the orography data to calculate gradients for
        Returns:
        A numpy array that consists of the gradient from each point in all 8 directions (plus
        zero for the gradient to self). Thus the array has 1 more dimension than the input 
        orography (i.e. 2+1 = 3 dimensions); the length of this extra dimension is 9 for
        a 9 by nlong by nlat array. The order of the 9 different kinds of gradient change for
        each grid point matches the 1-9 keypad compass rose numbering used to denote flow 
        directions
        
        First adds extra columns to the start/end of the input orography that are equal to the 
        last/first row respectively; thus allowing gradients that wrap in the longitudal direction
        to be calculated. Then calculate arrays of the edge differences using numpy's diff method. 
        Calculate the corner differences by overlaying two offset version of the same orography 
        field. Return all these gradients array stacked up to form the final array; inserting
        extra rows of zeros where no gradient exists as it would go over the pole (over the
        pole gradients are set to zero) and trimming off one (or where necessary both) of the
        extra columns added to facilitate wrapping. Notice the gradients calculated are actually
        just the differences between neighbouring cells and don't take into account the distances
        between the point at which the gradient is defined; if this point was the cell centre then
        the gradients would need to calculated different for diagonal neighbours from the direct 
        (NESW) neighbours.
        """
        
        #should be equal to nlat and nlong but redefining here allows for
        #easy testing of this method isolation 
        naxis0 = np.size(orography,axis=0)
        naxis1 = np.size(orography,axis=1)
        #For an  unknown reason insert only works as required if the array axis are 
        #swapped so axis 0 is being inserted into.
        orography = orography.swapaxes(0,1)
        orography = np.insert(orography,obj=(0,np.size(orography,axis=0)),
                                 values=(orography[-1,:],orography[0,:]),
                                 axis=0)
        orography = orography.swapaxes(0,1)
        edge_differences_zeroth_axis = np.diff(orography,n=1,axis=0)
        edge_differences_first_axis  = np.diff(orography,n=1,axis=1)
        corner_differences_pp_mm_indices = orography[1:,1:] - orography[:-1,:-1] 
        corner_differences_pm_mp_indices = orography[1:,:-1] - orography[:-1,1:]
        #To simplify later calculations allow two copies of each gradient thus
        #allow us to associate set of gradients to each grid cell, note the 
        #use of naxisx-0 instead of naxis0 to add to the final row because the 
        #individual gradient arrays have all be reduced in size along the relevant
        #axis by 1. 
        return np.stack([
                         #keypad direction 1
                         #extra row added to end of zeroth axis; an extra column is not
                         #required due to wrapping
                         np.insert(corner_differences_pm_mp_indices[:,:-1],obj=naxis0-1,
                                   values=0,axis=0),
                         #keypad direction 2
                         #extra row added to end of zeroth axis
                         np.insert(edge_differences_zeroth_axis[:,1:-1],obj=naxis0-1,values=0,
                                   axis=0),
                         #keypad direction 3
                         #extra row added to end of zeroth axis; an extra column is not
                         #required due to wrapping 
                         np.insert(corner_differences_pp_mm_indices[:,1:],obj=naxis0-1,
                                   values=0,axis=0),
                         #keypad direction 4
                         #extra column not required due to wrapping
                         -edge_differences_first_axis[:,:-1],
                         #keypad direction 5. No gradient is defined but fill
                         #with zero to produce a stack of 9 numbers for each
                         #cell to correspond to the 9 flow directions
                         np.zeros((naxis0,naxis1)),
                         #keypad direction 6
                         #extra column not required due to wrapping
                         edge_differences_first_axis[:,1:],
                         #keypad direction 7
                         #extra row added to start of zeroth axis; an extra column is not
                         #required due to wrapping
                         np.insert(-corner_differences_pp_mm_indices[:,:-1],obj=0,
                                   values=0,axis=0),
                         #keypad direction 8
                         #extra row added to start of zeroth axis
                         np.insert(-edge_differences_zeroth_axis[:,1:-1],obj=0,values=0,
                                   axis=0),
                         #keypad direction 9
                         #extra row added to start of zeroth axis; an extra column is not
                         #required due to wrapping
                         np.insert(-corner_differences_pm_mp_indices[:,1:],obj=0,
                                   values=0,axis=0)],
                        axis=0) 

    def mark_river_mouths(self,flow_direction_data):
        """Mark all sea points that a river flows into as river mouth points
        
        Arguments:
        flow_direction_data: ndarray; field of flow direction data to mark river mouths on
        Returns: ndarray of flow direction data with the river mouths marked
        
        Calls a fortran module as the kernel to ndimage generic filter in order to actually 
        mark the river mouths
        """
        #For an  unknown reason insert only works as required if the array axis are 
        #swapped so axis 0 is being inserted into.
        flow_direction_data = flow_direction_data.swapaxes(0,1)
        flow_direction_data = np.insert(flow_direction_data,obj=(0,np.size(flow_direction_data,axis=0)),
                                 values=(flow_direction_data[-1,:],flow_direction_data[0,:]),
                                 axis=0)
        flow_direction_data = flow_direction_data.swapaxes(0,1)
        
        f2py_mngr = f2py_mg.f2py_manager(path.join(fortran_source_path,
                                                   'mod_river_mouth_kernels.f90'), 
                                             func_name='latlongrid_river_mouth_kernel')
         
        flow_direction_data = ndi.generic_filter(flow_direction_data,
                                                 f2py_mngr.run_current_function_or_subroutine,
                                                 size=(3,3),
                                                 mode = 'constant',
                                                 cval = float(self.sea_point_flow_direction_value))
        return flow_direction_data[:,1:-1]
    
    def flip_ud(self,data):
        """Flips input data in the up down (north south) direction
        
        Arguments:
        data: ndarray; the data object to be flipped
        Returns: an ndarray contain the data flipped upside down in the up/down (north/south)
        direction
        """

        return np.flipud(data)
    
    def one_hundred_eighty_degree_longitude_translation(self,data):
        """Translate the longitude of a grid by 180 degrees
        
        Arguments:
        data: ndarray; array of data to be translated 
        
        Implementation should return an ndarray of data of the same size and shape containing
        the translated data This method effectively translate between a using the Greenwich 
        Meridan as the origin of the x axis and placing it in the centre of the x axis.
        """
        #The number of longitude points is already defined but calculate it again here to 
        #facilitate easy testing with smaller arrays;
        nlon = np.size(data,axis=1)
        return np.roll(data,nlon/2,axis=1)
    
    def find_area_minima(self,data,area_corner_coords_list,area_size):
        """Find the minimum for each of a set of (multigrid point) rectangles of a regular size
        
        Arguments:
        data: ndarray; the field to find minima in
        area_corner_coords_list: a list of the coordinates of the reference corner of each rectangle
        area_size: the size of the rectangle (as a tuple comprising of the length along each axis)
        Returns: ndarray; with the minima in the supplied rectangles marked as True in a field 
        that is otherwise false. This method is intended to position true sinks points derived 
        from a courser grid at the minima of the areas on the fine grid corresponding to the 
        position/cell of the points on the course grid.
        """

        minima = np.zeros_like(data,dtype=np.bool)
        for area_corner_coords in area_corner_coords_list:
            area = data[area_corner_coords[0]:area_corner_coords[0]+area_size[0],
                        area_corner_coords[1]:area_corner_coords[1]+area_size[1]]
            coords_within_area = np.unravel_index(area.argmin(),(area_size[0],area_size[1]))
            min_lat = coords_within_area[0] + area_corner_coords[0]
            min_lon = coords_within_area[1] + area_corner_coords[1]
            minima[min_lat,min_lon] = True
        return minima
    
    def get_flagged_points_coords(self,data):
        """Get the coordinates of points that are flagged true in a boolean field
       
        Arguments: ndarray; the field (to be interpreted as a boolean field) to find flagged points in
        Return: list of coordinate pairs (lat,lon) of the flagged (as True) points within the supplied
        data field.
         
        It is necessary to manipulate the output of numpy's nonzero function to 
        obtain the correct output format. 
        """
        
        coords_org_format = data.nonzero()  
        return zip(coords_org_format[0],coords_org_format[1])

    def flag_point(self,data,coords):
        """Flag the point at the supplied coordinates as True (or equivalently 1)
        
        Arguments:
        data: ndarray; the field (to be interpreted as a boolean field) to flag points in
        coords: tuple; a pair of coordinates in the format (lat,lon)
        """
    
        data[coords[0],coords[1]] = True
        
    def create_empty_field(self,dtype):
        """Creates a field full of zeros/False on this objects latitude-longitude grid
        
        Arguments:
        dtype: the numpy data type of the field to create
        
        Returns: ndarray; an empty field of zeros with the size and shape of this grid
        """
        
        return np.zeros((self.nlat,self.nlong),dtype=dtype)
    
    def mask_outside_region(self, field, region):
        """Mask a supplied field outside a supplied region and return it
        
        Arguments:
        field: ndarray, the field to be masked
        region: dictionary; the appropriate set of coordinates required
            to specify a region on the grid child class in question.
            The dictionary needs entries for lat_min,lat_max,
            lon_min and lon_max. These are the edge of the area that 
            is left unmasked. Outside these coordinates the field is 
            masked. Any other entries in the dictionary are ignored
        Returns: masked ndarray; the field with the area outside the supplied
            region masked
        
        Any existing mask is kept during the masking process.
        """

        field = np.ma.array(field,keep_mask=True)
        field[:region['lat_min'],:]   = np.ma.masked
        field[region['lat_max']+1:,:] = np.ma.masked
        field[:,:region['lon_min']]   = np.ma.masked
        field[:,region['lon_max']+1:] = np.ma.masked
        return field
    
    def replace_zeros_with_highest_valued_neighbor(self,data):
        """Replace any zeros with the value of the highest value out of any (non negative) neighboring cell
        
        Arguments: 
        field: 2d ndarray;  ndarray containing the field to replace the zeros in
        Returns:
        field: 2d ndarray;  ndarray containing the field with the zeros replaced 
            by the highest (non negative) neighboring cells value
        
        If all neighboring cell are negative or zero then a zero valued cell is left unchanged
        """ 
        
        data = np.insert(data,obj=(0,np.size(data,axis=0)),values=-1.0, axis=0)
        
        f2py_mngr = f2py_mg.f2py_manager(path.join(fortran_source_path,
                                                   'mod_res_size_init_kernel.f90'), 
                                             func_name='latlongrid_res_size_init_krnl')
        data = ndi.generic_filter(data,f2py_mngr.run_current_function_or_subroutine,size=(3,3),
                                  mode = 'wrap')
        return data[1:-1,:]
    
    def get_scale_factor_for_geographic_coords(self): 
        """Get the scale factor for this size of grid used to convert to geographical coordinates
        
        Arguments: none
        Returns:
        The scale factor used to convert coordinates values for this grid to geographical coordinates
        (actual latitude and longitude)
        """
        return 360.0/self.nlong
        
        
def makeGrid(grid_type,**kwargs):
    """Factory function that creates an object of the correct grid type given a keyword
    
    Arguments:
    grid_type: string; a keyword giving the type of grid required. Can either be the
        keyword of an actual grid type or a shortcut for a particular grid type and
        setup
    kwargs: keyword dictionary; keyword arguments to pass to the Grid constructor giving
    parameters of the grid. Only required if you are using an actual grid name and not
    a shortcut
    
    Check if a shortcut is being used; if so then use the type and parameters in the triple
    nested shortcuts dictionary; if not then use the grid type from the grid types dictionary
    and pass the keyword arguments to the grid constructor.
    """
    
    shortcuts = {'HD':{'type':LatLongGrid,'params':{'nlat':360,'nlong':720}},
                 'LatLong5min':{'type':LatLongGrid,'params':{'nlat':2160,'nlong':4320}},
                 'LatLong10min':{'type':LatLongGrid,'params':{'nlat':1080,'nlong':2160,
                                                              'longitude_offset_adjustment':-0.5/3.0}},
                 'LatLong1min':{'type':LatLongGrid,'params':{'nlat':10800,'nlong':21600}},
                 'LatLong30sec':{'type':LatLongGrid,'params':{'nlat':21600,'nlong':43200}},
                 'T63':{'type':LatLongGrid,'params':{'nlat':96,'nlong':192}},
                 'T106':{'type':LatLongGrid,'params':{'nlat':160,'nlong':320}}}
    grid_types = {'LatLong':LatLongGrid}
    if grid_type in shortcuts.keys():
        underlying_grid_type = shortcuts[grid_type]['type']
        if len(kwargs) > 0:
            warnings.warn("{0} is a shortcut for grid type {1} with set params; "
                          "user defined parameters passed in as keyword argument "
                          "will be ignored".format(grid_type,underlying_grid_type))
        kwargs = shortcuts[grid_type]['params']
        return underlying_grid_type(**kwargs)
    else:
        try:  
            return grid_types[grid_type](**kwargs) 
        except KeyError:
            raise RuntimeError('Invalid Grid Type')
        
class LatLongGridGradientChangeMaskingHelper(object):
    """A helper class containing methods for calculating when gradient changes are significant.
    
    Class methods:
    all_neighbours_method
    """
       
    frac_value_zero_offset_default = 1
    
    @classmethod
    def all_neighbours_method(self,gradients_changes,old_gradients,
                              gc_relative_tol=None,gc_absolute_tol=None,
                              gc_frac_value_zero_offset=frac_value_zero_offset_default):
        """Mask gradients using the maximum gradient for each grid point
        
        Arguments:
        gradient_changes: numpy array; a numpy array with gradient changes in 8 direction from 
            each grid cell plus flow to self gradient changes set to zero to make 9 elements 
            per grid cell in an order matching the river flow direction 1-9 compass rose keypad
            labelling scheme. Thus this should be 9 by nlat by nlong sized numpy array.  
        old_gradients: numpy array; the gradients from the base orography in the same format as
            the gradient changes. These will be used as a denominator in the calculation of relative
            gradient changes.
        gc_relative_tol: float; the relative tolerance to use. Zero means zero tolerance. None means 
            that relative tolerances are not considered at all.
        gc_absolute_tol: float; the absolute tolerance to use. Zero means zero tolerance. None means 
            that absolute tolerances are not considered at all.
        gc_frac_value_zero_offset: The mimimum value to default the absolute value of the old gradient 
            to before using it as the denominator to calculate the relative gradients
        Returns:
        The gradient changes array (a 9 by lat by long sized numpy array) with an identical copy of the 
        new mask applied to each of the 9 direction
        
        Mask a set of gradients using the maximum value for each grid point out of the set of 8 values
        that grid point corresponding to the 8 direction (including the diagonals) using either relative
        or absolute tolerances or a combination of the two. First calculates the maximum absolute 
        gradient change for each point and the calculate all the relative gradient changes (by 
        dividing by the old gradients - first defaulting the old gradients to a minimum absolute
        value to avoid divisions by zero) and from these calculate the maximum relative gradient 
        changes. Then apply the tolerances; if both absolute and relative tolerances are required
        combine them using an AND. (True means masked and we want to mask if the absolute and the
        relative tolerance is not exceeded.) Copy (tile) the mask (after prepartory manipulation) 9 times
        so it can be applied to the gradient change array (which is 9 by lat by long). (Note that each of 
        the 9 directions has the same mask based on whichever gradient is maximum at each grid point; the
        9 copies of the mask are identical.). Return the gradient change array with the mask applied.
        """
    
        max_grad_change_magnitudes = np.amax(np.absolute(gradients_changes),axis=0)
        #offset gradients slightly from zero to avoid errors due to dividing by zero
        old_gradients = np.maximum(np.absolute(old_gradients),gc_frac_value_zero_offset)
        #For clarity across python versions use true divide explicitly
        frac_grad_changes = np.true_divide(gradients_changes,old_gradients)
        max_frac_grad_change_magnitudes = np.amax(np.absolute(frac_grad_changes),
                                                  axis=0)
        if gc_relative_tol is None and gc_absolute_tol is None:
            sig_grad_change_mask = None
        else:
            if gc_relative_tol is None:
                sig_grad_change_mask = max_grad_change_magnitudes <= gc_absolute_tol
            elif gc_absolute_tol is None:
                sig_grad_change_mask = max_frac_grad_change_magnitudes <= gc_relative_tol
            else:
                sig_grad_change_mask = np.logical_and(max_grad_change_magnitudes <= gc_absolute_tol,
                                                     max_frac_grad_change_magnitudes <= gc_relative_tol)
            sig_grad_change_mask = np.reshape(sig_grad_change_mask,(1,) + sig_grad_change_mask.shape)
            sig_grad_change_mask = np.tile(sig_grad_change_mask,(9,1,1))
        return np.ma.array(gradients_changes,mask=sig_grad_change_mask,keep_mask=False)