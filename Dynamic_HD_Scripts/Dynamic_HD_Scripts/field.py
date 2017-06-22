'''
Contains a class to store and manipulate fields

Created on Jan 14, 2016

@author: thomasriddick
'''
import grid as gd
import sys
import numpy as np
import warnings

class Field(object):
    """A general class to store and manipulate fields on a given grid type
   
    Public methods:
    get_data
    set_data
    get_grid
    get_mask
    update_field_with_partially_masked_data
    flip_data_ud
    rotate_field_by_a_hundred_and_eighty_degrees
    get_flagged_points_coords
    flag_listed_points
    change_dtype
    mask_field_with_external_mask
    fill_mask
    subtract
    add
    invert_data
    copy
    convert_to_binary_mask
    """
    
    def __init__(self,input_field,grid='HD',**grid_kwargs):
        """Class constructor. Set data variable and create and set a grid if needed
       
        Arguments:
        input_field: numpy array-like; the raw input data
        grid: either an object that is a subclass of Grid or a string; the grid to use
        **grid_kwargs: dictionary; keyword arguments to pass to the makeGrid factory
            function
            
        If the grid argument is a string use the makeGrid factory function to make grid object
        otherwise just an existing grid object as the grid.
        """
        
        self.data = input_field
        if type(grid) is str:
            self.grid = gd.makeGrid(grid,**grid_kwargs)
        else:
            self.grid = grid
            
    def __repr__(self):
        """Overloaded representation method. Returns at representation of this Field instance"""
        return "<" + ".".join((self.__class__.__module__,self.__class__.__name__)) + \
            "\ndata: " + repr(self.data) + "grid: " + repr(self.grid) + ">"
        
    def get_data(self):
        """Get the data attribute"""
        return self.data
    
    def set_data(self,data):
        """Set the data attribute"""
        self.data = data
    
    def get_grid(self):
        """Get the grid attribute"""
        return self.grid
    
    def get_mask(self):
        """Get the mask of this field"""
        return np.ma.getmaskarray(self.get_data())

    def update_field_with_partially_masked_data(self,new_data):
        """Update this field with given input data only where the input data is not masked
       
        Arguments:
        new_data: a Field object or Field subclass object; new data that may be partially masked
        
        If the new_data Field is unmasked then simply replace this fields data with the new data;
        if it is masked replace this fields data with the new data only where the new field is not
        masked.
        """
        
        if type(new_data.get_data()) is np.ma.MaskedArray:  
            inverted_new_data_mask = np.invert(np.ma.getmaskarray(new_data.get_data()))
            field_data_with_inverted_mask = np.ma.array(self.data, copy=False, mask= inverted_new_data_mask, keep_mask=False)
            self.data = new_data.get_data().filled(0.0) + field_data_with_inverted_mask.filled(0.0)
        else:
            self.data = new_data.get_data()
            
    def flip_data_ud(self):
        """Flip this field in the up down (north south) direction"""
        self.data = self.grid.flip_ud(self.data)
        
    def rotate_field_by_a_hundred_and_eighty_degrees(self):
        """Rotate this field by 180 degrees in the longitude direction"""
        self.data = self.grid.one_hundred_eighty_degree_longitude_translation(self.data)
        
    def get_flagged_points_coords(self):
        """Get the coordinates of points that are flagged true in a boolean field
                
        Arguments and return value are discussed in underlying grid function
        """
        return self.grid.get_flagged_points_coords(self.data)
    
    def flag_listed_points(self,points_list):
        """Flag the points in a supplied list as true in a boolean field
        
        Arguments: 
        points_list: list; the coordinates of the points that are to be flagged
            as True. 
        Returns: nothing
        """
        
        for point in points_list:
            self.grid.flag_point(self.data,point)
            
    def change_dtype(self,dtype):
        """Convert the fields data to another numpy data type.
        
        Arguments: dtype; the numpy data type to convert the field to
        Returns: nothing
        """
        
        self.data = np.ascontiguousarray(self.data, dtype)
    
    def mask_field_with_external_mask(self,mask):
        """Mask a field using a supplied mask
       
        Arguments:
        mask: ndarray; an external mask to mask the field with
        Returns: nothing
        Any existing mask is discarded
        """

        self.data = np.ma.array(self.data,mask=mask,keep_mask=False)
        
    def fill_mask(self,fill_value):
        """Fills a mask with the supplied fill value
        
        Arguments
        fill_value: value to fill the mask elements of the mask with
        Returns: nothing
        """

        self.data = self.data.filled(fill_value)
        
    def subtract(self,field):
        """Subtract the values of another field from this field
        
        Arguments:
        field: ndarray; same shape as this field, the field to take away from this one
        Returns: nothing
        """
       
        if (type(self.grid) != type(field.get_grid()) or 
            self.data.shape != field.get_data().shape):
            raise RuntimeError("Field to subtract is not for same grid type")  
        self.data = self.data - field.get_data()
        
    def add(self,field):
        """Add the values of another field to this field
        
        Arguments:
        field: ndarray; same shape as this field, the field to add to this one
        Returns: nothing
        """
       
        if (type(self.grid) != type(field.get_grid()) or 
            self.data.shape != field.get_data().shape):
            raise RuntimeError("Field to add is not for same grid type")  
        self.data = self.data + field.get_data()
        
    def invert_data(self):
        """Invert a boolean or integer field (switch False/True or negative/postive)
        
        Arguments: none
        Returns: nothing
        """
        
        data_type = self.data.dtype
        self.data = np.logical_not(self.data.astype(np.bool_)).astype(data_type)
        
    def copy(self):
        """Return a copy of this field with a deep copy of its data
        
        Arguments: none
        Return: A field or field subclass object with the same type as this field and
        with the same grid type. The data in the returned field is a deep copy of its 
        data
        """

        return type(self)(input_field=self.data.copy(),grid=self.grid)
    
    def convert_to_binary_mask(self,threshold=50):
        """Convert this field to binary mask by converting points above threshold to 1 and all others to zero
        
        Arguments:
        threshold: float; threshold above which to convert field to 1's, all others points are converted to 0
        Return: Nothing
        """
        temporary_data_copy = np.zeros(self.data.shape)
        temporary_data_copy[self.data >= threshold] = 1
        temporary_data_copy[self.data <  threshold] = 0
        self.data = temporary_data_copy
   
class Orography(Field):
    """A subclass of Field with various method specific to orographies.
    
    Public methods:
    get_no_data_value
    mask_no_data_points
    mask_outside_region
    compute_flow_directions
    mask_new_orography_using_base_orography
    extend_mask
    generate_ls_mask
    find_area_minima
    mask_where_greater_than
    """
    
    #Best to use a whole number an order of magnitude deeper than the 
    #deepest ocean point
    no_data_value = -99999.0
    
    def __init__(self,input_field,grid='HD',**grid_kwargs):
        """Class constructor. Call parent constructor and set floating point error tolerance
        
        Arguments:
        As for parent class
        """

        super(Orography,self).__init__(input_field,grid,**grid_kwargs) 
        self.tiny = abs(self.data.max())*sys.float_info.epsilon*100.0  

    def get_no_data_value(self):
        """Get this object's no data value"""
        return self.no_data_value
    
    def mask_no_data_points(self):
        """Mask points where the orography value is set to the no data
        
        Arguments:None
        Returns: Nothing
        Modifies field in-place
        """
        
        self.data = np.ma.masked_values(self.data,self.no_data_value)
        
    def mask_outside_region(self,region):
        """Mask points outside a supplied region of the field
        
        Arguments:
        region: dictionary; the appropriate set of coordinates required
            to specify a region on the underlying grid
        Returns: Nothing
        """
        
        self.data = self.grid.mask_outside_region(self.data,region)
        

    def compute_flow_directions(self):
        """Use a method from grid object to compute flow directions on this orography
        
        Return:
        Flow direction numpy array calculated with this Orography object's data
        """

        return self.grid.compute_flow_directions(self.data)
    
    def calculate_differences_from_base_orography(self,base_orography):
        """Calculate differences from a supplied base orography and return them as masked field
        
        Arguments:
        base_orography: Orography object; the base orography to calculate differences from
        
        The masked array of differences returned is masked where differences are more than the largest
        plausible floating point error (plus a further margin of tolerance)
        """

        return np.ma.masked_inside(self.data - base_orography.get_data(),-self.tiny,self.tiny,copy=True)
   
    def calculate_significant_gradient_changes(self,base_orography,**grad_changes_kwargs):
        """Calculate and return gradient changes; masking them where they are not significant
        
        Arguments:
        base_orography: Orography object; the base orography to calculate significant changes from
        **grad_changes_kwargs: keyword dictionary; parameters to pass onto the mask_insigficant_changes
            method of grid
            
        First calculate the gradients of this orography and the base orography supplied and subtract them
        to give the gradient changes. Then call a method of Grid (passing in these gradient changes and
        the set of gradient on the original orography) that will mask these changes if they not considered 
        significant according to criterion supplied in the grad_changes_kwargs (or as defaults in the Grid 
        subclass called if grad_changes_kwargs are not supplied) and returned the masked changes.
        """
        
        base_orography_gradients = self.grid.calculate_gradients(base_orography.get_data())
        gradient_changes = self.grid.calculate_gradients(self.data) - base_orography_gradients
        return self.grid.mask_insignificant_gradient_changes(gradient_changes,base_orography_gradients,
                                                            **grad_changes_kwargs)
    
    def mask_new_orography_using_base_orography(self,base_orography,use_gradient=False,
                                                **grad_changes_kwargs):
        """Compare to a base orography and mask this orography where it does not need to updated.
        
        Arguments:
        base_orography: Orography object; a base orography to compare this orography to
        use_gradient: boolean; a flag whether to mask this orography everywhere it differs
            from the base orography by more than a small tolerance or only to mask it where 
            significant changes in gradient occur
        **grad_changes_kwargs: keyword dictionary; only required if use_gradient is true;
            parameter for deciding what a significant change in gradient is
        
        This routine creates a mask for this orography by using a method of this class to
        calculate either (depending on the value of the flag use_gradient) where significant 
        changes in gradient have not occured or where difference greater than a tolerance 
        have not occurred. Then it applies this newly generated mask to this orography 
        (discarding any prior mask).
        """
        
        if use_gradient:
            #The gradients are in a 9 by lat by long array with 9 indentical copies of the mask; take only the
            #first
            orography_mask = np.ma.getmaskarray(self.calculate_significant_gradient_changes(base_orography,
                                                                                            **grad_changes_kwargs)[0])
        else:
            orography_mask = np.ma.getmaskarray(self.calculate_differences_from_base_orography(base_orography))
        self.data = np.ma.array(self.data, copy=False, mask=orography_mask, keep_mask=False)
        
    def extend_mask(self):
        """Extend the mask to include all points neighbouring points in the current mask.
        
        First create an extended mask using a method from Grid and the current mask then apply 
        this new mask to this orography's data while simultaneously discarding the current mask.
        Note the potentially confusing terminology; it is the unmasked area that is extended 
        by this function while the masked area is reduced.
        
        """
        if type(self.data) is np.ma.MaskedArray:
            #Using np.ma.getmaskedarray instead of np.MaskedArray.mask ensures that an array of boolean 
            #and not a singular false (for no mask) is returned
            new_mask = self.grid.extend_mask_to_neighbours(np.ma.getmaskarray(self.data))
            self.data = np.ma.array(self.data,mask=new_mask,copy=False,keep_mask=False)
        else:
            warnings.warn("Trying to extend the mask of an object that is not a MaskedArray")
            
    def generate_ls_mask(self,sea_level):
        """Naively generate a land-sea mask for a given sea level
        
        Arguments:
        sea_level: float; the sea level to use to generate this land sea mask
        Returns: a numpy bool array where sea points are set to True and land points
        to False
        
        Simply take all points below (or at) the given sea-level to be sea and return this
        crude land-sea mask
        """
        
        data_with_sea_masked = np.ma.masked_less_equal(self.data,
                                                       sea_level,
                                                       copy=True)
        return np.ma.getmaskarray(data_with_sea_masked)
    
    def find_area_minima(self,area_corner_coords_list,area_size):
        """Find the minimum for each of a set of areas of a regular size
        
        Arguments and return value are discussed in underlying grid function
        """

        return self.grid.find_area_minima(self.data,area_corner_coords_list,area_size)
    
        
    def mask_where_greater_than(self,second_orography_field):
        """Mask this field where it greater than the supplied orography field
        
        Arguments: ndarray; a second field to mask the field belonging
            to this field object where this second field is lower.
        Returns: Nothing
        
        Preserve any existing mask
        """
        self.data = np.ma.masked_where(np.logical_or(np.ma.getmaskarray(self.data),
                                                     np.greater(self.data, 
                                                        second_orography_field.get_data())),
                                       self.data,copy=False)
    
class RiverDirections(Field):
    """A subclass of field with various method specific to river directions
    
    Public Methods:
    mark_river_mouths
    get_river_mouths
    get_lsmask
    extract_truesinks
    """

    def mark_river_mouths(self,lsmask=None):
        """Mark points where a river flows into the sea
        
        Arguments:
        lsmask (optional): ndarray; a land sea mask: an array of either bools or 0 and 1's with 
            the sea marked by True/or 1/or any other value that converts to true
        Returns: Nothing
            
        First any points in the optional input landsea mask are converted to sea points in the 
        river direction field. Then a check is made that there are sea points within the river
        direction field, if not then a warning is raised and the function returns. Finally the
        river directions are passed to this RiverDirections instances grid instance to be 
        processed."""
        if lsmask is not None:
            data_as_masked_array = np.ma.array(self.data,mask=lsmask,copy=False,keep_mask=False)
            self.data = np.ma.filled(data_as_masked_array, 
                                     float(self.grid.get_sea_point_flow_direction_value()))
        if not -1.0 in np.unique(self.data):
            raise UserWarning("No sea points are marked in the river flow direction field thus"
                              " no river mouths can be marked")
            return
        self.data = self.grid.mark_river_mouths(self.data)
        
    def get_river_mouths(self):
        """Return an array with the position of river mouth points in this field marked as true
        
        Returns: an ndarray with river mouths marked as True and other points as False
        
        Note this doesn't find river mouths not already present in the field; merely scans which
        cells have been marked as river mouths are returns an array with those cells set to true
        and all others set to false
        """
        #As the floating point flow direction are just used as a labels there should be no
        #scope for them to drift away exact integer values and no need for a tolerance
        river_mouths = np.ma.masked_equal(self.data,0.0,copy=True)
        return np.ma.getmaskarray(river_mouths)
    
    def get_lsmask(self):
        """Extract a landsea mask from a set of river flow directions
        
        Arguments: None
        
        Returns: an ndarray of 1's and 0's where 1's are sea points and 0's are land points
        
        This will only work on river flow direction where sea and coast points have been marked
        already. No warning is issued if it applied to a file where such points are not marked.
        """
        
        lsmask = np.zeros(self.data.shape,dtype=np.int64)
        lsmask[np.logical_or(self.data == -1, self.data == 0)] = 1
        return lsmask
    
    def extract_truesinks(self):
        """Extract true sinks from an river direction field
        
        Arguments: None
        Returns: a numpy bool array where true sinks (i.e. flow direction 5 cells) are True and
            all other cells are True
        """
        
        return (self.data == 5)
    
class CumulativeFlow(Field):
    """A subclass of field with various methods specific to cumulative flows
    
    Public Methods:
    find_cumulative_flow_at_outlets
    generate_cumulative_flow_threshold_mask
    """
    
    def generate_cumulative_flow_threshold_mask(self,threshold):
        """Return a mask of where the cumulative flow doesn't exceed a given threshold
       
        Arguments:
        threshold: integer, threshold cumulative flow below which mask is set to true 
        Returns: an ndarray containing the mask (true/masked meaning below the threshold)
            false/unmasked meaning above the threshold)
        """
        
        return self.data < threshold
    
    def find_cumulative_flow_at_outlets(self,river_mouths):
        """Returns the cumulative flow for given river mouth points
        
        Arguments: 
        river_mouths: ndarray of bools; a mask of river mouth points (marked as True) and non
            river mouth points (marked as False)
        Returns: an ndarray containing the value of the cumulative flow at river mouth points and the 
            zero elsewhere
        """

        cumulative_flow_at_outlets = np.ma.array(self.data,
                                                 mask=np.logical_not(river_mouths),
                                                 copy=True,
                                                 keep_mask=False)
        #Could also potentially return a list of locations and values however that would not 
        #use numpy and thus be processed very slowly
        return cumulative_flow_at_outlets.filled(0)
    
class ReservoirSize(Field):
    """A subclass of field with various methods specific to field of initial reserviour size data

    Public Methods:
    replace_zeros_with_global_postive_value_average
    replace_zeros_with_highest_valued_neighbor
    """
    
    def replace_zeros_with_highest_valued_neighbor(self):
        """Replace any zeros with the value of the highest value out of any (non negative) neighboring cell
        
        Arguments: None
        Returns: Nothing
        
        If all neighbor of a cell are negative or zero then a cell is left as zero
        """

        self.data = self.grid.replace_zeros_with_highest_valued_neighbor(self.data)

    def replace_zeros_with_global_postive_value_average(self):
        """Replace any values that are (exactly) zero with the global average of postive values
        
        Arguments: None
        Returns: None 
        """
        
        masked_data = np.ma.array(self.data,mask=(self.data <= 0.0),copy=False,keep_mask=False)
        self.data[self.data == 0.0] = np.ma.average(masked_data)
        
class RiverDischarge(Field):
    """A subclass of field for fields of discharge data generated by a model run
    
    Public methods:
    set_non_outflow_points_to_zero
    sum_river_outflow
    """
    
    def set_non_outflow_points_to_zero(self,rdirs_field):
        """Set all points to zero that aren't outflow points in the supplied river direction field
       
        Arguments:
        rdirs_field: Field object or subclass; the river direction to use to decide which points are
            outflow points
        Returns: nothing
        """

        self.data[np.logical_and(rdirs_field.get_data() != 0,rdirs_field.get_data() != 5)] = 0.0
        
    def sum_river_outflow(self):
        """Find the global sum of the river discharge at outflow points
        
        Arguments: 
        none
        Returns: double (assuming that is the type of this classes data array); the global sum of the
        river discharge at outflow points
        """
        
        return np.sum(self.data,dtype=np.float128)
        
#add any new Field subclasses to the list of field_types along with an appropriate key
def makeField(raw_field,field_type,grid_type,**grid_kwargs):
    """Factory function to create Field and Field subclass objects
    
    Arguments:
    raw_field: numpy array-like object; raw data to use for field
    field_type: string; key to which type of Field to use (either Field itself or a subclass)
    grid_type: string; key to which type of grid to use
    **grid_kwargs: keyword dictionary; additional parameters for grid (if necessary)
    """
    field_types = {'Orography':Orography,'Generic':Field,'RiverDirections':RiverDirections,
                   'CumulativeFlow':CumulativeFlow,'ReservoirSize':ReservoirSize,
                   'RiverDischarge':RiverDischarge}
    try:  
        return field_types[field_type](raw_field,grid_type,**grid_kwargs)
    except KeyError:
        raise RuntimeError('Invalid Field Type')
    
def makeEmptyField(field_type,dtype,grid_type,**kwargs):
    """Factory function to create a field filled with zeros
   
    Arguments same as makeField except that this function has
    no raw field argument
    """
    grid = gd.makeGrid(grid_type=grid_type,**kwargs)
    raw_field = grid.create_empty_field(dtype)
    return makeField(raw_field, field_type, grid_type,**kwargs)