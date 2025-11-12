use coords_mod
use subfield_mod
use field_section_mod
use cotat_parameters_mod, only: MUFP, area_threshold, run_check_for_sinks, &
                                yamazaki_max_range, yamazaki_wrap

// This file contains comments using a special markup for doxygen

//> An abstract class containing sections of a number of fields that all correspond to the same section of
// the underlying grid of pixel and (mostly virtual) operator this grid section and the various
// field sections for it.

class area {
    protected:
        //! A section of a field of pixel level river direction
        ?? field_section* river_directions => nullptr;
        //! A section of a field of pixel level total cumulative flow
        ?? field_section* total_cumulative_flow => nullptr;
        //! A section of a field used only by the yamazaki style algorithm to mark
        //! pixels as outlet pixels
        ?? field_section* yamazaki_outlet_pixels => nullptr;
        //The following variables are only used by latlon based area's
        //(Have to insert here due to lack of multiple inheritance)
        //! The minimum latitude defining the section's lower edge (only for
        //! latitude-longitude grids - here due to lack of multiple inheritance
        //! in Fortran)
        int section_min_lat;
        //! The minimum longitude defining the section's left edge (only for
        //! latitude-longitude grids - here due to lack of multiple inheritance
        //! in Fortran)
        int section_min_lon;
        //! The maximum latitude defining the section's upper edge (only for
        //! latitude-longitude grids - here due to lack of multiple inheritance
        //! in Fortran)
        int section_max_lat;
        //! The maximum longitude defining the section's right edge (only for
        //! latitude-longitude grids - here due to lack of multiple inheritance
        //! in Fortran)
        int section_max_lon;
        //! Yamazaki style algorithm code for a normal outlet pixel
        int yamazaki_normal_outlet = 1;
        //! Yamazaki style algorithm code for the outlet pixel of a cell
        //! where the outlet is not the LCDA (and thus a cell that won't
        //! receive inflow from other cells; any cells flows entering it
        //! will pass through and look for suitable outlets in the cell
        //! beyond; this is because this cells own flow needs to exit via
        //! a pixel that is not its outlet pixel
        int yamazaki_source_cell_outlet = 2;
        //! Yamazaki style algorithm code for a sink outlet pixel
        int yamazaki_sink_outlet = -1;
        //! Yamazaki style algorithm code for a river mouth pixel
        int yamazaki_rmouth_outlet = -2;
        //! Subroutine to find the next pixel upstream from a given pixel and return it via the
        //! input coordinate object; also return a flag if path has left the section and optionally
        //! if an outlet pixel has been found for the yamazaki-style algorithm
        find_next_pixel_upstream
        //! Return a list of neighbours that upstream of the pixel located at the input coordinates
        virtual vector<coords*>* find_upstream_neighbors(coords* coords_in);
        //! Check if a given set of coordinates are within the bounds of this area
        virtual bool check_if_coords_are_in_area(coords* coords_in);
        //! Check if neighbor at a given coordinates flows to a pixel which lies in a given
        //! direction from that neighbor (this direction being expressed in reverse, i.e. pointing
        //! from the pixel to the neighbor)
        virtual bool neighbor_flows_to_pixel(coords* coords_in,
                                             direction_indicator* flipped_direction_from_neighbor);
        //! Subroutine to find the next pixel downstream from the input pixel and returns it
        //! via the same variable as used for input. Also flag if the cell next coords would
        //! be outside the area and for the yamazaki style algorithm if the next pixel is an
        //! outflow pixel (this is an optional intent OUT argument).
        virtual void find_next_pixel_downstream(coords* coords_inout,
                                                bool coords_not_in_area,
                                                bool outflow_pixel_reached=true);
        //! Work out the length of a path through this pixel but looking up its flow direction
        virtual double calculate_length_through_pixel(coords* coords_in);
        //! Check if the pixel at the input coordinates is a diagonal or not by looking up its flow direction
        virtual bool is_diagonal(coords* coords_in);
};

//! A neighbourhood includes a centre cell and a number of cells surrounding it
//! and it contains functions that need to run over that entire neighborhood for the
//! COTAT+ and yamazaki-style upscaling algorithms

class neighborhood : public area {
    protected:
        int center_cell_id;
        //! Find the downstream cell for the COTAT+ algorithm from a given starting pixel
        //! returns an integer indicating the downstream cell found
        procedure :: find_downstream_cell
        //! Trace the path from this cell upstream to see if it re-enters the centre
        //! cell before exiting this neighborhood. Returns a flag indicating if this
        //! occurs
        procedure :: path_reenters_region
        //! Check if the cell specified by the input cell id is the centre cell
        procedure :: is_center_cell
        //! Find the downstream cell for the Yamazaki
        procedure :: yamazaki_find_downstream_cell
        //! Calculate an appropriate direction indicator from the id of the downstream cell
        virtual direction_indicator* calculate_direction_indicator(int downstream_cell);
        //! Calculate which cell a given pixel is in; return an integer identifying that cell
        //! from those in the neighborhood
        virtual int in_which_cell(coords* pixel);
        //! Get the coordinates of the cell that the input pixel lies in
        virtual coords* yamazaki_get_cell_coords(coords* pixel_coords);
        //! Subroutine that wraps a set of input pixel coordinates round the international date
        //! line if required; return the wrapped coordinates via the same variable as used for
        //! inputting them
        virtual void yamazaki_wrap_coordinates(coords* pixel_coords);
}

//! An object to contain the fine field sections corresponding to a single cell of the
//! coarse grid; also includes the necessary functions that act on that cell to
//! produce upscaled COTAT+ and yamazaki style upscalings. Each cell contains a
//! neighborhood object that contains both it and its surrounding cells.

class cell : public area {
    protected:
        //! An array marking pixels that have already been rejected as the
        //! outlet pixel of this cell
        class(subfield), pointer :: rejected_pixels => nullptr;
        //! An array containing the cumulative flow calculated across this
        //! cell only (and thus not counting pixels external to this cell)
        class(subfield), pointer :: cell_cumulative_flow => nullptr;
        //! A neighborhood object containing the neighborhood of this cell
        class(neighborhood), allocatable :: cell_neighborhood
        //! The number of pixels that lie along the edge of this cell
        int number_of_edge_pixels;
        //! Flag if this cell contains river mouth on the fine grid
        bool contains_river_mouths;
        //! Flag if this an ocean cell - i.e. the fine pixels inside it are
        //! all marked sea in the land sea mask
        bool ocean_cell;
        //! The coordinates of the edges of this section within the wider fine fields
        class(section_coords), allocatable :: cell_section_coords
        //! Process the cell represented by this cell object to produce a river direction
        procedure, public :: process_cell
        //! Find and mark the outlet pixel of this cell for the yamazaki style algorithm;
        //! an input flag indicates if the LCDA criterion is being used
        procedure, public :: yamazaki_mark_cell_outlet_pixel
        //! Setter for the logical variable contain_river_mouths
        procedure, public :: set_contains_river_mouths
        //! Print details of this cell and neighborhoods
        procedure, public :: print_cell_and_neighborhood
        //! Find and return the position of the outlet pixel for this cell. Also returns
        //! via an argument a flag to indicate if there was no remaining outlet pixels and
        //! takes a flag whether we should LCDA or not and optionally returns a
        //! flag indicating if this is the LCDA or not
        procedure :: find_outlet_pixel
        //! Subroutine to check if either the sum of the flows to sinks or the sum of the
        //! flows to river mouths is greater than the outflow at the choosen normal outlet
        //! for the cell. Takes as arguments the position of the current normal outlet and
        //! a flag to indicate if there were no remaining outlets (and a default was used)
        //! return two flags indicating whether cell should be marked a sink or a river
        //! mouth outflow. Note only one (or zero) of the flags can be true at a time
        procedure :: check_for_sinks_and_rmouth_outflows
        //! Find the downstream cell for the COTAT+ algorithm and calculate the direction
        //! to it. Takes the position of the outlet pixel and a flag to indicate if
        //! no valid outlets were found and a default is being used. Returns the flow
        //! direction
        procedure :: find_cell_flow_direction
        //! Find the next non-rejected pixel with the Largest Upstream Drainage Area
        //! and return its coordinates. If there is no remaining valid LUDA then return
        //! a default position along the cell's edge and indicate this through the
        //! no remaning outlets flag that is also returned.
        procedure :: find_pixel_with_LUDA
        //! Find the pixel with the Largest Cell Drainage Area; the pixel that drains the
        //! largest area of the cell itself and return its coordinates.
        procedure :: find_pixel_with_LCDA
        //! Measure the upstream path length through the cell of an outlet pixel at the supplied
        //! coordinates. Return the path length as a double
        procedure :: measure_upstream_path
        //! Check if a given outlet pixel either meets the minimum upstream flow path threshold
        //! or if the outlet pixel is the LCDA. If one of these checks is passed return true
        //! otherwise returns false. Must supply the position of the outlet pixel - the current
        //! LUDA - via the first argument and the position of the LCDA via the second. An optional
        //! third argument can be used to output a flag indicating whether the LUCA is the
        //! LCDA or not.
        procedure :: check_MUFP
        //! Test generating cumulative flow field for this cell. Has no arguments (apart from this)
        //! and returns an array containing the cumulative flow field generated.
        procedure, public :: test_generate_cell_cumulative_flow
        //! Test finding a downstream cell for the yamazaki style algorithm using the location
        //! of a given initial outlet pixel; returns the location of the downstream cell found.
        procedure, public :: yamazaki_test_find_downstream_cell
        //! Generates the cumulative flow for the section of fine orography river directions
        //! covered by this cell. Takes no arguments other than this; returns nothing (except
        //! via modifications to this - i.e. by filling in the cumulative flow array for this
        //! cell object).
        procedure :: generate_cell_cumulative_flow
        //! Function that retrieves a previously marked outlet pixel in this cell and returns
        //! its location. Also returns the outlet pixels type using a intent OUT argument.
        procedure :: yamazaki_retrieve_initial_outlet_pixel
        //! Find and return a list of the location of all the pixels along the edges of this
        //! cell
        virtual vector<coords*>* find_edge_pixels();
        //! Initialize a subfield object to zero to hold the cell cumulative flow.Takes no
        //! arguments other than this; returns nothing (except via modifications to this)
        virtual void initialize_cell_cumulative_flow_subfield();
        //! Initialize a subfield object to false to hold the rejected pixels flow.Takes no
        //! arguments other than this; returns nothing (except via modifications to this)
        virtual void initialize_rejected_pixels_subfield();
        //! Returns a list of the position of the pixels in the cell; takes no arguments
        //! except this
        virtual vector<coords*>* generate_list_of_pixels_in_cell();
        //! Return the appropriate direction indicator for a sink
        virtual direction_indicator* get_flow_direction_for_sink(bool is_for_cell=true);
        //! Return the appropriate direction indicator for an outflow
        virtual direction_indicator* get_flow_direction_for_outflow(bool is_for_cell=true);
        //! Return the appropriate direction indicator for an ocean point
        virtual direction_indicator* get_flow_direction_for_ocean_point(bool is_for_cell=true);
        //! Function to calculate the combined cumulative flow into all the sinks in this
        //! cell and return it; takes no arguments other than this
        virtual int get_sink_combined_cumulative_flow(coords* main_sink_coords = nullptr);
        //! Function to calculate the combined cumulative flow from all the river mouth in this
        //! cell and return it; takes no arguments other than this
        virtual int get_rmouth_outflow_combined_cumulative_flow(main_outflow_coords = nullptr);
        //! Subroutine that set the contains river mouths and ocean cell flags by scanning over
        //! the cell. If no none sea points are found it is marked as an ocean cell. If one or
        //! more river mouths are found then the contains river mouths flag is set to TRUE
        virtual void mark_ocean_and_river_mouth_points();
        //! Subroutine to calculate the river direction as set of indices for the yamazaki
        //! style algorithm and mark it in array of indices pointing to the next cell which
        //! is both given and returned via an argument with intent INOUT.
        virtual void yamazaki_calculate_river_directions_as_indices(
                        int* coarse_river_direction_indices,
                        section_coords* required_coarse_section_coords = nullptr);
}

//! An abstract implementation of the neighborhood class for latitude longitude grids

class latlon_neighborhood : public neighborhood {
    protected:
        //! The bottom row of pixels that are within the center cell this neighborhood
        //! is centered on
        int center_cell_min_lat;
        //! The leftmost row of pixels that are within the center cell this neighborhood
        //! is centered on
        int center_cell_min_lon;
        //! The top row of pixels that are within the center cell this neighborhood
        //! is centered on
        int center_cell_max_lat;
        //! The rightmost row of pixels that are within the center cell this neighborhood
        //! is centered on
        int center_cell_max_lon;
        //! For convience keep the latitudinal width of a cell in the neighborhood
        //! for the yamazaki-style algorithm
        int yamazaki_cell_width_lat;
        //! For convience keep the longitudinal width of a cell in the neighborhood
        //! for the yamazaki-style algorithm
        int yamazaki_cell_width_lon;
        //! Initialize a latitude longitude neighborhood object. Arguments are a set of
        //! coordinates of the bounds of the neighborhood's center cell (the center cell's
        //! section coordinates) and a fine river direction field and a fine cumulative flow
        //! field
        procedure :: init_latlon_neighborhood
        //! Initalize a latitude longitude neighborhood object for the the yamazaki-style
        //! algorithm. First three arguments (after the this argument) are the same as for
        //! init_latlon_neighborhood but has two additional arguments at end of argument
        //! list; one is array of the outlet pixels; second is the coordinates of the edges
        //! of the neighborhood as a whole - as yamazaki-style algorithm is non-local this
        //! is more than direct neighbors - can be the whole grid or a wider subsection of it
        procedure :: yamazaki_init_latlon_neighborhood
        //! Function that given the coordinates of a pixel returns a number indicating which
        //! cell of the neighborhood it is in. This works only for the COTAT+ algorithm
        procedure :: in_which_cell => latlon_in_which_cell
        //! Function that given a the coordinates of a pixel (on the fine grid) returns the
        //! coordinates of the cell that pixel is in on the coarse grid
        procedure :: yamazaki_get_cell_coords => yamazaki_latlon_get_cell_coords
        //! Subroutine to wrap pixel coords around globe for the yamazaki-style
        //! algorithm; takes the pixels location as an argument and returns the
        //! wrapped version of the pixels through the same arguement.
        procedure :: yamazaki_wrap_coordinates => yamazaki_latlon_wrap_coordinates
        //! These next four functions are all shared with various other subclasses as a workaround
        //! for Fortran 2003's lack of multiple inheritance.
        //! Function that given the location of a pixel return the length of the river path
        //! through it
        procedure, nopass :: calculate_length_through_pixel => latlon_calculate_length_through_pixel
        //! Function that given the location of a pixel returns a list of the locations of its
        //! those of it neighbors that are upstream of it.
        procedure, nopass :: find_upstream_neighbors => latlon_find_upstream_neighbors
        //! Function that returns boolean to indicate if a given coordinate is within the neighborhood
        //! (TRUE) or not (FALSE)
        procedure, nopass :: check_if_coords_are_in_area => latlon_check_if_coords_are_in_area
        //! Print information about this neighborhood object out
        procedure, nopass :: print_area => latlon_print_area
}

class irregular_latlon_neighborhood : public neighborhood {
    protected:
        class(latlon_field_section),pointer :: cell_numbers
        type(doubly_linked_list),pointer :: list_of_cells_in_neighborhood
        procedure :: init_irregular_latlon_neighborhood
        procedure :: in_which_cell => irregular_latlon_in_which_cell
        procedure, nopass :: check_if_coords_are_in_area => generic_check_if_coords_are_in_area
        procedure, nopass :: print_area => latlon_print_area
        procedure, nopass :: calculate_length_through_pixel => latlon_calculate_length_through_pixel
        procedure, nopass :: find_upstream_neighbors => latlon_find_upstream_neighbors
        procedure :: yamazaki_get_cell_coords => yamazaki_irregular_latlon_get_cell_coords_dummy
        procedure :: yamazaki_wrap_coordinates => yamazaki_irregular_latlon_wrap_coordinates_dummy
}

//! An abstract implementation of the cell class for latitude longitude grids
class latlon_cell :: public cell {
    protected:
        //! Latitudinal width of the cell in pixels
        int section_width_lat;
        //! Longitudinal width of the cell in pixels
        int section_width_lon;
        //! Initialize a latitude longitude cell object. Arguments are a set of
        //! coordinates of the bounds of the cell and a fine river direction field
        //! and fine cumulative flow field
        procedure :: init_latlon_cell
        //! Initialize a latitude longitude cell object for the yamazaki-style algorithm.
        //! The first three arguements (after the this argument) are the same as
        //! init_latlon_cell. The last argument is an array of outlet pixels.
        procedure :: yamazaki_init_latlon_cell
        //! Function to produce a list of the coordinates of edge pixels of this cell. Takes no
        //! arguments; returns a list of coordinates.
        procedure :: find_edge_pixels => latlon_find_edge_pixels
        //! Function that given the location of a pixel return the length of the river path
        //! through it
        procedure, nopass :: calculate_length_through_pixel => latlon_calculate_length_through_pixel
        //! Function that returns boolean to indicate if a given coordinate is within the cell
        //! (TRUE) or not (FALSE)
        procedure, nopass :: check_if_coords_are_in_area => latlon_check_if_coords_are_in_area
        //! Function that given the location of a pixel returns a list of the locations of its
        //! those of it neighbors that are upstream of it.
        procedure, nopass :: find_upstream_neighbors => latlon_find_upstream_neighbors
        //! Subroutine to initialize the cell cumulative flow subfield to zero. The only
        //! argument is this with intent INOUT. Returns this with the subfield initialized.
        procedure :: initialize_cell_cumulative_flow_subfield => &
            latlon_initialize_cell_cumulative_flow_subfield
        //! Subroutine to initialize the rejected pixels subfield to FALSE. The only argument
        //! is this with intent INOUT. Return this with the subfield initialized.
        procedure :: initialize_rejected_pixels_subfield => latlon_initialize_rejected_pixels_subfield
        //! Returns a list of the position of the pixels in the cell; takes no arguments other than
        //! this
        procedure :: generate_list_of_pixels_in_cell => latlon_generate_list_of_pixels_in_cell
        //! Function to calculate the combined cumulative flow into all the sinks in this
        //! cell and return it; takes no arguments other than this
        procedure :: get_sink_combined_cumulative_flow => latlon_get_sink_combined_cumulative_flow
        //! Function to calculate the combined cumulative flow from all the river mouth in this
        //! cell and return it; takes no arguments other than this
        procedure :: get_rmouth_outflow_combined_cumulative_flow => &
            latlon_get_rmouth_outflow_combined_cumulative_flow
        //! Print information about this cell object out
        procedure, nopass :: print_area => latlon_print_area
        //! Subroutine to calculate the river direction as set of indices for the yamazaki
        //! style algorithm and mark it in array of indices pointing to the next cell which
        //! is both given and returned via an argument with intent INOUT.
        procedure, public :: yamazaki_calculate_river_directions_as_indices => &
            yamazaki_latlon_calculate_river_directions_as_indices
}

//! An abstract implementation of the cell class for latitude longitude grids
class irregular_latlon_cell : public latlon_cell {
    class(latlon_field_section),pointer :: cell_numbers
        protected:
            int cell_number;
            //! Function to produce a list of the coordinates of edge pixels of this cell. Takes no
            //! arguments; returns a list of coordinates.
            procedure :: find_edge_pixels => irregular_latlon_find_edge_pixels
            //! Function that given the location of a pixel return the length of the river path
            //! through it
            procedure, nopass :: check_if_coords_are_in_area => generic_check_if_coords_are_in_area
            //! Returns a list of the position of the pixels in the cell; takes no arguments other than
            //! this
            procedure :: generate_list_of_pixels_in_cell => &
                         irregular_latlon_generate_list_of_pixels_in_cell
            //! Function to calculate the combined cumulative flow into all the sinks in this
            //! cell and return it; takes no arguments other than this
            procedure :: get_sink_combined_cumulative_flow => &
                irregular_latlon_get_sink_combined_cumulative_flow
            //! Function to calculate the combined cumulative flow from all the river mouth in this
            //! cell and return it; takes no arguments other than this
            procedure :: get_rmouth_outflow_combined_cumulative_flow => &
                irregular_latlon_get_rmouth_outflow_combined_cumulative_flow
            procedure ::  init_irregular_latlon_cell
}

//! A concerte subclass of latitude longitude field that uses direction (i.e. a 1-9
//! direction code like a keypad) indicators to specify river directions
class latlon_dir_based_rdirs_field : public latlon_field {
    protected:
        //! Subroutine to find the next pixel downstream from the input pixel and returns it
        //! via the same variable as used for input. Also flag if the cell next coords would
        //! be outside the area and for the yamazaki style algorithm if the next pixel is an
        //! outflow pixel (this is an optional intent OUT argument).
        procedure, nopass :: find_next_pixel_downstream => latlon_dir_based_rdirs_find_next_pixel_downstream
        //! Check if neighbor at a given coordinates flows to a pixel which lies in a given
        //! direction from that neighbor (this direction being expressed in reverse, i.e. pointing
        //! from the pixel to the neighbor)
        procedure, nopass :: neighbor_flows_to_pixel => latlon_dir_based_rdirs_neighbor_flows_to_pixel
        //! Check if the pixel at the input coordinates is a diagonal or not by looking up its flow direction
        procedure, nopass :: is_diagonal => dir_based_rdirs_is_diagonal
}

//! A concerte subclass of icosohedral field that uses an index to specify river directions
class icon_single_index_index_based_rdirs_field : public icon_single_index_field {
    protected:
        int index_for_sink = -5;
        int index_for_outflow = -1;
        int index_for_ocean_point = -2;
        //! Subroutine to find the next pixel downstream from the input pixel and returns it
        //! via the same variable as used for input. Also flag if the cell next coords would
        //! be outside the area and for the yamazaki style algorithm if the next pixel is an
        //! outflow pixel (this is an optional intent OUT argument).
        procedure, nopass :: find_next_pixel_downstream => &
            icon_single_index_index_based_rdirs_find_next_pixel_downstream
        //! Check if neighbor at a given coordinates flows to a pixel which lies in a given
        //! direction from that neighbor (this direction being expressed in reverse, i.e. pointing
        //! from the pixel to the neighbor)
        procedure, nopass :: neighbor_flows_to_pixel => &
            icon_si_index_based_rdirs_neighbor_flows_to_pixel_dummy
        //! Check if the pixel at the input coordinates is a diagonal or not by looking up its flow direction
        procedure, nopass :: is_diagonal => icon_si_index_based_rdirs_is_diagonal_dummy
}

//! A concrete subclass of latitude longitude neighborhood that uses direction
//! (i.e. a 1-9 direction code like a keypad) indicators to specify river directions
class latlon_dir_based_rdirs_neighborhood : public latlon_neighborhood {
    protected:
        //! Check if the pixel at the input coordinates is a diagonal or not by looking up its flow direction
        procedure, nopass :: is_diagonal => dir_based_rdirs_is_diagonal
        //! Subroutine to find the next pixel downstream from the input pixel and returns it
        //! via the same variable as used for input. Also flag if the cell next coords would
        //! be outside the area and for the yamazaki style algorithm if the next pixel is an
        //! outflow pixel (this is an optional intent OUT argument).
        procedure, nopass :: find_next_pixel_downstream => latlon_dir_based_rdirs_find_next_pixel_downstream
        //! Check if neighbor at a given coordinates flows to a pixel which lies in a given
        //! direction from that neighbor (this direction being expressed in reverse, i.e. pointing
        //! from the pixel to the neighbor)
        procedure, nopass :: neighbor_flows_to_pixel => latlon_dir_based_rdirs_neighbor_flows_to_pixel
        //! Calculate a river dierction (1-9 numeric keypad style) from the id of the downstream cell
        procedure :: calculate_direction_indicator => dir_based_rdirs_calculate_direction_indicator
}

//! A concrete subclass of latitude longitude cell that uses direction (i.e a 1-9
//! direction code like a keypad) indicators to specify river directions
class latlon_dir_based_rdirs_cell : public latlon_cell {
    protected:
        //! Flow direction to use for a sink point
        int flow_direction_for_sink = 5;
        //! Flow direction to use for an outflow point
        int flow_direction_for_outflow = 0;
        //! Flow direction to use for a (non-outflow) ocean point
        int flow_direction_for_ocean_point = -1;
        //! Check if the pixel at the input coordinates is a diagonal or not by looking up its flow direction
        procedure, nopass :: is_diagonal => dir_based_rdirs_is_diagonal
        //! Check if neighbor at a given coordinates flows to a pixel which lies in a given
        //! direction from that neighbor (this direction being expressed in reverse, i.e. pointing
        //! from the pixel to the neighbor)
        procedure, nopass :: neighbor_flows_to_pixel => &
            latlon_dir_based_rdirs_neighbor_flows_to_pixel
        //! Initialize a direction based latitude longitude cell object. Arguments are
        //! a set of coordinates of the bounds of the cell and a fine river direction
        //! field and fine cumulative flow field
        procedure :: init_dir_based_rdirs_latlon_cell
        //! Initialize a latitude longitude cell object for the yamazaki-style algorithm.
        //! The first three arguements (after the this argument) are the same as
        //! init_latlon_cell. The fourth argument is an array of outlet pixels. The last
        //! argument is the section coordinates of the neighborhood the cell is going to
        //! be placed in.
        procedure :: yamazaki_init_dir_based_rdirs_latlon_cell
        //! Get the direction code for an sink point
        procedure :: get_flow_direction_for_sink => latlon_dir_based_rdirs_get_flow_direction_for_sink
        //! Get the direction code for an outflow point
        procedure :: get_flow_direction_for_outflow => latlon_dir_based_rdirs_get_flow_direction_for_outflow
        //! Get the direction code for an ocean point
        procedure :: get_flow_direction_for_ocean_point => latlon_dir_based_rdirs_get_flow_direction_for_ocean_point
        //! Subroutine to find the next pixel downstream from the input pixel and returns it
        //! via the same variable as used for input. Also flag if the cell next coords would
        //! be outside the area and for the yamazaki style algorithm if the next pixel is an
        //! outflow pixel (this is an optional intent OUT argument).
        procedure, nopass :: find_next_pixel_downstream => latlon_dir_based_rdirs_find_next_pixel_downstream
        //! Subroutine that set the contains river mouths and ocean cell flags by scanning over
        //! the cell. If no none sea points are found it is marked as an ocean cell. If one or
        //! more river mouths are found then the contains river mouths flag is set to TRUE
        procedure, public :: mark_ocean_and_river_mouth_points => latlon_dir_based_rdirs_mark_ocean_and_river_mouth_points
}

class irregular_latlon_dir_based_rdirs_neighborhood : public irregular_latlon_neighborhood {
    private:
        procedure, nopass :: is_diagonal => dir_based_rdirs_is_diagonal
        procedure, nopass :: find_next_pixel_downstream => latlon_dir_based_rdirs_find_next_pixel_downstream
        procedure, nopass :: neighbor_flows_to_pixel => latlon_dir_based_rdirs_neighbor_flows_to_pixel
        procedure :: calculate_direction_indicator => irregular_latlon_calculate_direction_indicator
}

class irregular_latlon_dir_based_rdirs_cell : public irregular_latlon_cell {
    private:
        int flow_direction_for_sink = 5;
        int flow_direction_for_outflow = 0;
        int flow_direction_for_ocean_point = -1;
        int cell_flow_direction_for_sink = -5;
        int cell_flow_direction_for_outflow = -1;
        int cell_flow_direction_for_ocean_point = -2;
        procedure :: init_irregular_latlon_dir_based_rdirs_cell
        procedure, nopass :: is_diagonal => dir_based_rdirs_is_diagonal
        procedure, nopass :: neighbor_flows_to_pixel => &
            latlon_dir_based_rdirs_neighbor_flows_to_pixel
        procedure :: get_flow_direction_for_sink => &
            irregular_latlon_dir_based_rdirs_get_flow_direction_for_sink
        procedure :: get_flow_direction_for_outflow => &
            irregular_latlon_dir_based_rdirs_get_flow_direction_for_outflow
        procedure :: get_flow_direction_for_ocean_point => &
            irregular_latlon_dir_based_rdirs_get_flow_direction_for_o_point
        procedure, nopass :: find_next_pixel_downstream => latlon_dir_based_rdirs_find_next_pixel_downstream
        procedure, public :: mark_ocean_and_river_mouth_points => &
            irregular_latlon_dir_based_rdirs_mark_o_and_rm_points
}
