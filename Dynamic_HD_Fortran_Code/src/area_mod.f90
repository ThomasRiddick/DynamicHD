module area_mod
use coords_mod
use doubly_linked_list_mod
use subfield_mod
use field_section_mod
use precision_mod
use cotat_parameters_mod, only: MUFP, area_threshold, run_check_for_sinks, &
                                yamazaki_max_range, yamazaki_wrap
implicit none
private

!! This file contains comments using a special markup for doxygen

!> An abstract class containing sections of a number of fields that all correspond to the same section of
!! the underlying grid of pixel and (mostly virtual) operations on this grid section and the various
!! field sections for it.

type, abstract :: area
    private
        !> A section of a field of pixel level river direction
        class(field_section), pointer :: river_directions => null()
        !> A section of a field of pixel level total cumulative flow
        class(field_section), pointer :: total_cumulative_flow => null()
        !> A section of a field used only by the yamazaki style algorithm to mark
        !! pixels as outlet pixels
        class(field_section), pointer :: yamazaki_outlet_pixels => null()
        !The following variables are only used by latlon based area's
        !(Have to insert here due to lack of multiple inheritance)
        !> The minimum latitude defining the section's lower edge (only for
        !! latitude-longitude grids - here due to lack of multiple inheritance
        !! in Fortran)
        integer :: section_min_lat
        !> The minimum longitude defining the section's left edge (only for
        !! latitude-longitude grids - here due to lack of multiple inheritance
        !! in Fortran)
        integer :: section_min_lon
        !> The maximum latitude defining the section's upper edge (only for
        !! latitude-longitude grids - here due to lack of multiple inheritance
        !! in Fortran)
        integer :: section_max_lat
        !> The maximum longitude defining the section's right edge (only for
        !! latitude-longitude grids - here due to lack of multiple inheritance
        !! in Fortran)
        integer :: section_max_lon
        !> Yamazaki style algorithm code for a normal outlet pixel
        integer :: yamazaki_normal_outlet = 1
        !> Yamazaki style algorithm code for the outlet pixel of a cell
        !! where the outlet is not the LCDA (and thus a cell that won't
        !! receive inflow from other cells; any cells flows entering it
        !! will pass through and look for suitable outlets in the cell
        !! beyond; this is because this cells own flow needs to exit via
        !! a pixel that is not its outlet pixel
        integer :: yamazaki_source_cell_outlet = 2
        !> Yamazaki style algorithm code for a sink outlet pixel
        integer :: yamazaki_sink_outlet = -1
        !> Yamazaki style algorithm code for a river mouth pixel
        integer :: yamazaki_rmouth_outlet = -2
    contains
        private
        ! In lieu of a final routine as this feature is not currently (August 2016)
        ! supported by all fortran compilers
        ! final :: destructor
        !> Class destructor
        procedure(destructor), deferred, public :: destructor
        !> Subroutine to find the next pixel upstream from a given pixel and return it via the
        !! input coordinate object; also return a flag if path has left the section and optionally
        !! if an outlet pixel has been found for the yamazaki-style algorithm
        procedure :: find_next_pixel_upstream
        !> Return a list of neighbours that upstream of the pixel located at the input coordinates
        procedure(find_upstream_neighbors), deferred, nopass :: find_upstream_neighbors
        !> Check if a given set of coordinates are within the bounds of this area
        procedure(check_if_coords_are_in_area), deferred, nopass :: check_if_coords_are_in_area
        !> Check if neighbor at a given coordinates flows to a pixel which lies in a given
        !! direction from that neighbor (this direction being expressed in reverse, i.e. pointing
        !! from the pixel to the neighbor)
        procedure(neighbor_flows_to_pixel), deferred, nopass :: neighbor_flows_to_pixel
        !> Subroutine to find the next pixel downstream from the input pixel and returns it
        !! via the same variable as used for input. Also flag if the cell next coords would
        !! be outside the area and for the yamazaki style algorithm if the next pixel is an
        !! outflow pixel (this is an optional intent OUT argument).
        procedure(find_next_pixel_downstream), deferred, nopass :: find_next_pixel_downstream
        !> Work out the length of a path through this pixel but looking up its flow direction
        procedure(calculate_length_through_pixel), deferred, nopass :: calculate_length_through_pixel
        !> Check if the pixel at the input coordinates is a diagonal or not by looking up its flow direction
        procedure(is_diagonal), deferred, nopass :: is_diagonal
        !> Print information about this area object out
        procedure(print_area), deferred, nopass :: print_area
end type area

abstract interface
    !Must explicit pass the object this to this function as pointers to it have the nopass
    !attribute
    function check_if_coords_are_in_area(this,coords_in) result(within_area)
        import area
        import coords
        implicit none
        class(area), intent(in) :: this
        class(coords), intent(in) :: coords_in
        logical :: within_area
    end function check_if_coords_are_in_area

    !Must explicit pass the object this to this function as pointers to it have the nopass
    !attribute
    function find_upstream_neighbors(this,coords_in) result(list_of_neighbors)
        import area
        import coords
        implicit none
        class(area), intent(in) :: this
        class(coords), intent(in) :: coords_in
        class(coords), pointer :: list_of_neighbors(:)
    end function find_upstream_neighbors

    !Must explicit pass the object this to this function as pointers to it have the nopass
    !attribute
    function neighbor_flows_to_pixel(this,coords_in,flipped_direction_from_neighbor)
        import area
        import coords
        import direction_indicator
        implicit none
        class(area), intent(in) :: this
        class(coords), intent(in) :: coords_in
        class(direction_indicator), intent(in) :: flipped_direction_from_neighbor
        logical :: neighbor_flows_to_pixel
    end function neighbor_flows_to_pixel

    !Must explicitly pass the object this to this function as pointers to it have the nopass
    !attribute
    subroutine find_next_pixel_downstream(this,coords_inout,coords_not_in_area,outflow_pixel_reached)
        import area
        import coords
        implicit none
        class(area), intent(in) :: this
        class(coords), intent(inout) :: coords_inout
        logical, intent(out) :: coords_not_in_area
        logical, intent(out), optional :: outflow_pixel_reached
    end subroutine find_next_pixel_downstream

    !Must explicitly pass the object this to this function as pointers to it have the nopass
    !attribute
    subroutine print_area(this)
        import area
        implicit none
        class(area), intent(in) :: this
    end subroutine print_area

    !Must explicitly pass the object this to this function as pointers to it have the nopass
    !attribute
    function calculate_length_through_pixel(this,coords_in) result(length_through_pixel)
        import area
        import coords
        import double_precision
        implicit none
        class(area), intent(in) :: this
        class(coords), intent(in) :: coords_in
        real(kind=double_precision) :: length_through_pixel
    end function calculate_length_through_pixel

    !Must explicitly pass the object this to this function as pointers to it have the nopass
    !attribute
    function is_diagonal(this,coords_in) result(diagonal)
        import area
        import coords
        implicit none
        class(area), intent(in) :: this
        class(coords), intent(in) :: coords_in
        logical :: diagonal
    end function is_diagonal

    subroutine destructor(this)
        import area
        class(area), intent(inout) :: this
    end subroutine destructor

end interface

!> An abstract class (from which concrete subclasses are implemented for different grid type
!! of a area that covers an entire field. This is used by the loop removal tool

type, extends(area), abstract :: field
    private
    !Despite the name this field section actually covers the entire field
    !> Cells that will require to be reprocessed because of loops
    class(field_section), pointer :: cells_to_reprocess => null()
    !> Coordinates of field section to process - usually the entire field
    class(section_coords), allocatable :: field_section_coords
    contains
        private
        ! In lieu of a final routine as this feature is not currently (August 2016)
        ! supported by all fortran compilers
        ! final :: destructor => field_destructor
        !> Class destructor
        procedure, public :: destructor => field_destructor
        !> Run a check for localized loops on a particular cell within field
        procedure :: check_cell_for_localized_loops
        !> Run a check for localized loops within this fields cells
        procedure(check_field_for_localized_loops), deferred, public :: check_field_for_localized_loops
        !> Initialize the cells to reprocess field section object
        procedure(initialize_cells_to_reprocess_field_section), deferred :: &
            initialize_cells_to_reprocess_field_section
end type field

abstract interface
    function check_field_for_localized_loops(this) result(cells_to_reprocess)
        import field
        implicit none
        class(field) :: this
        logical, dimension(:,:), pointer :: cells_to_reprocess
    end function check_field_for_localized_loops

    subroutine initialize_cells_to_reprocess_field_section(this)
        import field
        implicit none
        class(field) :: this
    end subroutine initialize_cells_to_reprocess_field_section
end interface

!> A neighbourhood includes a centre cell and a number of cells surrounding it
!! and it contains functions that need to run over that entire neighborhood for the
!! COTAT+ and yamazaki-style upscaling algorithms

type, extends(area), abstract :: neighborhood
    private
    integer :: center_cell_id
    contains
        private
        ! In lieu of a final routine as this feature is not currently (August 2016)
        ! supported by all fortran compilers
        ! final :: destructor => neighborhood_destructor
        !> Class destructor
        procedure, public :: destructor => neighborhood_destructor
        !> Find the downstream cell for the COTAT+ algorithm from a given starting pixel
        !! returns an integer indicating the downstream cell found
        procedure :: find_downstream_cell
        !> Trace the path from this cell upstream to see if it re-enters the centre
        !! cell before exiting this neighborhood. Returns a flag indicating if this
        !! occurs
        procedure :: path_reenters_region
        !> Check if the cell specified by the input cell id is the centre cell
        procedure :: is_center_cell
        !> Find the downstream cell for the Yamazaki
        procedure :: yamazaki_find_downstream_cell
        !> Calculate an appropriate direction indicator from the id of the downstream cell
        procedure(calculate_direction_indicator), deferred :: calculate_direction_indicator
        !> Calculate which cell a given pixel is in; return an integer identifying that cell
        !! from those in the neighborhood
        procedure(in_which_cell), deferred :: in_which_cell
        !> Get the coordinates of the cell that the input pixel lies in
        procedure(yamazaki_get_cell_coords), deferred :: yamazaki_get_cell_coords
        !> Subroutine that wraps a set of input pixel coordinates round the international date
        !! line if required; return the wrapped coordinates via the same variable as used for
        !! inputting them
        procedure(yamazaki_wrap_coordinates), deferred :: yamazaki_wrap_coordinates
end type neighborhood

abstract interface

    function in_which_cell(this,pixel) result(in_cell)
        import neighborhood
        import coords
        implicit none
        class(neighborhood), intent(in) :: this
        class(coords), intent(in) :: pixel
        integer :: in_cell
    end function in_which_cell

    pure function calculate_direction_indicator(this,downstream_cell) result(flow_direction)
        import neighborhood
        import direction_indicator
        implicit none
        class(neighborhood), intent(in) :: this
        integer, intent(in) :: downstream_cell
        class(direction_indicator), pointer :: flow_direction
    end function calculate_direction_indicator

    pure function yamazaki_get_cell_coords(this,pixel_coords) result(cell_coords)
        import neighborhood
        import coords
        implicit none
        class(neighborhood), intent(in) :: this
        class(coords), intent(in) :: pixel_coords
        class(coords), pointer :: cell_coords
    end function yamazaki_get_cell_coords

    subroutine yamazaki_wrap_coordinates(this,pixel_coords)
        import neighborhood
        import coords
        implicit none
        class(neighborhood), intent(in) :: this
        class(coords), intent(inout) :: pixel_coords
    end subroutine yamazaki_wrap_coordinates

end interface

!> An object to contain the fine field sections corresponding to a single cell of the
!! coarse grid; also includes the necessary functions that act on that cell to
!! produce upscaled COTAT+ and yamazaki style upscalings. Each cell contains a
!! neighborhood object that contains both it and its surrounding cells.

type, extends(area), abstract :: cell
    private
        !> An array marking pixels that have already been rejected as the
        !! outlet pixel of this cell
        class(subfield), pointer :: rejected_pixels => null()
        !> An array containing the cumulative flow calculated across this
        !! cell only (and thus not counting pixels external to this cell)
        class(subfield), pointer :: cell_cumulative_flow => null()
        !> A neighborhood object containing the neighborhood of this cell
        class(neighborhood), allocatable :: cell_neighborhood
        !> The number of pixels that lie along the edge of this cell
        integer :: number_of_edge_pixels
        !> Flag if this cell contains river mouth on the fine grid
        logical :: contains_river_mouths
        !> Flag if this an ocean cell - i.e. the fine pixels inside it are
        !! all marked sea in the land sea mask
        logical :: ocean_cell
        !> The coordinates of the edges of this section within the wider fine fields
        class(section_coords), allocatable :: cell_section_coords
    contains
    private
        !> Process the cell represented by this cell object to produce a river direction
        procedure, public :: process_cell
        !> Find and mark the outlet pixel of this cell for the yamazaki style algorithm;
        !! an input flag indicates if the LCDA criterion is being used
        procedure, public :: yamazaki_mark_cell_outlet_pixel
        !> Setter for the logical variable contain_river_mouths
        procedure, public :: set_contains_river_mouths
        !> Print details of this cell and neighborhoods
        procedure, public :: print_cell_and_neighborhood
        ! In lieu of a final routine as this feature is not currently (August 2016)
        ! supported by all fortran compilers
        ! final :: destructor => cell_destructor
        !> Class destructor
        procedure, public :: destructor => cell_destructor
        !> Find and return the position of the outlet pixel for this cell. Also returns
        !! via an argument a flag to indicate if there was no remaining outlet pixels and
        !! takes a flag whether we should LCDA or not and optionally returns a
        !! flag indicating if this is the LCDA or not
        procedure :: find_outlet_pixel
        !> Subroutine to check if either the sum of the flows to sinks or the sum of the
        !! flows to river mouths is greater than the outflow at the choosen normal outlet
        !! for the cell. Takes as arguments the position of the current normal outlet and
        !! a flag to indicate if there were no remaining outlets (and a default was used)
        !! return two flags indicating whether cell should be marked a sink or a river
        !! mouth outflow. Note only one (or zero) of the flags can be true at a time
        procedure :: check_for_sinks_and_rmouth_outflows
        !> Find the downstream cell for the COTAT+ algorithm and calculate the direction
        !! to it. Takes the position of the outlet pixel and a flag to indicate if
        !! no valid outlets were found and a default is being used. Returns the flow
        !! direction
        procedure :: find_cell_flow_direction
        !> Find the next non-rejected pixel with the Largest Upstream Drainage Area
        !! and return its coordinates. If there is no remaining valid LUDA then return
        !! a default position along the cell's edge and indicate this through the
        !! no remaning outlets flag that is also returned.
        procedure :: find_pixel_with_LUDA
        !> Find the pixel with the Largest Cell Drainage Area; the pixel that drains the
        !! largest area of the cell itself and return its coordinates.
        procedure :: find_pixel_with_LCDA
        !> Measure the upstream path length through the cell of an outlet pixel at the supplied
        !! coordinates. Return the path length as a double
        procedure :: measure_upstream_path
        !> Check if a given outlet pixel either meets the minimum upstream flow path threshold
        !! or if the outlet pixel is the LCDA. If one of these checks is passed return true
        !! otherwise returns false. Must supply the position of the outlet pixel - the current
        !! LUDA - via the first argument and the position of the LCDA via the second. An optional
        !! third argument can be used to output a flag indicating whether the LUCA is the
        !! LCDA or not.
        procedure :: check_MUFP
        !> Test generating cumulative flow field for this cell. Has no arguments (apart from this)
        !! and returns an array containing the cumulative flow field generated.
        procedure, public :: test_generate_cell_cumulative_flow
        !> Test finding a downstream cell for the yamazaki style algorithm using the location
        !! of a given initial outlet pixel; returns the location of the downstream cell found.
        procedure, public :: yamazaki_test_find_downstream_cell
        !> Generates the cumulative flow for the section of fine orography river directions
        !! covered by this cell. Takes no arguments other than this; returns nothing (except
        !! via modifications to this - i.e. by filling in the cumulative flow array for this
        !! cell object).
        procedure :: generate_cell_cumulative_flow
        !> Function that retrieves a previously marked outlet pixel in this cell and returns
        !! its location. Also returns the outlet pixels type using a intent OUT argument.
        procedure :: yamazaki_retrieve_initial_outlet_pixel
        !> Find and return a list of the location of all the pixels along the edges of this
        !! cell
        procedure(find_edge_pixels), deferred :: find_edge_pixels
        !> Initialize a subfield object to zero to hold the cell cumulative flow.Takes no
        !! arguments other than this; returns nothing (except via modifications to this)
        procedure(initialize_cell_cumulative_flow_subfield), deferred :: &
            initialize_cell_cumulative_flow_subfield
        !> Initialize a subfield object to false to hold the rejected pixels flow.Takes no
        !! arguments other than this; returns nothing (except via modifications to this)
        procedure(initialize_rejected_pixels_subfield), deferred :: &
            initialize_rejected_pixels_subfield
        !> Returns a list of the position of the pixels in the cell; takes no arguments
        !! except this
        procedure(generate_list_of_pixels_in_cell), deferred :: generate_list_of_pixels_in_cell
        !> Return the appropriate direction indicator for a sink
        procedure(get_flow_direction_for_sink), deferred :: get_flow_direction_for_sink
        !> Return the appropriate direction indicator for an outflow
        procedure(get_flow_direction_for_outflow), deferred :: get_flow_direction_for_outflow
        !> Return the appropriate direction indicator for an ocean point
        procedure(get_flow_direction_for_ocean_point), deferred :: get_flow_direction_for_ocean_point
        !> Function to calculate the combined cumulative flow into all the sinks in this
        !! cell and return it; takes no arguments other than this
        procedure(get_sink_combined_cumulative_flow), deferred :: get_sink_combined_cumulative_flow
        !> Function to calculate the combined cumulative flow from all the river mouth in this
        !! cell and return it; takes no arguments other than this
        procedure(get_rmouth_outflow_combined_cumulative_flow), deferred :: &
            get_rmouth_outflow_combined_cumulative_flow
        !> Subroutine that set the contains river mouths and ocean cell flags by scanning over
        !! the cell. If no none sea points are found it is marked as an ocean cell. If one or
        !! more river mouths are found then the contains river mouths flag is set to TRUE
        procedure(mark_ocean_and_river_mouth_points), deferred, public :: &
            mark_ocean_and_river_mouth_points
        !> Subroutine to calculate the river direction as set of indices for the yamazaki
        !! style algorithm and mark it in array of indices pointing to the next cell which
        !! is both given and returned via an argument with intent INOUT.
        procedure(yamazaki_calculate_river_directions_as_indices), deferred, public :: &
            yamazaki_calculate_river_directions_as_indices
end type cell

abstract interface
    function find_edge_pixels(this) result(edge_pixel_coords_list)
        import cell
        import coords
        implicit none
        class(cell), intent(inout) :: this
        class(coords), pointer :: edge_pixel_coords_list(:)
    end function find_edge_pixels

    subroutine initialize_cell_cumulative_flow_subfield(this)
        import cell
        implicit none
        class(cell), intent(inout) :: this
    end subroutine initialize_cell_cumulative_flow_subfield

    subroutine initialize_rejected_pixels_subfield(this)
        import cell
        implicit none
        class(cell), intent(inout) :: this
    end subroutine initialize_rejected_pixels_subfield

    function generate_list_of_pixels_in_cell(this) result(list_of_pixels_in_cell)
        import cell
        import doubly_linked_list
        implicit none
        class(cell), intent(in) :: this
        type(doubly_linked_list), pointer :: list_of_pixels_in_cell
    end function generate_list_of_pixels_in_cell

    pure function get_flow_direction_for_sink(this,is_for_cell) result(flow_direction_for_sink)
        import cell
        import coords
        import direction_indicator
        implicit none
        class(cell), intent(in) :: this
        class(direction_indicator), pointer :: flow_direction_for_sink
        logical, optional, intent(in) :: is_for_cell
    end function get_flow_direction_for_sink

    pure function get_flow_direction_for_outflow(this,is_for_cell) result(flow_direction_for_outflow)
        import cell
        import coords
        import direction_indicator
        implicit none
        class(cell), intent(in) :: this
        class(direction_indicator), pointer :: flow_direction_for_outflow
        logical, optional, intent(in) :: is_for_cell
    end function get_flow_direction_for_outflow

    pure function get_flow_direction_for_ocean_point(this,is_for_cell) result(flow_direction_for_ocean_point)
        import cell
        import coords
        import direction_indicator
        implicit none
        class(cell), intent(in) :: this
        class(direction_indicator), pointer :: flow_direction_for_ocean_point
        logical, optional, intent(in) :: is_for_cell
    end function get_flow_direction_for_ocean_point

    function get_sink_combined_cumulative_flow(this,main_sink_coords) &
        result(combined_cumulative_flow_to_sinks)
        import cell
        import coords
        class(cell), intent(in) :: this
        class(coords), pointer, optional,intent(out) :: main_sink_coords
        integer :: combined_cumulative_flow_to_sinks
    end function get_sink_combined_cumulative_flow

    function get_rmouth_outflow_combined_cumulative_flow(this,main_outflow_coords) &
        result(combined_cumulative_rmouth_outflow)
        import cell
        import coords
        class(cell), intent(in) :: this
        class(coords), pointer, optional,intent(out) :: main_outflow_coords
        integer :: combined_cumulative_rmouth_outflow
    end function get_rmouth_outflow_combined_cumulative_flow

    subroutine mark_ocean_and_river_mouth_points(this)
        import cell
        class(cell), intent(inout) :: this
    end subroutine mark_ocean_and_river_mouth_points

    subroutine yamazaki_calculate_river_directions_as_indices(this,coarse_river_direction_indices, &
                                                              required_coarse_section_coords)
        import cell
        import section_coords
        class(cell), intent(inout) :: this
        integer, dimension(:,:,:), intent(inout) :: coarse_river_direction_indices
        class(section_coords), intent(in), optional :: required_coarse_section_coords
    end subroutine yamazaki_calculate_river_directions_as_indices
end interface

!> A abstract implementation of the field class for latitude longitude grids

type, extends(field), abstract :: latlon_field
    !> Width of this field in terms of number of latitudinal cells
    integer :: section_width_lat
    !> Width of this field in terms of number of longitudinal cells
    integer :: section_width_lon
    contains
        private
        !> Initialized a latlon field object from a set of coordinates for the edges of the field section
        !! and a 2D array of river directions
        procedure, public :: init_latlon_field
        !> Check this field objects field for localized loops; return a list of cells with localized
        !! loops that need to be reprocessed as a 2D boolean array
        procedure, public :: check_field_for_localized_loops => latlon_check_field_for_localized_loops
        !! These next four functions are all shared with various other subclasses as a workaround
        !! for Fortran 2003's lack of multiple inheritance.
        !> Function that returns boolean to indicate if a given coordinate is within the field
        !! (TRUE) or not (FALSE)
        procedure, nopass :: check_if_coords_are_in_area => latlon_check_if_coords_are_in_area
        !> Print information about this field object out
        procedure, nopass :: print_area => latlon_print_area
        !> Function that given the location of a pixel returns a list of the locations of its
        !! those of it neighbors that are upstream of it.
        procedure, nopass :: find_upstream_neighbors => latlon_find_upstream_neighbors
        !> Function that given the location of a pixel return the length of the river path
        !! through it
        procedure, nopass :: calculate_length_through_pixel => latlon_calculate_length_through_pixel
        !> Initialize a 2D array to store flags whether cells need to be reprocessed or not; set all
        !! entries initial value to false. A subroutine that takes no arguments (and hence also
        !! returns nothing.
        procedure :: initialize_cells_to_reprocess_field_section => &
            latlon_initialize_cells_to_reprocess_field_section
end type latlon_field

!> A abstract implementation of the field class for icon icosohedral grids

type, extends(field), abstract :: icon_single_index_field
    integer :: ncells
    contains
        private
        !> Initialized a icon icosohedral field object from a set of coordinates for the edges of the field section
        !! and a 2D array of river directions
        procedure, public :: init_icon_single_index_field
        !> Check this field objects field for localized loops; return a list of cells with localized
        !! loops that need to be reprocessed as a 2D boolean array
        procedure, public :: check_field_for_localized_loops => icon_single_index_check_field_for_localized_loops
        !! These next four functions are all shared with various other subclasses as a workaround
        !! for Fortran 2003's lack of multiple inheritance.
        !> Function that returns boolean to indicate if a given coordinate is within the field
        !! (TRUE) or not (FALSE)
        procedure, nopass :: check_if_coords_are_in_area => generic_check_if_coords_are_in_area
        !> Print information about this field object out
        procedure, nopass :: print_area => icon_single_index_field_dummy_subroutine
        !> Function that given the location of a pixel returns a list of the locations of its
        !! those of it neighbors that are upstream of it.
        procedure, nopass :: find_upstream_neighbors => icon_single_index_field_dummy_coords_pointer_function
        !> Function that given the location of a pixel return the length of the river path
        !! through it
        procedure, nopass :: calculate_length_through_pixel => icon_single_index_field_dummy_real_function
        !> Initialize a 2D array to store flags whether cells need to be reprocessed or not; set all
        !! entries initial value to false. A subroutine that takes no arguments (and hence also
        !! returns nothing.
        procedure :: initialize_cells_to_reprocess_field_section => &
            icon_single_index_initialize_cells_to_reprocess_field_section
end type icon_single_index_field

!> An abstract implementation of the neighborhood class for latitude longitude grids

type, extends(neighborhood), abstract :: latlon_neighborhood
    private
        !> The bottom row of pixels that are within the center cell this neighborhood
        !! is centered on
        integer :: center_cell_min_lat
        !> The leftmost row of pixels that are within the center cell this neighborhood
        !! is centered on
        integer :: center_cell_min_lon
        !> The top row of pixels that are within the center cell this neighborhood
        !! is centered on
        integer :: center_cell_max_lat
        !> The rightmost row of pixels that are within the center cell this neighborhood
        !! is centered on
        integer :: center_cell_max_lon
        !> For convience keep the latitudinal width of a cell in the neighborhood
        !! for the yamazaki-style algorithm
        integer :: yamazaki_cell_width_lat
        !> For convience keep the longitudinal width of a cell in the neighborhood
        !! for the yamazaki-style algorithm
        integer :: yamazaki_cell_width_lon
    contains
        !> Initialize a latitude longitude neighborhood object. Arguments are a set of
        !! coordinates of the bounds of the neighborhood's center cell (the center cell's
        !! section coordinates) and a fine river direction field and a fine cumulative flow
        !! field
        procedure :: init_latlon_neighborhood
        !> Initalize a latitude longitude neighborhood object for the the yamazaki-style
        !! algorithm. First three arguments (after the this argument) are the same as for
        !! init_latlon_neighborhood but has two additional arguments at end of argument
        !! list; one is array of the outlet pixels; second is the coordinates of the edges
        !! of the neighborhood as a whole - as yamazaki-style algorithm is non-local this
        !! is more than direct neighbors - can be the whole grid or a wider subsection of it
        procedure :: yamazaki_init_latlon_neighborhood
        !> Function that given the coordinates of a pixel returns a number indicating which
        !! cell of the neighborhood it is in. This works only for the COTAT+ algorithm
        procedure :: in_which_cell => latlon_in_which_cell
        !> Function that given a the coordinates of a pixel (on the fine grid) returns the
        !! coordinates of the cell that pixel is in on the coarse grid
        procedure :: yamazaki_get_cell_coords => yamazaki_latlon_get_cell_coords
        !> Subroutine to wrap pixel coords around globe for the yamazaki-style
        !! algorithm; takes the pixels location as an argument and returns the
        !! wrapped version of the pixels through the same arguement.
        procedure :: yamazaki_wrap_coordinates => yamazaki_latlon_wrap_coordinates
        !! These next four functions are all shared with various other subclasses as a workaround
        !! for Fortran 2003's lack of multiple inheritance.
        !> Function that given the location of a pixel return the length of the river path
        !! through it
        procedure, nopass :: calculate_length_through_pixel => latlon_calculate_length_through_pixel
        !> Function that given the location of a pixel returns a list of the locations of its
        !! those of it neighbors that are upstream of it.
        procedure, nopass :: find_upstream_neighbors => latlon_find_upstream_neighbors
        !> Function that returns boolean to indicate if a given coordinate is within the neighborhood
        !! (TRUE) or not (FALSE)
        procedure, nopass :: check_if_coords_are_in_area => latlon_check_if_coords_are_in_area
        !> Print information about this neighborhood object out
        procedure, nopass :: print_area => latlon_print_area
end type latlon_neighborhood

type, extends(neighborhood), abstract :: irregular_latlon_neighborhood
    private
        class(latlon_field_section),pointer :: cell_numbers
        type(doubly_linked_list),pointer :: list_of_cells_in_neighborhood
    contains
        procedure :: init_irregular_latlon_neighborhood
        procedure :: in_which_cell => irregular_latlon_in_which_cell
        procedure, nopass :: check_if_coords_are_in_area => generic_check_if_coords_are_in_area
        procedure, nopass :: print_area => latlon_print_area
        procedure, nopass :: calculate_length_through_pixel => latlon_calculate_length_through_pixel
        procedure, nopass :: find_upstream_neighbors => latlon_find_upstream_neighbors
        procedure :: yamazaki_get_cell_coords => yamazaki_irregular_latlon_get_cell_coords_dummy
        procedure :: yamazaki_wrap_coordinates => yamazaki_irregular_latlon_wrap_coordinates_dummy
        procedure :: destructor => irregular_latlon_neighborhood_destructor
end type irregular_latlon_neighborhood

!> An abstract implementation of the cell class for latitude longitude grids
type, extends(cell), abstract :: latlon_cell
    !> Latitudinal width of the cell in pixels
    integer :: section_width_lat
    !> Longitudinal width of the cell in pixels
    integer :: section_width_lon
    contains
        !> Initialize a latitude longitude cell object. Arguments are a set of
        !! coordinates of the bounds of the cell and a fine river direction field
        !! and fine cumulative flow field
        procedure :: init_latlon_cell
        !> Initialize a latitude longitude cell object for the yamazaki-style algorithm.
        !! The first three arguements (after the this argument) are the same as
        !! init_latlon_cell. The last argument is an array of outlet pixels.
        procedure :: yamazaki_init_latlon_cell
        !> Function to produce a list of the coordinates of edge pixels of this cell. Takes no
        !! arguments; returns a list of coordinates.
        procedure :: find_edge_pixels => latlon_find_edge_pixels
        !> Function that given the location of a pixel return the length of the river path
        !! through it
        procedure, nopass :: calculate_length_through_pixel => latlon_calculate_length_through_pixel
        !> Function that returns boolean to indicate if a given coordinate is within the cell
        !! (TRUE) or not (FALSE)
        procedure, nopass :: check_if_coords_are_in_area => latlon_check_if_coords_are_in_area
        !> Function that given the location of a pixel returns a list of the locations of its
        !! those of it neighbors that are upstream of it.
        procedure, nopass :: find_upstream_neighbors => latlon_find_upstream_neighbors
        !> Subroutine to initialize the cell cumulative flow subfield to zero. The only
        !! argument is this with intent INOUT. Returns this with the subfield initialized.
        procedure :: initialize_cell_cumulative_flow_subfield => &
            latlon_initialize_cell_cumulative_flow_subfield
        !> Subroutine to initialize the rejected pixels subfield to FALSE. The only argument
        !! is this with intent INOUT. Return this with the subfield initialized.
        procedure :: initialize_rejected_pixels_subfield => latlon_initialize_rejected_pixels_subfield
        !> Returns a list of the position of the pixels in the cell; takes no arguments other than
        !! this
        procedure :: generate_list_of_pixels_in_cell => latlon_generate_list_of_pixels_in_cell
        !> Function to calculate the combined cumulative flow into all the sinks in this
        !! cell and return it; takes no arguments other than this
        procedure :: get_sink_combined_cumulative_flow => latlon_get_sink_combined_cumulative_flow
        !> Function to calculate the combined cumulative flow from all the river mouth in this
        !! cell and return it; takes no arguments other than this
        procedure :: get_rmouth_outflow_combined_cumulative_flow => &
            latlon_get_rmouth_outflow_combined_cumulative_flow
        !> Print information about this cell object out
        procedure, nopass :: print_area => latlon_print_area
        !> Subroutine to calculate the river direction as set of indices for the yamazaki
        !! style algorithm and mark it in array of indices pointing to the next cell which
        !! is both given and returned via an argument with intent INOUT.
        procedure, public :: yamazaki_calculate_river_directions_as_indices => &
            yamazaki_latlon_calculate_river_directions_as_indices
end type latlon_cell

!> An abstract implementation of the cell class for latitude longitude grids
type, extends(latlon_cell), abstract :: irregular_latlon_cell
    class(latlon_field_section),pointer :: cell_numbers
    integer :: cell_number
    contains
        !> Function to produce a list of the coordinates of edge pixels of this cell. Takes no
        !! arguments; returns a list of coordinates.
        procedure :: find_edge_pixels => irregular_latlon_find_edge_pixels
        !> Function that given the location of a pixel return the length of the river path
        !! through it
        procedure, nopass :: check_if_coords_are_in_area => generic_check_if_coords_are_in_area
        !> Returns a list of the position of the pixels in the cell; takes no arguments other than
        !! this
        procedure :: generate_list_of_pixels_in_cell => &
                     irregular_latlon_generate_list_of_pixels_in_cell
        !> Function to calculate the combined cumulative flow into all the sinks in this
        !! cell and return it; takes no arguments other than this
        procedure :: get_sink_combined_cumulative_flow => &
            irregular_latlon_get_sink_combined_cumulative_flow
        !> Function to calculate the combined cumulative flow from all the river mouth in this
        !! cell and return it; takes no arguments other than this
        procedure :: get_rmouth_outflow_combined_cumulative_flow => &
            irregular_latlon_get_rmouth_outflow_combined_cumulative_flow
        procedure ::  init_irregular_latlon_cell
        procedure :: destructor => irregular_latlon_cell_destructor
end type irregular_latlon_cell

!> A concerte subclass of latitude longitude field that uses direction (i.e. a 1-9
!! direction code like a keypad) indicators to specify river directions
type, public, extends(latlon_field) :: latlon_dir_based_rdirs_field
    private
    contains
        private
        !> Subroutine to find the next pixel downstream from the input pixel and returns it
        !! via the same variable as used for input. Also flag if the cell next coords would
        !! be outside the area and for the yamazaki style algorithm if the next pixel is an
        !! outflow pixel (this is an optional intent OUT argument).
        procedure, nopass :: find_next_pixel_downstream => latlon_dir_based_rdirs_find_next_pixel_downstream
        !> Check if neighbor at a given coordinates flows to a pixel which lies in a given
        !! direction from that neighbor (this direction being expressed in reverse, i.e. pointing
        !! from the pixel to the neighbor)
        procedure, nopass :: neighbor_flows_to_pixel => latlon_dir_based_rdirs_neighbor_flows_to_pixel
        !> Check if the pixel at the input coordinates is a diagonal or not by looking up its flow direction
        procedure, nopass :: is_diagonal => dir_based_rdirs_is_diagonal
end type latlon_dir_based_rdirs_field

interface  latlon_dir_based_rdirs_field
    procedure latlon_dir_based_rdirs_field_constructor
end interface

!> A concerte subclass of icosohedral field that uses an index to specify river directions
type, public, extends(icon_single_index_field) :: icon_single_index_index_based_rdirs_field
    private
    integer :: index_for_sink = -5
    integer :: index_for_outflow = -1
    integer :: index_for_ocean_point = -2
    contains
        private
        !> Subroutine to find the next pixel downstream from the input pixel and returns it
        !! via the same variable as used for input. Also flag if the cell next coords would
        !! be outside the area and for the yamazaki style algorithm if the next pixel is an
        !! outflow pixel (this is an optional intent OUT argument).
        procedure, nopass :: find_next_pixel_downstream => &
            icon_single_index_index_based_rdirs_find_next_pixel_downstream
        !> Check if neighbor at a given coordinates flows to a pixel which lies in a given
        !! direction from that neighbor (this direction being expressed in reverse, i.e. pointing
        !! from the pixel to the neighbor)
        procedure, nopass :: neighbor_flows_to_pixel => &
            icon_si_index_based_rdirs_neighbor_flows_to_pixel_dummy
        !> Check if the pixel at the input coordinates is a diagonal or not by looking up its flow direction
        procedure, nopass :: is_diagonal => icon_si_index_based_rdirs_is_diagonal_dummy
end type icon_single_index_index_based_rdirs_field

interface  icon_single_index_index_based_rdirs_field
    procedure icon_single_index_index_based_rdirs_field_constructor
end interface

!> A concrete subclass of latitude longitude neighborhood that uses direction
!! (i.e. a 1-9 direction code like a keypad) indicators to specify river directions
type, extends(latlon_neighborhood) :: latlon_dir_based_rdirs_neighborhood
    private
    contains
    private
        !> Check if the pixel at the input coordinates is a diagonal or not by looking up its flow direction
        procedure, nopass :: is_diagonal => dir_based_rdirs_is_diagonal
        !> Subroutine to find the next pixel downstream from the input pixel and returns it
        !! via the same variable as used for input. Also flag if the cell next coords would
        !! be outside the area and for the yamazaki style algorithm if the next pixel is an
        !! outflow pixel (this is an optional intent OUT argument).
        procedure, nopass :: find_next_pixel_downstream => latlon_dir_based_rdirs_find_next_pixel_downstream
        !> Check if neighbor at a given coordinates flows to a pixel which lies in a given
        !! direction from that neighbor (this direction being expressed in reverse, i.e. pointing
        !! from the pixel to the neighbor)
        procedure, nopass :: neighbor_flows_to_pixel => latlon_dir_based_rdirs_neighbor_flows_to_pixel
        !> Calculate a river dierction (1-9 numeric keypad style) from the id of the downstream cell
        procedure :: calculate_direction_indicator => dir_based_rdirs_calculate_direction_indicator
end type latlon_dir_based_rdirs_neighborhood

interface latlon_dir_based_rdirs_neighborhood
    procedure latlon_dir_based_rdirs_neighborhood_constructor, &
              yamazaki_latlon_dir_based_rdirs_neighborhood_constructor
end interface latlon_dir_based_rdirs_neighborhood

!> A concrete subclass of latitude longitude cell that uses direction (i.e a 1-9
!! direction code like a keypad) indicators to specify river directions
type, public, extends(latlon_cell) :: latlon_dir_based_rdirs_cell
    !> Flow direction to use for a sink point
    integer :: flow_direction_for_sink = 5
    !> Flow direction to use for an outflow point
    integer :: flow_direction_for_outflow = 0
    !> Flow direction to use for a (non-outflow) ocean point
    integer :: flow_direction_for_ocean_point = -1
contains
    !> Check if the pixel at the input coordinates is a diagonal or not by looking up its flow direction
    procedure, nopass :: is_diagonal => dir_based_rdirs_is_diagonal
    !> Check if neighbor at a given coordinates flows to a pixel which lies in a given
    !! direction from that neighbor (this direction being expressed in reverse, i.e. pointing
    !! from the pixel to the neighbor)
    procedure, nopass :: neighbor_flows_to_pixel => &
        latlon_dir_based_rdirs_neighbor_flows_to_pixel
    !> Initialize a direction based latitude longitude cell object. Arguments are
    !! a set of coordinates of the bounds of the cell and a fine river direction
    !! field and fine cumulative flow field
    procedure :: init_dir_based_rdirs_latlon_cell
    !> Initialize a latitude longitude cell object for the yamazaki-style algorithm.
    !! The first three arguements (after the this argument) are the same as
    !! init_latlon_cell. The fourth argument is an array of outlet pixels. The last
    !! argument is the section coordinates of the neighborhood the cell is going to
    !! be placed in.
    procedure :: yamazaki_init_dir_based_rdirs_latlon_cell
    !> Get the direction code for an sink point
    procedure :: get_flow_direction_for_sink => latlon_dir_based_rdirs_get_flow_direction_for_sink
    !> Get the direction code for an outflow point
    procedure :: get_flow_direction_for_outflow => latlon_dir_based_rdirs_get_flow_direction_for_outflow
    !> Get the direction code for an ocean point
    procedure :: get_flow_direction_for_ocean_point => latlon_dir_based_rdirs_get_flow_direction_for_ocean_point
    !> Subroutine to find the next pixel downstream from the input pixel and returns it
    !! via the same variable as used for input. Also flag if the cell next coords would
    !! be outside the area and for the yamazaki style algorithm if the next pixel is an
    !! outflow pixel (this is an optional intent OUT argument).
    procedure, nopass :: find_next_pixel_downstream => latlon_dir_based_rdirs_find_next_pixel_downstream
    !> Subroutine that set the contains river mouths and ocean cell flags by scanning over
    !! the cell. If no none sea points are found it is marked as an ocean cell. If one or
    !! more river mouths are found then the contains river mouths flag is set to TRUE
    procedure, public :: mark_ocean_and_river_mouth_points => latlon_dir_based_rdirs_mark_ocean_and_river_mouth_points
end type latlon_dir_based_rdirs_cell

interface latlon_dir_based_rdirs_cell
    procedure latlon_dir_based_rdirs_cell_constructor,yamazaki_latlon_dir_based_rdirs_cell_constructor
end interface latlon_dir_based_rdirs_cell

type, public, extends(irregular_latlon_neighborhood) :: irregular_latlon_dir_based_rdirs_neighborhood
    private
    contains
    private
        procedure, nopass :: is_diagonal => dir_based_rdirs_is_diagonal
        procedure, nopass :: find_next_pixel_downstream => latlon_dir_based_rdirs_find_next_pixel_downstream
        procedure, nopass :: neighbor_flows_to_pixel => latlon_dir_based_rdirs_neighbor_flows_to_pixel
        procedure :: calculate_direction_indicator => irregular_latlon_calculate_direction_indicator

end type irregular_latlon_dir_based_rdirs_neighborhood

interface irregular_latlon_dir_based_rdirs_neighborhood
    procedure :: irregular_latlon_dir_based_rdirs_neighborhood_constructor
end interface irregular_latlon_dir_based_rdirs_neighborhood

type, public, extends(irregular_latlon_cell) :: irregular_latlon_dir_based_rdirs_cell
    integer :: flow_direction_for_sink = 5
    integer :: flow_direction_for_outflow = 0
    integer :: flow_direction_for_ocean_point = -1
    integer :: cell_flow_direction_for_sink = -5
    integer :: cell_flow_direction_for_outflow = -1
    integer :: cell_flow_direction_for_ocean_point = -2
contains
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
end type irregular_latlon_dir_based_rdirs_cell

interface irregular_latlon_dir_based_rdirs_cell
    procedure :: irregular_latlon_dir_based_rdirs_cell_constructor
end interface irregular_latlon_dir_based_rdirs_cell

contains

    subroutine find_next_pixel_upstream(this,coords_inout,coords_not_in_area)
        class(area), intent(in) :: this
        class(coords), allocatable, intent(inout) :: coords_inout
        logical, intent(out) :: coords_not_in_area
        class(coords), pointer :: upstream_neighbors_coords_list(:)
        class(*), pointer :: upstream_neighbor_total_cumulative_flow
        integer :: max_cmltv_flow_working_value
        integer :: i
        max_cmltv_flow_working_value = 0
        upstream_neighbors_coords_list => this%find_upstream_neighbors(this,coords_inout)
        if (size(upstream_neighbors_coords_list) > 0) then
            do i = 1, size(upstream_neighbors_coords_list)
                upstream_neighbor_total_cumulative_flow => &
                    this%total_cumulative_flow%get_value(upstream_neighbors_coords_list(i))
                select type (upstream_neighbor_total_cumulative_flow)
                type is (integer)
                    if ( upstream_neighbor_total_cumulative_flow > max_cmltv_flow_working_value) then
                        if (allocated(coords_inout)) deallocate(coords_inout)
                        allocate(coords_inout,source=upstream_neighbors_coords_list(i))
                        max_cmltv_flow_working_value = upstream_neighbor_total_cumulative_flow
                    end if
                end select
                deallocate(upstream_neighbor_total_cumulative_flow)
            end do
            coords_not_in_area = .not. this%check_if_coords_are_in_area(this,coords_inout)
        else
            coords_not_in_area = .TRUE.
        end if
        deallocate(upstream_neighbors_coords_list)
    end subroutine find_next_pixel_upstream

    subroutine field_destructor(this)
        class(field), intent(inout) :: this
            if (associated(this%cells_to_reprocess)) call this%cells_to_reprocess%deallocate_data()
            if (associated(this%river_directions)) deallocate(this%river_directions)
            if (associated(this%total_cumulative_flow)) deallocate(this%total_cumulative_flow)
            if (associated(this%cells_to_reprocess)) deallocate(this%cells_to_reprocess)
    end subroutine field_destructor

    function check_cell_for_localized_loops(this,coords_inout) result(localized_loop_found)
        class(field)  :: this
        class(coords) :: coords_inout
        logical :: localized_loop_found
        class(coords),allocatable :: initial_cell_coords
        logical :: coords_not_in_area_dummy
        logical :: is_sea_point_or_horizontal_edge_or_sink
            allocate(initial_cell_coords,source=coords_inout)
            call this%find_next_pixel_downstream(this,coords_inout,coords_not_in_area_dummy)
            if (coords_inout%are_equal_to(initial_cell_coords)) then
                is_sea_point_or_horizontal_edge_or_sink = .True.
            else
                is_sea_point_or_horizontal_edge_or_sink = .False.
            end if
            call this%find_next_pixel_downstream(this,coords_inout,coords_not_in_area_dummy)
            if (coords_inout%are_equal_to(initial_cell_coords) .and. &
                .not. is_sea_point_or_horizontal_edge_or_sink) then
                localized_loop_found = .True.
            else
                localized_loop_found = .False.
            end if
            deallocate(initial_cell_coords)
    end function check_cell_for_localized_loops

    function find_outlet_pixel(this,no_remaining_outlets,use_LCDA_criterion,outlet_is_LCDA) &
        result(LUDA_pixel_coords)
        class(cell) :: this
        logical, intent(inout) :: no_remaining_outlets
        logical, intent(in)    :: use_LCDA_criterion
        logical, intent(inout), optional :: outlet_is_LCDA
        class(coords), pointer :: LUDA_pixel_coords
        class(coords), pointer :: LCDA_pixel_coords => null()
            nullify(LUDA_pixel_coords)
            if(use_LCDA_criterion) then
                call this%generate_cell_cumulative_flow()
                LCDA_pixel_coords=>this%find_pixel_with_LCDA()
            else
                nullify(LCDA_pixel_coords)
            end if
            do
                if (associated(LUDA_pixel_coords)) deallocate(LUDA_pixel_coords)
                LUDA_pixel_coords=>this%find_pixel_with_LUDA(no_remaining_outlets)
                if (no_remaining_outlets) then
                    if (present(outlet_is_LCDA)) outlet_is_LCDA = .False.
                    exit
                else
                    if (present(outlet_is_LCDA)) then
                        if (this%check_MUFP(LUDA_pixel_coords,LCDA_pixel_coords,outlet_is_LCDA)) exit
                    else
                        if (this%check_MUFP(LUDA_pixel_coords,LCDA_pixel_coords)) exit
                    end if
                end if
            end do
            if (use_LCDA_criterion) deallocate(LCDA_pixel_coords)
    end function find_outlet_pixel

    subroutine check_for_sinks_and_rmouth_outflows(this,outlet_pixel_coords,no_remaining_outlets, &
                                                   mark_sink,mark_outflow)
        class(cell) :: this
        class(coords), pointer, intent(inout) :: outlet_pixel_coords
        logical, intent(in) :: no_remaining_outlets
        logical, intent(out) :: mark_sink
        logical, intent(out) :: mark_outflow
        class(*), pointer :: outlet_pixel_cumulative_flow_value
        class(coords), pointer :: main_sink_coords
        class(coords), pointer :: main_river_mouth_coords
        integer :: outlet_pixel_cumulative_flow
        integer :: cumulative_sink_outflow
        integer :: cumulative_rmouth_outflow
            mark_sink = .False.
            mark_outflow = .False.
            if (no_remaining_outlets) then
                outlet_pixel_cumulative_flow = -1
            else
                outlet_pixel_cumulative_flow_value => &
                    this%total_cumulative_flow%get_value(outlet_pixel_coords)
                select type(outlet_pixel_cumulative_flow_value)
                type is (integer)
                    outlet_pixel_cumulative_flow = outlet_pixel_cumulative_flow_value
                end select
                deallocate(outlet_pixel_cumulative_flow_value)
            end if
            if (run_check_for_sinks) then
                cumulative_sink_outflow = &
                    this%get_sink_combined_cumulative_flow(main_sink_coords)
            else
                cumulative_sink_outflow = 0
            end if
            if (this%contains_river_mouths) then
                cumulative_rmouth_outflow = &
                    this%get_rmouth_outflow_combined_cumulative_flow(main_river_mouth_coords)
            else
                cumulative_rmouth_outflow = -1
            end if
            if ( cumulative_sink_outflow > cumulative_rmouth_outflow .and. &
                 cumulative_sink_outflow > outlet_pixel_cumulative_flow ) then
                 mark_sink = .True.
                 deallocate(outlet_pixel_coords)
                 if (this%contains_river_mouths) deallocate(main_river_mouth_coords)
                 outlet_pixel_coords =>  main_sink_coords
            else if (  cumulative_rmouth_outflow > outlet_pixel_cumulative_flow) then
                 mark_outflow = .True.
                 deallocate(outlet_pixel_coords)
                 if (run_check_for_sinks) deallocate(main_sink_coords)
                 outlet_pixel_coords =>  main_river_mouth_coords
            else
                if (this%contains_river_mouths) deallocate(main_river_mouth_coords)
                if (run_check_for_sinks) deallocate(main_sink_coords)
            end if
    end subroutine check_for_sinks_and_rmouth_outflows


    function find_cell_flow_direction(this,outlet_pixel_coords,no_remaining_outlets) &
        result(flow_direction)
        class(cell) :: this
        class(coords), pointer, intent(inout) :: outlet_pixel_coords
        logical, intent(in) :: no_remaining_outlets
        class(direction_indicator), pointer :: flow_direction
        class(*), allocatable :: downstream_cell
        logical :: mark_sink
        logical :: mark_outflow
            call this%check_for_sinks_and_rmouth_outflows(outlet_pixel_coords,no_remaining_outlets,mark_sink, &
                                                          mark_outflow)
            if (mark_sink) then
                flow_direction => this%get_flow_direction_for_sink(is_for_cell=.true.)
                return
            else if (mark_outflow) then
                flow_direction => this%get_flow_direction_for_outflow(is_for_cell=.true.)
                return
            end if
            allocate(downstream_cell,source=this%cell_neighborhood%find_downstream_cell(outlet_pixel_coords))
            select type (downstream_cell)
            type is (integer)
                flow_direction => this%cell_neighborhood%calculate_direction_indicator(downstream_cell)
            end select
    end function

    function process_cell(this) result(flow_direction)
        class(cell) :: this
        class(direction_indicator), pointer :: flow_direction
        class(coords), pointer :: outlet_pixel_coords => null()
        logical :: no_remaining_outlets
        logical :: use_LCDA_criterion = .True.
            if (.not. this%ocean_cell) then
                outlet_pixel_coords => this%find_outlet_pixel(no_remaining_outlets,use_LCDA_criterion)
                flow_direction => this%find_cell_flow_direction(outlet_pixel_coords,no_remaining_outlets)
                deallocate(outlet_pixel_coords)
            else
                flow_direction => this%get_flow_direction_for_ocean_point(is_for_cell=.true.)
            end if
    end function process_cell

    subroutine yamazaki_mark_cell_outlet_pixel(this,use_LCDA_criterion)
        class(cell) :: this
        logical, intent(in) :: use_LCDA_criterion
        class(coords), pointer :: outlet_pixel_coords => null()
        logical :: no_remaining_outlets
        logical :: mark_sink
        logical :: mark_outflow
        logical :: outlet_is_LCDA
        integer :: outlet_type
            if (.not. this%ocean_cell) then
                outlet_is_LCDA = .False.
                outlet_pixel_coords => this%find_outlet_pixel(no_remaining_outlets,use_LCDA_criterion,&
                                                              outlet_is_LCDA)
                call this%check_for_sinks_and_rmouth_outflows(outlet_pixel_coords,no_remaining_outlets,mark_sink, &
                                                              mark_outflow)
                if (mark_sink) then
                    outlet_type = this%yamazaki_sink_outlet
                else if (mark_outflow) then
                    outlet_type = this%yamazaki_rmouth_outlet
                else
                    if ( .not. (no_remaining_outlets .or. outlet_is_LCDA) ) then
                        outlet_type = this%yamazaki_normal_outlet
                    else
                        outlet_type = this%yamazaki_source_cell_outlet
                    end if
                end if
                call this%yamazaki_outlet_pixels%set_value(outlet_pixel_coords,outlet_type)
                deallocate(outlet_pixel_coords)
            end if
    end subroutine yamazaki_mark_cell_outlet_pixel

    subroutine print_cell_and_neighborhood(this)
    class(cell), intent(in) :: this
        write(*,*) 'For cell'

        call this%print_area(this)
        write (*,*) 'contains_river_mouths= ', this%contains_river_mouths
        write (*,*) 'ocean_cell= ', this%ocean_cell
        write(*,*) 'For neighborhood'
        call this%cell_neighborhood%print_area(this%cell_neighborhood)
    end subroutine

    subroutine cell_destructor(this)
        class(cell), intent(inout) :: this
            if (associated(this%rejected_pixels)) then
                call this%rejected_pixels%destructor()
                deallocate(this%rejected_pixels)
            end if
            if (associated(this%cell_cumulative_flow)) then
                call this%cell_cumulative_flow%destructor()
                deallocate(this%cell_cumulative_flow)
            end if
            if (associated(this%river_directions)) deallocate(this%river_directions)
            if (associated(this%total_cumulative_flow)) deallocate(this%total_cumulative_flow)
            if (allocated(this%cell_neighborhood)) call this%cell_neighborhood%destructor()
            if (associated(this%yamazaki_outlet_pixels)) deallocate(this%yamazaki_outlet_pixels)
    end subroutine cell_destructor

    function yamazaki_retrieve_initial_outlet_pixel(this,outlet_pixel_type) result(initial_outlet_pixel)
        class(cell), intent(inout) :: this
        integer, intent(out) :: outlet_pixel_type
        class(coords), pointer  :: initial_outlet_pixel
        class(coords), pointer :: working_pixel
        class(*), pointer :: working_pixel_ptr
        class(coords), pointer :: edge_pixel_coords_list(:)
        class(doubly_linked_list), pointer :: pixels_to_process
        class(*), pointer :: working_pixel_value
        logical :: pixel_found
        integer :: i
            pixel_found = .false.
            edge_pixel_coords_list => this%find_edge_pixels()
            do i = 1,size(edge_pixel_coords_list)
                working_pixel => edge_pixel_coords_list(i)
                if ( .not. pixel_found) then
                    working_pixel_value => this%yamazaki_outlet_pixels%get_value(working_pixel)
                    select type(working_pixel_value)
                    type is (integer)
                        if (working_pixel_value == this%yamazaki_source_cell_outlet .or. &
                            working_pixel_value == this%yamazaki_normal_outlet .or. &
                            working_pixel_value == this%yamazaki_sink_outlet .or. &
                            working_pixel_value == this%yamazaki_rmouth_outlet) then
                            outlet_pixel_type = working_pixel_value
                            allocate(initial_outlet_pixel,source=working_pixel)
                            pixel_found = .true.
                        end if
                    end select
                    deallocate(working_pixel_value)
                end if
            end do
            deallocate(edge_pixel_coords_list)
            if ( .not. pixel_found) then
                pixels_to_process => this%generate_list_of_pixels_in_cell()
                do
                    !Combine iteration and check to see if we have reached the end
                    !of the list
                    if (pixels_to_process%iterate_forward()) then
                        if (pixel_found) then
                            exit
                        else
                            stop 'no outlet pixel could be retrieved for cell'
                        end if
                    end if
                    working_pixel_ptr => pixels_to_process%get_value_at_iterator_position()
                    select type(working_pixel_ptr)
                        class is (coords)
                            working_pixel => working_pixel_ptr
                    end select
                    if ( .not. pixel_found) then
                        working_pixel_value => this%yamazaki_outlet_pixels%get_value(working_pixel)
                        select type(working_pixel_value)
                        type is (integer)
                            if (working_pixel_value == this%yamazaki_sink_outlet .or. &
                                working_pixel_value == this%yamazaki_rmouth_outlet) then
                                outlet_pixel_type = working_pixel_value
                                allocate(initial_outlet_pixel,source=working_pixel)
                                pixel_found = .true.
                            end if
                        end select
                        deallocate(working_pixel_value)
                    end if
                    call pixels_to_process%remove_element_at_iterator_position()
                end do
            deallocate(pixels_to_process)
            end if
    end function yamazaki_retrieve_initial_outlet_pixel

    function find_pixel_with_LUDA(this,no_remaining_outlets) result(LUDA_pixel_coords)
        class(cell), intent(inout) :: this
        class(coords), pointer :: LUDA_pixel_coords
        logical, intent(out) :: no_remaining_outlets
        class(coords), allocatable :: working_pixel_coords
        integer :: LUDA_working_value
        integer :: i
        class(coords), pointer :: edge_pixel_coords_list(:)
        class(*), pointer :: rejected_pixel
        class(*), pointer :: upstream_drainage
        logical :: coords_not_in_cell
        nullify(LUDA_pixel_coords)
        edge_pixel_coords_list => this%find_edge_pixels()
            LUDA_working_value = 0
            do i = 1,size(edge_pixel_coords_list)
                rejected_pixel => this%rejected_pixels%get_value(edge_pixel_coords_list(i))
                select type(rejected_pixel)
                type is (logical)
                    if ( .NOT. rejected_pixel) then
                        upstream_drainage => this%total_cumulative_flow%get_value(edge_pixel_coords_list(i))
                        select type (upstream_drainage)
                        type is (integer)
                        if (upstream_drainage > LUDA_working_value) then
                                allocate(working_pixel_coords,source=edge_pixel_coords_list(i))
                                call this%find_next_pixel_downstream(this,working_pixel_coords,coords_not_in_cell)
                                deallocate(working_pixel_coords)
                                if(coords_not_in_cell) then
                                    LUDA_working_value = upstream_drainage
                                    if (associated(LUDA_pixel_coords)) deallocate(LUDA_pixel_coords)
                                    allocate(LUDA_pixel_coords,source=edge_pixel_coords_list(i))
                                end if
                            end if
                        end select
                        deallocate(upstream_drainage)
                    end if
                end select
                deallocate(rejected_pixel)
            end do
            if (.not. associated(LUDA_pixel_coords)) then
                allocate(LUDA_pixel_coords,source=edge_pixel_coords_list(1))
                no_remaining_outlets=.True.
            else
                no_remaining_outlets=.False.
            end if
            deallocate(edge_pixel_coords_list)
    end function find_pixel_with_LUDA

    function find_pixel_with_LCDA(this) result(LCDA_pixel_coords)
        class(cell), intent(inout) :: this
        class(coords), pointer :: LCDA_pixel_coords
        class(coords), allocatable :: working_pixel_coords
        integer :: LCDA_pixel_working_value
        class(coords), pointer :: edge_pixel_coords_list(:)
        class(*), pointer :: cell_drainage_area
        logical :: coords_not_in_cell
        integer :: i
            nullify(LCDA_pixel_coords)
            edge_pixel_coords_list => this%find_edge_pixels()
            LCDA_pixel_working_value = 0
            do i = 1,size(edge_pixel_coords_list)
                cell_drainage_area => this%cell_cumulative_flow%get_value(edge_pixel_coords_list(i))
                select type(cell_drainage_area)
                type is (integer)
                    if (cell_drainage_area > LCDA_pixel_working_value) then
                        allocate(working_pixel_coords,source=edge_pixel_coords_list(i))
                        call this%find_next_pixel_downstream(this,working_pixel_coords,coords_not_in_cell)
                        deallocate(working_pixel_coords)
                        if(coords_not_in_cell) then
                            LCDA_pixel_working_value = cell_drainage_area
                            if (associated(LCDA_pixel_coords)) deallocate(LCDA_pixel_coords)
                            allocate(LCDA_pixel_coords,source=edge_pixel_coords_list(i))
                        end if
                    end if
                end select
                deallocate(cell_drainage_area)
            end do
            if (.not. associated(LCDA_pixel_coords)) then
                allocate(LCDA_pixel_coords,source=edge_pixel_coords_list(1))
            end if
            deallocate(edge_pixel_coords_list)
    end function find_pixel_with_LCDA

    function measure_upstream_path(this,outlet_pixel_coords) result(upstream_path_length)
        class(cell), intent(in) :: this
        class(coords), intent(in), target :: outlet_pixel_coords
        class(coords), allocatable :: working_coords
        logical :: coords_not_in_cell
        real(kind=double_precision) :: length_through_pixel
        real(kind=double_precision) :: upstream_path_length
            upstream_path_length = 0.0_double_precision
            allocate(working_coords, source=outlet_pixel_coords)
            do
                length_through_pixel = this%calculate_length_through_pixel(this,working_coords)
                do
                    upstream_path_length = upstream_path_length + length_through_pixel
                    call this%find_next_pixel_upstream(working_coords,coords_not_in_cell)
                    length_through_pixel = this%calculate_length_through_pixel(this,working_coords)
                    if (coords_not_in_cell) exit
                end do
                if (.NOT. this%cell_neighborhood%path_reenters_region(working_coords)) exit
            end do
            deallocate(working_coords)
    end function measure_upstream_path

    function check_MUFP(this,LUDA_pixel_coords,LCDA_pixel_coords,LUDA_is_LCDA) result(accept_pixel)
        class(cell) :: this
        class(coords), intent(in) :: LUDA_pixel_coords
        class(coords), intent(in), pointer :: LCDA_pixel_coords
        logical, optional, intent(inout) :: LUDA_is_LCDA
        logical :: accept_pixel
            if (present(LUDA_is_LCDA)) LUDA_is_LCDA = .FALSE.
            if (this%measure_upstream_path(LUDA_pixel_coords) > MUFP) then
                accept_pixel = .TRUE.
                return
            else if (associated(LCDA_pixel_coords)) then
                if(LUDA_pixel_coords%are_equal_to(LCDA_pixel_coords)) then
                    accept_pixel = .TRUE.
                    if (present(LUDA_is_LCDA)) LUDA_is_LCDA = .TRUE.
                    return
                end if
            end if
            accept_pixel = .FALSE.
            call this%rejected_pixels%set_value(LUDA_pixel_coords,.TRUE.)
    end function check_MUFP

    function yamazaki_test_find_downstream_cell(this,initial_outlet_pixel) result(downstream_cell)
        class(cell), intent(in) :: this
        class(coords), pointer, intent(in) :: initial_outlet_pixel
        class(coords), pointer :: downstream_cell
            downstream_cell => this%cell_neighborhood%yamazaki_find_downstream_cell(initial_outlet_pixel)
    end function yamazaki_test_find_downstream_cell

    function test_generate_cell_cumulative_flow(this) result(cmltv_flow_array)
        class(cell), intent(inout) :: this
        class(*), dimension(:,:), pointer :: cmltv_flow_array
        call this%generate_cell_cumulative_flow()
        select type(cmltv_flow => this%cell_cumulative_flow)
        type is (latlon_subfield)
            cmltv_flow_array => cmltv_flow%latlon_get_data()
        end select
    end function test_generate_cell_cumulative_flow

    subroutine generate_cell_cumulative_flow(this)
        class(cell), intent(inout) :: this
        class(doubly_linked_list), pointer :: pixels_to_process
        class(coords), pointer :: upstream_neighbors_coords_list(:)
        class(*), pointer :: pixel_coords
        class(*), allocatable :: neighbor_coords
        logical :: unprocessed_inflowing_pixels
        integer :: pixel_cumulative_flow
        class(*), pointer :: neighbor_cumulative_flow
        integer :: i
            call this%initialize_cell_cumulative_flow_subfield()
            pixels_to_process => this%generate_list_of_pixels_in_cell()
            do
                if (pixels_to_process%get_length() <= 0) exit
                do
                    !Combine iteration and check to see if we have reached the end
                    !of the list
                    if(pixels_to_process%iterate_forward()) exit
                    pixel_cumulative_flow = 1
                    unprocessed_inflowing_pixels = .FALSE.
                    pixel_coords => pixels_to_process%get_value_at_iterator_position()
                    select type (pixel_coords)
                    class is (coords)
                        upstream_neighbors_coords_list => this%find_upstream_neighbors(this,pixel_coords)
                    end select
                    if (size(upstream_neighbors_coords_list) > 0) then
                        do i = 1,size(upstream_neighbors_coords_list)
                            if (allocated(neighbor_coords)) deallocate(neighbor_coords)
                            allocate(neighbor_coords,source=upstream_neighbors_coords_list(i))
                            select type (neighbor_coords)
                            class is (coords)
                                if( .not. this%check_if_coords_are_in_area(this,neighbor_coords)) then
                                    cycle
                                end if
                                neighbor_cumulative_flow => this%cell_cumulative_flow%get_value(neighbor_coords)
                            end select
                            select type (neighbor_cumulative_flow)
                            type is (integer)
                                if (neighbor_cumulative_flow == 0) then
                                    unprocessed_inflowing_pixels = .TRUE.
                                else
                                    pixel_cumulative_flow = pixel_cumulative_flow + neighbor_cumulative_flow
                                end if
                            end select
                            deallocate(neighbor_cumulative_flow)
                            if (unprocessed_inflowing_pixels) exit
                        end do
                        if (allocated(neighbor_coords)) deallocate(neighbor_coords)
                    end if
                    if (.not. unprocessed_inflowing_pixels) then
                        select type (pixel_coords)
                        class is (coords)
                            call this%cell_cumulative_flow%set_value(pixel_coords,pixel_cumulative_flow)
                        end select
                        call pixels_to_process%remove_element_at_iterator_position()
                        if (pixels_to_process%get_length() <= 0) exit
                    end if
                    deallocate(upstream_neighbors_coords_list)
                end do
                if (associated(upstream_neighbors_coords_list)) deallocate(upstream_neighbors_coords_list)
                call pixels_to_process%reset_iterator()
            end do
            deallocate(pixels_to_process)
    end subroutine generate_cell_cumulative_flow

    subroutine set_contains_river_mouths(this,value)
        class(cell) :: this
        logical :: value
            this%contains_river_mouths = value
    end subroutine

    subroutine init_latlon_field(this,field_section_coords,river_directions)
        class(latlon_field) :: this
        class(*), dimension(:,:), pointer :: river_directions
        type(latlon_section_coords) :: field_section_coords
            this%section_min_lat = field_section_coords%section_min_lat
            this%section_min_lon = field_section_coords%section_min_lon
            this%section_width_lat = field_section_coords%section_width_lat
            this%section_width_lon = field_section_coords%section_width_lon
            this%section_max_lat =  field_section_coords%section_min_lat + field_section_coords%section_width_lat - 1
            this%section_max_lon =  field_section_coords%section_min_lon + field_section_coords%section_width_lon - 1
            allocate(this%field_section_coords,source=field_section_coords)
            this%total_cumulative_flow => null()
            this%river_directions => latlon_field_section(river_directions,field_section_coords)
            call this%initialize_cells_to_reprocess_field_section()
    end subroutine init_latlon_field

    subroutine init_icon_single_index_field(this,field_section_coords,river_directions)
        class(icon_single_index_field) :: this
        class(*), dimension(:), pointer :: river_directions
        type(generic_1d_section_coords) :: field_section_coords
            this%ncells = size(field_section_coords%cell_neighbors,1)
            allocate(this%field_section_coords,source=field_section_coords)
            this%total_cumulative_flow => null()
            this%river_directions => icon_single_index_field_section(river_directions,field_section_coords)
            call this%initialize_cells_to_reprocess_field_section()
    end subroutine init_icon_single_index_field

    function latlon_check_field_for_localized_loops(this) result(cells_to_reprocess)
        implicit none
        class(latlon_field) :: this
        class(*), dimension(:,:), pointer :: cells_to_reprocess_data
        logical, dimension(:,:), pointer :: cells_to_reprocess
        type(latlon_coords) :: cell_coords
        integer :: i,j
            do j = this%section_min_lon, this%section_max_lon
                do i = this%section_min_lat, this%section_max_lat
                    cell_coords = latlon_coords(i,j)
                    if (this%check_cell_for_localized_loops(cell_coords)) then
                        call this%cells_to_reprocess%set_value(cell_coords,.True.)
                    end if
                end do
            end do
            select type (cells_to_reprocess_field_section => this%cells_to_reprocess)
            type is (latlon_field_section)
                cells_to_reprocess_data => cells_to_reprocess_field_section%get_data()
                select type (cells_to_reprocess_data)
                type is (logical)
                    cells_to_reprocess => cells_to_reprocess_data
                end select
            end select
    end function latlon_check_field_for_localized_loops

    function icon_single_index_check_field_for_localized_loops(this) result(cells_to_reprocess)
        implicit none
        class(icon_single_index_field) :: this
        class(*), dimension(:), pointer :: cells_to_reprocess_data
        !Retain the dimension from the lat lon version but only ever use a single column for the
        !second index
        logical, dimension(:,:), pointer :: cells_to_reprocess
        type(generic_1d_coords) :: cell_coords
        integer :: i
            do i = 1, this%ncells
                cell_coords = generic_1d_coords(i)
                if (this%check_cell_for_localized_loops(cell_coords)) then
                    call this%cells_to_reprocess%set_value(cell_coords,.True.)
                end if
            end do
            allocate(cells_to_reprocess(this%ncells,1))
            select type (cells_to_reprocess_field_section => this%cells_to_reprocess)
            type is (icon_single_index_field_section)
                cells_to_reprocess_data => cells_to_reprocess_field_section%get_data()
                select type (cells_to_reprocess_data)
                type is (logical)
                    cells_to_reprocess(:,1) = cells_to_reprocess_data
                end select
            end select
    end function icon_single_index_check_field_for_localized_loops

    subroutine latlon_initialize_cells_to_reprocess_field_section(this)
        class(latlon_field) :: this
        class(*), dimension(:,:), pointer :: cells_to_reprocess_initialization_data
            allocate(logical:: cells_to_reprocess_initialization_data(this%section_width_lat,&
                                                                      this%section_width_lon))
            select type(cells_to_reprocess_initialization_data)
            type is (logical)
                cells_to_reprocess_initialization_data = .FALSE.
            end select
            select type (field_section_coords => this%field_section_coords)
            type is (latlon_section_coords)
                this%cells_to_reprocess => latlon_field_section(cells_to_reprocess_initialization_data, &
                                                                field_section_coords)
            end select
    end subroutine latlon_initialize_cells_to_reprocess_field_section

    subroutine icon_single_index_initialize_cells_to_reprocess_field_section(this)
        class(icon_single_index_field) :: this
        class(*), dimension(:), pointer :: cells_to_reprocess_initialization_data
            allocate(logical:: cells_to_reprocess_initialization_data(this%ncells))
            select type(cells_to_reprocess_initialization_data)
            type is (logical)
                cells_to_reprocess_initialization_data = .FALSE.
            end select
            select type (field_section_coords => this%field_section_coords)
            type is (generic_1d_section_coords)
                this%cells_to_reprocess => &
                    icon_single_index_field_section(cells_to_reprocess_initialization_data, &
                                                    field_section_coords)
            end select
    end subroutine icon_single_index_initialize_cells_to_reprocess_field_section

    subroutine init_latlon_cell(this,cell_section_coords,river_directions,&
                                total_cumulative_flow)
        class(latlon_cell), intent(inout) :: this
        type(latlon_section_coords) :: cell_section_coords
        integer, dimension(:,:), target :: river_directions
        integer, dimension(:,:), target :: total_cumulative_flow
        class(*), dimension(:,:), pointer :: river_directions_pointer
        class(*), dimension(:,:), pointer :: total_cumulative_flow_pointer
            allocate(this%cell_section_coords,source=cell_section_coords)
            this%section_min_lat = cell_section_coords%section_min_lat
            this%section_min_lon = cell_section_coords%section_min_lon
            this%section_width_lat = cell_section_coords%section_width_lat
            this%section_width_lon = cell_section_coords%section_width_lon
            this%section_max_lat =  cell_section_coords%section_min_lat + cell_section_coords%section_width_lat - 1
            this%section_max_lon =  cell_section_coords%section_min_lon + cell_section_coords%section_width_lon - 1
            river_directions_pointer => river_directions
            total_cumulative_flow_pointer => total_cumulative_flow
            this%river_directions => latlon_field_section(river_directions_pointer,cell_section_coords)
            this%total_cumulative_flow => latlon_field_section(total_cumulative_flow_pointer,cell_section_coords)
            !Minus four to avoid double counting the corners
            this%number_of_edge_pixels = 2*this%section_width_lat + 2*this%section_width_lon - 4
            call this%initialize_rejected_pixels_subfield()
            this%contains_river_mouths = .FALSE.
            this%ocean_cell = .FALSE.
    end subroutine init_latlon_cell

    subroutine init_irregular_latlon_cell(this,cell_section_coords,river_directions,&
                                          total_cumulative_flow)
        class(irregular_latlon_cell), intent(inout) :: this
        class(irregular_latlon_section_coords) :: cell_section_coords
        integer, dimension(:,:), target :: river_directions
        integer, dimension(:,:), target :: total_cumulative_flow
        class(*), dimension(:,:), pointer :: river_directions_pointer
        class(*), dimension(:,:), pointer :: total_cumulative_flow_pointer
        class(*), dimension(:,:), pointer :: cell_numbers_pointer
            allocate(this%cell_section_coords,source=cell_section_coords)
            this%section_min_lat = cell_section_coords%section_min_lat
            this%section_min_lon = cell_section_coords%section_min_lon
            this%section_width_lat = cell_section_coords%section_width_lat
            this%section_width_lon = cell_section_coords%section_width_lon
            this%section_max_lat =  cell_section_coords%section_min_lat + cell_section_coords%section_width_lat - 1
            this%section_max_lon =  cell_section_coords%section_min_lon + cell_section_coords%section_width_lon - 1
            cell_numbers_pointer => cell_section_coords%cell_numbers
            this%cell_numbers => latlon_field_section(cell_numbers_pointer,cell_section_coords)
                this%cell_number = cell_section_coords%cell_number
            river_directions_pointer => river_directions
            total_cumulative_flow_pointer => total_cumulative_flow
            this%river_directions => latlon_field_section(river_directions_pointer,cell_section_coords)
            this%total_cumulative_flow => latlon_field_section(total_cumulative_flow_pointer,cell_section_coords)
            this%number_of_edge_pixels = 0
            call this%initialize_rejected_pixels_subfield()
            this%contains_river_mouths = .FALSE.
            this%ocean_cell = .FALSE.
    end subroutine init_irregular_latlon_cell

    subroutine irregular_latlon_cell_destructor(this)
        class(irregular_latlon_cell), intent(inout) :: this
            call cell_destructor(this)
            if(associated(this%cell_numbers)) deallocate(this%cell_numbers)
            call this%cell_neighborhood%destructor()
    end subroutine irregular_latlon_cell_destructor

    subroutine yamazaki_init_latlon_cell(this,cell_section_coords,river_directions, &
                                         total_cumulative_flow,yamazaki_outlet_pixels)
        class(latlon_cell), intent(inout) :: this
        type(latlon_section_coords) :: cell_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
        integer, dimension(:,:), target :: yamazaki_outlet_pixels
        class(*), dimension(:,:), pointer :: yamazaki_outlet_pixels_pointer
            call this%init_latlon_cell(cell_section_coords,river_directions,&
                                       total_cumulative_flow)
            yamazaki_outlet_pixels_pointer => yamazaki_outlet_pixels
            this%yamazaki_outlet_pixels => latlon_field_section(yamazaki_outlet_pixels_pointer, &
                                                                cell_section_coords)
    end subroutine yamazaki_init_latlon_cell

    function latlon_find_edge_pixels(this) result(edge_pixel_coords_list)
        class(latlon_cell), intent(inout) :: this
        class(coords), pointer :: edge_pixel_coords_list(:)
        integer :: i
        integer :: list_index
        list_index = 1
        allocate(latlon_coords::edge_pixel_coords_list(this%number_of_edge_pixels))
        select type (edge_pixel_coords_list)
        type is (latlon_coords)
            do i = this%section_min_lat,this%section_max_lat
                edge_pixel_coords_list(list_index) = latlon_coords(i,this%section_min_lon)
                edge_pixel_coords_list(list_index+1) = latlon_coords(i,this%section_max_lon)
                list_index = list_index + 2
            end do
            do  i = this%section_min_lon+1,this%section_max_lon-1
                edge_pixel_coords_list(list_index) = latlon_coords(this%section_min_lat,i)
                edge_pixel_coords_list(list_index+1) = latlon_coords(this%section_max_lat,i)
                list_index = list_index + 2
            end do
        end select
    end function latlon_find_edge_pixels

    function irregular_latlon_find_edge_pixels(this) result(edge_pixel_coords_list)
        class(irregular_latlon_cell), intent(inout) :: this
        class(coords), pointer :: edge_pixel_coords_list(:)
        type(doubly_linked_list), pointer :: list_of_edge_pixels
        class(*), pointer :: working_pixel_ptr
        type(latlon_field_section), pointer :: cell_numbers
        integer, pointer, dimension(:,:) :: cell_numbers_data
        integer :: i,j
        integer :: list_index
            cell_numbers => this%cell_numbers
            list_index = 1
            allocate(list_of_edge_pixels)
            select type(cell_numbers_data_ptr => cell_numbers%data)
            type is (integer)
                cell_numbers_data => cell_numbers_data_ptr
            end select
            this%number_of_edge_pixels = 0
            do j = this%section_min_lon, this%section_max_lon
                do i = this%section_min_lat, this%section_max_lat
                    if (cell_numbers_data(i,j) == this%cell_number) then
                        if(cell_numbers_data(i-1,j-1) /= this%cell_number .or. &
                           cell_numbers_data(i-1,j)   /= this%cell_number .or. &
                           cell_numbers_data(i-1,j+1) /= this%cell_number .or. &
                           cell_numbers_data(i,j-1)   /= this%cell_number .or. &
                           cell_numbers_data(i,j+1)   /= this%cell_number .or. &
                           cell_numbers_data(i+1,j-1) /= this%cell_number .or. &
                           cell_numbers_data(i+1,j)   /= this%cell_number .or. &
                           cell_numbers_data(i+1,j+1) /= this%cell_number) then
                            call list_of_edge_pixels%add_value_to_back(latlon_coords(i,j))
                            this%number_of_edge_pixels = this%number_of_edge_pixels + 1
                        end if
                    end if
                end do
            end do
            allocate(latlon_coords::edge_pixel_coords_list(this%number_of_edge_pixels))
            call list_of_edge_pixels%reset_iterator()
            do while (.not. list_of_edge_pixels%iterate_forward())
                working_pixel_ptr => list_of_edge_pixels%get_value_at_iterator_position()
                select type(working_pixel_ptr)
                type is (latlon_coords)
                    select type(edge_pixel_coords_list)
                    type is (latlon_coords)
                        edge_pixel_coords_list(list_index) = working_pixel_ptr
                    end select
                end select
                list_index = list_index + 1
                call list_of_edge_pixels%remove_element_at_iterator_position()
            end do
            deallocate(list_of_edge_pixels)
    end function irregular_latlon_find_edge_pixels

    ! Note must explicit pass this function the object this as it has the nopass attribute
    ! so that it can be shared by latlon_cell and latlon_neighborhood
    function latlon_calculate_length_through_pixel(this,coords_in) result(length_through_pixel)
        class(area), intent(in) :: this
        class(coords), intent(in) :: coords_in
        real(kind=double_precision) length_through_pixel
        if (this%is_diagonal(this,coords_in)) then
            length_through_pixel = 1.414_double_precision
        else
            length_through_pixel = 1.0_double_precision
        end if
    end function latlon_calculate_length_through_pixel

    ! Note must explicit pass this function the object this as it has the nopass attribute
    ! so that it can be shared by latlon_cell and latlon_neighborhood
    function latlon_check_if_coords_are_in_area(this,coords_in) result(within_area)
        class(area), intent(in) :: this
        class(coords), intent(in) :: coords_in
        logical :: within_area
        select type (coords_in)
        type is (latlon_coords)
            if (coords_in%lat >= this%section_min_lat .and. &
                coords_in%lat <= this%section_max_lat .and. &
                coords_in%lon >= this%section_min_lon .and. &
                coords_in%lon <= this%section_max_lon) then
                within_area = .TRUE.
            else
                within_area = .FALSE.
            end if
        end select
    end function latlon_check_if_coords_are_in_area

    ! Note must explicit pass this function the object this as it has the nopass attribute
    ! so that it can be shared by latlon_cell and latlon_neighborhood
    function generic_check_if_coords_are_in_area(this,coords_in) result(within_area)
        class(area), intent(in) :: this
        class(coords), intent(in) :: coords_in
        class(field_section), pointer :: cell_numbers
        type(doubly_linked_list), pointer :: list_of_cells_in_neighborhood
        class (*), pointer :: cell_number_ptr
        class (*), pointer :: working_cell_number_ptr
        integer    :: working_cell_number
        logical :: within_area
        integer :: cell_number
            select type(this)
            class is (irregular_latlon_cell)
                cell_numbers => this%cell_numbers
                cell_number = this%cell_number
            class is (irregular_latlon_neighborhood)
                list_of_cells_in_neighborhood => this%list_of_cells_in_neighborhood
                cell_numbers => this%cell_numbers
            end select
            within_area = .FALSE.
            select type(this)
            class is (field)
                within_area = .TRUE.
            class is (cell)
                working_cell_number_ptr => cell_numbers%get_value(coords_in)
                select type (working_cell_number_ptr)
                    type is (integer)
                        working_cell_number = working_cell_number_ptr
                end select
                deallocate(working_cell_number_ptr)
                if(working_cell_number == cell_number) then
                    within_area = .TRUE.
                end if
            class is (neighborhood)
                call list_of_cells_in_neighborhood%reset_iterator()
                do while (.not. list_of_cells_in_neighborhood%iterate_forward())
                    cell_number_ptr => &
                        list_of_cells_in_neighborhood%get_value_at_iterator_position()
                    select type (cell_number_ptr)
                    type is (integer)
                        working_cell_number_ptr => cell_numbers%get_value(coords_in)
                        select type (working_cell_number_ptr)
                        type is (integer)
                            working_cell_number = working_cell_number_ptr
                        end select
                        deallocate(working_cell_number_ptr)
                        if(working_cell_number == cell_number_ptr) then
                            within_area = .TRUE.
                        end if
                    end select
                end do
            end select
    end function generic_check_if_coords_are_in_area

    ! Note must explicit pass this function the object this as it has the nopass attribute
    ! so that it can be shared by latlon_cell and latlon_neighborhood
    function latlon_find_upstream_neighbors(this,coords_in) result(list_of_neighbors)
        class(area), intent(in) :: this
        class(coords), intent(in) :: coords_in
        type(latlon_coords), dimension(9) :: list_of_neighbors_temp
        class(coords), dimension(:), pointer :: list_of_neighbors
        integer :: i,j
        integer :: counter
        integer :: n
        counter = 0
        n = 0
        select type (coords_in)
        type is (latlon_coords)
            do i = coords_in%lat + 1, coords_in%lat - 1,-1
                do j = coords_in%lon - 1, coords_in%lon + 1
                    counter = counter + 1
                    if (counter == 5) cycle
                    if (this%neighbor_flows_to_pixel(this,latlon_coords(lat=i,lon=j),&
                        dir_based_direction_indicator(counter))) then
                        n = n + 1
                        list_of_neighbors_temp(n) = latlon_coords(lat=i,lon=j)
                    end if
                end do
            end do
        end select
        allocate(latlon_coords::list_of_neighbors(n))
        select type(list_of_neighbors)
        type is (latlon_coords)
            do counter = 1,n
                list_of_neighbors(counter) = list_of_neighbors_temp(counter)
            end do
        end select
    end function latlon_find_upstream_neighbors

    subroutine latlon_initialize_cell_cumulative_flow_subfield(this)
        class(latlon_cell), intent(inout) :: this
        class(*), dimension(:,:), pointer :: input_data
        logical :: wrap
            allocate(integer::input_data(this%section_width_lat,this%section_width_lon))
            select type(input_data)
            type is (integer)
                input_data = 0
            end select
            select type (river_directions => this%river_directions)
            class is (latlon_field_section)
                wrap = (this%section_width_lon == river_directions%get_nlon() .and. &
                        river_directions%get_wrap())
            end select
            select type(cell_section_coords => this%cell_section_coords)
            class is (latlon_section_coords)
                this%cell_cumulative_flow => latlon_subfield(input_data,cell_section_coords,&
                                                             wrap)
            end select
    end subroutine latlon_initialize_cell_cumulative_flow_subfield

    subroutine latlon_initialize_rejected_pixels_subfield(this)
        class(latlon_cell), intent(inout) :: this
        class(*), dimension(:,:), pointer :: rejected_pixel_initialization_data
        logical :: wrap
            allocate(logical::rejected_pixel_initialization_data(this%section_width_lat,&
                                                                 this%section_width_lon))
            select type(rejected_pixel_initialization_data)
            type is (logical)
                rejected_pixel_initialization_data = .FALSE.
            end select
            select type (river_directions => this%river_directions)
            class is (latlon_field_section)
                wrap = (this%section_width_lon == river_directions%get_nlon() .and. &
                        river_directions%get_wrap())
            end select
            select type (cell_section_coords => this%cell_section_coords)
            class is (latlon_section_coords)
                this%rejected_pixels => latlon_subfield(rejected_pixel_initialization_data,cell_section_coords,wrap)
            end select
    end subroutine latlon_initialize_rejected_pixels_subfield

    function latlon_generate_list_of_pixels_in_cell(this) result(list_of_pixels_in_cell)
        class(latlon_cell), intent(in) :: this
        type(doubly_linked_list), pointer :: list_of_pixels_in_cell
        integer :: i,j
            allocate(list_of_pixels_in_cell)
            do j = this%section_min_lon, this%section_max_lon
                do i = this%section_min_lat, this%section_max_lat
                    call list_of_pixels_in_cell%add_value_to_back(latlon_coords(i,j))
                end do
            end do
    end function

    function irregular_latlon_generate_list_of_pixels_in_cell(this) result(list_of_pixels_in_cell)
        class(irregular_latlon_cell), intent(in) :: this
        type(doubly_linked_list), pointer :: list_of_pixels_in_cell
        class(latlon_field_section), pointer :: cell_numbers
        integer, pointer, dimension(:,:) :: cell_numbers_data
        integer :: i,j
            cell_numbers => this%cell_numbers
            select type (cell_numbers_data_ptr => cell_numbers%data)
            type is (integer)
                cell_numbers_data => cell_numbers_data_ptr
            end select
            allocate(list_of_pixels_in_cell)
            do j = this%section_min_lon, this%section_max_lon
                do i = this%section_min_lat, this%section_max_lat
                    if (cell_numbers_data(i,j) == this%cell_number) then
                        call list_of_pixels_in_cell%add_value_to_back(latlon_coords(i,j))
                    end if
                end do
            end do
    end function

    function latlon_get_sink_combined_cumulative_flow(this,main_sink_coords) &
        result(combined_cumulative_flow_to_sinks)
        class(latlon_cell), intent(in) :: this
        class(coords), pointer, optional,intent(out) :: main_sink_coords
        integer :: combined_cumulative_flow_to_sinks
        integer :: greatest_flow_to_single_sink
        class(*), pointer :: current_pixel_total_cumulative_flow
        class(*), pointer :: current_pixel_flow_direction
        class(direction_indicator), pointer :: flow_direction_for_sink
        integer :: highest_outflow_sink_lat,highest_outflow_sink_lon
        integer :: i,j
            highest_outflow_sink_lat = 0
            highest_outflow_sink_lon = 0
            combined_cumulative_flow_to_sinks = 0
            greatest_flow_to_single_sink = 0
            flow_direction_for_sink => this%get_flow_direction_for_sink()
            do j = this%section_min_lon, this%section_min_lon + this%section_width_lon - 1
                do i = this%section_min_lat, this%section_min_lat + this%section_width_lat - 1
                    current_pixel_total_cumulative_flow => this%total_cumulative_flow%get_value(latlon_coords(i,j))
                    current_pixel_flow_direction => this%river_directions%get_value(latlon_coords(i,j))
                    select type (current_pixel_total_cumulative_flow)
                    type is (integer)
                        if (flow_direction_for_sink%is_equal_to(current_pixel_flow_direction)) then
                            combined_cumulative_flow_to_sinks = current_pixel_total_cumulative_flow + &
                                combined_cumulative_flow_to_sinks
                            if (present(main_sink_coords)) then
                                if (greatest_flow_to_single_sink < &
                                    current_pixel_total_cumulative_flow) then
                                    current_pixel_total_cumulative_flow = &
                                        greatest_flow_to_single_sink
                                    highest_outflow_sink_lat = i
                                    highest_outflow_sink_lon = j
                                end if
                            end if
                        end if
                        deallocate(current_pixel_flow_direction)
                    end select
                    deallocate(current_pixel_total_cumulative_flow)
                end do
            end do
            deallocate(flow_direction_for_sink)
            if(present(main_sink_coords)) then
                allocate(main_sink_coords,source=latlon_coords(highest_outflow_sink_lat,&
                                                               highest_outflow_sink_lon))
            end if
    end function latlon_get_sink_combined_cumulative_flow

    function latlon_get_rmouth_outflow_combined_cumulative_flow(this,main_outflow_coords) &
        result(combined_cumulative_rmouth_outflow)
        class(latlon_cell), intent(in) :: this
        class(coords), pointer, optional,intent(out) :: main_outflow_coords
        class(*), pointer :: current_pixel_total_cumulative_flow
        class(*), pointer :: current_pixel_flow_direction
        class(direction_indicator), pointer :: flow_direction_for_outflow
        integer :: greatest_flow_to_single_outflow
        integer :: combined_cumulative_rmouth_outflow
        integer :: highest_outflow_rmouth_lat,highest_outflow_rmouth_lon
        integer :: i,j
            highest_outflow_rmouth_lat = 0
            highest_outflow_rmouth_lon = 0
            combined_cumulative_rmouth_outflow = 0
            greatest_flow_to_single_outflow = 0
            flow_direction_for_outflow => this%get_flow_direction_for_outflow()
            do j = this%section_min_lon, this%section_min_lon + this%section_width_lon - 1
                do i = this%section_min_lat, this%section_min_lat + this%section_width_lat - 1
                    current_pixel_total_cumulative_flow => &
                        this%total_cumulative_flow%get_value(latlon_coords(i,j))
                    current_pixel_flow_direction => this%river_directions%get_value(latlon_coords(i,j))
                    select type (current_pixel_total_cumulative_flow)
                    type is (integer)
                        if (flow_direction_for_outflow%is_equal_to(current_pixel_flow_direction)) then
                            combined_cumulative_rmouth_outflow = current_pixel_total_cumulative_flow + &
                                combined_cumulative_rmouth_outflow
                            if (present(main_outflow_coords)) then
                                if (greatest_flow_to_single_outflow < &
                                    current_pixel_total_cumulative_flow) then
                                    current_pixel_total_cumulative_flow = &
                                        greatest_flow_to_single_outflow
                                    highest_outflow_rmouth_lat = i
                                    highest_outflow_rmouth_lon = j
                                end if
                            end if
                        end if
                    end select
                    deallocate(current_pixel_total_cumulative_flow)
                    deallocate(current_pixel_flow_direction)
                end do
            end do
            deallocate(flow_direction_for_outflow)
            if(present(main_outflow_coords)) then
                allocate(main_outflow_coords,source=latlon_coords(highest_outflow_rmouth_lat,&
                                                                  highest_outflow_rmouth_lon))
            end if
    end function latlon_get_rmouth_outflow_combined_cumulative_flow

   function irregular_latlon_get_sink_combined_cumulative_flow(this,main_sink_coords) &
        result(combined_cumulative_flow_to_sinks)
        class(irregular_latlon_cell), intent(in) :: this
        class(coords), pointer, optional,intent(out) :: main_sink_coords
        integer :: combined_cumulative_flow_to_sinks
        integer :: greatest_flow_to_single_sink
        class(*), pointer :: current_pixel_total_cumulative_flow
        class(*), pointer :: current_pixel_flow_direction
        class(direction_indicator), pointer :: flow_direction_for_sink
        class(latlon_field_section), pointer :: cell_numbers
        integer, pointer, dimension(:,:) :: cell_numbers_data
        integer :: highest_outflow_sink_lat,highest_outflow_sink_lon
        integer :: i,j
            cell_numbers => this%cell_numbers
            select type (cell_numbers_data_ptr => cell_numbers%data)
            type is (integer)
                cell_numbers_data => cell_numbers_data_ptr
            end select
            highest_outflow_sink_lat = 0
            highest_outflow_sink_lon = 0
            combined_cumulative_flow_to_sinks = 0
            greatest_flow_to_single_sink = 0
            flow_direction_for_sink => this%get_flow_direction_for_sink()
            do j = this%section_min_lon, this%section_max_lon
                do i = this%section_min_lat, this%section_max_lat
                    if (cell_numbers_data(i,j) == this%cell_number) then
                        current_pixel_total_cumulative_flow => this%total_cumulative_flow%get_value(latlon_coords(i,j))
                        current_pixel_flow_direction => this%river_directions%get_value(latlon_coords(i,j))
                        select type (current_pixel_total_cumulative_flow)
                        type is (integer)
                            if (flow_direction_for_sink%is_equal_to(current_pixel_flow_direction)) then
                                combined_cumulative_flow_to_sinks = current_pixel_total_cumulative_flow + &
                                    combined_cumulative_flow_to_sinks
                                if (present(main_sink_coords)) then
                                    if (greatest_flow_to_single_sink < &
                                        current_pixel_total_cumulative_flow) then
                                        current_pixel_total_cumulative_flow = &
                                            greatest_flow_to_single_sink
                                        highest_outflow_sink_lat = i
                                        highest_outflow_sink_lon = j
                                    end if
                                end if
                            end if
                            deallocate(current_pixel_flow_direction)
                        end select
                        deallocate(current_pixel_total_cumulative_flow)
                    end if
                end do
            end do
            deallocate(flow_direction_for_sink)
            if(present(main_sink_coords)) then
                allocate(main_sink_coords,source=latlon_coords(highest_outflow_sink_lat,&
                                                               highest_outflow_sink_lon))
            end if
    end function irregular_latlon_get_sink_combined_cumulative_flow

    function irregular_latlon_get_rmouth_outflow_combined_cumulative_flow(this,main_outflow_coords) &
        result(combined_cumulative_rmouth_outflow)
        class(irregular_latlon_cell), intent(in) :: this
        class(coords), pointer, optional,intent(out) :: main_outflow_coords
        class(*), pointer :: current_pixel_total_cumulative_flow
        class(*), pointer :: current_pixel_flow_direction
        class(direction_indicator), pointer :: flow_direction_for_outflow
        class(latlon_field_section), pointer :: cell_numbers
        integer, pointer, dimension(:,:) :: cell_numbers_data
        integer :: greatest_flow_to_single_outflow
        integer :: combined_cumulative_rmouth_outflow
        integer :: highest_outflow_rmouth_lat,highest_outflow_rmouth_lon
        integer :: i,j
            cell_numbers => this%cell_numbers
            select type (cell_numbers_data_ptr => cell_numbers%data)
            type is (integer)
                cell_numbers_data => cell_numbers_data_ptr
            end select
            highest_outflow_rmouth_lat = 0
            highest_outflow_rmouth_lon = 0
            combined_cumulative_rmouth_outflow = 0
            greatest_flow_to_single_outflow = 0
            flow_direction_for_outflow => this%get_flow_direction_for_outflow()
            do j = this%section_min_lon, this%section_max_lon
                do i = this%section_min_lat, this%section_max_lat
                    if (cell_numbers_data(i,j) == this%cell_number) then
                        current_pixel_total_cumulative_flow => this%total_cumulative_flow%get_value(latlon_coords(i,j))
                        current_pixel_flow_direction => this%river_directions%get_value(latlon_coords(i,j))
                        select type (current_pixel_total_cumulative_flow)
                        type is (integer)
                            if (flow_direction_for_outflow%is_equal_to(current_pixel_flow_direction)) then
                                combined_cumulative_rmouth_outflow = current_pixel_total_cumulative_flow + &
                                    combined_cumulative_rmouth_outflow
                                if (present(main_outflow_coords)) then
                                    if (greatest_flow_to_single_outflow < &
                                        current_pixel_total_cumulative_flow) then
                                        current_pixel_total_cumulative_flow = &
                                            greatest_flow_to_single_outflow
                                        highest_outflow_rmouth_lat = i
                                        highest_outflow_rmouth_lon = j
                                    end if
                                end if
                            end if
                        end select
                        deallocate(current_pixel_total_cumulative_flow)
                        deallocate(current_pixel_flow_direction)
                    end if
                end do
            end do
            deallocate(flow_direction_for_outflow)
            if(present(main_outflow_coords)) then
                allocate(main_outflow_coords,source=latlon_coords(highest_outflow_rmouth_lat,&
                                                                  highest_outflow_rmouth_lon))
            end if
    end function irregular_latlon_get_rmouth_outflow_combined_cumulative_flow


    subroutine latlon_print_area(this)
    class(area), intent(in) :: this
    character(len=160) :: line
    class(*), pointer :: value
    integer :: i,j
        write(*,*) 'Area Information'
        write(*,*) 'min lat=', this%section_min_lat
        write(*,*) 'min lon=', this%section_min_lon
        write(*,*) 'max lat=', this%section_max_lat
        write(*,*) 'max lon=', this%section_max_lon
        write(*,*) 'area rdirs'
        select type(rdirs => this%river_directions)
        type is (latlon_field_section)
            do i = this%section_min_lat,this%section_max_lat
                line = ''
                do j = this%section_min_lon,this%section_max_lon
                    value => rdirs%get_value(latlon_coords(i,j))
                    select type(value)
                    type is (integer)
                        write(line,'(A,I5)') trim(line), value
                    end select
                    deallocate(value)
                end do
                write (*,'(A)') line
            end do
        end select
        write(*,*) 'area total cumulative flow'
        select type(total_cumulative_flow => this%total_cumulative_flow)
        type is (latlon_field_section)
            do i = this%section_min_lat,this%section_max_lat
                line=''
                do j = this%section_min_lon,this%section_max_lon
                    value => total_cumulative_flow%get_value(latlon_coords(i,j))
                    select type(value)
                    type is (integer)
                        write(line,'(A,I5)') trim(line), value
                    end select
                    deallocate(value)
                end do
                write (*,'(A)') line
            end do
        end select
    end subroutine latlon_print_area

    function icon_single_index_field_dummy_real_function(this,coords_in) result(dummy_result)
        class(area), intent(in) :: this
        class(coords), intent(in) :: coords_in
        real(kind=double_precision) :: dummy_result
            select type (this)
            class is (icon_single_index_field)
                continue
            end select
            select type (coords_in)
            type is (generic_1d_coords)
                continue
            end select
            dummy_result = 0.0
            stop
    end function icon_single_index_field_dummy_real_function

    subroutine icon_single_index_field_dummy_subroutine(this)
        class(area), intent(in) :: this
            select type (this)
            class is (icon_single_index_field)
                continue
            end select
            stop
    end subroutine icon_single_index_field_dummy_subroutine

    function icon_single_index_field_dummy_coords_pointer_function(this,coords_in) &
            result(list_of_neighbors)
        class(area), intent(in) :: this
        class(coords), intent(in) :: coords_in
        class(coords), pointer :: list_of_neighbors(:)
            select type (this)
            class is (icon_single_index_field)
                continue
            end select
            select type (coords_in)
            type is (generic_1d_coords)
                continue
            end select
            allocate(generic_1d_coords::list_of_neighbors(1))
            stop
    end function icon_single_index_field_dummy_coords_pointer_function

    function latlon_dir_based_rdirs_cell_constructor(cell_section_coords,river_directions,&
                                                    total_cumulative_flow) result(constructor)
        type(latlon_dir_based_rdirs_cell), allocatable :: constructor
        class(latlon_section_coords) :: cell_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
            allocate(constructor)
            call constructor%init_dir_based_rdirs_latlon_cell(cell_section_coords,river_directions,&
                                                              total_cumulative_flow)
    end function

    function irregular_latlon_dir_based_rdirs_cell_constructor(cell_section_coords, &
                                                               river_directions, &
                                                               total_cumulative_flow, &
                                                               cell_neighbors) &
                                                                result(constructor)
        type(irregular_latlon_dir_based_rdirs_cell), allocatable :: constructor
        class(irregular_latlon_section_coords) :: cell_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
        integer, dimension(:,:), pointer, intent(in) :: cell_neighbors
            allocate(constructor)
            call constructor%init_irregular_latlon_dir_based_rdirs_cell(cell_section_coords,&
                                                                        river_directions,&
                                                                        total_cumulative_flow,&
                                                                        cell_neighbors)
    end function


    function yamazaki_latlon_dir_based_rdirs_cell_constructor(cell_section_coords,river_directions,&
                                                              total_cumulative_flow,yamazaki_outlet_pixels, &
                                                              yamazaki_section_coords) &
        result(constructor)
        type(latlon_dir_based_rdirs_cell), allocatable :: constructor
        class(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: yamazaki_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
        integer, dimension(:,:) :: yamazaki_outlet_pixels
            allocate(constructor)
            call constructor%yamazaki_init_dir_based_rdirs_latlon_cell(cell_section_coords,river_directions,&
                                                                       total_cumulative_flow,yamazaki_outlet_pixels, &
                                                                       yamazaki_section_coords)
    end function yamazaki_latlon_dir_based_rdirs_cell_constructor

    subroutine init_dir_based_rdirs_latlon_cell(this,cell_section_coords,river_directions,&
                                                total_cumulative_flow)
        class(latlon_dir_based_rdirs_cell) :: this
        type(latlon_section_coords) :: cell_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
            call this%init_latlon_cell(cell_section_coords,river_directions,total_cumulative_flow)
            allocate(this%cell_neighborhood, source=latlon_dir_based_rdirs_neighborhood(cell_section_coords, &
                     river_directions, total_cumulative_flow))
    end subroutine init_dir_based_rdirs_latlon_cell

    subroutine init_irregular_latlon_dir_based_rdirs_cell(this,cell_section_coords,river_directions, &
                                                          total_cumulative_flow,cell_neighbors)
        class(irregular_latlon_dir_based_rdirs_cell), intent(inout) :: this
        type(irregular_latlon_section_coords), intent(in) :: cell_section_coords
        integer, dimension(:,:), intent(in) :: river_directions
        integer, dimension(:,:), intent(in) :: total_cumulative_flow
        integer, dimension(:,:), pointer, intent(in) :: cell_neighbors
            call this%init_irregular_latlon_cell(cell_section_coords,river_directions,total_cumulative_flow)
            allocate(this%cell_neighborhood, &
                     source=irregular_latlon_dir_based_rdirs_neighborhood(cell_section_coords,&
                                                                          cell_neighbors, &
                                                                          river_directions,&
                                                                          total_cumulative_flow))
    end subroutine init_irregular_latlon_dir_based_rdirs_cell

    subroutine yamazaki_init_dir_based_rdirs_latlon_cell(this,cell_section_coords,river_directions,&
                                                         total_cumulative_flow,yamazaki_outlet_pixels, &
                                                         yamazaki_section_coords)
        class(latlon_dir_based_rdirs_cell) :: this
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: yamazaki_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
        integer, dimension(:,:) :: yamazaki_outlet_pixels
            call this%yamazaki_init_latlon_cell(cell_section_coords,river_directions,total_cumulative_flow,&
                                                yamazaki_outlet_pixels)
            allocate(this%cell_neighborhood, source=&
                     latlon_dir_based_rdirs_neighborhood(cell_section_coords,river_directions,&
                     total_cumulative_flow,yamazaki_outlet_pixels,yamazaki_section_coords))
    end subroutine yamazaki_init_dir_based_rdirs_latlon_cell

    subroutine yamazaki_latlon_calculate_river_directions_as_indices(this,coarse_river_direction_indices, &
                                                                     required_coarse_section_coords)
        class(latlon_cell), intent(inout) :: this
        integer, dimension(:,:,:), intent(inout) :: coarse_river_direction_indices
        class(section_coords), intent(in), optional :: required_coarse_section_coords
        class(coords), pointer :: destination_cell_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: initial_cell_coords
        integer :: outlet_pixel_type
        integer :: lat_offset, lon_offset
            if (present(required_coarse_section_coords)) then
                select type (required_coarse_section_coords)
                    type is (latlon_section_coords)
                        lat_offset = required_coarse_section_coords%section_min_lat - 1
                        lon_offset = required_coarse_section_coords%section_min_lon - 1
                end select
            else
                lat_offset = 0
                lon_offset = 0
            end if
            initial_outlet_pixel => this%yamazaki_retrieve_initial_outlet_pixel(outlet_pixel_type)
            initial_cell_coords => this%cell_neighborhood%yamazaki_get_cell_coords(initial_outlet_pixel)
            if (outlet_pixel_type == this%yamazaki_sink_outlet .or. outlet_pixel_type == this%yamazaki_rmouth_outlet) then
                select type (initial_cell_coords)
                type is (latlon_coords)
                    coarse_river_direction_indices(initial_cell_coords%lat,initial_cell_coords%lon,1) = &
                        -abs(outlet_pixel_type)
                    coarse_river_direction_indices(initial_cell_coords%lat,initial_cell_coords%lon,2) = &
                        -abs(outlet_pixel_type)
                end select
                deallocate(initial_cell_coords)
                deallocate(initial_outlet_pixel)
                return
            end if
            destination_cell_coords => this%cell_neighborhood%yamazaki_find_downstream_cell(initial_outlet_pixel)
            select type (initial_cell_coords)
            type is (latlon_coords)
                select type (destination_cell_coords)
                type is (latlon_coords)
                    coarse_river_direction_indices(initial_cell_coords%lat,initial_cell_coords%lon,1) = &
                        destination_cell_coords%lat - lat_offset
                    coarse_river_direction_indices(initial_cell_coords%lat,initial_cell_coords%lon,2) = &
                        destination_cell_coords%lon - lon_offset
                end select
            end select
            deallocate(initial_cell_coords)
            deallocate(destination_cell_coords)
            deallocate(initial_outlet_pixel)
    end subroutine yamazaki_latlon_calculate_river_directions_as_indices

    ! Note must explicit pass this function the object this as it has the nopass attribute
    ! so that it can be shared by latlon_cell and latlon_neighborhood
    function latlon_dir_based_rdirs_neighbor_flows_to_pixel(this,coords_in, &
                  flipped_direction_from_neighbor) result(neighbor_flows_to_pixel)
        class(area), intent(in) :: this
        class(coords), intent(in) :: coords_in
        class(direction_indicator), intent(in) :: flipped_direction_from_neighbor
        class(*), pointer :: rdir
        logical neighbor_flows_to_pixel
        integer, dimension(9) :: directions_pointing_to_pixel
        directions_pointing_to_pixel = (/9,8,7,6,5,4,3,2,1/)
        select type(flipped_direction_from_neighbor)
        type is (dir_based_direction_indicator)
            rdir => this%river_directions%get_value(coords_in)
            select type(rdir)
            type is (integer)
                if ( rdir == &
                    directions_pointing_to_pixel(flipped_direction_from_neighbor%get_direction())) then
                    neighbor_flows_to_pixel = .TRUE.
                else
                    neighbor_flows_to_pixel = .FALSE.
                end if
            end select
            deallocate(rdir)
        end select
    end function latlon_dir_based_rdirs_neighbor_flows_to_pixel

    ! Note must explicit pass this function the object this as it has the nopass attribute
    ! so that it can be shared by latlon_cell and latlon_neighborhood
    function icon_si_index_based_rdirs_neighbor_flows_to_pixel_dummy(this,coords_in, &
                  flipped_direction_from_neighbor) result(neighbor_flows_to_pixel)
        class(area), intent(in) :: this
        class(coords), intent(in) :: coords_in
        class(direction_indicator), intent(in) :: flipped_direction_from_neighbor
        logical :: neighbor_flows_to_pixel
            select type (this)
            class is (icon_single_index_field)
                continue
            end select
            select type (coords_in)
            type is (generic_1d_coords)
                continue
            end select
            select type (flipped_direction_from_neighbor)
            type is (index_based_direction_indicator)
                continue
            end select
            neighbor_flows_to_pixel = .true.
    end function icon_si_index_based_rdirs_neighbor_flows_to_pixel_dummy


    pure function latlon_dir_based_rdirs_get_flow_direction_for_sink(this,is_for_cell) &
        result(flow_direction_for_sink)
        class(latlon_dir_based_rdirs_cell), intent(in) :: this
        class(direction_indicator), pointer :: flow_direction_for_sink
        logical, optional, intent(in) :: is_for_cell
            if (present(is_for_cell)) continue
            allocate(flow_direction_for_sink,source=&
                dir_based_direction_indicator(this%flow_direction_for_sink))
    end function latlon_dir_based_rdirs_get_flow_direction_for_sink

    pure function latlon_dir_based_rdirs_get_flow_direction_for_outflow(this,is_for_cell) &
        result(flow_direction_for_outflow)
        class(latlon_dir_based_rdirs_cell), intent(in) :: this
        class(direction_indicator), pointer :: flow_direction_for_outflow
        logical, optional, intent(in) :: is_for_cell
            if (present(is_for_cell)) continue
            allocate(flow_direction_for_outflow,source=&
                dir_based_direction_indicator(this%flow_direction_for_outflow))
    end function latlon_dir_based_rdirs_get_flow_direction_for_outflow

    pure function latlon_dir_based_rdirs_get_flow_direction_for_ocean_point(this,is_for_cell) &
        result(flow_direction_for_ocean_point)
        class(latlon_dir_based_rdirs_cell), intent(in) :: this
        class(direction_indicator), pointer :: flow_direction_for_ocean_point
        logical, optional, intent(in) :: is_for_cell
            if (present(is_for_cell)) continue
            allocate(flow_direction_for_ocean_point,source=&
                dir_based_direction_indicator(this%flow_direction_for_ocean_point))
    end function latlon_dir_based_rdirs_get_flow_direction_for_ocean_point

    pure function irregular_latlon_dir_based_rdirs_get_flow_direction_for_sink(this,is_for_cell) &
        result(flow_direction_for_sink)
        class(irregular_latlon_dir_based_rdirs_cell), intent(in) :: this
        class(direction_indicator), pointer :: flow_direction_for_sink
        logical, optional, intent(in) :: is_for_cell
            if (present(is_for_cell)) then
                if (is_for_cell) then
                    allocate(flow_direction_for_sink,source=&
                        index_based_direction_indicator(this%cell_flow_direction_for_sink))
                else
                    allocate(flow_direction_for_sink,source=&
                        index_based_direction_indicator(this%flow_direction_for_sink))
                end if
            else
                allocate(flow_direction_for_sink,source=&
                    index_based_direction_indicator(this%flow_direction_for_sink))
            end if
    end function irregular_latlon_dir_based_rdirs_get_flow_direction_for_sink

    pure function irregular_latlon_dir_based_rdirs_get_flow_direction_for_outflow(this,is_for_cell) &
        result(flow_direction_for_outflow)
        class(irregular_latlon_dir_based_rdirs_cell), intent(in) :: this
        class(direction_indicator), pointer :: flow_direction_for_outflow
        logical, optional, intent(in) :: is_for_cell
            if (present(is_for_cell)) then
                if (is_for_cell) then
                    allocate(flow_direction_for_outflow,source=&
                        index_based_direction_indicator(this%cell_flow_direction_for_outflow))
                else
                    allocate(flow_direction_for_outflow,source=&
                        index_based_direction_indicator(this%flow_direction_for_outflow))
                end if
            else
                allocate(flow_direction_for_outflow,source=&
                    index_based_direction_indicator(this%flow_direction_for_outflow))
            end if
    end function irregular_latlon_dir_based_rdirs_get_flow_direction_for_outflow

    pure function irregular_latlon_dir_based_rdirs_get_flow_direction_for_o_point(this,is_for_cell) &
        result(flow_direction_for_ocean_point)
        class(irregular_latlon_dir_based_rdirs_cell), intent(in) :: this
        class(direction_indicator), pointer :: flow_direction_for_ocean_point
        logical, optional, intent(in) :: is_for_cell
            if (present(is_for_cell)) then
                if (is_for_cell) then
                    allocate(flow_direction_for_ocean_point,source=&
                        index_based_direction_indicator(this%cell_flow_direction_for_ocean_point))
                else
                    allocate(flow_direction_for_ocean_point,source=&
                        index_based_direction_indicator(this%flow_direction_for_ocean_point))
                end if
            else
                allocate(flow_direction_for_ocean_point,source=&
                    index_based_direction_indicator(this%flow_direction_for_ocean_point))
            end if
    end function irregular_latlon_dir_based_rdirs_get_flow_direction_for_o_point

    ! Note must explicit pass this function the object this as it has the nopass attribute
    ! so that it can be shared by latlon_cell and latlon_neighborhood
    function dir_based_rdirs_is_diagonal(this,coords_in) result(diagonal)
        class(area),intent(in) :: this
        class(coords), intent(in) :: coords_in
        class(*), pointer :: rdir
        logical :: diagonal
            rdir => this%river_directions%get_value(coords_in)
            select type (rdir)
                type is (integer)
                if (rdir == 7 .or. rdir == 9 .or. rdir == 1 .or. rdir == 3) then
                    diagonal = .TRUE.
                else
                    diagonal = .FALSE.
                end if
            end select
            deallocate(rdir)
    end function dir_based_rdirs_is_diagonal


    ! Note must explicit pass this function the object this as it has the nopass attribute
    ! so that it can be shared by icon_single_index_cell and icon_single_index_neighborhood
    function icon_si_index_based_rdirs_is_diagonal_dummy(this,coords_in) result(diagonal)
        class(area),intent(in) :: this
        class(coords), intent(in) :: coords_in
        logical :: diagonal
            select type (this)
            type is (icon_single_index_index_based_rdirs_field)
                continue
            end select
            select type (coords_in)
            type is (generic_1d_coords)
                continue
            end select
            diagonal = .false.
    end function icon_si_index_based_rdirs_is_diagonal_dummy

    subroutine latlon_dir_based_rdirs_mark_ocean_and_river_mouth_points(this)
        class(latlon_dir_based_rdirs_cell), intent(inout) :: this
        class(*), pointer :: rdir
        logical :: non_ocean_cell_found
        integer :: i,j
            non_ocean_cell_found = .FALSE.
            do j = this%section_min_lon, this%section_max_lon
                do i = this%section_min_lat, this%section_max_lat
                    rdir => this%river_directions%get_value(latlon_coords(i,j))
                    select type (rdir)
                    type is (integer)
                        if( rdir == this%flow_direction_for_outflow ) then
                            this%contains_river_mouths = .TRUE.
                            this%ocean_cell = .FALSE.
                        else if( rdir /= this%flow_direction_for_ocean_point ) then
                            non_ocean_cell_found = .TRUE.
                        end if
                    end select
                    deallocate(rdir)
                    if(this%contains_river_mouths) return
                end do
            end do
            this%contains_river_mouths = .FALSE.
            this%ocean_cell = .not. non_ocean_cell_found
    end subroutine latlon_dir_based_rdirs_mark_ocean_and_river_mouth_points

    subroutine irregular_latlon_dir_based_rdirs_mark_o_and_rm_points(this)
        class(irregular_latlon_dir_based_rdirs_cell), intent(inout) :: this
        class(latlon_field_section), pointer :: cell_numbers
        integer, pointer, dimension(:,:) :: cell_numbers_data
        class(*), pointer :: rdir
        logical :: non_ocean_cell_found
        integer :: i,j
            cell_numbers => this%cell_numbers
            select type (cell_numbers_data_ptr => cell_numbers%data)
            type is (integer)
                cell_numbers_data => cell_numbers_data_ptr
            end select
            non_ocean_cell_found = .FALSE.
            do j = this%section_min_lon, this%section_max_lon
                do i = this%section_min_lat, this%section_max_lat
                    if (cell_numbers_data(i,j) == this%cell_number) then
                        rdir => this%river_directions%get_value(latlon_coords(i,j))
                        select type (rdir)
                        type is (integer)
                            if( rdir == this%flow_direction_for_outflow ) then
                                this%contains_river_mouths = .TRUE.
                                this%ocean_cell = .FALSE.
                            else if( rdir /= this%flow_direction_for_ocean_point ) then
                                non_ocean_cell_found = .TRUE.
                            end if
                        end select
                        deallocate(rdir)
                        if(this%contains_river_mouths) return
                    end if
                end do
            end do
            this%contains_river_mouths = .FALSE.
            this%ocean_cell = .not. non_ocean_cell_found
    end subroutine irregular_latlon_dir_based_rdirs_mark_o_and_rm_points

    subroutine neighborhood_destructor(this)
        class(neighborhood), intent(inout) :: this
            if (associated(this%river_directions)) deallocate(this%river_directions)
            if (associated(this%total_cumulative_flow)) deallocate(this%total_cumulative_flow)
            if (associated(this%yamazaki_outlet_pixels)) deallocate(this%yamazaki_outlet_pixels)
    end subroutine neighborhood_destructor

    function find_downstream_cell(this,initial_outlet_pixel) result(downstream_cell)
        class(neighborhood), intent(in) :: this
        class(coords), intent(in) :: initial_outlet_pixel
        class(coords), allocatable :: working_pixel
        integer :: downstream_cell
        class(*), pointer :: initial_cumulative_flow
        class(*), pointer :: last_cell_cumulative_flow
        logical :: coords_not_in_neighborhood
        logical :: outflow_pixel_reached
            initial_cumulative_flow => this%total_cumulative_flow%get_value(initial_outlet_pixel)
            coords_not_in_neighborhood = .FALSE.
            allocate(working_pixel,source=initial_outlet_pixel)
            call this%find_next_pixel_downstream(this,working_pixel,coords_not_in_neighborhood,outflow_pixel_reached)
            do
                downstream_cell = this%in_which_cell(working_pixel)
                do
                    last_cell_cumulative_flow => this%total_cumulative_flow%get_value(working_pixel)
                    call this%find_next_pixel_downstream(this,working_pixel,coords_not_in_neighborhood,outflow_pixel_reached)
                    if (coords_not_in_neighborhood .or. .not. downstream_cell == this%in_which_cell(working_pixel) .or. &
                        outflow_pixel_reached) exit
                    deallocate(last_cell_cumulative_flow)
                end do
                if (coords_not_in_neighborhood .or. outflow_pixel_reached) exit
                select type(last_cell_cumulative_flow)
                type is (integer)
                    select type(initial_cumulative_flow)
                    type is (integer)
                        if ((last_cell_cumulative_flow  - initial_cumulative_flow > area_threshold) .and. &
                            ( .not. this%is_center_cell(downstream_cell))) exit
                    end select
                end select
                deallocate(last_cell_cumulative_flow)
            end do
            deallocate(last_cell_cumulative_flow)
            deallocate(initial_cumulative_flow)
            deallocate(working_pixel)
    end function find_downstream_cell

    function yamazaki_find_downstream_cell(this,initial_outlet_pixel) result(cell_coords)
        class(neighborhood), intent(in) :: this
        class(coords), intent(in), pointer :: initial_outlet_pixel
        class(coords), pointer :: cell_coords
        class(coords), pointer :: new_cell_coords
        class(coords), pointer :: working_pixel
        class(*), pointer :: yamazaki_outlet_pixel_type
        logical :: coords_not_in_neighborhood
        logical :: outflow_pixel_reached
        logical :: downstream_cell_found
        real(double_precision) :: downstream_path_length
        integer :: count_of_cells_passed_through
            downstream_cell_found = .false.
            allocate(working_pixel,source=initial_outlet_pixel)
            cell_coords => this%yamazaki_get_cell_coords(initial_outlet_pixel)
            downstream_path_length = 0.0
            count_of_cells_passed_through = 0
            do
                call this%find_next_pixel_downstream(this,working_pixel,coords_not_in_neighborhood,outflow_pixel_reached)
                downstream_path_length = downstream_path_length + this%calculate_length_through_pixel(this,working_pixel)
                if (coords_not_in_neighborhood) call this%yamazaki_wrap_coordinates(working_pixel)
                new_cell_coords => this%yamazaki_get_cell_coords(working_pixel)
                if ( .not. cell_coords%are_equal_to(new_cell_coords)) then
                    deallocate(cell_coords)
                    allocate(cell_coords,source=new_cell_coords)
                    count_of_cells_passed_through = count_of_cells_passed_through + 1
                end if
                deallocate(new_cell_coords)
                yamazaki_outlet_pixel_type => this%yamazaki_outlet_pixels%get_value(working_pixel)
                select type (yamazaki_outlet_pixel_type)
                type is (integer)
                    if (yamazaki_outlet_pixel_type == this%yamazaki_normal_outlet .and. &
                        downstream_path_length > MUFP) downstream_cell_found = .true.
                    if (yamazaki_outlet_pixel_type == this%yamazaki_rmouth_outlet .or. &
                        yamazaki_outlet_pixel_type == this%yamazaki_sink_outlet) &
                        downstream_cell_found = .true.
                end select
                deallocate(yamazaki_outlet_pixel_type)
                if (downstream_cell_found) exit
                if (count_of_cells_passed_through > yamazaki_max_range) then
                    deallocate(working_pixel)
                    allocate(working_pixel,source=initial_outlet_pixel)
                    call this%find_next_pixel_downstream(this,working_pixel,coords_not_in_neighborhood,outflow_pixel_reached)
                    if (coords_not_in_neighborhood) call this%yamazaki_wrap_coordinates(working_pixel)
                    deallocate(cell_coords)
                    cell_coords => this%yamazaki_get_cell_coords(working_pixel)
                    exit
                end if
            end do
            deallocate(working_pixel)
    end function yamazaki_find_downstream_cell

    function path_reenters_region(this,coords_inout)
        class(neighborhood), intent(in) :: this
        class(coords), allocatable, intent(inout) :: coords_inout
        logical path_reenters_region
        logical :: coords_not_in_neighborhood
        do
            call this%find_next_pixel_upstream(coords_inout,coords_not_in_neighborhood)
            if (coords_not_in_neighborhood) then
                path_reenters_region = .FALSE.
                exit
            else if (this%in_which_cell(coords_inout) == this%center_cell_id) then
                path_reenters_region = .TRUE.
                exit
            end if
        end do
    end function path_reenters_region

    pure function is_center_cell(this,cell_id)
        implicit none
        class(neighborhood), intent(in) :: this
        integer, intent(in) :: cell_id
        logical :: is_center_cell
            if (cell_id == this%center_cell_id) then
                is_center_cell = .TRUE.
            else
                is_center_cell = .FALSE.
            end if
    end function is_center_cell

    subroutine init_latlon_neighborhood(this,center_cell_section_coords,river_directions,&
                                        total_cumulative_flow)
        class(latlon_neighborhood) :: this
        type(latlon_section_coords) :: center_cell_section_coords
        type(latlon_section_coords) :: section_coords
        integer, target, dimension(:,:) :: river_directions
        integer, target, dimension(:,:) :: total_cumulative_flow
        class (*), pointer, dimension(:,:) :: river_directions_pointer
        class (*), pointer, dimension(:,:) :: total_cumulative_flow_pointer
            this%center_cell_min_lat = center_cell_section_coords%section_min_lat
            this%center_cell_min_lon = center_cell_section_coords%section_min_lon
            this%center_cell_max_lat =  center_cell_section_coords%section_min_lat + &
                                        center_cell_section_coords%section_width_lat - 1
            this%center_cell_max_lon =  center_cell_section_coords%section_min_lon + &
                                        center_cell_section_coords%section_width_lon - 1
            this%section_min_lat = center_cell_section_coords%section_min_lat - &
                                   center_cell_section_coords%section_width_lat
            this%section_min_lon = center_cell_section_coords%section_min_lon - &
                                   center_cell_section_coords%section_width_lon
            this%section_max_lat = this%center_cell_max_lat + &
                                   center_cell_section_coords%section_width_lat
            this%section_max_lon = this%center_cell_max_lon + &
                                   center_cell_section_coords%section_width_lon
            section_coords = latlon_section_coords(this%section_min_lat,this%section_min_lon,&
                                                   this%section_max_lat,this%section_max_lon)
            river_directions_pointer => river_directions
            this%river_directions => latlon_field_section(river_directions_pointer,&
                                                          section_coords)
            total_cumulative_flow_pointer => total_cumulative_flow
            this%total_cumulative_flow => latlon_field_section(total_cumulative_flow_pointer,&
                                                               section_coords)
            this%center_cell_id = 5
    end subroutine init_latlon_neighborhood

    subroutine init_irregular_latlon_neighborhood(this,center_cell_section_coords,&
                                                  coarse_cell_neighbors, &
                                                  river_directions,&
                                                  total_cumulative_flow)
        class(irregular_latlon_neighborhood), intent(inout) :: this
        class(irregular_latlon_section_coords) :: center_cell_section_coords
        type(irregular_latlon_section_coords) :: neighborhood_section_coords
        integer, dimension(:,:), target :: river_directions
        integer, dimension(:,:), target :: total_cumulative_flow
        integer, dimension(:,:), pointer :: coarse_cell_neighbors
        integer, dimension(:), pointer :: list_of_cell_numbers
        class(*), dimension(:,:), pointer :: river_directions_pointer
        class(*), dimension(:,:), pointer :: total_cumulative_flow_pointer
        class(*), dimension(:,:), pointer :: cell_numbers_pointer
        class(*), pointer :: cell_number_pointer
        integer :: i
        integer :: max_nneigh_coarse,nneigh_coarse
        integer :: nbr_cell_number
            max_nneigh_coarse = size(coarse_cell_neighbors,2)
            this%center_cell_id = center_cell_section_coords%cell_number
            allocate(this%list_of_cells_in_neighborhood)
            nneigh_coarse = 0
            do i=1,max_nneigh_coarse
                nbr_cell_number = coarse_cell_neighbors(this%center_cell_id,i)
                if (nbr_cell_number > 0) then
                    call this%list_of_cells_in_neighborhood%&
                        &add_value_to_back(nbr_cell_number)
                    nneigh_coarse = nneigh_coarse+1
                end if
            end do
            allocate(list_of_cell_numbers(nneigh_coarse+1))
            list_of_cell_numbers(1) = this%center_cell_id
            call this%list_of_cells_in_neighborhood%reset_iterator()
            i = 1
            do while (.not. this%list_of_cells_in_neighborhood%iterate_forward())
                i = i + 1
                cell_number_pointer => this%list_of_cells_in_neighborhood%get_value_at_iterator_position()
                select type (cell_number_pointer)
                type is (integer)
                    list_of_cell_numbers(i) = cell_number_pointer
                end select
            end do
            neighborhood_section_coords = &
                irregular_latlon_section_coords(list_of_cell_numbers(i),&
                                                center_cell_section_coords%cell_numbers, &
                                                center_cell_section_coords%section_min_lats, &
                                                center_cell_section_coords%section_min_lons, &
                                                center_cell_section_coords%section_max_lats, &
                                                center_cell_section_coords%section_max_lons )
            cell_numbers_pointer => center_cell_section_coords%cell_numbers
            this%cell_numbers => latlon_field_section(cell_numbers_pointer,neighborhood_section_coords)
            river_directions_pointer => river_directions
            this%river_directions => latlon_field_section(river_directions_pointer,&
                                                          neighborhood_section_coords)
            total_cumulative_flow_pointer => total_cumulative_flow
            this%total_cumulative_flow => latlon_field_section(total_cumulative_flow_pointer,&
                                                               neighborhood_section_coords)
            call neighborhood_section_coords%irregular_latlon_section_coords_destructor()
            deallocate(list_of_cell_numbers)
    end subroutine init_irregular_latlon_neighborhood

    subroutine irregular_latlon_neighborhood_destructor(this)
        class(irregular_latlon_neighborhood), intent(inout) :: this
            call neighborhood_destructor(this)
            if (associated(this%list_of_cells_in_neighborhood)) then
                call this%list_of_cells_in_neighborhood%reset_iterator()
                do while ( .not. this%list_of_cells_in_neighborhood%iterate_forward())
                    call this%list_of_cells_in_neighborhood%remove_element_at_iterator_position()
                end do
                deallocate(this%list_of_cells_in_neighborhood)
            end if
            if (associated(this%cell_numbers)) deallocate(this%cell_numbers)
    end subroutine irregular_latlon_neighborhood_destructor

    subroutine yamazaki_init_latlon_neighborhood(this,center_cell_section_coords,river_directions,&
                                                 total_cumulative_flow,yamazaki_outlet_pixels, &
                                                 yamazaki_section_coords)
        class(latlon_neighborhood) :: this
        type(latlon_section_coords) :: center_cell_section_coords
        type(latlon_section_coords) :: yamazaki_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
        integer, dimension(:,:), target :: yamazaki_outlet_pixels
        class(*), dimension(:,:), pointer :: yamazaki_outlet_pixels_pointer
            call this%init_latlon_neighborhood(center_cell_section_coords,river_directions,&
                                               total_cumulative_flow)
            yamazaki_outlet_pixels_pointer => yamazaki_outlet_pixels
            this%section_min_lat = yamazaki_section_coords%section_min_lat
            this%section_min_lon = yamazaki_section_coords%section_min_lon
            this%section_max_lat = yamazaki_section_coords%section_min_lat + yamazaki_section_coords%section_width_lat - 1
            this%section_max_lon = yamazaki_section_coords%section_min_lon + yamazaki_section_coords%section_width_lon - 1
            this%yamazaki_outlet_pixels => latlon_field_section(yamazaki_outlet_pixels_pointer, &
                                                                yamazaki_section_coords)
            this%yamazaki_cell_width_lat = this%center_cell_max_lat + 1 - this%center_cell_min_lat
            this%yamazaki_cell_width_lon = this%center_cell_max_lon + 1 - this%center_cell_min_lon
    end subroutine yamazaki_init_latlon_neighborhood

    function latlon_in_which_cell(this,pixel) result (in_cell)
        class(latlon_neighborhood), intent(in) :: this
        class(coords), intent(in) :: pixel
        integer :: in_cell
        integer :: internal_cell_lat
        integer :: internal_cell_lon
            select type(pixel)
            class is (latlon_coords)
                if (pixel%lat < this%center_cell_min_lat) then
                    internal_cell_lat = 1
                else if (pixel%lat > this%center_cell_max_lat) then
                    internal_cell_lat = 3
                else
                    internal_cell_lat = 2
                end if
                if (pixel%lon < this%center_cell_min_lon) then
                    internal_cell_lon = 1
                else if (pixel%lon > this%center_cell_max_lon) then
                    internal_cell_lon = 3
                else
                    internal_cell_lon = 2
                end if
                in_cell = internal_cell_lon + 9 - 3*internal_cell_lat
            end select
    end function latlon_in_which_cell

    function irregular_latlon_in_which_cell(this,pixel) result (in_cell)
        class(irregular_latlon_neighborhood), intent(in) :: this
        class(coords), intent(in) :: pixel
        class(*), pointer :: cell_number_ptr
        integer :: in_cell
            cell_number_ptr => this%cell_numbers%get_value(pixel)
            select type (cell_number_ptr )
            type is (integer)
                in_cell = cell_number_ptr
            end select
            deallocate(cell_number_ptr)
    end function irregular_latlon_in_which_cell

    pure function yamazaki_latlon_get_cell_coords(this,pixel_coords) result(cell_coords)
        class(latlon_neighborhood), intent(in) :: this
        class(coords), pointer, intent(in) :: pixel_coords
        class(coords), pointer :: cell_coords
            select type (pixel_coords)
            class is (latlon_coords)
                allocate(cell_coords,source=latlon_coords(ceiling(real(pixel_coords%lat)/this%yamazaki_cell_width_lat),&
                                                          ceiling(real(pixel_coords%lon)/this%yamazaki_cell_width_lon)))
            end select
    end function yamazaki_latlon_get_cell_coords


    pure function yamazaki_irregular_latlon_get_cell_coords_dummy(this,pixel_coords) &
            result(cell_coords)
        class(irregular_latlon_neighborhood), intent(in) :: this
        class(coords), pointer, intent(in) :: pixel_coords
        class(coords), pointer :: cell_coords
            !Prevents compiler warnings
            select type (pixel_coords)
            end select
            select type (this)
            end select
            cell_coords => null()
    end function yamazaki_irregular_latlon_get_cell_coords_dummy

    !This will neeed updating if we use neighborhood smaller than the full grid size for the
    !yamazaki-style algorithm - at the moment it assumes at neighborhood edges are edges of the
    !full latitude-longitude grid and need longitudinal wrapping.
    subroutine yamazaki_latlon_wrap_coordinates(this,pixel_coords)
        class(latlon_neighborhood), intent(in) :: this
        class(coords), intent(inout) :: pixel_coords
            if ( .not. yamazaki_wrap ) return
            select type (pixel_coords)
            class is (latlon_coords)
                if (pixel_coords%lon < this%section_min_lon) then
                    pixel_coords%lon = this%section_max_lon + pixel_coords%lon + 1 - this%section_min_lon
                else if (pixel_coords%lon > this%section_max_lon) then
                    pixel_coords%lon = this%section_min_lon + pixel_coords%lon - 1 - this%section_max_lon
                end if
            end select
    end subroutine yamazaki_latlon_wrap_coordinates

    subroutine yamazaki_irregular_latlon_wrap_coordinates_dummy(this,pixel_coords)
        class(irregular_latlon_neighborhood), intent(in) :: this
        class(coords), intent(inout) :: pixel_coords
            !Prevents compiler warnings
            select type (pixel_coords)
            end select
            select type (this)
            end select
    end subroutine yamazaki_irregular_latlon_wrap_coordinates_dummy

    function latlon_dir_based_rdirs_field_constructor(field_section_coords,river_directions) &
        result(constructor)
        type(latlon_dir_based_rdirs_field), allocatable :: constructor
        type(latlon_section_coords) :: field_section_coords
        integer, dimension(:,:), target :: river_directions
        class(*), dimension(:,:), pointer :: river_directions_pointer
            allocate(constructor)
            river_directions_pointer => river_directions
            call constructor%init_latlon_field(field_section_coords,river_directions_pointer)
    end function

    function latlon_dir_based_rdirs_neighborhood_constructor(center_cell_section_coords,river_directions, &
                                                               total_cumulative_flow) result(constructor)
        type(latlon_dir_based_rdirs_neighborhood), allocatable :: constructor
        type(latlon_section_coords) :: center_cell_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
            allocate(constructor)
            call constructor%init_latlon_neighborhood(center_cell_section_coords,river_directions,&
                                                      total_cumulative_flow)
    end function latlon_dir_based_rdirs_neighborhood_constructor

    function irregular_latlon_dir_based_rdirs_neighborhood_constructor(center_cell_section_coords,&
                                                                       coarse_cell_neighbors, &
                                                                       river_directions,&
                                                                       total_cumulative_flow) &
                                                                        result(constructor)
        type(irregular_latlon_dir_based_rdirs_neighborhood), allocatable :: constructor
        type(irregular_latlon_section_coords) :: center_cell_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
        integer, dimension(:,:), pointer :: coarse_celL_neighbors
            allocate(constructor)
            call constructor%init_irregular_latlon_neighborhood(center_cell_section_coords, &
                                                                coarse_cell_neighbors, &
                                                                river_directions, &
                                                                total_cumulative_flow)
    end function irregular_latlon_dir_based_rdirs_neighborhood_constructor

    function icon_single_index_index_based_rdirs_field_constructor(field_section_coords,river_directions) &
            result(constructor)
        type(icon_single_index_index_based_rdirs_field), allocatable :: constructor
        type(generic_1d_section_coords) :: field_section_coords
        integer, dimension(:), target :: river_directions
        class(*), dimension(:), pointer :: river_directions_pointer
            allocate(constructor)
            river_directions_pointer => river_directions
            call constructor%init_icon_single_index_field(field_section_coords,river_directions_pointer)
    end function icon_single_index_index_based_rdirs_field_constructor

    function yamazaki_latlon_dir_based_rdirs_neighborhood_constructor(center_cell_section_coords,river_directions, &
                                                                      total_cumulative_flow,yamazaki_outlet_pixels,&
                                                                      yamazaki_section_coords) &
        result(constructor)
        type(latlon_dir_based_rdirs_neighborhood), allocatable :: constructor
        type(latlon_section_coords) :: center_cell_section_coords
        type(latlon_section_coords) :: yamazaki_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
        integer, dimension(:,:) :: yamazaki_outlet_pixels
            allocate(constructor)
            call constructor%yamazaki_init_latlon_neighborhood(center_cell_section_coords,river_directions,&
                                                               total_cumulative_flow,yamazaki_outlet_pixels,&
                                                               yamazaki_section_coords)
    end function yamazaki_latlon_dir_based_rdirs_neighborhood_constructor

    subroutine latlon_dir_based_rdirs_find_next_pixel_downstream(this,coords_inout,coords_not_in_area,&
                                                                 outflow_pixel_reached)
        class(area), intent(in) :: this
        class(coords), intent(inout) :: coords_inout
        logical, intent(out) :: coords_not_in_area
        logical, intent(out), optional :: outflow_pixel_reached
        class(*), pointer :: rdir
        select type (coords_inout)
        type is (latlon_coords)
            rdir => this%river_directions%get_value(coords_inout)
            select type (rdir)
            type is (integer)
                if ( rdir == 7 .or. rdir == 8 .or. rdir == 9) then
                    coords_inout%lat = coords_inout%lat - 1
                else if ( rdir == 1 .or. rdir == 2 .or. rdir == 3) then
                    coords_inout%lat = coords_inout%lat + 1
                end if
                if ( rdir == 7 .or. rdir == 4 .or. rdir == 1 ) then
                    coords_inout%lon = coords_inout%lon - 1
                else if ( rdir == 9 .or. rdir == 6 .or. rdir == 3 ) then
                    coords_inout%lon = coords_inout%lon + 1
                end if
                if ( rdir == 5 .or. rdir == 0 .or. rdir == -1 ) then
                    coords_not_in_area = .FALSE.
                    if (present(outflow_pixel_reached)) outflow_pixel_reached =  .TRUE.
                else
                    if (present(outflow_pixel_reached)) outflow_pixel_reached =  .FALSE.
                    if (this%check_if_coords_are_in_area(this,coords_inout)) then
                        coords_not_in_area = .FALSE.
                    else
                        coords_not_in_area = .TRUE.
                    end if
                end if
            end select
            deallocate(rdir)
        end select
    end subroutine latlon_dir_based_rdirs_find_next_pixel_downstream

    subroutine icon_single_index_index_based_rdirs_find_next_pixel_downstream(this,&
                                                                              coords_inout,&
                                                                              coords_not_in_area,&
                                                                              outflow_pixel_reached)
        class(area), intent(in) :: this
        class(coords), intent(inout) :: coords_inout
        logical, intent(out) :: coords_not_in_area
        logical, intent(out), optional :: outflow_pixel_reached
        class(*), pointer :: next_cell_index
        select type (coords_inout)
        type is (generic_1d_coords)
            select type (this)
            type is (icon_single_index_index_based_rdirs_field)
                next_cell_index => this%river_directions%get_value(coords_inout)
                select type (next_cell_index)
                type is (integer)
                    if ( next_cell_index == this%index_for_sink .or. &
                         next_cell_index == this%index_for_outflow .or. &
                         next_cell_index == this%index_for_ocean_point ) then
                        coords_not_in_area = .FALSE.
                        if (present(outflow_pixel_reached)) outflow_pixel_reached =  .TRUE.
                    else
                        if (present(outflow_pixel_reached)) outflow_pixel_reached =  .FALSE.
                        if (this%check_if_coords_are_in_area(this,coords_inout)) then
                            coords_inout%index = next_cell_index
                            coords_not_in_area = .FALSE.
                        else
                            coords_not_in_area = .TRUE.
                        end if
                    end if
                end select
            end select
            deallocate(next_cell_index)
        end select
    end subroutine icon_single_index_index_based_rdirs_find_next_pixel_downstream

    pure function dir_based_rdirs_calculate_direction_indicator(this,downstream_cell) result(flow_direction)
        class(latlon_dir_based_rdirs_neighborhood), intent(in) :: this
        integer, intent(in) :: downstream_cell
        class(direction_indicator), pointer :: flow_direction
            !Would be better to assign this directly but has not yet been implemented
            !in all Fortran compilers
            allocate(flow_direction,source=dir_based_direction_indicator(downstream_cell))
            !Pointless test to get rid of compiler warning that 'this' is unused
            if(this%section_min_lat /= 0) continue
    end function dir_based_rdirs_calculate_direction_indicator

    pure function irregular_latlon_calculate_direction_indicator(this,downstream_cell) &
            result(flow_direction)
        class(irregular_latlon_dir_based_rdirs_neighborhood), intent(in) :: this
        integer, intent(in) :: downstream_cell
        class(direction_indicator), pointer :: flow_direction
            allocate(flow_direction,source=index_based_direction_indicator(downstream_cell))
            !Pointless test to get rid of compiler warning that 'this' is unused
            if(this%section_min_lat /= 0) continue
    end function irregular_latlon_calculate_direction_indicator

end module area_mod
