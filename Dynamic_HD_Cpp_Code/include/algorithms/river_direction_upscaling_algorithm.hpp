#include "base/coords.hpp"
#include "base/section_coords.hpp"
#include "base/field_section.hpp"
#include "base/subfield.hpp"
#include "base/grid.hpp"
#include <forward_list>
#include <list>

// This file contains comments using a special markup for doxygen

//> An abstract class containing sections of a number of fields that all correspond to the same section of
// the underlying grid of pixel and (mostly virtual) operator this grid section and the various
// field sections for it.

class rdir_upscaling_parameters {
    public:
        rdir_upscaling_parameters(double MUFP_in = 1.5,
                                  int area_threshold_in = 9,
                                  bool run_check_for_sinks_in = true) :
            MUFP(MUFP_in), area_threshold(area_threshold_in),
            run_check_for_sinks(run_check_for_sinks_in),
            flow_max_range(9),
            flow_wrap(true),
            upscale_k_parameters(false) {}
        //! Given with default values
        //! The minimum upstream flow path; the minimum distance through the cell that the
        //! highest flow path leading to a given exit pixel must have passed through
        double MUFP = 1.5;
        //! The miminum additional area a downstream pixel must have gathered before
        //! the cell it is in can be chosen as the downstream cell of a given centre cell
        int area_threshold = 9;
        //! Check for sinks within a cell; this can be turned off for the sake of efficiency
        //! when using a set of fine river directions known not to contain sinks
        bool run_check_for_sinks = true;
        //! The maximum number of cells to let a path go for between outlet pixels for the
        //! FLOW algorithm
        int flow_max_range = 9;
        //! Wrap the grid when using the flow algorithm?
        bool flow_wrap = true;
        bool upscale_k_parameters;
};

class area {
    protected:
        grid_params* _grid_params;
        //! A section of a field of pixel level river direction
        field_section<int>* river_directions;
        //! A section of a field of pixel level total cumulative flow
        field_section<int>* total_cumulative_flow;
        //! A section of a field used only by the flow style algorithm to mark
        //! pixels as outlet pixels
        field_section<pixel_type>* flow_outlet_pixels;

        // //! FLOW algorithm code for a normal outlet pixel
        // int flow_normal_outlet = 1;
        // //! FLOW algorithm code for the outlet pixel of a cell
        // //! where the outlet is not the LCDA (and thus a cell that won't
        // //! receive inflow from other cells; any cells flows entering it
        // //! will pass through and look for suitable outlets in the cell
        // //! beyond; this is because this cells own flow needs to exit via
        // //! a pixel that is not its outlet pixel
        // int flow_source_cell_outlet = 2;
        // //! FLOW algorithm code for a sink outlet pixel
        // int flow_sink_outlet = -1;
        // //! FLOW algorithm code for a river mouth pixel
        // int flow_rmouth_outlet = -2;
        //! Object containing cotat/flow parameters to use
        rdir_upscaling_parameters* parameters;
        bool delete_parameters_on_exit;
        //! Subroutine to find the next pixel upstream from a given pixel and return it via the
        //! input coordinate object; also return a flag if path has left the section and optionally
        //! if an outlet pixel has been found for the flow-style algorithm
        void find_next_pixel_upstream(coords*& coords_inout,
                                      bool& coords_not_in_area);
        //! Return a list of neighbours that upstream of the pixel located at the input coordinates
        virtual list<coords*>* find_upstream_neighbors(coords* coords_in) = 0;
        //! Check if a given set of coordinates are within the bounds of this area
        virtual bool check_if_coords_are_in_area(coords* coords_in) = 0;
        //! Check if neighbor at a given coordinates flows to a pixel which lies in a given
        //! direction from that neighbor (this direction being expressed in reverse, i.e. pointing
        //! from the pixel to the neighbor)
        virtual bool neighbor_flows_to_pixel(coords* coords_in,
                                             direction_indicator* flipped_direction_from_neighbor) = 0;
        //! Subroutine to find the next pixel downstream from the input pixel and returns it
        //! Also flag if the cell next coords would be outside the area and for the flow style
        //! algorithm if the next pixel is an outflow pixel (this is an optional intent OUT argument).
        virtual coords* find_next_pixel_downstream(coords* coords_in,
                                                   bool& coords_not_in_area,
                                                   bool& outflow_pixel_reached) = 0;
        //! Work out the length of a path through this pixel but looking up its flow direction
        virtual double calculate_length_through_pixel(coords* coords_in) = 0;
        //! Check if the pixel at the input coordinates is a diagonal or not by looking up its flow direction
        virtual bool is_diagonal(coords* coords_in) = 0;
    public:
        ~area();
        //virtual void print_area();
};

class latlon_area : public virtual area {
    public:
        virtual ~latlon_area() {};
    protected:
        //! The minimum latitude defining the section's lower edge
        int section_min_lat;
        //! The minimum longitude defining the section's left edge
        int section_min_lon;
        //! The maximum latitude defining the section's upper edge
        int section_max_lat;
        //! The maximum longitude defining the section's right edge
        int section_max_lon;
        double calculate_length_through_pixel(coords* coords_in);
        bool check_if_coords_are_in_area(coords* coords_in);
        //void print_area();
        bool is_diagonal(coords* coords_in);
        list<coords*>* find_upstream_neighbors(coords* coords_in);
};

class irregular_latlon_area : public virtual latlon_area {
    public:
        virtual ~irregular_latlon_area() {};

};

class icon_single_index_area : public virtual area {
    public:
        virtual ~icon_single_index_area() {}
    protected:
        int ncells;
        double calculate_length_through_pixel(coords* coords_in) { return 1.0; };
        bool check_if_coords_are_in_area(coords* coords_in) { return true; };
        //void print_area();
        bool is_diagonal(coords* coords_in) { return false; };
        list<coords*>* find_upstream_neighbors(coords* coords_in) { return nullptr; };
};

class extended_area : virtual public area {
public:
        virtual ~extended_area();
    protected:
        // Cells that will require to be reprocessed because of loops
        field<bool>* cells_to_reprocess;
        bool check_cell_for_localized_loops(coords* coords_in);
        public:
        virtual bool* check_area_for_localized_loops() = 0;
};

//! A neighbourhood includes a centre cell and a number of cells surrounding it
//! and it contains functions that need to run over that entire neighborhood for the
//! COTAT+ and flow-style upscaling algorithms

class neighborhood : public virtual area {
    protected:
        section_coords* neighborhood_section_coords;
        int center_cell_id;
        //! Find the downstream cell for the COTAT+ algorithm from a given starting pixel
        //! returns an integer indicating the downstream cell found
    public:
        virtual ~neighborhood();
        int find_downstream_cell(coords* initial_outlet_pixel);
        //! Trace the path from this cell upstream to see if it re-enters the centre
        //! cell before exiting this neighborhood. Returns a flag indicating if this
        //! occurs
        bool path_reenters_region(coords*& coords_inout);
        //! Check if the cell specified by the input cell id is the centre cell
        bool is_center_cell(int cell_id);
        //! Find the downstream cell for the FLOW algorithm
        coords* flow_find_downstream_cell(coords* initial_outlet_pixel);
        //! Calculate an appropriate direction indicator from the id of the downstream cell
        virtual direction_indicator* calculate_direction_indicator(int downstream_cell) = 0;
        //! Calculate which cell a given pixel is in; return an integer identifying that cell
        //! from those in the neighborhood
        virtual int in_which_cell(coords* pixel) = 0;
        //! Get the coordinates of the cell that the input pixel lies in
        virtual coords* flow_get_cell_coords(coords* pixel_coords) = 0;
        //! Subroutine that wraps a set of input pixel coordinates round the international date
        //! line if required
        virtual coords* flow_wrap_coordinates(coords* pixel_coords) = 0;
};

//! An object to contain the fine field sections corresponding to a single cell of the
//! coarse grid; also includes the necessary functions that act on that cell to
//! produce upscaled COTAT+ and flow style upscalings. Each cell contains a
//! neighborhood object that contains both it and its surrounding cells.

class cell : public virtual area {
    protected:
        //! An array marking pixels that have already been rejected as the
        //! outlet pixel of this cell
        subfield<bool>* rejected_pixels;
        //! An array containing the cumulative flow calculated across this
        //! cell only (and thus not counting pixels external to this cell)
        subfield<int>* cell_cumulative_flow;
        //! A neighborhood object containing the neighborhood of this cell
        neighborhood* cell_neighborhood;
        //! The number of pixels that lie along the edge of this cell
        int number_of_edge_pixels;
        //! Flag if this cell contains river mouth on the fine grid
        bool contains_river_mouths;
        //! Flag if this an ocean cell - i.e. the fine pixels inside it are
        //! all marked sea in the land sea mask
        bool ocean_cell;
        //! The coordinates of the edges of this section within the wider fine fields
        section_coords* cell_section_coords;
    public:
        virtual ~cell();
        //! Process the cell represented by this cell object to produce a river direction
        direction_indicator* process_cell();
        //! Find and mark the outlet pixel of this cell for the flow style algorithm;
        //! an input flag indicates if the LCDA criterion is being used
        void flow_mark_cell_outlet_pixel(bool use_LCDA_criterion);
        //! Setter for the logical variable contain_river_mouths
        void set_contains_river_mouths(bool value);
        //! Print details of this cell and neighborhoods
        //virtual void print_cell_and_neighborhood();
        //! Find and return the position of the outlet pixel for this cell. Also returns
        //! via an argument a flag to indicate if there was no remaining outlet pixels and
        //! takes a flag whether we should LCDA or not and optionally returns a
        //! flag indicating if this is the LCDA or not
        coords* find_outlet_pixel(bool& no_remaining_outlets,
                                  bool use_LCDA_criterion,
                                  bool& outlet_is_LCDA);
        void calculate_k_parameters(coords* outlet_pixel_coords);
        //! Subroutine to check if either the sum of the flows to sinks or the sum of the
        //! flows to river mouths is greater than the outflow at the choosen normal outlet
        //! for the cell. Takes as arguments the position of the current normal outlet and
        //! a flag to indicate if there were no remaining outlets (and a default was used)
        //! return two flags indicating whether cell should be marked a sink or a river
        //! mouth outflow. Note only one (or zero) of the flags can be true at a time
        void check_for_sinks_and_rmouth_outflows(coords*& outlet_pixel_coords,
                                                 bool no_remaining_outlets,
                                                 bool& mark_sink,
                                                 bool& mark_outflow);
        //! Find the downstream cell for the COTAT+ algorithm and calculate the direction
        //! to it. Takes the position of the outlet pixel and a flag to indicate if
        //! no valid outlets were found and a default is being used. Returns the flow
        //! direction
        direction_indicator* find_cell_flow_direction(coords*& outlet_pixel_coords,
                                                      bool& no_remaining_outlets);
        //! Find the next non-rejected pixel with the Largest Upstream Drainage Area
        //! and return its coordinates. If there is no remaining valid LUDA then return
        //! a default position along the cell's edge and indicate this through the
        //! no remaning outlets flag that is also returned.
        coords* find_pixel_with_LUDA(bool& no_remaining_outlets);
        //! Find the pixel with the Largest Cell Drainage Area; the pixel that drains the
        //! largest area of the cell itself and return its coordinates.
        coords* find_pixel_with_LCDA();
        //! Measure the upstream path length through the cell of an outlet pixel at the supplied
        //! coordinates. Return the path length as a double
        double measure_upstream_path(coords* outlet_pixel_coords);
        //! Check if a given outlet pixel either meets the minimum upstream flow path threshold
        //! or if the outlet pixel is the LCDA. If one of these checks is passed return true
        //! otherwise returns false. Must supply the position of the outlet pixel - the current
        //! LUDA - via the first argument and the position of the LCDA via the second. An optional
        //! third argument can be used to output a flag indicating whether the LUCA is the
        //! LCDA or not.
        bool check_MUFP(coords* LUDA_pixel_coords,
                        coords* LCDA_pixel_coords,
                        bool& LUDA_is_LCDA);
        //! Test generating cumulative flow field for this cell. Has no arguments (apart from this)
        //! and returns an array containing the cumulative flow field generated.
        int* test_generate_cell_cumulative_flow();
        //! Test finding a downstream cell for the flow style algorithm using the location
        //! of a given initial outlet pixel; returns the location of the downstream cell found.
        coords* flow_test_find_downstream_cell(coords* initial_outlet_pixel);
        //! Generates the cumulative flow for the section of fine orography river directions
        //! covered by this cell. Takes no arguments other than this; returns nothing (except
        //! via modifications to this - i.e. by filling in the cumulative flow array for this
        //! cell object).
        void generate_cell_cumulative_flow();
        //! Function that retrieves a previously marked outlet pixel in this cell and returns
        //! its location. Also returns the outlet pixels type using a intent OUT argument.
        coords* flow_retrieve_initial_outlet_pixel(pixel_type& outlet_pixel_type);
        //! Find and return a list of the location of all the pixels along the edges of this
        //! cell
        virtual forward_list<coords*>* find_edge_pixels() = 0;
        //! Initialize a subfield object to zero to hold the cell cumulative flow.Takes no
        //! arguments other than this; returns nothing (except via modifications to this)
        virtual void initialize_cell_cumulative_flow_subfield() = 0;
        //! Initialize a subfield object to false to hold the rejected pixels flow.Takes no
        //! arguments other than this; returns nothing (except via modifications to this)
        virtual void initialize_rejected_pixels_subfield() = 0;
        //! Returns a list of the position of the pixels in the cell; takes no arguments
        //! except this
        virtual list<coords*>* generate_list_of_pixels_in_cell() = 0;
        //! Return the appropriate direction indicator for a sink
        virtual direction_indicator* get_flow_direction_for_sink(bool is_for_cell=false) = 0;
        //! Return the appropriate direction indicator for an outflow
        virtual direction_indicator* get_flow_direction_for_outflow(bool is_for_cell=false) = 0;
        //! Return the appropriate direction indicator for an ocean point
        virtual direction_indicator* get_flow_direction_for_ocean_point(bool is_for_cell=false) = 0;
        //! Function to calculate the combined cumulative flow into all the sinks in this
        //! cell and return it; takes no arguments other than this
        virtual int get_sink_combined_cumulative_flow(coords*& main_sink_coords) = 0;
        //! Function to calculate the combined cumulative flow from all the river mouth in this
        //! cell and return it; takes no arguments other than this
        virtual int get_rmouth_outflow_combined_cumulative_flow(coords*& main_outflow_coords) = 0;
        //! Subroutine that set the contains river mouths and ocean cell flags by scanning over
        //! the cell. If no none sea points are found it is marked as an ocean cell. If one or
        //! more river mouths are found then the contains river mouths flag is set to TRUE
        virtual void mark_ocean_and_river_mouth_points() = 0;
        //! Subroutine to calculate the river direction as set of indices for the flow
        //! style algorithm and mark it in array of indices pointing to the next cell which
        //! is both given and returned via an argument with intent INOUT.
        // virtual void flow_calculate_river_directions_as_indices(
        //                 int* coarse_river_direction_indices,
        //                 section_coords* required_coarse_section_coords = nullptr) = 0;
};

//! An abstract implementation of the neighborhood class for latitude longitude grids

class latlon_neighborhood : public virtual neighborhood,latlon_area {
    public:
        latlon_neighborhood(section_coords* center_cell_section_coords,
                            grid_params* grid_params_in,
                            int* river_directions_in,
                            int* total_cumulative_flow_in,
                            rdir_upscaling_parameters* parameters_in=nullptr);
        virtual ~latlon_neighborhood() {};
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
        //! for the flow-style algorithm
        int flow_cell_width_lat;
        //! For convience keep the longitudinal width of a cell in the neighborhood
        //! for the flow-style algorithm
        int flow_cell_width_lon;
        //! Initalize a latitude longitude neighborhood object for the the flow-style
        //! algorithm. First three arguments (after the this argument) are the same as for
        //! init_latlon_neighborhood but has two additional arguments at end of argument
        //! list; one is array of the outlet pixels; second is the coordinates of the edges
        //! of the neighborhood as a whole - as flow-style algorithm is non-local this
        //! is more than direct neighbors - can be the whole grid or a wider subsection of it
        // void flow_init_latlon_neighborhood();
        int in_which_cell(coords* pixel);
        coords* flow_get_cell_coords(coords* pixel_coords);
        coords* flow_wrap_coordinates(coords* pixel_coords);
};

class irregular_latlon_neighborhood : public neighborhood, public latlon_area {
    public:
    irregular_latlon_neighborhood(section_coords* center_cell_section_coords,
                                  grid_params* grid_params_in,
                                  int* coarse_cell_neighbors,
                                  int* river_directions,
                                  int* total_cumulative_flow,
                                  int max_nneigh_coarse,
                                  rdir_upscaling_parameters* parameters_in=nullptr);
        virtual ~irregular_latlon_neighborhood();
    protected:
        field_section<int>* cell_numbers;
        list<int> list_of_cells_in_neighborhood;
        bool check_if_coords_are_in_area(coords* coords_in);
        int in_which_cell(coords* pixel);
        coords* flow_get_cell_coords(coords* pixel_coords);
        coords* flow_wrap_coordinates(coords* pixel_coords) { return nullptr; };
};

//! An abstract implementation of the cell class for latitude longitude grids
class latlon_cell : public cell, public virtual latlon_area {
    public:
        //! Initialize a latitude longitude cell object. Arguments are a set of
        //! coordinates of the bounds of the cell and a fine river direction field
        //! and fine cumulative flow field
        latlon_cell(latlon_section_coords* cell_section_coords_in,
                    grid_params* _grid_params_in,
                    int* river_directions_in,
                    int* total_cumulative_flow_in,
                    rdir_upscaling_parameters* parameters_in=nullptr);
        latlon_cell();
        virtual ~latlon_cell() {};
    protected:
        //! Latitudinal width of the cell in pixels
        int section_width_lat;
        //! Longitudinal width of the cell in pixels
        int section_width_lon;

        void init_latlon_cell();
        //! Initialize a latitude longitude cell object for the flow-style algorithm.
        //! The first three arguements (after the this argument) are the same as
        //! init_latlon_cell. The last argument is an array of outlet pixels.
        void flow_init_latlon_cell();
        virtual forward_list<coords*>* find_edge_pixels();
        void initialize_cell_cumulative_flow_subfield();
        void initialize_rejected_pixels_subfield();
        virtual list<coords*>* generate_list_of_pixels_in_cell();
        virtual int get_sink_combined_cumulative_flow(coords*& main_sink_coords);
        virtual int get_rmouth_outflow_combined_cumulative_flow(
                        coords*& main_outflow_coords);
        // virtual void flow_calculate_river_directions_as_indices(
        //                 int* coarse_river_direction_indices,
        //                 section_coords* required_coarse_section_coords = nullptr);
        //void print_cell_and_neighborhood();
};

//! An abstract implementation of the cell class for latitude longitude grids
class irregular_latlon_cell : public latlon_cell, public irregular_latlon_area {
    public:
        irregular_latlon_cell(irregular_latlon_section_coords* cell_section_coords,
                              grid_params* _grid_params_in,
                              int* river_directions,
                              int* total_cumulative_flow,
                              rdir_upscaling_parameters* parameters_in=nullptr);
        int cell_number;
        virtual ~irregular_latlon_cell();
    protected:
        int cell_flow_direction_for_sink = -5;
        int cell_flow_direction_for_outflow = -1;
        int cell_flow_direction_for_ocean_point = -2;
        field_section<int>* cell_numbers;
        forward_list<coords*>* find_edge_pixels();
        list<coords*>* generate_list_of_pixels_in_cell();
        bool check_if_coords_are_in_area(coords* coords_in);
        int get_sink_combined_cumulative_flow(coords*& main_sink_coords);
        int get_rmouth_outflow_combined_cumulative_flow(coords*& main_outflow_coords);
        // void flow_calculate_river_directions_as_indices(
        //                 int* coarse_river_direction_indices,
        //                 section_coords* required_coarse_section_coords = nullptr){};
};

class dir_based_rdirs_area : public virtual area {
    protected:
        //! Flow direction to use for a sink point
        int flow_direction_for_sink = 5;
        //! Flow direction to use for an outflow point
        int flow_direction_for_outflow = 0;
        //! Flow direction to use for a (non-outflow) ocean point
        int flow_direction_for_ocean_point = -1;
        bool neighbor_flows_to_pixel(coords* coords_in,
                                     direction_indicator*
                                     flipped_direction_from_neighbor);

        coords* find_next_pixel_downstream(coords* coords_in,
                                       bool& coords_not_in_area,
                                       bool& outflow_pixel_reached);
};

class index_based_rdirs_area : public virtual area {
    protected:
        //! Flow direction to use for a sink point
        int index_for_sink = -5;
        //! Flow direction to use for an outflow point
        int index_for_outflow = -1;
        //! Flow direction to use for a (non-outflow) ocean point
        int index_for_ocean_point = -2;
        bool neighbor_flows_to_pixel(coords* coords_in,
                                     direction_indicator*
                                     flipped_direction_from_neighbor);
        coords* find_next_pixel_downstream(coords* coords_in,
                                           bool& coords_not_in_area,
                                           bool& outflow_pixel_reached);
};

class dir_based_rdirs_latlon_neighborhood : public latlon_neighborhood,
                                            public dir_based_rdirs_area {
    public:
        direction_indicator* calculate_direction_indicator(int downstream_cell);
        dir_based_rdirs_latlon_neighborhood(latlon_section_coords* cell_section_coords_in,
                                            grid_params* grid_params_in,
                                            int* river_directions_in,
                                            int* total_cumulative_flow_in,
                                            rdir_upscaling_parameters*
                                            parameters_in=nullptr);
        ~dir_based_rdirs_latlon_neighborhood() {};
};

class dir_based_rdirs_latlon_cell : public latlon_cell,
                                    public dir_based_rdirs_area {
    public:
        dir_based_rdirs_latlon_cell(latlon_section_coords* cell_section_coords_in,
                                    grid_params* grid_params_in,
                                    int* river_directions_in,
                                    int* total_cumulative_flow_in,
                                    rdir_upscaling_parameters* parameters_in=nullptr);
        void mark_ocean_and_river_mouth_points();
        ~dir_based_rdirs_latlon_cell() {};
    protected:
        //! Return the appropriate direction indicator for a sink
        direction_indicator* get_flow_direction_for_sink(bool is_for_cell=false);
        //! Return the appropriate direction indicator for an outflow
        direction_indicator* get_flow_direction_for_outflow(bool is_for_cell=false);
        //! Return the appropriate direction indicator for an ocean point
        direction_indicator* get_flow_direction_for_ocean_point(bool is_for_cell=false);

};

class dir_based_rdirs_irregular_latlon_cell : public irregular_latlon_cell,
                                              public dir_based_rdirs_area {
    public:
        dir_based_rdirs_irregular_latlon_cell(irregular_latlon_section_coords* cell_section_coords_in,
                                              grid_params* grid_params_in,
                                              int* coarse_cell_neighbors_in,
                                              int* river_directions_in,
                                              int* total_cumulative_flow_in,
                                              int max_nneigh_coarse_in,
                                              rdir_upscaling_parameters* parameters_in=nullptr);
        ~dir_based_rdirs_irregular_latlon_cell() {};
        void mark_ocean_and_river_mouth_points();
    protected:
        //! Return the appropriate direction indicator for a sink
        direction_indicator* get_flow_direction_for_sink(bool is_for_cell=false);
        //! Return the appropriate direction indicator for an outflow
        direction_indicator* get_flow_direction_for_outflow(bool is_for_cell=false);
        //! Return the appropriate direction indicator for an ocean point
        direction_indicator* get_flow_direction_for_ocean_point(bool is_for_cell=false);
};

class dir_based_rdirs_irregular_latlon_neighborhood :
        public irregular_latlon_neighborhood,
        public dir_based_rdirs_area {
    public:
        dir_based_rdirs_irregular_latlon_neighborhood(latlon_section_coords* cell_section_coords_in,
                                                      grid_params* grid_params_in,
                                                      int* coarse_cell_neighbors_in,
                                                      int* river_directions_in,
                                                      int* total_cumulative_flow_in,
                                                      int max_nneigh_coarse_in,
                                                      rdir_upscaling_parameters*
                                                      parameters_in=nullptr);
        ~dir_based_rdirs_irregular_latlon_neighborhood() {};
        direction_indicator* calculate_direction_indicator(int downstream_cell);
};

class latlon_extended_area : public extended_area, public latlon_area,
                             public dir_based_rdirs_area {
    public:
        latlon_extended_area(section_coords* area_section_coords_in,
                             grid_params* area_grid_params_in,
                             int* river_directions_in);
        ~latlon_extended_area() {};
        bool* check_area_for_localized_loops();
};

class icon_single_index_extended_area : public extended_area, public icon_single_index_area,
                                        public index_based_rdirs_area {
    public:
        icon_single_index_extended_area(section_coords* area_section_coords_in,
                                        grid_params* area_grid_params_in,
                                        int* river_directions_in);
        ~icon_single_index_extended_area() {};
        bool* check_area_for_localized_loops();
};
