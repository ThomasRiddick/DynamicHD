module latlon_lake_model_mod

use latlon_lake_model_tree_mod
#ifdef USE_LOGGING
  use latlon_lake_logger_mod
#endif

implicit none

integer,parameter :: dp = selected_real_kind(12)

integer, parameter :: filling_lake_type = 1
integer, parameter :: overflowing_lake_type = 2
integer, parameter :: subsumed_lake_type = 3
integer, parameter :: no_merge_mtype = 0
integer, parameter :: primary_merge_mtype   = 1
integer, parameter :: secondary_merge_mtype = 2
integer, parameter :: null_mtype = 3

type :: coordslist
  integer, pointer, dimension(:) :: lat_coords
  integer, pointer, dimension(:) :: lon_coords
end type coordslist

interface coordslist
  procedure :: coordslistconstructor
end interface coordslist

type :: mergeandredirectindices
  logical :: is_primary_merge
  logical :: merged
  logical :: local_redirect
  integer :: merge_target_lat_index
  integer :: merge_target_lon_index
  integer :: redirect_lat_index
  integer :: redirect_lon_index
  contains
    procedure :: initialisemergeandredirectindices
    procedure :: add_offset_to_merge_indices
    procedure :: reset_merge_indices
    procedure :: get_merge_type
    procedure :: get_merge_target_coords
    procedure :: get_outflow_redirect_coords
    procedure :: is_equal_to
end type mergeandredirectindices

interface mergeandredirectindices
  procedure :: mergeandredirectindicesconstructor
  procedure :: mergeandredirectindicesconstructorfromarray
end interface mergeandredirectindices

type :: mergeandredirectindicespointer
  type(mergeandredirectindices), pointer :: ptr
end type mergeandredirectindicespointer

type :: mergeandredirectindicescollection
  logical :: primary_merge
  logical :: secondary_merge
  type(mergeandredirectindicespointer), pointer, dimension(:) :: &
    primary_merge_and_redirect_indices
  integer :: primary_merge_and_redirect_indices_count
  type(mergeandredirectindices), pointer :: secondary_merge_and_redirect_indices
  contains
    procedure :: initialisemergeandredirectindicescollection
    procedure :: mergeandredirectindicescollectiondestructor
    procedure :: add_offset_to_collection
    procedure :: reset_collection
end type mergeandredirectindicescollection

interface mergeandredirectindicescollection
  procedure :: mergeandredirectindicescollectionconstructor
end interface mergeandredirectindicescollection

type :: mergeandredirectindicescollectionpointer
  type(mergeandredirectindicescollection), pointer :: ptr
end type mergeandredirectindicescollectionpointer

type :: lakeparameters
  logical :: instant_throughflow
  real(dp) :: lake_retention_coefficient
  real(dp) :: minimum_lake_volume_threshold
  logical, pointer, dimension(:,:) :: lake_centers
  real(dp), pointer, dimension(:,:) :: connection_volume_thresholds
  real(dp), pointer, dimension(:,:) :: flood_volume_thresholds
  logical, pointer, dimension(:,:) :: flood_only
  integer, pointer, dimension(:,:) :: connect_merge_and_redirect_indices_index
  integer, pointer, dimension(:,:) :: flood_merge_and_redirect_indices_index
  type(mergeandredirectindicescollectionpointer), pointer, dimension(:) :: &
    connect_merge_and_redirect_indices_collections
  type(mergeandredirectindicescollectionpointer), pointer, dimension(:) :: &
    flood_merge_and_redirect_indices_collections
  real(dp), pointer, dimension(:,:) :: cell_areas_on_surface_model_grid
  real(dp), pointer, dimension(:,:) :: cell_areas
  real(dp), pointer, dimension(:,:) :: raw_heights
  real(dp), pointer, dimension(:,:) :: corrected_heights
  real(dp), pointer, dimension(:,:) :: flood_heights
  real(dp), pointer, dimension(:,:) :: connection_heights
  integer, pointer, dimension(:,:) :: basin_numbers
  integer, pointer, dimension(:,:) :: number_fine_grid_cells
  type(coordslist), pointer, dimension(:) :: basins
  type(coordslist), pointer, dimension(:) :: surface_cell_to_fine_cell_maps
  integer, pointer, dimension(:,:) :: surface_cell_to_fine_cell_map_numbers
  integer, pointer, dimension(:,:) :: flood_next_cell_lat_index
  integer, pointer, dimension(:,:) :: flood_next_cell_lon_index
  integer, pointer, dimension(:,:) :: connect_next_cell_lat_index
  integer, pointer, dimension(:,:) :: connect_next_cell_lon_index
  integer, pointer, dimension(:,:) :: corresponding_surface_cell_lat_index
  integer, pointer, dimension(:,:) :: corresponding_surface_cell_lon_index
  integer :: nlat,nlon
  integer :: nlat_coarse,nlon_coarse
  integer :: nlat_surface_model,nlon_surface_model
  contains
     procedure :: initialiselakeparameters
     procedure :: lakeparametersdestructor
end type lakeparameters

interface lakeparameters
  procedure :: lakeparametersconstructor
end interface lakeparameters

type :: lakepointer
  type(lake), pointer :: lake_pointer
end type lakepointer

type :: lakefields
  logical, pointer, dimension(:,:) :: connected_lake_cells
  logical, pointer, dimension(:,:) :: flooded_lake_cells
  integer, pointer, dimension(:,:) :: lake_numbers
  integer, pointer, dimension(:,:) :: buried_lake_numbers
  real(dp), pointer, dimension(:,:) :: effective_volume_per_cell_on_surface_grid
  real(dp), pointer, dimension(:,:) :: new_effective_volume_per_cell_on_surface_grid
  real(dp), pointer, dimension(:,:) :: effective_lake_height_on_surface_grid_from_lakes
  real(dp), pointer, dimension(:,:) :: effective_lake_height_on_surface_grid_to_lakes
  real(dp), pointer, dimension(:,:) :: evaporation_on_surface_grid
  real(dp), pointer, dimension(:,:) :: water_to_lakes
  real(dp), pointer, dimension(:,:) :: water_to_hd
  real(dp), pointer, dimension(:,:) :: lake_water_from_ocean
  integer, pointer, dimension(:,:) :: number_lake_cells
  real(dp), pointer, dimension(:,:) :: true_lake_depths
  integer, pointer, dimension(:) :: cells_with_lakes_lat
  integer, pointer, dimension(:) :: cells_with_lakes_lon
  real(dp), pointer, dimension(:) :: evaporation_from_lakes
  logical, pointer, dimension(:) :: evaporation_applied
  type(rooted_tree_forest), pointer :: set_forest
  type(lakepointer), pointer, dimension(:) :: other_lakes
  contains
    procedure :: initialiselakefields
    procedure :: lakefieldsdestructor
end type lakefields

interface lakefields
  procedure :: lakefieldsconstructor
end interface lakefields

type lakeprognostics
  type(lakepointer), pointer, dimension(:) :: lakes
  real(dp) :: total_lake_volume
  contains
    procedure :: initialiselakeprognostics
    procedure :: lakeprognosticsdestructor
end type lakeprognostics

interface lakeprognostics
  procedure :: lakeprognosticsconstructor
end interface lakeprognostics

type :: lake
  integer :: lake_type
  integer :: lake_number
  integer :: center_cell_lat
  integer :: center_cell_lon
  real(dp)    :: lake_volume
  real(dp)    :: secondary_lake_volume
  real(dp)    :: unprocessed_water
  integer :: current_cell_to_fill_lat
  integer :: current_cell_to_fill_lon
  integer :: center_cell_coarse_lat
  integer :: center_cell_coarse_lon
  integer :: previous_cell_to_fill_lat
  integer :: previous_cell_to_fill_lon
  ! Need to nullify these pointers so tests for
  ! a lack of association work
  integer, dimension(:), pointer :: filled_lake_cell_lats => null()
  integer, dimension(:), pointer :: filled_lake_cell_lons => null()
  integer :: filled_lake_cell_index
  type(lakeparameters), pointer :: lake_parameters
  type(lakefields), pointer :: lake_fields
  integer :: number_of_flooded_cells
  integer :: secondary_number_of_flooded_cells
  ! There are no specific filling lake variables
  ! Overflowing lake variables
  real(dp) :: excess_water
  integer :: outflow_redirect_lat
  integer :: outflow_redirect_lon
  integer :: next_merge_target_lat
  integer :: next_merge_target_lon
  logical :: local_redirect
  real(dp)    :: lake_retention_coefficient
  ! Subsumed lake variables
  integer :: primary_lake_number
  contains
    procedure :: add_water
    procedure :: remove_water
    procedure :: accept_merge
    procedure :: store_water
    procedure :: find_true_primary_lake
    procedure :: find_true_rolledback_primary_lake
    procedure :: get_merge_indices_index
    procedure :: get_merge_indices_collection
    procedure :: check_if_merge_is_possible
    procedure :: initialiselake
    procedure :: set_overflowing_lake_parameter_to_null_values
    procedure :: set_subsumed_lake_parameter_to_null_values
    procedure :: accept_split
    procedure :: drain_current_cell
    procedure :: release_negative_water
    procedure :: generate_lake_numbers
    procedure :: calculate_effective_lake_volume_per_cell
    procedure :: calculate_true_lake_depth
    !Filling lake procedures
    procedure :: initialisefillinglake
    procedure :: perform_primary_merge
    procedure :: perform_secondary_merge
    procedure :: conditionally_rollback_primary_merge
    procedure :: fill_current_cell
    procedure :: change_to_overflowing_lake
    procedure :: update_filling_cell
    procedure :: rollback_filling_cell
    !Overflowing lake procedures
    procedure :: initialiseoverflowinglake
    procedure :: change_overflowing_lake_to_filling_lake
    procedure :: drain_excess_water
    !Subsumed lake procedures
    procedure :: initialisesubsumedlake
    procedure :: change_subsumed_lake_to_filling_lake
end type lake

interface lake
  procedure :: lakeconstructor
end interface lake

contains

subroutine initialisemergeandredirectindices(this, &
                                             is_primary_merge_in, &
                                             local_redirect_in, &
                                             merge_target_lat_index_in, &
                                             merge_target_lon_index_in, &
                                             redirect_lat_index_in, &
                                             redirect_lon_index_in)
  class(mergeandredirectindices) :: this
  logical :: is_primary_merge_in
  logical :: local_redirect_in
  integer :: merge_target_lat_index_in
  integer :: merge_target_lon_index_in
  integer :: redirect_lat_index_in
  integer :: redirect_lon_index_in
    this%is_primary_merge = is_primary_merge_in
    this%local_redirect = local_redirect_in
    this%merged = .False.
    this%merge_target_lat_index = merge_target_lat_index_in
    this%merge_target_lon_index = merge_target_lon_index_in
    this%redirect_lat_index = redirect_lat_index_in
    this%redirect_lon_index = redirect_lon_index_in
end subroutine initialisemergeandredirectindices

function mergeandredirectindicesconstructor(is_primary_merge_in, &
                                            local_redirect_in, &
                                            merge_target_lat_index_in, &
                                            merge_target_lon_index_in, &
                                            redirect_lat_index_in, &
                                            redirect_lon_index_in) &
    result(constructor)
  type(mergeandredirectindices), pointer :: constructor
  logical :: is_primary_merge_in
  logical :: local_redirect_in
  integer :: merge_target_lat_index_in
  integer :: merge_target_lon_index_in
  integer :: redirect_lat_index_in
  integer :: redirect_lon_index_in
    allocate(constructor)
    call constructor%initialisemergeandredirectindices(is_primary_merge_in, &
                                                       local_redirect_in, &
                                                       merge_target_lat_index_in, &
                                                       merge_target_lon_index_in, &
                                                       redirect_lat_index_in, &
                                                       redirect_lon_index_in)
end function mergeandredirectindicesconstructor

function mergeandredirectindicesconstructorfromarray(is_primary_merge,array_in) &
    result(constructor)
  type(mergeandredirectindices), pointer :: constructor
  integer, dimension(:), pointer :: array_in
  logical :: is_primary_merge
    allocate(constructor)
    call constructor%initialisemergeandredirectindices(is_primary_merge, &
                                                       (array_in(2)==1), &
                                                       array_in(3), &
                                                       array_in(4), &
                                                       array_in(5), &
                                                       array_in(6))
end function mergeandredirectindicesconstructorfromarray

subroutine add_offset_to_merge_indices(this,offset)
  class(mergeandredirectindices) :: this
  integer, intent(in) :: offset
    this%merge_target_lat_index = this%merge_target_lat_index + offset
    this%merge_target_lon_index = this%merge_target_lon_index + offset
    this%redirect_lat_index = this%redirect_lat_index + offset
    this%redirect_lon_index = this%redirect_lon_index + offset
end subroutine add_offset_to_merge_indices

subroutine reset_merge_indices(this)
  class(mergeandredirectindices) :: this
    this%merged = .false.
end subroutine reset_merge_indices

function is_equal_to(this,rhs) result(equals)
  class(mergeandredirectindices) :: this
  type(mergeandredirectindices), pointer :: rhs
  logical :: equals
    equals = (this%is_primary_merge .eqv. rhs%is_primary_merge)
    equals = equals .and. (this%local_redirect .eqv. rhs%local_redirect)
    equals = equals .and. (this%merge_target_lat_index == rhs%merge_target_lat_index)
    equals = equals .and. (this%merge_target_lon_index == rhs%merge_target_lon_index)
    equals = equals .and. (this%redirect_lat_index == rhs%redirect_lat_index)
    equals = equals .and. (this%redirect_lon_index == rhs%redirect_lon_index)
end function is_equal_to


subroutine add_offset_to_collection(this,offset)
  class(mergeandredirectindicescollection) :: this
  class(mergeandredirectindices), pointer :: working_merge_and_redirect_indices
  integer, intent(in) :: offset
  integer :: i
    if(this%primary_merge) then
      do i = 1,this%primary_merge_and_redirect_indices_count
        working_merge_and_redirect_indices => &
          this%primary_merge_and_redirect_indices(i)%ptr
        call working_merge_and_redirect_indices%add_offset_to_merge_indices(offset)
      end do
    end if
    if(this%secondary_merge) then
      call this%secondary_merge_and_redirect_indices%add_offset_to_merge_indices(offset)
    end if
end subroutine add_offset_to_collection

subroutine reset_collection(this)
  class(mergeandredirectindicescollection) :: this
  class(mergeandredirectindices), pointer :: working_merge_and_redirect_indices
  integer :: i
    if(this%primary_merge) then
      do i = 1,this%primary_merge_and_redirect_indices_count
        working_merge_and_redirect_indices => &
          this%primary_merge_and_redirect_indices(i)%ptr
        call working_merge_and_redirect_indices%reset_merge_indices()
      end do
    end if
    if(this%secondary_merge) then
      call this%secondary_merge_and_redirect_indices%reset_merge_indices()
    end if
end subroutine reset_collection

subroutine initialisemergeandredirectindicescollection(this, &
                                                       primary_merge_and_redirect_indices_in, &
                                                       secondary_merge_and_redirect_indices_in)
  class(mergeandredirectindicescollection) :: this
  type(mergeandredirectindicespointer), pointer, dimension(:) :: &
    primary_merge_and_redirect_indices_in
  type(mergeandredirectindices), pointer :: secondary_merge_and_redirect_indices_in
    this%primary_merge_and_redirect_indices => primary_merge_and_redirect_indices_in
    this%secondary_merge_and_redirect_indices => secondary_merge_and_redirect_indices_in
    if (associated(this%primary_merge_and_redirect_indices)) then
      this%primary_merge_and_redirect_indices_count = size(this%primary_merge_and_redirect_indices)
      this%primary_merge = (this%primary_merge_and_redirect_indices_count > 0)
    else
      this%primary_merge_and_redirect_indices_count = 0
      this%primary_merge = .false.
    end if
    this%secondary_merge = associated(this%secondary_merge_and_redirect_indices)
end subroutine initialisemergeandredirectindicescollection

function mergeandredirectindicescollectionconstructor(primary_merge_and_redirect_indices_in, &
                                                        secondary_merge_and_redirect_indices_in) &
    result(constructor)
  type(mergeandredirectindicescollection), pointer :: constructor
  type(mergeandredirectindicespointer), pointer, dimension(:) :: &
    primary_merge_and_redirect_indices_in
  type(mergeandredirectindices), pointer :: secondary_merge_and_redirect_indices_in
    allocate(constructor)
    call constructor%initialisemergeandredirectindicescollection(primary_merge_and_redirect_indices_in, &
                                                                 secondary_merge_and_redirect_indices_in)
end function mergeandredirectindicescollectionconstructor

subroutine mergeandredirectindicescollectiondestructor(this)
  class(mergeandredirectindicescollection) :: this
  type(mergeandredirectindices), pointer :: working_merge_and_redirect_indices
  integer :: i
    if(this%primary_merge) then
      do i = 1,this%primary_merge_and_redirect_indices_count
        working_merge_and_redirect_indices => &
          this%primary_merge_and_redirect_indices(i)%ptr
        deallocate(working_merge_and_redirect_indices)
      end do
      deallocate(this%primary_merge_and_redirect_indices)
    end if
    if(this%secondary_merge) then
      deallocate(this%secondary_merge_and_redirect_indices)
    end if
end subroutine mergeandredirectindicescollectiondestructor

function createmergeindicescollectionsfromarray(array_in) &
    result(merge_and_redirect_indices_collections)
  integer, dimension(:,:,:), pointer :: array_in
  type(mergeandredirectindicescollectionpointer), pointer, dimension(:) :: &
      merge_and_redirect_indices_collections
  type(mergeandredirectindicescollection), pointer :: &
      working_merge_and_redirect_indices_collection
  type(mergeandredirectindices), pointer :: working_merge_and_redirect_indices
  type(mergeandredirectindices), pointer :: secondary_merge_and_redirect_indices
  type(mergeandredirectindicespointer), pointer, dimension(:) :: &
    primary_merge_and_redirect_indices
  integer, dimension(:), pointer :: array_slice
  integer :: i,j
  integer :: primary_merge_count
    allocate(merge_and_redirect_indices_collections(size(array_in,1)))
    do i = 1,size(array_in,1)
      if (size(array_in,2) > 1) then
        primary_merge_count = 0
        do j = 2,size(array_in,2)
          if (array_in(i,j,1) == 1) then
            primary_merge_count = j - 1
          end if
        end do
        if (primary_merge_count > 0) then
          allocate(primary_merge_and_redirect_indices(primary_merge_count))
        else
          primary_merge_and_redirect_indices => null()
        end if
      else
        primary_merge_and_redirect_indices => null()
      end if
      do j = 1,size(array_in,2)
        if (array_in(i,j,1) == 1) then
          array_slice => array_in(i,j,1:size(array_in,3))
          working_merge_and_redirect_indices => &
            mergeandredirectindices((j /= 1),array_slice)
          if (j == 1) then
            secondary_merge_and_redirect_indices => working_merge_and_redirect_indices
          else
            primary_merge_and_redirect_indices(j-1)%ptr => working_merge_and_redirect_indices
          end if
        else if (j == 1) then
          secondary_merge_and_redirect_indices => null()
        end if
      end do
      working_merge_and_redirect_indices_collection => &
        mergeandredirectindicescollection(primary_merge_and_redirect_indices, &
                                          secondary_merge_and_redirect_indices)
      merge_and_redirect_indices_collections(i)%ptr => &
        working_merge_and_redirect_indices_collection
    end do
end function createmergeindicescollectionsfromarray

subroutine initialiselake(this,center_cell_lat_in,center_cell_lon_in, &
                          current_cell_to_fill_lat_in, &
                          current_cell_to_fill_lon_in, &
                          center_cell_coarse_lat_in, &
                          center_cell_coarse_lon_in, &
                          lake_number_in,lake_volume_in,&
                          secondary_lake_volume_in,&
                          unprocessed_water_in,&
                          lake_parameters_in,lake_fields_in)
  class(lake),intent(inout) :: this
  integer, intent(in) :: center_cell_lat_in
  integer, intent(in) :: center_cell_lon_in
  integer, intent(in) :: current_cell_to_fill_lat_in
  integer, intent(in) :: current_cell_to_fill_lon_in
  integer, intent(in) :: center_cell_coarse_lat_in
  integer, intent(in) :: center_cell_coarse_lon_in
  integer, intent(in) :: lake_number_in
  real(dp), intent(in)    :: lake_volume_in
  real(dp), intent(in)    :: secondary_lake_volume_in
  real(dp), intent(in)    :: unprocessed_water_in
  type(lakeparameters), target, intent(in) :: lake_parameters_in
  type(lakefields), pointer, intent(inout) :: lake_fields_in
  type(mergeandredirectindicescollection), pointer :: merge_indices_collection
  logical :: is_flood_merge
  integer :: merge_indices_index
  integer :: i
    this%lake_parameters => lake_parameters_in
    this%lake_fields => lake_fields_in
    this%lake_number = lake_number_in
    this%lake_volume = lake_volume_in
    this%secondary_lake_volume = secondary_lake_volume_in
    this%center_cell_lat = center_cell_lat_in
    this%center_cell_lon = center_cell_lon_in
    this%center_cell_coarse_lat = center_cell_coarse_lat_in
    this%center_cell_coarse_lon = center_cell_coarse_lon_in
    this%current_cell_to_fill_lat = current_cell_to_fill_lat_in
    this%current_cell_to_fill_lon = current_cell_to_fill_lon_in
    this%unprocessed_water = unprocessed_water_in
    if (.not. associated(this%filled_lake_cell_lats)) then
      i = 0
      do
        merge_indices_index = this%get_merge_indices_index(is_flood_merge)
        if (merge_indices_index /= 0) then
          merge_indices_collection => &
            this%get_merge_indices_collection(merge_indices_index, &
                                              is_flood_merge)
          if (merge_indices_collection%secondary_merge) exit
        end if
        call this%update_filling_cell(.true.)
        i = i + 1
      end do
      if (i == 0) then
        i = 1
      end if
      allocate(this%filled_lake_cell_lats(i))
      allocate(this%filled_lake_cell_lons(i))
      this%filled_lake_cell_lats(:) = -1
      this%filled_lake_cell_lons(:) = -1
      this%filled_lake_cell_index = 0
      this%current_cell_to_fill_lat = current_cell_to_fill_lat_in
      this%current_cell_to_fill_lon = current_cell_to_fill_lon_in
      this%lake_fields%connected_lake_cells(:,:) = .false.
      this%lake_fields%flooded_lake_cells(:,:) = .false.
      this%number_of_flooded_cells = 0
      this%secondary_number_of_flooded_cells = 0
    end if
end subroutine initialiselake

subroutine generate_lake_numbers(this)
  class(lake),intent(inout) :: this
  type(mergeandredirectindicescollection), pointer :: merge_indices_collection
  integer :: merge_indices_index
  logical :: is_flood_merge
    do
      merge_indices_index = this%get_merge_indices_index(is_flood_merge)
      if (merge_indices_index /= 0) then
        merge_indices_collection => &
          this%get_merge_indices_collection(merge_indices_index, &
                                            is_flood_merge)
        if (merge_indices_collection%secondary_merge) exit
      end if
      call this%update_filling_cell(.false.)
    end do
end subroutine generate_lake_numbers

function coordslistconstructor(lat_coords_in,lon_coords_in) &
    result(constructor)
  type(coordslist) :: constructor
  integer, pointer, dimension(:) :: lat_coords_in
  integer, pointer, dimension(:) :: lon_coords_in
    constructor%lat_coords => lat_coords_in
    constructor%lon_coords => lon_coords_in
end function coordslistconstructor

subroutine initialiselakeparameters(this,lake_centers_in, &
                                    connection_volume_thresholds_in, &
                                    flood_volume_thresholds_in, &
                                    cell_areas_on_surface_model_grid_in, &
                                    cell_areas_in, &
                                    raw_heights_in, &
                                    corrected_heights_in, &
                                    flood_heights_in, &
                                    connection_heights_in, &
                                    flood_next_cell_lat_index_in, &
                                    flood_next_cell_lon_index_in, &
                                    connect_next_cell_lat_index_in, &
                                    connect_next_cell_lon_index_in, &
                                    connect_merge_and_redirect_indices_index_in, &
                                    flood_merge_and_redirect_indices_index_in, &
                                    connect_merge_and_redirect_indices_collections_in, &
                                    flood_merge_and_redirect_indices_collections_in, &
                                    corresponding_surface_cell_lat_index_in, &
                                    corresponding_surface_cell_lon_index_in, &
                                    nlat_in,nlon_in, &
                                    nlat_coarse_in,nlon_coarse_in, &
                                    nlat_surface_model_in,nlon_surface_model_in, &
                                    instant_throughflow_in, &
                                    lake_retention_coefficient_in)
  class(lakeparameters),intent(inout) :: this
  logical :: instant_throughflow_in
  real(dp) :: lake_retention_coefficient_in
  logical, pointer, dimension(:,:) :: lake_centers_in
  real(dp), pointer, dimension(:,:) :: connection_volume_thresholds_in
  real(dp), pointer, dimension(:,:) :: flood_volume_thresholds_in
  real(dp), pointer, dimension(:,:) :: cell_areas_on_surface_model_grid_in
  real(dp), pointer, dimension(:,:) :: cell_areas_in
  real(dp), pointer, dimension(:,:) :: raw_heights_in
  real(dp), pointer, dimension(:,:) :: corrected_heights_in
  real(dp), pointer, dimension(:,:) :: flood_heights_in
  real(dp), pointer, dimension(:,:) :: connection_heights_in
  integer, pointer, dimension(:,:) :: flood_next_cell_lat_index_in
  integer, pointer, dimension(:,:) :: flood_next_cell_lon_index_in
  integer, pointer, dimension(:,:) :: connect_next_cell_lat_index_in
  integer, pointer, dimension(:,:) :: connect_next_cell_lon_index_in
  integer, pointer, dimension(:,:) :: connect_merge_and_redirect_indices_index_in
  integer, pointer, dimension(:,:) :: flood_merge_and_redirect_indices_index_in
  type(mergeandredirectindicescollectionpointer), pointer, dimension(:) :: &
    connect_merge_and_redirect_indices_collections_in
  type(mergeandredirectindicescollectionpointer), pointer, dimension(:) :: &
    flood_merge_and_redirect_indices_collections_in
  integer, pointer, dimension(:,:) :: corresponding_surface_cell_lat_index_in
  integer, pointer, dimension(:,:) :: corresponding_surface_cell_lon_index_in
  integer, pointer, dimension(:,:) :: number_of_lake_cells_temp
  integer, pointer, dimension(:) :: basins_in_coarse_cell_lat_temp
  integer, pointer, dimension(:) :: basins_in_coarse_cell_lon_temp
  integer, pointer, dimension(:) :: basins_in_coarse_cell_lat
  integer, pointer, dimension(:) :: basins_in_coarse_cell_lon
  integer, pointer, dimension(:) :: surface_cell_to_fine_cell_map_lats_temp
  integer, pointer, dimension(:) :: surface_cell_to_fine_cell_map_lons_temp
  logical, pointer, dimension(:,:) :: needs_map
  type(coordslist), pointer, dimension(:) :: basins_temp
  integer :: nlat_in,nlon_in
  integer :: nlat_coarse_in,nlon_coarse_in
  integer :: nlat_surface_model_in,nlon_surface_model_in
  integer :: basin_number
  integer :: number_of_basins_in_coarse_cell
  integer :: i,j,k,l,m
  integer :: lat_scale_factor,lon_scale_factor
  integer :: lat_surface_model,lon_surface_model
  integer :: surface_cell_to_fine_cell_map_number
  integer :: map_number,number_of_lake_cells
    number_of_basins_in_coarse_cell = 0
    this%minimum_lake_volume_threshold = 0.0000001
    this%lake_centers => lake_centers_in
    this%connection_volume_thresholds => connection_volume_thresholds_in
    this%flood_volume_thresholds => flood_volume_thresholds_in
    this%cell_areas_on_surface_model_grid => cell_areas_on_surface_model_grid_in
    this%cell_areas => cell_areas_in
    this%raw_heights => raw_heights_in
    this%corrected_heights => corrected_heights_in
    this%flood_heights => flood_heights_in
    this%connection_heights => connection_heights_in
    this%flood_next_cell_lat_index => flood_next_cell_lat_index_in
    this%flood_next_cell_lon_index => flood_next_cell_lon_index_in
    this%connect_next_cell_lat_index => connect_next_cell_lat_index_in
    this%connect_next_cell_lon_index => connect_next_cell_lon_index_in
    this%connect_merge_and_redirect_indices_index => &
        connect_merge_and_redirect_indices_index_in
    this%flood_merge_and_redirect_indices_index => &
        flood_merge_and_redirect_indices_index_in
    this%connect_merge_and_redirect_indices_collections => &
        connect_merge_and_redirect_indices_collections_in
    this%flood_merge_and_redirect_indices_collections => &
        flood_merge_and_redirect_indices_collections_in
    this%corresponding_surface_cell_lat_index => corresponding_surface_cell_lat_index_in
    this%corresponding_surface_cell_lon_index => corresponding_surface_cell_lon_index_in
    this%nlat = nlat_in
    this%nlon = nlon_in
    this%nlat_coarse = nlat_coarse_in
    this%nlon_coarse = nlon_coarse_in
    this%nlat_surface_model = nlat_surface_model_in
    this%nlon_surface_model = nlon_surface_model_in
    this%instant_throughflow = instant_throughflow_in
    this%lake_retention_coefficient = lake_retention_coefficient_in
    allocate(this%flood_only(nlat_in,nlon_in))
    this%flood_only(:,:) = .false.
    allocate(this%number_fine_grid_cells(nlat_surface_model_in, &
                                         nlon_surface_model_in))
    this%number_fine_grid_cells(:,:) = 0
    where (this%connection_volume_thresholds == -1.0_dp)
      this%flood_only = .true.
    else where
      this%flood_only = .false.
    end where
    allocate(needs_map(nlat_surface_model_in, &
                       nlon_surface_model_in))
    needs_map(:,:) = .false.
    allocate(number_of_lake_cells_temp(nlat_surface_model_in, &
                                  nlon_surface_model_in))
    number_of_lake_cells_temp(:,:) = 0
    do j = 1,nlon_in
      do i = 1,nlat_in
        lat_surface_model = this%corresponding_surface_cell_lat_index(i,j)
        lon_surface_model = this%corresponding_surface_cell_lon_index(i,j)
        this%number_fine_grid_cells(lat_surface_model,lon_surface_model) = &
          this%number_fine_grid_cells(lat_surface_model,lon_surface_model) + 1
        if (this%flood_volume_thresholds(i,j) /= -1.0_dp) then
          needs_map(lat_surface_model,lon_surface_model) = .true.
          number_of_lake_cells_temp(lat_surface_model,lon_surface_model) = &
            number_of_lake_cells_temp(lat_surface_model,lon_surface_model) + 1
        end if
      end do
    end do
    allocate(this%surface_cell_to_fine_cell_map_numbers(nlat_surface_model_in, &
                                                        nlon_surface_model_in))
    this%surface_cell_to_fine_cell_map_numbers(:,:) = 0
    map_number = 0
    do j = 1,nlon_surface_model_in
      do i = 1,nlat_surface_model_in
        if (needs_map(i,j)) then
          map_number = map_number + 1
        end if
      end do
    end do
    allocate(this%surface_cell_to_fine_cell_maps(map_number))
    map_number = 0
    do j = 1,nlon_surface_model_in
      do i = 1,nlat_surface_model_in
        if (needs_map(i,j)) then
          map_number = map_number + 1
          number_of_lake_cells = number_of_lake_cells_temp(i,j)
          allocate(surface_cell_to_fine_cell_map_lats_temp(number_of_lake_cells))
          allocate(surface_cell_to_fine_cell_map_lons_temp(number_of_lake_cells))
          surface_cell_to_fine_cell_map_lats_temp(:) = -1
          surface_cell_to_fine_cell_map_lons_temp(:) = -1
          this%surface_cell_to_fine_cell_maps(map_number) = &
            coordslist(surface_cell_to_fine_cell_map_lats_temp, &
                       surface_cell_to_fine_cell_map_lons_temp)
          this%surface_cell_to_fine_cell_map_numbers(i,j) = map_number
        end if
      end do
    end do
    deallocate(number_of_lake_cells_temp)
    do j = 1,nlon_in
      do i = 1,nlat_in
        if (this%flood_volume_thresholds(i,j) /= -1.0) then
          lat_surface_model = this%corresponding_surface_cell_lat_index(i,j)
          lon_surface_model = this%corresponding_surface_cell_lon_index(i,j)
          surface_cell_to_fine_cell_map_number = &
            this%surface_cell_to_fine_cell_map_numbers(lat_surface_model, &
                                                       lon_surface_model)
          k = 1
          do while (this%surface_cell_to_fine_cell_maps(&
                    surface_cell_to_fine_cell_map_number)%lat_coords(k) /= -1)
            k = k + 1
          end do
          this%surface_cell_to_fine_cell_maps(surface_cell_to_fine_cell_map_number)%lat_coords(k) = i
          this%surface_cell_to_fine_cell_maps(surface_cell_to_fine_cell_map_number)%lon_coords(k) = j
        end if
      end do
    end do
    allocate(basins_in_coarse_cell_lat_temp((this%nlat*this%nlon)&
                                            /(this%nlat_coarse*this%nlon_coarse)))
    allocate(basins_in_coarse_cell_lon_temp((this%nlat*this%nlon)&
                                            /(this%nlat_coarse*this%nlon_coarse)))
    allocate(this%basin_numbers(this%nlat_coarse,this%nlon_coarse))
    this%basin_numbers(:,:) = 0
    allocate(basins_temp(this%nlat_coarse*this%nlon_coarse))
    basin_number = 0
    lat_scale_factor = nlat_in/nlat_coarse_in
    lon_scale_factor = nlon_in/nlon_coarse_in
    do j=1,nlon_coarse_in
      do i=1,nlat_coarse_in
        number_of_basins_in_coarse_cell = 0
        do l = 1+(j-1)*lon_scale_factor,j*lon_scale_factor
          do k = 1+(i-1)*lat_scale_factor,i*lat_scale_factor
            if (this%lake_centers(k,l)) then
              number_of_basins_in_coarse_cell = number_of_basins_in_coarse_cell + 1
              basins_in_coarse_cell_lat_temp(number_of_basins_in_coarse_cell) = k
              basins_in_coarse_cell_lon_temp(number_of_basins_in_coarse_cell) = l
            end if
         end do
        end do
        if(number_of_basins_in_coarse_cell > 0) then
          allocate(basins_in_coarse_cell_lat(number_of_basins_in_coarse_cell))
          allocate(basins_in_coarse_cell_lon(number_of_basins_in_coarse_cell))
          do m=1,number_of_basins_in_coarse_cell
            basins_in_coarse_cell_lat(m) = basins_in_coarse_cell_lat_temp(m)
            basins_in_coarse_cell_lon(m) = basins_in_coarse_cell_lon_temp(m)
          end do
          basin_number = basin_number + 1
          basins_temp(basin_number) = &
            coordslist(basins_in_coarse_cell_lat,basins_in_coarse_cell_lon)
          this%basin_numbers(i,j) = basin_number
        end if
      end do
    end do
    allocate(this%basins(basin_number))
    do m=1,basin_number
      this%basins(m) = basins_temp(m)
    end do
    deallocate(needs_map)
    deallocate(basins_temp)
    deallocate(basins_in_coarse_cell_lat_temp)
    deallocate(basins_in_coarse_cell_lon_temp)
end subroutine initialiselakeparameters

function lakeparametersconstructor(lake_centers_in, &
                                   connection_volume_thresholds_in, &
                                   flood_volume_thresholds_in, &
                                   cell_areas_on_surface_model_grid_in, &
                                   cell_areas_in, &
                                   raw_heights_in, &
                                   corrected_heights_in, &
                                   flood_heights_in, &
                                   connection_heights_in, &
                                   flood_next_cell_lat_index_in, &
                                   flood_next_cell_lon_index_in, &
                                   connect_next_cell_lat_index_in, &
                                   connect_next_cell_lon_index_in, &
                                   connect_merge_and_redirect_indices_index_in, &
                                   flood_merge_and_redirect_indices_index_in, &
                                   connect_merge_and_redirect_indices_collections_in, &
                                   flood_merge_and_redirect_indices_collections_in, &
                                   corresponding_surface_cell_lat_index_in, &
                                   corresponding_surface_cell_lon_index_in, &
                                   nlat_in,nlon_in, &
                                   nlat_coarse_in,nlon_coarse_in, &
                                   nlat_surface_model_in,nlon_surface_model_in, &
                                   instant_throughflow_in, &
                                   lake_retention_coefficient_in) result(constructor)
  type(lakeparameters), pointer :: constructor
  logical, pointer, dimension(:,:), intent(in) :: lake_centers_in
  real(dp), pointer, dimension(:,:), intent(in) :: connection_volume_thresholds_in
  real(dp), pointer, dimension(:,:), intent(in) :: flood_volume_thresholds_in
  real(dp), pointer, dimension(:,:), intent(in) :: cell_areas_on_surface_model_grid_in
  real(dp), pointer, dimension(:,:), intent(in) :: cell_areas_in
  real(dp), pointer, dimension(:,:), intent(in) :: raw_heights_in
  real(dp), pointer, dimension(:,:), intent(in) :: corrected_heights_in
  real(dp), pointer, dimension(:,:), intent(in) :: flood_heights_in
  real(dp), pointer, dimension(:,:), intent(in) :: connection_heights_in
  integer, pointer, dimension(:,:), intent(in) :: flood_next_cell_lat_index_in
  integer, pointer, dimension(:,:), intent(in) :: flood_next_cell_lon_index_in
  integer, pointer, dimension(:,:), intent(in) :: connect_next_cell_lat_index_in
  integer, pointer, dimension(:,:), intent(in) :: connect_next_cell_lon_index_in
  integer, pointer, dimension(:,:) :: connect_merge_and_redirect_indices_index_in
  integer, pointer, dimension(:,:) :: flood_merge_and_redirect_indices_index_in
  type(mergeandredirectindicescollectionpointer), pointer, dimension(:) :: &
    connect_merge_and_redirect_indices_collections_in
  type(mergeandredirectindicescollectionpointer), pointer, dimension(:) :: &
    flood_merge_and_redirect_indices_collections_in
  integer, pointer, dimension(:,:) :: corresponding_surface_cell_lat_index_in
  integer, pointer, dimension(:,:) :: corresponding_surface_cell_lon_index_in
  integer, intent(in) :: nlat_in,nlon_in
  integer, intent(in) :: nlat_coarse_in,nlon_coarse_in
  integer, intent(in) :: nlat_surface_model_in,nlon_surface_model_in
  logical, intent(in) :: instant_throughflow_in
  real(dp), intent(in) :: lake_retention_coefficient_in
    allocate(constructor)
    call constructor%initialiselakeparameters(lake_centers_in, &
                                              connection_volume_thresholds_in, &
                                              flood_volume_thresholds_in, &
                                              cell_areas_on_surface_model_grid_in, &
                                              cell_areas_in, &
                                              raw_heights_in, &
                                              corrected_heights_in, &
                                              flood_heights_in, &
                                              connection_heights_in, &
                                              flood_next_cell_lat_index_in, &
                                              flood_next_cell_lon_index_in, &
                                              connect_next_cell_lat_index_in, &
                                              connect_next_cell_lon_index_in, &
                                              connect_merge_and_redirect_indices_index_in, &
                                              flood_merge_and_redirect_indices_index_in, &
                                              connect_merge_and_redirect_indices_collections_in, &
                                              flood_merge_and_redirect_indices_collections_in, &
                                              corresponding_surface_cell_lat_index_in, &
                                              corresponding_surface_cell_lon_index_in, &
                                              nlat_in,nlon_in, &
                                              nlat_coarse_in,nlon_coarse_in, &
                                              nlat_surface_model_in,nlon_surface_model_in, &
                                              instant_throughflow_in, &
                                              lake_retention_coefficient_in)
end function lakeparametersconstructor

subroutine lakeparametersdestructor(this)
  class(lakeparameters),intent(inout) :: this
  type(mergeandredirectindicescollection), pointer :: collected_indices
  integer :: i
    deallocate(this%flood_only)
    deallocate(this%basin_numbers)
    do i = 1,size(this%basins)
      deallocate(this%basins(i)%lat_coords)
      deallocate(this%basins(i)%lon_coords)
    end do
    deallocate(this%basins)
    do i = 1,size(this%surface_cell_to_fine_cell_maps)
      deallocate(this%surface_cell_to_fine_cell_maps(i)%lat_coords)
      deallocate(this%surface_cell_to_fine_cell_maps(i)%lon_coords)
    end do
    if (associated(this%flood_merge_and_redirect_indices_collections)) then
      do i = 1,size(this%flood_merge_and_redirect_indices_collections)
          collected_indices => this%flood_merge_and_redirect_indices_collections(i)%ptr
          call collected_indices%mergeandredirectindicescollectiondestructor()
          deallocate(collected_indices)
      end do
      deallocate(this%flood_merge_and_redirect_indices_collections)
    end if
    if (associated(this%connect_merge_and_redirect_indices_collections)) then
      do i = 1,size(this%connect_merge_and_redirect_indices_collections)
          collected_indices => this%connect_merge_and_redirect_indices_collections(i)%ptr
          call collected_indices%mergeandredirectindicescollectiondestructor()
          deallocate(collected_indices)
      end do
      deallocate(this%connect_merge_and_redirect_indices_collections)
    end if
    deallocate(this%surface_cell_to_fine_cell_maps)
    deallocate(this%lake_centers)
    deallocate(this%connection_volume_thresholds)
    deallocate(this%flood_volume_thresholds)
    deallocate(this%cell_areas_on_surface_model_grid)
    deallocate(this%flood_next_cell_lat_index)
    deallocate(this%flood_next_cell_lon_index)
    deallocate(this%connect_next_cell_lat_index)
    deallocate(this%connect_next_cell_lon_index)
    deallocate(this%connect_merge_and_redirect_indices_index)
    deallocate(this%flood_merge_and_redirect_indices_index)
    deallocate(this%corresponding_surface_cell_lat_index)
    deallocate(this%corresponding_surface_cell_lon_index)
    deallocate(this%number_fine_grid_cells)
    deallocate(this%surface_cell_to_fine_cell_map_numbers)
end subroutine lakeparametersdestructor

subroutine initialiselakefields(this,lake_parameters)
    class(lakefields), intent(inout) :: this
    type(lakeparameters) :: lake_parameters
    integer :: lat_scale_factor, lon_scale_factor
    integer, pointer, dimension(:) :: cells_with_lakes_lat_temp
    integer, pointer, dimension(:) :: cells_with_lakes_lon_temp
    integer :: i,j,k,l
    integer :: number_of_cells_containing_lakes
    logical :: contains_lake
      allocate(this%flooded_lake_cells(lake_parameters%nlat,lake_parameters%nlon))
      this%flooded_lake_cells(:,:) = .false.
      allocate(this%connected_lake_cells(lake_parameters%nlat,lake_parameters%nlon))
      this%connected_lake_cells(:,:) = .false.
      allocate(this%lake_numbers(lake_parameters%nlat,lake_parameters%nlon))
      this%lake_numbers(:,:) = 0
      allocate(this%buried_lake_numbers(lake_parameters%nlat,lake_parameters%nlon))
      this%buried_lake_numbers(:,:) = 0
      allocate(this%effective_volume_per_cell_on_surface_grid(lake_parameters%nlat_surface_model,&
                                                              lake_parameters%nlon_surface_model))
      this%effective_volume_per_cell_on_surface_grid(:,:) = 0.0_dp
      allocate(this%effective_lake_height_on_surface_grid_to_lakes(lake_parameters%nlat_surface_model,&
                                                                   lake_parameters%nlon_surface_model))
      this%effective_lake_height_on_surface_grid_to_lakes(:,:) = 0.0_dp
      allocate(this%effective_lake_height_on_surface_grid_from_lakes(lake_parameters%nlat_surface_model, &
                                                                     lake_parameters%nlon_surface_model))
      this%effective_lake_height_on_surface_grid_from_lakes(:,:) = 0.0_dp
      allocate(this%new_effective_volume_per_cell_on_surface_grid(lake_parameters%nlat_surface_model,&
                                                                  lake_parameters%nlon_surface_model))
      this%new_effective_volume_per_cell_on_surface_grid(:,:) = 0.0_dp
      allocate(this%evaporation_on_surface_grid(lake_parameters%nlat_surface_model,&
                                                lake_parameters%nlon_surface_model))
      this%evaporation_on_surface_grid(:,:) = 0.0_dp
      allocate(this%water_to_lakes(lake_parameters%nlat_coarse,lake_parameters%nlon_coarse))
      this%water_to_lakes(:,:) = 0.0_dp
      allocate(this%lake_water_from_ocean(lake_parameters%nlat_coarse,lake_parameters%nlon_coarse))
      this%lake_water_from_ocean(:,:) = 0.0_dp
      allocate(this%water_to_hd(lake_parameters%nlat_coarse,lake_parameters%nlon_coarse))
      this%water_to_hd(:,:) = 0.0_dp
      allocate(this%number_lake_cells(lake_parameters%nlat_surface_model,&
                                      lake_parameters%nlon_surface_model))
      this%number_lake_cells(:,:) = 0
      allocate(this%true_lake_depths(lake_parameters%nlat,lake_parameters%nlon))
      this%true_lake_depths(:,:) = 0.0_dp
      allocate(cells_with_lakes_lat_temp(lake_parameters%nlat_coarse*lake_parameters%nlon_coarse))
      allocate(cells_with_lakes_lon_temp(lake_parameters%nlat_coarse*lake_parameters%nlon_coarse))
      number_of_cells_containing_lakes = 0
      lat_scale_factor = lake_parameters%nlat/lake_parameters%nlat_coarse
      lon_scale_factor = lake_parameters%nlon/lake_parameters%nlon_coarse
      do j=1,lake_parameters%nlon_coarse
        do i=1,lake_parameters%nlat_coarse
          contains_lake = .false.
          do l = 1+(j-1)*lon_scale_factor,j*lon_scale_factor
            do k = 1+(i-1)*lat_scale_factor,i*lat_scale_factor
              if (lake_parameters%lake_centers(k,l)) contains_lake = .true.
            end do
          end do
          if(contains_lake) then
            number_of_cells_containing_lakes = number_of_cells_containing_lakes + 1
            cells_with_lakes_lat_temp(number_of_cells_containing_lakes) = i
            cells_with_lakes_lon_temp(number_of_cells_containing_lakes) = j
          end if
        end do
      end do
      allocate(this%cells_with_lakes_lat(number_of_cells_containing_lakes))
      allocate(this%cells_with_lakes_lon(number_of_cells_containing_lakes))
      do i = 1,number_of_cells_containing_lakes
        this%cells_with_lakes_lat(i) = &
          & cells_with_lakes_lat_temp(i)
        this%cells_with_lakes_lon(i) = &
          & cells_with_lakes_lon_temp(i)
      end do
      allocate(this%evaporation_from_lakes(count(lake_parameters%lake_centers)))
      this%evaporation_from_lakes(:) = 0.0
      allocate(this%evaporation_applied(count(lake_parameters%lake_centers)))
      this%evaporation_applied(:) = .false.
      this%set_forest => rooted_tree_forest()
      deallocate(cells_with_lakes_lat_temp)
      deallocate(cells_with_lakes_lon_temp)
end subroutine initialiselakefields

function lakefieldsconstructor(lake_parameters) result(constructor)
  type(lakeparameters), intent(inout) :: lake_parameters
  type(lakefields), pointer :: constructor
    allocate(constructor)
    call constructor%initialiselakefields(lake_parameters)
end function lakefieldsconstructor

subroutine lakefieldsdestructor(this)
  class(lakefields), intent(inout) :: this
    deallocate(this%flooded_lake_cells)
    deallocate(this%connected_lake_cells)
    deallocate(this%lake_numbers)
    deallocate(this%buried_lake_numbers)
    deallocate(this%water_to_lakes)
    deallocate(this%water_to_hd)
    deallocate(this%lake_water_from_ocean)
    deallocate(this%cells_with_lakes_lat)
    deallocate(this%cells_with_lakes_lon)
    deallocate(this%number_lake_cells)
    deallocate(this%evaporation_on_surface_grid)
    deallocate(this%new_effective_volume_per_cell_on_surface_grid)
    deallocate(this%effective_lake_height_on_surface_grid_from_lakes)
    deallocate(this%effective_lake_height_on_surface_grid_to_lakes)
    deallocate(this%effective_volume_per_cell_on_surface_grid)
    deallocate(this%evaporation_applied)
    deallocate(this%evaporation_from_lakes)
    call this%set_forest%rooted_tree_forest_destructor()
    deallocate(this%set_forest)
end subroutine lakefieldsdestructor

subroutine initialiselakeprognostics(this,lake_parameters_in,lake_fields_in)
  class(lakeprognostics), intent(inout) :: this
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), pointer, intent(inout) :: lake_fields_in
  type(lakepointer), pointer, dimension(:) :: lakes_temp
  type(lake), pointer :: lake_temp
  integer :: lake_number
  integer :: center_cell_coarse_lat, center_cell_coarse_lon
  integer :: i,j
  integer :: fine_cells_per_coarse_cell_lat, fine_cells_per_coarse_cell_lon
  integer :: lat_surface_model, lon_surface_model
    this%total_lake_volume = 0.0_dp
    allocate(lakes_temp(lake_parameters_in%nlat*lake_parameters_in%nlon))
    lake_number = 0
    do j=1,lake_parameters_in%nlon
      do i=1,lake_parameters_in%nlat
        if(lake_parameters_in%lake_centers(i,j)) then
          fine_cells_per_coarse_cell_lat = &
            lake_parameters_in%nlat/lake_parameters_in%nlat_coarse
          fine_cells_per_coarse_cell_lon = &
            lake_parameters_in%nlon/lake_parameters_in%nlon_coarse
          center_cell_coarse_lat = ceiling(real(i)/real(fine_cells_per_coarse_cell_lat));
          center_cell_coarse_lon = ceiling(real(j)/real(fine_cells_per_coarse_cell_lon));
          lake_number = lake_number + 1
          call lake_fields_in%set_forest%add_set(lake_number)
          lake_temp => lake(lake_parameters_in,lake_fields_in,&
                             i,j,i,j,center_cell_coarse_lat, &
                             center_cell_coarse_lon, &
                             lake_number,0.0_dp,0.0_dp,0.0_dp)
          lakes_temp(lake_number) = lakepointer(lake_temp)
          lake_fields_in%lake_numbers(i,j) = lake_number
        end if
      end do
    end do
    !Need a seperate loop as the dry run messes up the number_lake_cells field
    lake_fields_in%number_lake_cells(:,:) = 0
    do j=1,lake_parameters_in%nlon
      do i=1,lake_parameters_in%nlat
        if(lake_parameters_in%lake_centers(i,j)) then
          if (lake_parameters_in%flood_only(i,j)) then
            lat_surface_model = lake_parameters_in%corresponding_surface_cell_lat_index(i,j)
            lon_surface_model = lake_parameters_in%corresponding_surface_cell_lon_index(i,j)
            lake_fields_in%number_lake_cells(lat_surface_model,lon_surface_model) = &
              lake_fields_in%number_lake_cells(lat_surface_model,lon_surface_model) + 1
          end if
        end if
      end do
    end do
    allocate(this%lakes(lake_number))
    do i=1,lake_number
      this%lakes(i) = lakes_temp(i)
    end do
    lake_fields_in%other_lakes => this%lakes
    deallocate(lakes_temp)
end subroutine initialiselakeprognostics

function lakeprognosticsconstructor(lake_parameters_in,lake_fields_in) result(constructor)
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), pointer, intent(inout) :: lake_fields_in
  type(lakeprognostics), pointer :: constructor
    allocate(constructor)
    call constructor%initialiselakeprognostics(lake_parameters_in,lake_fields_in)
end function lakeprognosticsconstructor

subroutine lakeprognosticsdestructor(this)
  class(lakeprognostics), intent(inout) :: this
  integer :: i
    do i = 1,size(this%lakes)
      deallocate(this%lakes(i)%lake_pointer%filled_lake_cell_lats)
      deallocate(this%lakes(i)%lake_pointer%filled_lake_cell_lons)
      deallocate(this%lakes(i)%lake_pointer)
    end do
    deallocate(this%lakes)
end subroutine lakeprognosticsdestructor

subroutine initialisefillinglake(this,lake_parameters_in,lake_fields_in,center_cell_lat_in, &
                                 center_cell_lon_in,current_cell_to_fill_lat_in, &
                                 current_cell_to_fill_lon_in, &
                                 center_cell_coarse_lat_in, &
                                 center_cell_coarse_lon_in, &
                                 lake_number_in, &
                                 lake_volume_in, &
                                 secondary_lake_volume_in,&
                                 unprocessed_water_in)
  class(lake) :: this
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), pointer, intent(inout) :: lake_fields_in
  integer, intent(in) :: center_cell_lat_in
  integer, intent(in) :: center_cell_lon_in
  integer, intent(in) :: current_cell_to_fill_lat_in
  integer, intent(in) :: current_cell_to_fill_lon_in
  integer, intent(in) :: center_cell_coarse_lat_in
  integer, intent(in) :: center_cell_coarse_lon_in
  integer, intent(in) :: lake_number_in
  real(dp),    intent(in) :: lake_volume_in
  real(dp), intent(in)    :: secondary_lake_volume_in
  real(dp),    intent(in) :: unprocessed_water_in
    call this%initialiselake(center_cell_lat_in,center_cell_lon_in, &
                             current_cell_to_fill_lat_in, &
                             current_cell_to_fill_lon_in, &
                             center_cell_coarse_lat_in, &
                             center_cell_coarse_lon_in, &
                             lake_number_in, &
                             lake_volume_in, &
                             secondary_lake_volume_in,&
                             unprocessed_water_in, &
                             lake_parameters_in, &
                             lake_fields_in)
    this%lake_type = filling_lake_type
    call this%set_overflowing_lake_parameter_to_null_values()
    call this%set_subsumed_lake_parameter_to_null_values()
end subroutine initialisefillinglake

function lakeconstructor(lake_parameters_in,lake_fields_in,center_cell_lat_in, &
                         center_cell_lon_in,current_cell_to_fill_lat_in, &
                         current_cell_to_fill_lon_in,center_cell_coarse_lat_in, &
                         center_cell_coarse_lon_in, lake_number_in, &
                         lake_volume_in,secondary_lake_volume_in,&
                         unprocessed_water_in) result(constructor)
  type(lake), pointer :: constructor
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), pointer, intent(inout) :: lake_fields_in
  integer, intent(in) :: center_cell_lat_in
  integer, intent(in) :: center_cell_lon_in
  integer, intent(in) :: current_cell_to_fill_lat_in
  integer, intent(in) :: current_cell_to_fill_lon_in
  integer, intent(in) :: center_cell_coarse_lat_in
  integer, intent(in) :: center_cell_coarse_lon_in
  integer, intent(in) :: lake_number_in
  real(dp),    intent(in) :: lake_volume_in
  real(dp), intent(in)    :: secondary_lake_volume_in
  real(dp),    intent(in) :: unprocessed_water_in
  allocate(constructor)
  call constructor%initialisefillinglake(lake_parameters_in, &
                                         lake_fields_in, &
                                         center_cell_lat_in, &
                                         center_cell_lon_in, &
                                         current_cell_to_fill_lat_in, &
                                         current_cell_to_fill_lon_in, &
                                         center_cell_coarse_lat_in, &
                                         center_cell_coarse_lon_in, &
                                         lake_number_in, &
                                         lake_volume_in, &
                                         secondary_lake_volume_in,&
                                         unprocessed_water_in)
end function lakeconstructor

subroutine initialiseoverflowinglake(this,outflow_redirect_lat_in, &
                                     outflow_redirect_lon_in, &
                                     local_redirect_in, &
                                     lake_retention_coefficient_in, &
                                     lake_parameters_in,lake_fields_in, &
                                     center_cell_lat_in, &
                                     center_cell_lon_in,&
                                     current_cell_to_fill_lat_in, &
                                     current_cell_to_fill_lon_in, &
                                     center_cell_coarse_lat_in, &
                                     center_cell_coarse_lon_in, &
                                     next_merge_target_lat_in, &
                                     next_merge_target_lon_in, &
                                     lake_number_in,lake_volume_in, &
                                     secondary_lake_volume_in,&
                                     unprocessed_water_in)
  class(lake) :: this
  integer, intent(in) :: outflow_redirect_lat_in
  integer, intent(in) :: outflow_redirect_lon_in
  logical, intent(in) :: local_redirect_in
  real(dp), intent(in)    :: lake_retention_coefficient_in
  integer, intent(in) :: center_cell_lat_in
  integer, intent(in) :: center_cell_lon_in
  integer, intent(in) :: current_cell_to_fill_lat_in
  integer, intent(in) :: current_cell_to_fill_lon_in
  integer, intent(in) :: center_cell_coarse_lat_in
  integer, intent(in) :: center_cell_coarse_lon_in
  integer, intent(in) :: next_merge_target_lat_in
  integer, intent(in) :: next_merge_target_lon_in
  integer, intent(in) :: lake_number_in
  real(dp),    intent(in) :: lake_volume_in
  real(dp), intent(in)    :: secondary_lake_volume_in
  real(dp),    intent(in) :: unprocessed_water_in
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), pointer, intent(inout) :: lake_fields_in
    call this%initialiselake(center_cell_lat_in,center_cell_lon_in, &
                             current_cell_to_fill_lat_in, &
                             current_cell_to_fill_lon_in, &
                             center_cell_coarse_lat_in, &
                             center_cell_coarse_lon_in, &
                             lake_number_in,lake_volume_in, &
                             secondary_lake_volume_in,&
                             unprocessed_water_in, &
                             lake_parameters_in, &
                             lake_fields_in)
    this%lake_type = overflowing_lake_type
    this%outflow_redirect_lat = outflow_redirect_lat_in
    this%outflow_redirect_lon = outflow_redirect_lon_in
    this%next_merge_target_lat = next_merge_target_lat_in
    this%next_merge_target_lon = next_merge_target_lon_in
    this%local_redirect = local_redirect_in
    this%lake_retention_coefficient = lake_retention_coefficient_in
    this%excess_water = 0.0_dp
    call this%set_subsumed_lake_parameter_to_null_values()
end subroutine initialiseoverflowinglake

subroutine set_overflowing_lake_parameter_to_null_values(this)
  class(lake) :: this
    this%outflow_redirect_lat = 0
    this%outflow_redirect_lon = 0
    this%next_merge_target_lat = 0
    this%next_merge_target_lon = 0
    this%local_redirect = .false.
    this%lake_retention_coefficient = 0
    this%excess_water = 0.0_dp
end subroutine set_overflowing_lake_parameter_to_null_values

subroutine initialisesubsumedlake(this,lake_parameters_in,lake_fields_in, &
                                  primary_lake_number_in,center_cell_lat_in, &
                                  center_cell_lon_in,current_cell_to_fill_lat_in, &
                                  current_cell_to_fill_lon_in, &
                                  center_cell_coarse_lat_in, &
                                  center_cell_coarse_lon_in, &
                                  lake_number_in,&
                                  lake_volume_in,&
                                  secondary_lake_volume_in,&
                                  unprocessed_water_in)
  class(lake) :: this
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), pointer, intent(inout) :: lake_fields_in
  integer, intent(in) :: primary_lake_number_in
  integer, intent(in) :: center_cell_lat_in
  integer, intent(in) :: center_cell_lon_in
  integer, intent(in) :: current_cell_to_fill_lat_in
  integer, intent(in) :: current_cell_to_fill_lon_in
  integer, intent(in) :: center_cell_coarse_lat_in
  integer, intent(in) :: center_cell_coarse_lon_in
  integer, intent(in) :: lake_number_in
  real(dp),    intent(in) :: lake_volume_in
  real(dp), intent(in)    :: secondary_lake_volume_in
  real(dp),    intent(in) :: unprocessed_water_in
    call this%initialiselake(center_cell_lat_in,center_cell_lon_in, &
                             current_cell_to_fill_lat_in, &
                             current_cell_to_fill_lon_in, &
                             center_cell_coarse_lat_in, &
                             center_cell_coarse_lon_in, &
                             lake_number_in,lake_volume_in, &
                             secondary_lake_volume_in,&
                             unprocessed_water_in, &
                             lake_parameters_in,lake_fields_in)
    this%lake_type = subsumed_lake_type
    this%primary_lake_number = primary_lake_number_in
    call this%set_overflowing_lake_parameter_to_null_values()
end subroutine initialisesubsumedlake

subroutine set_subsumed_lake_parameter_to_null_values(this)
  class(lake) :: this
    this%primary_lake_number = 0
end subroutine set_subsumed_lake_parameter_to_null_values

subroutine setup_lakes(lake_parameters,lake_prognostics,lake_fields,initial_water_to_lake_centers)
  type(lakeparameters), intent(in) :: lake_parameters
  type(lakeprognostics), intent(inout) :: lake_prognostics
  type(lakefields), intent(inout) :: lake_fields
  real(dp), pointer, dimension(:,:), intent(in) :: initial_water_to_lake_centers
  type(lake), pointer :: working_lake
  real(dp) :: initial_water_to_lake_center
  integer :: lake_index
  integer :: i,j
#ifdef USE_LOGGING
    call setup_logger
#endif
    do j=1,lake_parameters%nlon
      do i=1,lake_parameters%nlat
        if(lake_parameters%lake_centers(i,j)) then
          initial_water_to_lake_center = initial_water_to_lake_centers(i,j)
          if(initial_water_to_lake_center > 0.0_dp) then
            lake_index = lake_fields%lake_numbers(i,j)
            working_lake => lake_prognostics%lakes(lake_index)%lake_pointer
            call working_lake%add_water(initial_water_to_lake_center)
          end if
        end if
      end do
    end do
end subroutine setup_lakes

subroutine run_lakes(lake_parameters,lake_prognostics,lake_fields)
  type(lakeparameters), intent(in) :: lake_parameters
  type(lakeprognostics), intent(inout) :: lake_prognostics
  type(lakefields), intent(inout) :: lake_fields
  integer :: lat,lon,fine_lat,fine_lon
  integer :: basin_center_lat,basin_center_lon
  integer :: lake_index,target_lake_index
  integer :: i,j,k
  integer :: num_basins_in_cell
  integer :: basin_number
  integer :: number_lake_cells,map_index
  real(dp) :: share_to_each_lake,inflow_minus_evaporation
  real(dp) :: evaporation_per_lake_cell,evaporation
  type(lake), pointer :: working_lake
#ifdef USE_LOGGING
    call increment_timestep_wrapper
    if (is_logging()) then
      call create_info_dump_wrapper(lake_fields%water_to_lakes,&
                                    calculate_lake_volumes(lake_parameters,&
                                                           lake_prognostics),&
                                    lake_parameters%nlat,&
                                    lake_parameters%nlon,&
                                    lake_parameters%nlat_coarse, &
                                    lake_parameters%nlon_coarse)
    end if
#endif
    lake_fields%water_to_hd(:,:) = 0.0_dp
    lake_fields%lake_water_from_ocean(:,:) = 0.0_dp
    lake_fields%evaporation_from_lakes(:) = 0.0_dp
    lake_fields%new_effective_volume_per_cell_on_surface_grid = &
      lake_fields%effective_lake_height_on_surface_grid_to_lakes * &
      lake_parameters%cell_areas_on_surface_model_grid
    lake_fields%evaporation_on_surface_grid = &
      lake_fields%effective_volume_per_cell_on_surface_grid - &
      lake_fields%new_effective_volume_per_cell_on_surface_grid
    lake_fields%effective_volume_per_cell_on_surface_grid(:,:) = 0.0_dp
    do j = 1,lake_parameters%nlon_surface_model
      do i = 1,lake_parameters%nlat_surface_model
        number_lake_cells = lake_fields%number_lake_cells(i,j)
        if ((lake_fields%evaporation_on_surface_grid(i,j) /= 0.0) .and. &
            (number_lake_cells > 0)) then
          map_index = lake_parameters%surface_cell_to_fine_cell_map_numbers(i,j)
          evaporation_per_lake_cell = lake_fields%evaporation_on_surface_grid(i,j)/number_lake_cells
          if (map_index /= 0) then
            do k = 1,size(lake_parameters%surface_cell_to_fine_cell_maps(map_index)%lat_coords)
              fine_lat = lake_parameters%surface_cell_to_fine_cell_maps(map_index)%lat_coords(k)
              fine_lon = lake_parameters%surface_cell_to_fine_cell_maps(map_index)%lon_coords(k)
              target_lake_index = lake_fields%lake_numbers(fine_lat,fine_lon)
              if (target_lake_index /= 0) then
                working_lake => lake_prognostics%lakes(target_lake_index)%lake_pointer
                if (lake_fields%flooded_lake_cells(fine_lat,fine_lon) .or. &
                   (((working_lake%current_cell_to_fill_lat == fine_lat) .and. &
                     (working_lake%current_cell_to_fill_lon == fine_lon))  .and. &
                    (lake_parameters%flood_only(fine_lat,fine_lon) .or. &
                     lake_fields%connected_lake_cells(fine_lat,fine_lon)))) then
                  lake_fields%evaporation_from_lakes(target_lake_index) = &
                    lake_fields%evaporation_from_lakes(target_lake_index) + evaporation_per_lake_cell
                end if
              end if
            end do
          end if
        end if
      end do
    end do
    lake_fields%evaporation_applied(:) = .false.
    do i = 1,size(lake_fields%cells_with_lakes_lat)
      lat = lake_fields%cells_with_lakes_lat(i)
      lon = lake_fields%cells_with_lakes_lon(i)
      if (lake_fields%water_to_lakes(lat,lon) > 0.0_dp) then
        basin_number = lake_parameters%basin_numbers(lat,lon)
        num_basins_in_cell = &
          size(lake_parameters%basins(basin_number)%lat_coords)
        share_to_each_lake = lake_fields%water_to_lakes(lat,lon)/&
                             real(num_basins_in_cell,dp)
        do j = 1,num_basins_in_cell
          basin_center_lat = lake_parameters%basins(basin_number)%lat_coords(j)
          basin_center_lon = lake_parameters%basins(basin_number)%lon_coords(j)
          lake_index = lake_fields%lake_numbers(basin_center_lat,basin_center_lon)
          working_lake => lake_prognostics%lakes(lake_index)%lake_pointer
          if (.not. lake_fields%evaporation_applied(lake_index)) then
            inflow_minus_evaporation = share_to_each_lake - &
                                       lake_fields%evaporation_from_lakes(lake_index)
            if (inflow_minus_evaporation >= 0.0) then
              call working_lake%add_water(inflow_minus_evaporation)
            else
              call working_lake%remove_water(-1.0_dp*inflow_minus_evaporation)
            end if
            lake_fields%evaporation_applied(lake_index) = .true.
          else
            call working_lake%add_water(share_to_each_lake)
          end if
        end do
      else if (lake_fields%water_to_lakes(lat,lon) < 0.0_dp) then
        basin_number = lake_parameters%basin_numbers(lat,lon)
        num_basins_in_cell = \
          size(lake_parameters%basins(basin_number)%lat_coords)
        share_to_each_lake = -1.0_dp*lake_fields%water_to_lakes(lat,lon)/&
                              real(num_basins_in_cell,dp)
        do j = 1,num_basins_in_cell
          basin_center_lat = lake_parameters%basins(basin_number)%lat_coords(j)
          basin_center_lon = lake_parameters%basins(basin_number)%lon_coords(j)
          lake_index = lake_fields%lake_numbers(basin_center_lat,basin_center_lon)
          working_lake => lake_prognostics%lakes(lake_index)%lake_pointer
          call working_lake%remove_water(share_to_each_lake)
        end do
      end if
    end do
    do i = 1,size(lake_prognostics%lakes)
      working_lake => lake_prognostics%lakes(i)%lake_pointer
      if (working_lake%unprocessed_water > 0.0_dp) then
          call working_lake%add_water(0.0_dp)
      end if
    end do
    do i = 1,size(lake_prognostics%lakes)
      working_lake => lake_prognostics%lakes(i)%lake_pointer
      if (.not. lake_fields%evaporation_applied(i)) then
        evaporation = lake_fields%evaporation_from_lakes(i)
        if (evaporation > 0.0_dp) then
          call working_lake%remove_water(evaporation)
        else if (evaporation < 0.0_dp) then
          call working_lake%add_water(-1.0*evaporation)
        end if
        lake_fields%evaporation_applied(i) = .true.
      end if
    end do
    do i = 1,size(lake_prognostics%lakes)
      working_lake => lake_prognostics%lakes(i)%lake_pointer
      if (working_lake%lake_volume < 0.0_dp) then
        call working_lake%release_negative_water()
      end if
    end do
    do i = 1,size(lake_prognostics%lakes)
      working_lake => lake_prognostics%lakes(i)%lake_pointer
      if (working_lake%lake_type == overflowing_lake_type) then
          call working_lake%drain_excess_water()
      end if
    end do
    do i = 1,size(lake_prognostics%lakes)
      working_lake => lake_prognostics%lakes(i)%lake_pointer
      call working_lake%calculate_effective_lake_volume_per_cell(.false.)
    end do
end subroutine run_lakes

subroutine calculate_true_lake_depths(lake_prognostics)
  type(lakeprognostics), intent(inout) :: lake_prognostics
  type(lake), pointer :: working_lake
  integer :: i
    do i = 1,size(lake_prognostics%lakes)
      working_lake => lake_prognostics%lakes(i)%lake_pointer
      if (working_lake%lake_type == overflowing_lake_type .or. &
          working_lake%lake_type == filling_lake_type) then
        call working_lake%calculate_true_lake_depth()
      end if
    end do
end subroutine calculate_true_lake_depths

recursive subroutine add_water(this,inflow)
  class(lake), target, intent(inout) :: this
  real(dp), intent(in) :: inflow
  type(lake),  pointer :: other_lake
  type(mergeandredirectindicescollection), pointer :: merge_indices_collection
  type(mergeandredirectindices), pointer :: merge_indices
  integer :: merge_type
  integer :: other_lake_number
  integer :: merge_indices_index
  logical :: filled
  integer :: i
  logical :: merge_possible
  logical :: already_merged
  logical :: is_flood_merge
  real(dp) :: inflow_local
#ifdef USE_LOGGING
    call log_process_wrapper(this%lake_number,this%center_cell_lat, &
                             this%center_cell_lon,this%lake_type,&
                             this%lake_volume,"add_water")
#endif
    if (this%lake_type == filling_lake_type) then
      inflow_local = inflow + this%unprocessed_water
      this%unprocessed_water = 0.0_dp
      do while (inflow_local > 0.0_dp)
        filled = this%fill_current_cell(inflow_local)
        if(filled) then
          merge_indices_index = this%get_merge_indices_index(is_flood_merge)
          if (merge_indices_index /= 0) then
            merge_indices_collection => &
              this%get_merge_indices_collection(merge_indices_index, &
                                                is_flood_merge)
            do i = 1,merge_indices_collection%primary_merge_and_redirect_indices_count + 1
              if (i <= &
                  merge_indices_collection%primary_merge_and_redirect_indices_count) then
                merge_indices => &
                  merge_indices_collection%primary_merge_and_redirect_indices(i)%ptr
              else
                if (merge_indices_collection%secondary_merge) then
                  merge_indices => &
                    merge_indices_collection%secondary_merge_and_redirect_indices
                else
                  exit
                end if
              end if
              merge_possible = &
                this%check_if_merge_is_possible(merge_indices,already_merged,merge_type)
              if (merge_possible) then
                if (merge_type == secondary_merge_mtype) then
                  call this%perform_secondary_merge(merge_indices)
                  call this%store_water(inflow_local)
                  return
                else
                  call this%perform_primary_merge(merge_indices)
                end if
              else if (.not. already_merged) then
                !Note becoming an overflowing lake occur not only when the other basin
                !is not yet full enough but also when the other other basin is overflowing
                !but is filling another basin at the same height (at a tri basin meeting point)
                call this%change_to_overflowing_lake(merge_indices)
                call this%store_water(inflow_local)
                return
              end if
            end do
          end if
          call this%update_filling_cell()
        end if
      end do
    else if (this%lake_type == overflowing_lake_type) then
      inflow_local = inflow + this%unprocessed_water
      this%unprocessed_water = 0.0_dp
      if (this%local_redirect) then
        other_lake_number  = this%lake_fields%lake_numbers(this%outflow_redirect_lat, &
                                                           this%outflow_redirect_lon)
        other_lake => &
          this%lake_fields%other_lakes(other_lake_number)%lake_pointer
        if (this%lake_parameters%instant_throughflow) then
          call other_lake%add_water(inflow_local)
        else
          call other_lake%store_water(inflow_local)
        end if
      else
        this%excess_water = this%excess_water + inflow_local
      end if
    else if (this%lake_type == subsumed_lake_type) then
      inflow_local = inflow + this%unprocessed_water
      this%unprocessed_water = 0.0_dp
      other_lake => this%lake_fields%other_lakes(this%primary_lake_number)%lake_pointer
      call other_lake%add_water(inflow_local)
    end if
end subroutine add_water

subroutine remove_water(this,outflow)
  class(lake), target, intent(inout) :: this
  class(lake), pointer :: other_lake
  type(mergeandredirectindicescollection), pointer :: merge_indices_collection
  type(mergeandredirectindices), pointer :: merge_indices
  real(dp) :: outflow
  real(dp) :: outflow_local
  real(dp) :: new_outflow
  logical :: drained
  logical :: is_flood_merge
  integer :: merge_type
  integer :: merge_indices_index
  integer :: i
#ifdef USE_LOGGING
    call log_process_wrapper(this%lake_number,this%center_cell_lat, &
                             this%center_cell_lon,this%lake_type,&
                             this%lake_volume,"remove_water")
#endif
    if (this%lake_type == filling_lake_type) then
      drained = .true.
      if (outflow <= this%unprocessed_water) then
        this%unprocessed_water = this%unprocessed_water - outflow
        return
      end if
      outflow_local = outflow - this%unprocessed_water
      this%unprocessed_water = 0.0_dp
      do while (outflow_local > 0.0_dp .or. drained)
        outflow_local = this%drain_current_cell(outflow_local,drained)
        if (drained) then
          call this%rollback_filling_cell()
          merge_indices_index = this%get_merge_indices_index(is_flood_merge)
          if (merge_indices_index /= 0) then
            merge_indices_collection => &
              this%get_merge_indices_collection(merge_indices_index, &
                                                is_flood_merge)
            do i = merge_indices_collection%primary_merge_and_redirect_indices_count+1,1,-1
              if (i <= merge_indices_collection%primary_merge_and_redirect_indices_count) then
                merge_indices => &
                  merge_indices_collection%primary_merge_and_redirect_indices(i)%ptr
              else
                if(merge_indices_collection%secondary_merge) then
                  merge_indices => &
                    merge_indices_collection%secondary_merge_and_redirect_indices
                else
                  cycle
                end if
              end if
              merge_type = merge_indices%get_merge_type()
              if (merge_type == primary_merge_mtype) then
                new_outflow = outflow_local/2.0_dp
                if (this%conditionally_rollback_primary_merge(merge_indices,new_outflow)) then
                  outflow_local = outflow_local - new_outflow
                end if
              else if (merge_type == secondary_merge_mtype) then
                write(*,*) "Merge logic failure"
                stop
              end if
            end do
          end if
        end if
      end do
    else if (this%lake_type == overflowing_lake_type) then
      if (outflow <= this%unprocessed_water) then
        this%unprocessed_water = this%unprocessed_water - outflow
        return
      end if
      outflow_local = outflow - this%unprocessed_water
      this%unprocessed_water = 0.0_dp
      if (outflow_local <= this%excess_water) then
        this%excess_water = this%excess_water - outflow_local
        return
      end if
      outflow_local = outflow_local - this%excess_water
      this%excess_water = 0.0_dp
      if (this%lake_fields%flooded_lake_cells(this%current_cell_to_fill_lat, &
                                              this%current_cell_to_fill_lon)) then
        this%lake_fields%flooded_lake_cells(this%current_cell_to_fill_lat, &
                                            this%current_cell_to_fill_lon) = .false.
        this%number_of_flooded_cells = &
          this%number_of_flooded_cells - 1
      else
        this%lake_fields%connected_lake_cells(this%current_cell_to_fill_lat, &
                                              this%current_cell_to_fill_lon) = .false.
      end if
      call this%change_overflowing_lake_to_filling_lake()

      merge_indices_index = this%get_merge_indices_index(is_flood_merge)
      if (merge_indices_index /= 0) then
        merge_indices_collection => &
          this%get_merge_indices_collection(merge_indices_index, &
                                            is_flood_merge)
        do i = merge_indices_collection%primary_merge_and_redirect_indices_count+1,1,-1
          if (i <= merge_indices_collection%primary_merge_and_redirect_indices_count) then
            merge_indices => &
              merge_indices_collection%primary_merge_and_redirect_indices(i)%ptr
          else
            if(merge_indices_collection%secondary_merge) then
              merge_indices => &
                merge_indices_collection%secondary_merge_and_redirect_indices
            else
              cycle
            end if
          end if
          merge_type = merge_indices%get_merge_type()
          if (merge_type == primary_merge_mtype) then
            new_outflow = outflow_local/2.0_dp
            if(this%conditionally_rollback_primary_merge(merge_indices, &
                                                         new_outflow)) then
              outflow_local = outflow_local - new_outflow
            end if
          else if (merge_type == secondary_merge_mtype) then
            if(merge_indices%merged) then
              write(*,*) "Merge logic failure"
              stop
            end if
          end if
        end do
      end if
      call this%remove_water(outflow_local)
    else if (this%lake_type == subsumed_lake_type) then
      if (outflow <= this%unprocessed_water) then
        this%unprocessed_water = this%unprocessed_water - outflow
        return
      end if
      outflow_local = outflow - this%unprocessed_water
      this%unprocessed_water = 0.0_dp
      other_lake => this%lake_fields%other_lakes(this%primary_lake_number)%lake_pointer
      call other_lake%remove_water(outflow_local)
    end if
end subroutine remove_water

subroutine store_water(this,inflow)
  class(lake), target, intent(inout) :: this
  real(dp), intent(in) :: inflow
    this%unprocessed_water = this%unprocessed_water + inflow
end subroutine store_water

subroutine release_negative_water(this)
  class(lake), target, intent(inout) :: this
    this%lake_fields%lake_water_from_ocean(this%center_cell_coarse_lat, &
                                           this%center_cell_coarse_lon) = &
      this%lake_fields%lake_water_from_ocean(this%center_cell_coarse_lat, &
                                             this%center_cell_coarse_lon) - &
      this%lake_volume
    this%lake_volume = 0.0_dp
end subroutine release_negative_water

subroutine accept_merge(this,redirect_coords_lat,redirect_coords_lon, &
                        primary_lake_number)
  class(lake), intent(inout) :: this
  integer, intent(in) :: redirect_coords_lat
  integer, intent(in) :: redirect_coords_lon
  integer :: lake_type
  integer :: primary_lake_number
  real(dp) :: excess_water
    lake_type = this%lake_type
    excess_water = this%excess_water
    if (.not. (this%lake_fields%connected_lake_cells(this%current_cell_to_fill_lat,&
                                                     this%current_cell_to_fill_lon) .or. &
               this%lake_parameters%flood_only(this%current_cell_to_fill_lat,&
                                          this%current_cell_to_fill_lon))) then
      this%lake_fields%connected_lake_cells(this%current_cell_to_fill_lat, &
                                            this%current_cell_to_fill_lon) = .true.
    else if (.not. this%lake_fields%flooded_lake_cells(this%current_cell_to_fill_lat,&
                                                       this%current_cell_to_fill_lon)) then
      this%lake_fields%flooded_lake_cells(this%current_cell_to_fill_lat,&
                                          this%current_cell_to_fill_lon) = .true.
      this%number_of_flooded_cells = &
        this%number_of_flooded_cells + 1
    end if
    if (.not. this%lake_fields%set_forest%make_new_link_from_labels(primary_lake_number, &
                                                                    this%lake_number)) then
      write(*,*) "Lake merging logic failure"
      stop
    end if
    call this%initialisesubsumedlake(this%lake_parameters, &
                                     this%lake_fields, &
                                     this%lake_fields%lake_numbers(redirect_coords_lat, &
                                                                   redirect_coords_lon), &
                                     this%center_cell_lat, &
                                     this%center_cell_lon, &
                                     this%current_cell_to_fill_lat, &
                                     this%current_cell_to_fill_lon, &
                                     this%center_cell_coarse_lat, &
                                     this%center_cell_coarse_lon, &
                                     this%lake_number,this%lake_volume,&
                                     this%secondary_lake_volume,&
                                     this%unprocessed_water)
    if (lake_type == overflowing_lake_type) then
        call this%store_water(excess_water)
    end if
end subroutine accept_merge

subroutine accept_split(this,primary_lake_number)
  class(lake), intent(inout) :: this
  integer :: primary_lake_number
    if (this%lake_fields%flooded_lake_cells(this%current_cell_to_fill_lat, &
                                            this%current_cell_to_fill_lon)) then
      this%lake_fields%flooded_lake_cells(this%current_cell_to_fill_lat, &
                                          this%current_cell_to_fill_lon) = .false.
      this%number_of_flooded_cells = &
        this%number_of_flooded_cells - 1
    else
      this%lake_fields%connected_lake_cells(this%current_cell_to_fill_lat, &
                                            this%current_cell_to_fill_lon) = .false.
    end if
    if ( .not. this%lake_fields%set_forest%split_set(primary_lake_number, &
                                                     this%lake_number)) then
      write(*,*) "Lake splitting logic failure"
      stop
    end if
    call this%initialisefillinglake(this%lake_parameters,this%lake_fields, &
                                    this%center_cell_lat, &
                                    this%center_cell_lon, &
                                    this%current_cell_to_fill_lat, &
                                    this%current_cell_to_fill_lon, &
                                    this%center_cell_coarse_lat, &
                                    this%center_cell_coarse_lon, &
                                    this%lake_number,this%lake_volume, &
                                    this%secondary_lake_volume,&
                                    this%unprocessed_water)
end subroutine

subroutine change_to_overflowing_lake(this,merge_indices)
  class(lake), intent(inout) :: this
  type(mergeandredirectindices), pointer :: merge_indices
  integer :: outflow_redirect_lat
  integer :: outflow_redirect_lon
  integer :: target_cell_lat,target_cell_lon
  logical :: local_redirect
    call merge_indices%get_outflow_redirect_coords(outflow_redirect_lat, &
                                                   outflow_redirect_lon, &
                                                   local_redirect)
    call merge_indices%get_merge_target_coords(target_cell_lat,target_cell_lon)
    if (.not. (this%lake_fields%connected_lake_cells(this%current_cell_to_fill_lat, &
                                                     this%current_cell_to_fill_lon) .or. &
               this%lake_parameters%flood_only(this%current_cell_to_fill_lat, &
                                               this%current_cell_to_fill_lon))) then
      this%lake_fields%connected_lake_cells(this%current_cell_to_fill_lat, &
                                            this%current_cell_to_fill_lon) = .true.
    else if (.not. this%lake_fields%flooded_lake_cells(this%current_cell_to_fill_lat, &
                                                       this%current_cell_to_fill_lon)) then
      this%lake_fields%flooded_lake_cells(this%current_cell_to_fill_lat, &
                                          this%current_cell_to_fill_lon) = .true.
      this%number_of_flooded_cells = this%number_of_flooded_cells + 1
    end if
    call this%initialiseoverflowinglake(outflow_redirect_lat, &
                                        outflow_redirect_lon, &
                                        local_redirect, &
                                        this%lake_parameters%lake_retention_coefficient, &
                                        this%lake_parameters,this%lake_fields, &
                                        this%center_cell_lat, &
                                        this%center_cell_lon, &
                                        this%current_cell_to_fill_lat, &
                                        this%current_cell_to_fill_lon, &
                                        this%center_cell_coarse_lat, &
                                        this%center_cell_coarse_lon, &
                                        target_cell_lat, &
                                        target_cell_lon, &
                                        this%lake_number, &
                                        this%lake_volume, &
                                        this%secondary_lake_volume,&
                                        this%unprocessed_water)
end subroutine change_to_overflowing_lake

subroutine drain_excess_water(this)
  class(lake), intent(inout) :: this
  real(dp) :: flow
    if (this%excess_water > 0.0_dp) then
      this%excess_water = this%excess_water + this%unprocessed_water
      this%unprocessed_water = 0.0_dp
      flow = (this%excess_water+ &
              this%lake_volume+ &
              this%secondary_lake_volume)/ &
             (this%lake_retention_coefficient + 1.0_dp)
      flow = min(flow,this%excess_water)
      this%lake_fields%water_to_hd(this%outflow_redirect_lat,this%outflow_redirect_lon) = &
           this%lake_fields%water_to_hd(this%outflow_redirect_lat, &
                                        this%outflow_redirect_lon)+flow
      this%excess_water = this%excess_water - flow
    end if
end subroutine drain_excess_water

recursive subroutine calculate_effective_lake_volume_per_cell(this,consider_secondary_lake, &
                                                              lake_cell_lats_in, &
                                                              lake_cell_lons_in, &
                                                              filled_lake_cell_index_in)
  class(lake), intent(inout) :: this
  logical, intent(in) :: consider_secondary_lake
  integer, pointer, dimension(:), optional :: lake_cell_lats_in
  integer, pointer, dimension(:), optional :: lake_cell_lons_in
  integer, optional :: filled_lake_cell_index_in
  integer, pointer, dimension(:) :: working_filled_lake_cell_lats
  integer, pointer, dimension(:) :: working_filled_lake_cell_lons
  integer :: working_filled_lake_cell_index
  real(dp) :: effective_volume_per_cell
  integer :: total_number_of_flooded_cells
  integer :: lat,lon,lat_surface_model,lon_surface_model
  integer :: i
    if (this%lake_type == subsumed_lake_type) then
      call calculate_effective_lake_volume_per_cell(&
            this%lake_fields%other_lakes(this%primary_lake_number)%lake_pointer, &
            .true.,this%filled_lake_cell_lats,this%filled_lake_cell_lons, &
            this%filled_lake_cell_index)
    else
      total_number_of_flooded_cells = this%number_of_flooded_cells + &
                                      this%secondary_number_of_flooded_cells
      if ((this%lake_type == filling_lake_type) .or. &
          (total_number_of_flooded_cells == 0)) then
        effective_volume_per_cell = &
          (this%lake_volume + this%secondary_lake_volume) / &
          (total_number_of_flooded_cells + 1)
      else
        effective_volume_per_cell = &
          (this%lake_volume + this%secondary_lake_volume) / &
           total_number_of_flooded_cells
      end if
      if (consider_secondary_lake) then
        working_filled_lake_cell_lats => lake_cell_lats_in
        working_filled_lake_cell_lons => lake_cell_lons_in
        working_filled_lake_cell_index = filled_lake_cell_index_in
      else
        working_filled_lake_cell_lats => this%filled_lake_cell_lats
        working_filled_lake_cell_lons => this%filled_lake_cell_lons
        working_filled_lake_cell_index = this%filled_lake_cell_index
        lat_surface_model = &
            this%lake_parameters%corresponding_surface_cell_lat_index(this%current_cell_to_fill_lat, &
                                                                      this%current_cell_to_fill_lon)
        lon_surface_model = &
            this%lake_parameters%corresponding_surface_cell_lon_index(this%current_cell_to_fill_lat, &
                                                                      this%current_cell_to_fill_lon)
        this%lake_fields%effective_volume_per_cell_on_surface_grid(lat_surface_model, &
                                                                   lon_surface_model) = &
          this%lake_fields%effective_volume_per_cell_on_surface_grid(lat_surface_model, &
                                                                     lon_surface_model) + &
          effective_volume_per_cell
      end if
      if (working_filled_lake_cell_index > 0) then
        do i = 1,working_filled_lake_cell_index
          lat = working_filled_lake_cell_lats(i)
          lon = working_filled_lake_cell_lons(i)
          lat_surface_model = &
            this%lake_parameters%corresponding_surface_cell_lat_index(lat, &
                                                                      lon)
          lon_surface_model = &
            this%lake_parameters%corresponding_surface_cell_lon_index(lat, &
                                                                      lon)
          this%lake_fields%effective_volume_per_cell_on_surface_grid(lat_surface_model, &
                                                                     lon_surface_model) = &
            this%lake_fields%effective_volume_per_cell_on_surface_grid(lat_surface_model, &
                                                                       lon_surface_model) + &
            effective_volume_per_cell
        end do
      end if
    end if
end subroutine calculate_effective_lake_volume_per_cell

subroutine calculate_true_lake_depth(this)
  class(lake), intent(inout) :: this
  class(lake), pointer :: secondary_lake
  integer, pointer, dimension(:) :: working_cell_lats
  integer, pointer, dimension(:) :: working_cell_lons
  integer, dimension(:), pointer :: label_list
  real(dp) :: current_cell_to_fill_height
  real(dp) :: sill_height
  real(dp) :: volume_above_current_filling_cell
  real(dp) :: highest_threshold_reached
  real(dp) :: total_lake_area
  real(dp) :: volume_above_sill
  real(dp) :: depth_above_sill
  real(dp) :: depth_above_current_filling_cell
  integer :: secondary_lake_number
  integer :: working_cell_list_length
  integer :: working_cell_index
  integer :: i
  integer :: lat,lon
    if (this%lake_type == filling_lake_type) then
      if (this%lake_fields%connected_lake_cells(this%current_cell_to_fill_lat, &
                                                 this%current_cell_to_fill_lon) .or. &
          this%lake_parameters%flood_only(this%current_cell_to_fill_lat, &
                                          this%current_cell_to_fill_lon)) then
        current_cell_to_fill_height = &
          this%lake_parameters%raw_heights(this%current_cell_to_fill_lat, &
                                           this%current_cell_to_fill_lon)
      else
        current_cell_to_fill_height = &
          this%lake_parameters%corrected_heights(this%current_cell_to_fill_lat, &
                                                 this%current_cell_to_fill_lon)
      end if
      if (this%previous_cell_to_fill_lat /= this%current_cell_to_fill_lat .or. &
          this%previous_cell_to_fill_lon /= this%current_cell_to_fill_lon) then
        if (this%lake_fields%flooded_lake_cells(this%previous_cell_to_fill_lat, &
                                                 this%previous_cell_to_fill_lon) .or. &
             this%lake_parameters%flood_only(this%previous_cell_to_fill_lat, &
                                             this%previous_cell_to_fill_lon)) then
          highest_threshold_reached = &
            this%lake_parameters%flood_volume_thresholds(this%previous_cell_to_fill_lat, &
                                                         this%previous_cell_to_fill_lon)
        else
          highest_threshold_reached = &
            this%lake_parameters%connection_volume_thresholds(this%previous_cell_to_fill_lat, &
                                                              this%previous_cell_to_fill_lon)
        end if
      else
        highest_threshold_reached = 0.0_dp
      end if
      volume_above_current_filling_cell = &
        this%lake_volume + &
        this%unprocessed_water - &
        highest_threshold_reached
      label_list => this%lake_fields%set_forest%get_all_node_labels_of_set(this%lake_number)
      working_cell_list_length = this%filled_lake_cell_index + 1
      if (size(label_list) > 0) then
        do secondary_lake_number = 1,size(label_list)
          if (.not. (secondary_lake_number == this%lake_number)) then
            working_cell_list_length = &
              working_cell_list_length + &
              this%lake_fields%other_lakes(secondary_lake_number)&
              &%lake_pointer%filled_lake_cell_index + 1
          end if
        end do
      end if
      allocate(working_cell_lats(working_cell_list_length))
      allocate(working_cell_lons(working_cell_list_length))
      working_cell_index = 0
      if (this%filled_lake_cell_index > 0) then
        do i = 1,this%filled_lake_cell_index
          if (this%filled_lake_cell_lats(i) /= this%current_cell_to_fill_lat .or. &
              this%filled_lake_cell_lons(i) /= this%current_cell_to_fill_lon) then
            working_cell_index = working_cell_index + 1
            working_cell_lats(working_cell_index) = this%filled_lake_cell_lats(i)
            working_cell_lons(working_cell_index) = this%filled_lake_cell_lons(i)
          end if
        end do
      end if
      working_cell_index = working_cell_index + 1
      working_cell_lats(working_cell_index) = this%current_cell_to_fill_lat
      working_cell_lons(working_cell_index) = this%current_cell_to_fill_lon
      if (size(label_list) > 0) then
        do secondary_lake_number = 1,size(label_list)
          if (.not. (secondary_lake_number == this%lake_number)) then
            secondary_lake => this%lake_fields%&
                              &other_lakes(secondary_lake_number)%lake_pointer
            do i = 1,secondary_lake%filled_lake_cell_index
              if (this%lake_fields%lake_numbers(secondary_lake%filled_lake_cell_lats(i), &
                                                secondary_lake%filled_lake_cell_lons(i)) == &
                  secondary_lake_number) then
                working_cell_index = working_cell_index + 1
                working_cell_lats(working_cell_index) = &
                  secondary_lake%filled_lake_cell_lats(i)
                working_cell_lons(working_cell_index) = &
                  secondary_lake%filled_lake_cell_lons(i)
              end if
            end do
            if (this%lake_fields%lake_numbers(secondary_lake%current_cell_to_fill_lat, &
                                              secondary_lake%current_cell_to_fill_lon) == &
                secondary_lake_number) then
              working_cell_index = working_cell_index + 1
              working_cell_lats(working_cell_index) = &
                secondary_lake%current_cell_to_fill_lat
              working_cell_lons(working_cell_index) = &
                secondary_lake%current_cell_to_fill_lon
            end if
          end if
        end do
      end if
      total_lake_area = 0.0_dp
      do i = 1,working_cell_index
        lat = working_cell_lats(i)
        lon = working_cell_lons(i)
        if (this%lake_fields%flooded_lake_cells(lat,lon) .or. &
            this%lake_parameters%flood_only(lat,lon) .or. &
            (lat == this%current_cell_to_fill_lat .and. &
             lon == this%current_cell_to_fill_lon .and. &
             this%lake_fields%connected_lake_cells(lat,lon))) then
          total_lake_area = total_lake_area + &
                            this%lake_parameters%cell_areas(lat,lon)
        end if
      end do
      depth_above_current_filling_cell = &
        volume_above_current_filling_cell / total_lake_area
      do i = 1,working_cell_index
        lat = working_cell_lats(i)
        lon = working_cell_lons(i)
        if (this%lake_fields%flooded_lake_cells(lat,lon) .or. &
           this%lake_parameters%flood_only(lat,lon) .or. &
           (lat == this%current_cell_to_fill_lat .and. &
            lon == this%current_cell_to_fill_lon .and. &
            this%lake_fields%connected_lake_cells(lat,lon))) then
          this%lake_fields%true_lake_depths(lat,lon) =  &
            depth_above_current_filling_cell + current_cell_to_fill_height - &
            this%lake_parameters%raw_heights(lat,lon)
        end if
      end do
    else if (this%lake_type == overflowing_lake_type) then
      if (this%lake_fields%flooded_lake_cells(this%current_cell_to_fill_lat, &
                                              this%current_cell_to_fill_lon) .or. &
          this%lake_parameters%flood_only(this%current_cell_to_fill_lat, &
                                          this%current_cell_to_fill_lon)) then
        sill_height = &
          this%lake_parameters%flood_heights(this%current_cell_to_fill_lat, &
                                             this%current_cell_to_fill_lon)
      else
        sill_height = &
          this%lake_parameters%connection_heights(this%current_cell_to_fill_lat, &
                                                  this%current_cell_to_fill_lon)
      end if
      volume_above_sill = this%excess_water + &
                            this%unprocessed_water
      label_list => this%lake_fields%set_forest%get_all_node_labels_of_set(this%lake_number)
      working_cell_list_length = this%filled_lake_cell_index + 1
      if (size(label_list) > 0) then
        do secondary_lake_number = 1,size(label_list)
          if (.not. (secondary_lake_number == this%lake_number)) then
            working_cell_list_length = &
              working_cell_list_length + &
              this%lake_fields%other_lakes(secondary_lake_number)%&
              &lake_pointer%filled_lake_cell_index + 1
          end if
        end do
      end if
      allocate(working_cell_lats(working_cell_list_length))
      allocate(working_cell_lons(working_cell_list_length))
      working_cell_index = 0
      if (this%filled_lake_cell_index > 0) then
        do i = 1,this%filled_lake_cell_index
          working_cell_index = working_cell_index + 1
          working_cell_lats(working_cell_index) = this%filled_lake_cell_lats(i)
          working_cell_lons(working_cell_index) = this%filled_lake_cell_lons(i)
        end do
      end if
      working_cell_index = working_cell_index + 1
      working_cell_lats(working_cell_index) = this%current_cell_to_fill_lat
      working_cell_lons(working_cell_index) = this%current_cell_to_fill_lon
      if (size(label_list) > 0) then
        do secondary_lake_number = 1,size(label_list)
          if (.not. (secondary_lake_number == this%lake_number)) then
            secondary_lake => this%lake_fields%&
                              &other_lakes(secondary_lake_number)%lake_pointer
            do i = 1,secondary_lake%filled_lake_cell_index
              working_cell_index = working_cell_index + 1
              working_cell_lats(working_cell_index) = &
                secondary_lake%filled_lake_cell_lats(i)
              working_cell_lons(working_cell_index) = &
                secondary_lake%filled_lake_cell_lons(i)
            end do
            working_cell_index = working_cell_index + 1
            working_cell_lats(working_cell_index) = &
              secondary_lake%current_cell_to_fill_lat
            working_cell_lons(working_cell_index) = &
              secondary_lake%current_cell_to_fill_lon
          end if
        end do
      end if
      if (this%lake_fields%flooded_lake_cells(this%current_cell_to_fill_lat, &
                                              this%current_cell_to_fill_lon) .or. &
          this%lake_parameters%flood_only(this%current_cell_to_fill_lat, &
                                          this%current_cell_to_fill_lon)) then
        total_lake_area = this%lake_parameters%cell_areas(this%current_cell_to_fill_lat, &
                                                          this%current_cell_to_fill_lon)
      else
        total_lake_area = 0.0_dp
      end if
      do i = 1,working_cell_index
        lat = working_cell_lats(i)
        lon = working_cell_lons(i)
        if (this%lake_fields%flooded_lake_cells(lat,lon) .or. &
            this%lake_parameters%flood_only(lat,lon)) then
          total_lake_area = total_lake_area + &
                            this%lake_parameters%cell_areas(lat,lon)
        end if
      end do
      depth_above_sill = &
        volume_above_sill / total_lake_area
      do i = 1,working_cell_index
        lat = working_cell_lats(i)
        lon = working_cell_lons(i)
        if (this%lake_fields%flooded_lake_cells(lat,lon) .or. &
            this%lake_parameters%flood_only(lat,lon)) then
          this%lake_fields%true_lake_depths(lat,lon) = &
            depth_above_sill + sill_height - &
            this%lake_parameters%raw_heights(lat,lon)
        end if
      end do
    end if
end subroutine calculate_true_lake_depth

function fill_current_cell(this,inflow) result(filled)
  class(lake), intent(inout) :: this
  real(dp), intent(inout) :: inflow
  logical :: filled
  real(dp) :: new_lake_volume
  real(dp) :: maximum_new_lake_volume
    new_lake_volume = inflow + this%lake_volume
    if (this%lake_fields%connected_lake_cells(this%current_cell_to_fill_lat, &
                                              this%current_cell_to_fill_lon) .or. &
        this%lake_parameters%flood_only(this%current_cell_to_fill_lat, &
                                        this%current_cell_to_fill_lon)) then
      maximum_new_lake_volume = &
        this%lake_parameters%flood_volume_thresholds(this%current_cell_to_fill_lat, &
                                                     this%current_cell_to_fill_lon)
    else
      maximum_new_lake_volume = &
        this%lake_parameters%connection_volume_thresholds(this%current_cell_to_fill_lat, &
                                                          this%current_cell_to_fill_lon)
    end if
    if (new_lake_volume <= maximum_new_lake_volume) then
      this%lake_volume = new_lake_volume
      inflow = 0.0_dp
      filled = .false.
    else
      inflow = new_lake_volume - maximum_new_lake_volume
      this%lake_volume = maximum_new_lake_volume
      filled = .true.
    end if
end function fill_current_cell

function drain_current_cell(this,outflow,drained) result(outflow_out)
  class(lake), intent(inout) :: this
  real(dp) :: outflow
  real(dp) :: outflow_out
  real(dp) :: new_lake_volume
  real(dp) :: minimum_new_lake_volume
  logical, intent(out) :: drained
    if (this%filled_lake_cell_index == 0) then
      this%lake_volume = this%lake_volume - outflow
      outflow_out = 0.0_dp
      drained = .false.
      return
    end if
    new_lake_volume = this%lake_volume - outflow
    if (this%lake_fields%flooded_lake_cells(this%previous_cell_to_fill_lat, &
                                            this%previous_cell_to_fill_lon)) then
      minimum_new_lake_volume = &
        this%lake_parameters%flood_volume_thresholds(this%previous_cell_to_fill_lat, &
                                                     this%previous_cell_to_fill_lon)
    else
      minimum_new_lake_volume = &
        this%lake_parameters%connection_volume_thresholds(this%previous_cell_to_fill_lat, &
                                                          this%previous_cell_to_fill_lon)
    end if

    if (new_lake_volume <= 0.0_dp .and. minimum_new_lake_volume <= 0.0_dp) then
      this%lake_volume = new_lake_volume
      drained = .true.
      outflow_out = 0.0_dp
    else if (new_lake_volume <= this%lake_parameters%minimum_lake_volume_threshold &
             .and. minimum_new_lake_volume <= 0.0_dp) then
      this%lake_volume = 0.0_dp
      drained = .true.
      outflow_out = 0.0_dp
      this%lake_fields%lake_water_from_ocean(this%center_cell_coarse_lat, &
                                           this%center_cell_coarse_lon) = &
        this%lake_fields%lake_water_from_ocean(this%center_cell_coarse_lat, &
                                             this%center_cell_coarse_lon) - &
        new_lake_volume
    else if (new_lake_volume >= minimum_new_lake_volume) then
      this%lake_volume = new_lake_volume
      drained = .false.
      outflow_out = 0.0_dp
    else
      this%lake_volume = minimum_new_lake_volume
      drained = .true.
      outflow_out = minimum_new_lake_volume - new_lake_volume
    end if
end function

function get_merge_type(this) result(merge_type)
  class(mergeandredirectindices) :: this
  integer :: merge_type
    if (this%is_primary_merge) then
      merge_type = primary_merge_mtype
    else
      merge_type = secondary_merge_mtype
    end if
end function get_merge_type

function get_merge_indices_index(this,is_flood_merge) result(merge_indices_index)
  class(lake), intent(in) :: this
  logical, intent(out) :: is_flood_merge
  integer :: merge_indices_index
    if (this%lake_fields%connected_lake_cells(this%current_cell_to_fill_lat, &
                                              this%current_cell_to_fill_lon) .or. &
        this%lake_parameters%flood_only(this%current_cell_to_fill_lat, &
                                        this%current_cell_to_fill_lon)) then
      is_flood_merge = .True.
      merge_indices_index = this%lake_parameters%&
        &flood_merge_and_redirect_indices_index(this%current_cell_to_fill_lat, &
                                                this%current_cell_to_fill_lon)
    else
      is_flood_merge = .False.
      merge_indices_index = this%lake_parameters%&
        &connect_merge_and_redirect_indices_index(this%current_cell_to_fill_lat, &
                                                  this%current_cell_to_fill_lon)
    end if
end function get_merge_indices_index

function get_merge_indices_collection(this,merge_indices_index, &
                                      is_flood_merge) result(collection)
  class(lake), intent(in) :: this
  type(mergeandredirectindicescollection), pointer :: collection
  integer :: merge_indices_index
  logical :: is_flood_merge
  if (is_flood_merge) then
    collection => this%lake_parameters% &
           flood_merge_and_redirect_indices_collections(merge_indices_index)%ptr
  else
    collection => this%lake_parameters% &
           connect_merge_and_redirect_indices_collections(merge_indices_index)%ptr
  end if
end function get_merge_indices_collection

subroutine get_merge_target_coords(this,target_cell_lat,target_cell_lon)
  class(mergeandredirectindices), intent(inout) :: this
  integer, intent(out) :: target_cell_lat
  integer, intent(out) :: target_cell_lon
    target_cell_lat = this%merge_target_lat_index
    target_cell_lon = this%merge_target_lon_index
end subroutine get_merge_target_coords

subroutine get_outflow_redirect_coords(this, &
                                       outflow_redirect_lat, &
                                       outflow_redirect_lon, &
                                       local_redirect)
  class(mergeandredirectindices), intent(inout) :: this
  integer, intent(out) :: outflow_redirect_lat
  integer, intent(out) :: outflow_redirect_lon
  logical, intent(out) :: local_redirect
    outflow_redirect_lat = this%redirect_lat_index
    outflow_redirect_lon = this%redirect_lon_index
    local_redirect = this%local_redirect
end subroutine get_outflow_redirect_coords

function check_if_merge_is_possible(this,merge_indices,already_merged, &
                                    merge_type) &
   result(merge_possible)
  class(lake), intent(in) :: this
  type(mergeandredirectindices), pointer, intent(inout) :: merge_indices
  integer, intent(out) :: merge_type
  logical, intent(out) :: already_merged
  logical :: merge_possible
  type(lake), pointer :: other_lake
  type(lake), pointer :: other_lake_target_lake
  integer :: target_cell_lat
  integer :: target_cell_lon
  integer :: other_lake_number
  integer :: other_lake_target_lake_number
    merge_type = merge_indices%get_merge_type()
    if (merge_indices%merged) then
      merge_possible = .false.
      already_merged = .true.
      return
    end if
    call merge_indices%get_merge_target_coords(target_cell_lat,target_cell_lon)
    other_lake_number = this%lake_fields%lake_numbers(target_cell_lat, &
                                                      target_cell_lon)
    if (other_lake_number == 0) then
      merge_possible = .false.
      already_merged = .false.
      return
    end if
    other_lake => this%lake_fields%other_lakes(other_lake_number)%lake_pointer
    other_lake => other_lake%find_true_primary_lake()
    if (other_lake%lake_number == this%lake_number) then
      merge_possible = .false.
      already_merged = .true.
      return
    else if (other_lake%lake_type == overflowing_lake_type) then
      other_lake_target_lake_number = &
        this%lake_fields%lake_numbers(other_lake%next_merge_target_lat, &
                                      other_lake%next_merge_target_lon)
      if (other_lake_target_lake_number == 0) then
        merge_possible = .false.
        already_merged = .false.
        return
      end if
      other_lake_target_lake => &
        this%lake_fields%other_lakes(other_lake_target_lake_number)%lake_pointer
      other_lake_target_lake => find_true_primary_lake(other_lake_target_lake)
      if (other_lake_target_lake%lake_number == this%lake_number) then
        merge_possible = .true.
        already_merged = .false.
        return
      else
         merge_possible = .false.
         already_merged = .false.
         return
      end if
    else
      merge_possible = .false.
      already_merged = .false.
    end if
end function check_if_merge_is_possible

subroutine perform_primary_merge(this,merge_indices)
  class(lake), intent(inout) :: this
  type(mergeandredirectindices), pointer, intent(inout) :: merge_indices
  type(lake), pointer :: other_lake
  integer :: target_cell_lat,target_cell_lon
  integer :: other_lake_number
    call merge_indices%get_merge_target_coords(target_cell_lat,target_cell_lon)
    merge_indices%merged = .true.
    other_lake_number = this%lake_fields%lake_numbers(target_cell_lat,target_cell_lon)
    other_lake => this%lake_fields%other_lakes(other_lake_number)%lake_pointer
    other_lake => other_lake%find_true_primary_lake()
    this%secondary_lake_volume = this%secondary_lake_volume + other_lake%lake_volume + &
      other_lake%secondary_lake_volume
    this%secondary_number_of_flooded_cells = this%secondary_number_of_flooded_cells + &
      other_lake%number_of_flooded_cells + &
      other_lake%secondary_number_of_flooded_cells
    call other_lake%accept_merge(this%center_cell_lat,this%center_cell_lon, &
                                 this%lake_number)
end subroutine perform_primary_merge

subroutine perform_secondary_merge(this,merge_indices)
  class(lake), intent(inout) :: this
  type(mergeandredirectindices), pointer, intent(inout) :: merge_indices
  type(lake), pointer :: other_lake
  integer :: target_cell_lat,target_cell_lon
  integer :: other_lake_number
    call merge_indices%get_merge_target_coords(target_cell_lat,target_cell_lon)
    if (.not. (this%lake_fields%connected_lake_cells(this%current_cell_to_fill_lat, &
                                                     this%current_cell_to_fill_lon) .or. &
               this%lake_parameters%flood_only(this%current_cell_to_fill_lat, &
                                               this%current_cell_to_fill_lon))) then
      this%lake_fields%connected_lake_cells(this%current_cell_to_fill_lat, &
                                            this%current_cell_to_fill_lon) = .true.
    else if (.not. (this%lake_fields%flooded_lake_cells(this%current_cell_to_fill_lat, &
                                                        this%current_cell_to_fill_lon))) then
      this%lake_fields%flooded_lake_cells(this%current_cell_to_fill_lat, &
                                          this%current_cell_to_fill_lon) = .true.
      this%number_of_flooded_cells = this%number_of_flooded_cells + 1
    end if
    other_lake_number = this%lake_fields%lake_numbers(target_cell_lat,target_cell_lon)
    other_lake => this%lake_fields%other_lakes(other_lake_number)%lake_pointer
    other_lake => other_lake%find_true_primary_lake()
    other_lake%secondary_lake_volume = other_lake%secondary_lake_volume + &
      this%lake_volume + this%secondary_lake_volume
    other_lake%secondary_number_of_flooded_cells = other_lake%secondary_number_of_flooded_cells + &
                                                   this%number_of_flooded_cells + &
                                                   this%secondary_number_of_flooded_cells
    if (.not. (this%lake_fields%flooded_lake_cells(other_lake%current_cell_to_fill_lat, &
                                                   other_lake%current_cell_to_fill_lon) .or. &
               this%lake_parameters%flood_only(other_lake%current_cell_to_fill_lat, &
                                               other_lake%current_cell_to_fill_lon))) then
      this%lake_fields%connected_lake_cells(other_lake%current_cell_to_fill_lat, &
                                            other_lake%current_cell_to_fill_lon) = .false.
    else
      this%lake_fields%flooded_lake_cells(other_lake%current_cell_to_fill_lat, &
                                          other_lake%current_cell_to_fill_lon) = .false.
      other_lake%number_of_flooded_cells = other_lake%number_of_flooded_cells - 1
    end if
    if (other_lake%lake_type == overflowing_lake_type) then
        call other_lake%change_overflowing_lake_to_filling_lake()
    end if
    call this%accept_merge(other_lake%center_cell_lat, &
                           other_lake%center_cell_lon, &
                           other_lake%lake_number)
end subroutine perform_secondary_merge

recursive function conditionally_rollback_primary_merge(this,merge_indices,&
                                                        other_lake_outflow) &
    result(rollback_performed)
  class(lake), intent(inout) :: this
  type(mergeandredirectindices), pointer, intent(inout) :: merge_indices
  type(mergeandredirectindicescollection), pointer :: merge_indices_collection
  logical :: rollback_performed
  class(lake), pointer :: other_lake
  class(lake), pointer :: other_lake_true_primary_lake
  integer :: other_lake_number
  real(dp) :: other_lake_outflow
  integer :: target_cell_lat, target_cell_lon
  logical :: is_flood_merge, dummy
  integer :: i
  integer :: merge_type
  integer :: merge_indices_index
    call merge_indices%get_merge_target_coords(target_cell_lat,target_cell_lon)
    other_lake_number = this%lake_fields%lake_numbers(target_cell_lat, &
                                                      target_cell_lon)
    other_lake => this%lake_fields%other_lakes(other_lake_number)%lake_pointer
    if (other_lake%lake_type /= subsumed_lake_type) then
      rollback_performed = .false.
      return
    end if
    other_lake => find_true_rolledback_primary_lake(other_lake)
    other_lake_true_primary_lake => find_true_primary_lake(other_lake)
    if (other_lake_true_primary_lake%lake_number /= this%lake_number) then
      rollback_performed = .false.
      return
    end if
    merge_indices%merged = .false.
    this%secondary_lake_volume = this%secondary_lake_volume - other_lake%lake_volume - &
      other_lake%secondary_lake_volume
    this%secondary_number_of_flooded_cells = &
      this%secondary_number_of_flooded_cells - &
      (other_lake%number_of_flooded_cells + &
       other_lake%secondary_number_of_flooded_cells)
    call other_lake%accept_split(this%lake_number)
    merge_indices_index = other_lake%get_merge_indices_index(is_flood_merge)
    if (merge_indices_index /= 0) then
      merge_indices_collection => &
            this%get_merge_indices_collection(merge_indices_index, &
                                              is_flood_merge)
      do i = merge_indices_collection%primary_merge_and_redirect_indices_count+1,1,-1
        if (i <= merge_indices_collection%primary_merge_and_redirect_indices_count) then
          merge_indices => &
            merge_indices_collection%primary_merge_and_redirect_indices(i)%ptr
        else
          if(merge_indices_collection%secondary_merge) then
            merge_indices => &
              merge_indices_collection%secondary_merge_and_redirect_indices
          else
            cycle
          end if
        end if
        merge_type = merge_indices%get_merge_type()
        if (merge_type == primary_merge_mtype) then
          dummy = conditionally_rollback_primary_merge(other_lake,merge_indices,0.0_dp)
        else if(merge_type == secondary_merge_mtype) then
          if (merge_indices%merged) then
              write(*,*) "Merge logic failure"
              stop
          end if
        end if
      end do
    end if
    call other_lake%remove_water(other_lake_outflow)
    rollback_performed = .true.
end function conditionally_rollback_primary_merge

subroutine change_overflowing_lake_to_filling_lake(this)
  class(lake), intent(inout) :: this
    this%unprocessed_water = this%unprocessed_water + this%excess_water
    call this%initialisefillinglake(this%lake_parameters,this%lake_fields, &
                                    this%center_cell_lat, &
                                    this%center_cell_lon, &
                                    this%current_cell_to_fill_lat, &
                                    this%current_cell_to_fill_lon,&
                                    this%center_cell_coarse_lat, &
                                    this%center_cell_coarse_lon, &
                                    this%lake_number,this%lake_volume,&
                                    this%secondary_lake_volume,&
                                    this%unprocessed_water)
end subroutine change_overflowing_lake_to_filling_lake

subroutine change_subsumed_lake_to_filling_lake(this)
  class(lake), intent(inout) :: this
    call this%initialisefillinglake(this%lake_parameters,this%lake_fields, &
                                    this%center_cell_lat, &
                                    this%center_cell_lon, &
                                    this%current_cell_to_fill_lat, &
                                    this%current_cell_to_fill_lon,&
                                    this%center_cell_coarse_lat, &
                                    this%center_cell_coarse_lon, &
                                    this%lake_number,this%lake_volume, &
                                    this%secondary_lake_volume,&
                                    this%unprocessed_water)
end subroutine change_subsumed_lake_to_filling_lake

subroutine update_filling_cell(this,dry_run)
  class(lake), intent(inout) :: this
  logical, optional :: dry_run
  integer :: coords_lat, coords_lon
  integer :: new_lat_surface_model, new_lon_surface_model
  integer :: old_lake_number
  logical :: dry_run_local
    if (present(dry_run)) then
      dry_run_local = dry_run
    else
      dry_run_local = .false.
    end if
    coords_lat = this%current_cell_to_fill_lat
    coords_lon = this%current_cell_to_fill_lon
    if (this%lake_fields%connected_lake_cells(coords_lat,coords_lon) .or. &
        this%lake_parameters%flood_only(coords_lat,coords_lon)) then
      this%current_cell_to_fill_lat = &
        this%lake_parameters%flood_next_cell_lat_index(coords_lat,coords_lon)
      this%current_cell_to_fill_lon = &
        this%lake_parameters%flood_next_cell_lon_index(coords_lat,coords_lon)
    else
      this%current_cell_to_fill_lat = &
        this%lake_parameters%connect_next_cell_lat_index(coords_lat,coords_lon)
      this%current_cell_to_fill_lon = &
        this%lake_parameters%connect_next_cell_lon_index(coords_lat,coords_lon)
    end if
    if (.not. (this%lake_fields%connected_lake_cells(coords_lat, &
                                                     coords_lon) .or. &
               this%lake_parameters%flood_only(coords_lat, &
                                               coords_lon))) then
      this%lake_fields%connected_lake_cells(coords_lat, &
                                            coords_lon) = .true.
    else if (.not. this%lake_fields%flooded_lake_cells(coords_lat, &
                                                       coords_lon)) then
      this%lake_fields%flooded_lake_cells(coords_lat, &
                                          coords_lon) = .true.
      this%number_of_flooded_cells = this%number_of_flooded_cells + 1
    end if
    if ((this%lake_fields%connected_lake_cells(this%current_cell_to_fill_lat, &
                                               this%current_cell_to_fill_lon) .or. &
         this%lake_parameters%flood_only(this%current_cell_to_fill_lat, &
                                         this%current_cell_to_fill_lon)) .and. &
        .not. this%lake_fields%flooded_lake_cells(this%current_cell_to_fill_lat, &
                                                  this%current_cell_to_fill_lon)) then
      new_lat_surface_model = &
            this%lake_parameters%corresponding_surface_cell_lat_index(this%current_cell_to_fill_lat, &
                                                                      this%current_cell_to_fill_lon)
      new_lon_surface_model = &
            this%lake_parameters%corresponding_surface_cell_lon_index(this%current_cell_to_fill_lat, &
                                                                      this%current_cell_to_fill_lon)
      this%lake_fields%number_lake_cells(new_lat_surface_model, &
                                         new_lon_surface_model) = &
          this%lake_fields%number_lake_cells(new_lat_surface_model, &
                                             new_lon_surface_model) + 1
    end if
    if ( .not. dry_run_local) then
      old_lake_number = this%lake_fields%lake_numbers(this%current_cell_to_fill_lat, &
                                                      this%current_cell_to_fill_lon)
      if (old_lake_number /= 0 .and. &
          old_lake_number /= this%lake_number) then
        this%lake_fields%buried_lake_numbers(this%current_cell_to_fill_lat, &
                                             this%current_cell_to_fill_lon) = old_lake_number
      end if
      this%lake_fields%lake_numbers(this%current_cell_to_fill_lat, &
                                    this%current_cell_to_fill_lon) = this%lake_number
      this%previous_cell_to_fill_lat = coords_lat
      this%previous_cell_to_fill_lon = coords_lon
      this%filled_lake_cell_index = this%filled_lake_cell_index + 1
      this%filled_lake_cell_lats(this%filled_lake_cell_index) = coords_lat
      this%filled_lake_cell_lons(this%filled_lake_cell_index) = coords_lon
    end if
end subroutine update_filling_cell

subroutine rollback_filling_cell(this)
  class(lake), intent(inout) :: this
  integer :: lat_surface_model, lon_surface_model
    if (this%lake_parameters%flood_only(this%current_cell_to_fill_lat, &
                                        this%current_cell_to_fill_lon) .or. &
        (.not. this%lake_fields%connected_lake_cells(this%current_cell_to_fill_lat, &
                                                     this%current_cell_to_fill_lon))) then
      this%lake_fields%lake_numbers(this%current_cell_to_fill_lat, &
                                    this%current_cell_to_fill_lon) = 0
    else if (this%lake_fields%buried_lake_numbers(this%current_cell_to_fill_lat, &
                                                  this%current_cell_to_fill_lon) /= 0) then
      this%lake_fields%lake_numbers(this%current_cell_to_fill_lat, &
                                    this%current_cell_to_fill_lon) = &
        this%lake_fields%buried_lake_numbers(this%current_cell_to_fill_lat, &
                                             this%current_cell_to_fill_lon)
      this%lake_fields%buried_lake_numbers(this%current_cell_to_fill_lat, &
                                           this%current_cell_to_fill_lon) = 0
    end if
    if (.not. (this%lake_fields%flooded_lake_cells(this%previous_cell_to_fill_lat, &
                                                   this%previous_cell_to_fill_lon) .or. &
               this%lake_parameters%flood_only(this%previous_cell_to_fill_lat, &
                                               this%previous_cell_to_fill_lon))) then
      this%lake_fields%connected_lake_cells(this%previous_cell_to_fill_lat, &
                                            this%previous_cell_to_fill_lon) = .false.
    else
      this%lake_fields%flooded_lake_cells(this%previous_cell_to_fill_lat, &
                                          this%previous_cell_to_fill_lon) = .false.
      this%number_of_flooded_cells = this%number_of_flooded_cells - 1
    end if
    if (this%lake_fields%connected_lake_cells(this%current_cell_to_fill_lat, &
                                              this%current_cell_to_fill_lon) .or. &
        this%lake_parameters%flood_only(this%current_cell_to_fill_lat, &
                                        this%current_cell_to_fill_lon)) then
      lat_surface_model = &
          this%lake_parameters%corresponding_surface_cell_lat_index(this%current_cell_to_fill_lat, &
                                                                    this%current_cell_to_fill_lon)
      lon_surface_model = &
          this%lake_parameters%corresponding_surface_cell_lon_index(this%current_cell_to_fill_lat, &
                                                                    this%current_cell_to_fill_lon)
      this%lake_fields%number_lake_cells(lat_surface_model,&
                                         lon_surface_model) = &
           this%lake_fields%number_lake_cells(lat_surface_model, &
                                              lon_surface_model) - 1
    end if
    this%current_cell_to_fill_lat = this%previous_cell_to_fill_lat
    this%current_cell_to_fill_lon = this%previous_cell_to_fill_lon
    this%filled_lake_cell_lats(this%filled_lake_cell_index) = 0
    this%filled_lake_cell_lons(this%filled_lake_cell_index) = 0
    this%filled_lake_cell_index = this%filled_lake_cell_index - 1
    if (this%filled_lake_cell_index /= 0 ) then
      this%previous_cell_to_fill_lat = this%filled_lake_cell_lats(this%filled_lake_cell_index)
      this%previous_cell_to_fill_lon = this%filled_lake_cell_lons(this%filled_lake_cell_index)
    end if
end subroutine

recursive function find_true_rolledback_primary_lake(this) result(true_primary_lake)
  class(lake), target, intent(in) :: this
  type(lake), pointer :: true_primary_lake
  type(lake), pointer :: next_lake
    next_lake => this%lake_fields%other_lakes(this%primary_lake_number)%lake_pointer
    if (next_lake%lake_type == subsumed_lake_type) then
      true_primary_lake => next_lake%find_true_rolledback_primary_lake()
    else
      true_primary_lake => this
    end if
end function

recursive function find_true_primary_lake(this) result(true_primary_lake)
  class(lake), target, intent(in) :: this
  type(lake), pointer :: true_primary_lake
  integer :: true_primary_lake_number
    if (this%lake_type == subsumed_lake_type) then
        true_primary_lake_number = &
          this%lake_fields%set_forest%find_root_from_label(this%primary_lake_number)
        true_primary_lake => &
          this%lake_fields%other_lakes(true_primary_lake_number)%lake_pointer
    else
        true_primary_lake => this
    end if
end function find_true_primary_lake

function calculate_lake_volumes(lake_parameters,&
                                lake_prognostics) result(lake_volumes)
  type(lakeparameters) :: lake_parameters
  type(lakeprognostics) :: lake_prognostics
  real(dp), dimension(:,:), pointer :: lake_volumes
  type(lake), pointer :: working_lake
  real(dp) :: lake_overflow
  integer :: i
      allocate(lake_volumes(lake_parameters%nlat,&
                            lake_parameters%nlon))
      lake_volumes(:,:) = 0.0_dp
      do i = 1,size(lake_prognostics%lakes)
        working_lake => lake_prognostics%lakes(i)%lake_pointer
        if(working_lake%lake_type == overflowing_lake_type) then
          lake_overflow = working_lake%excess_water
        else
          lake_overflow = 0.0_dp
        end if
        lake_volumes(working_lake%center_cell_lat,&
                     working_lake%center_cell_lon) = working_lake%lake_volume + &
                                                     working_lake%unprocessed_water + &
                                                     lake_overflow
      end do
end function calculate_lake_volumes

function calculate_diagnostic_lake_volumes(lake_parameters,&
                                           lake_prognostics,&
                                           lake_fields) result(diagnostic_lake_volumes)
  type(lakeparameters), pointer, intent(in) :: lake_parameters
  type(lakeprognostics), pointer, intent(in) :: lake_prognostics
  type(lakefields), pointer, intent(in) :: lake_fields
  real(dp), dimension(:,:), pointer :: diagnostic_lake_volumes
  real(dp), dimension(:), allocatable :: lake_volumes_by_lake_number
  type(lake), pointer :: working_lake
  integer :: i,j
  integer :: original_lake_index,lake_number
    allocate(diagnostic_lake_volumes(lake_parameters%nlat,&
                                     lake_parameters%nlon))
    diagnostic_lake_volumes(:,:) = 0.0_dp
    allocate(lake_volumes_by_lake_number(size(lake_prognostics%lakes)))
    do i = 1,size(lake_prognostics%lakes)
      working_lake => lake_prognostics%lakes(i)%lake_pointer
      original_lake_index = working_lake%lake_number
      if (working_lake%lake_type == subsumed_lake_type) then
        working_lake => working_lake%find_true_primary_lake()
      end if
      lake_volumes_by_lake_number(original_lake_index) = &
        working_lake%lake_volume + &
        working_lake%secondary_lake_volume
    end do
    do j=1,lake_parameters%nlon
      do i=1,lake_parameters%nlat
        lake_number = lake_fields%lake_numbers(i,j)
        if (lake_number > 0) then
          diagnostic_lake_volumes(i,j) = lake_volumes_by_lake_number(lake_number)
        end if
      end do
    end do
    deallocate(lake_volumes_by_lake_number)
end function calculate_diagnostic_lake_volumes

function get_lake_volume_list(lake_prognostics) result(lake_volumes)
  type(lakeprognostics) :: lake_prognostics
  real(dp), dimension(:), pointer :: lake_volumes
  type(lake), pointer :: working_lake
  real(dp) :: lake_overflow
  integer :: i
      allocate(lake_volumes(size(lake_prognostics%lakes)))
      lake_volumes(:) = 0.0_dp
      do i = 1,size(lake_prognostics%lakes)
        working_lake => lake_prognostics%lakes(i)%lake_pointer
        if(working_lake%lake_type == overflowing_lake_type) then
          lake_overflow = working_lake%excess_water
        else
          lake_overflow = 0.0_dp
        end if
        lake_volumes(i) = working_lake%lake_volume + &
                          working_lake%unprocessed_water + &
                          lake_overflow
      end do
end function get_lake_volume_list

subroutine check_water_budget(lake_prognostics,lake_fields,initial_water_to_lakes)
  type(lakeprognostics), intent(inout) :: lake_prognostics
  type(lakefields), intent(inout) :: lake_fields
  real(dp), optional :: initial_water_to_lakes
  type(lake), pointer :: working_lake
  real(dp) :: new_total_lake_volume
  real(dp) :: change_in_total_lake_volume,total_inflow_minus_outflow,difference
  real(dp) :: total_water_to_lakes
  real(dp) :: tolerance
  integer :: i
  character*64 :: number_as_string
    new_total_lake_volume = 0.0_dp
    do i = 1,size(lake_prognostics%lakes)
      working_lake => lake_prognostics%lakes(i)%lake_pointer
      new_total_lake_volume = new_total_lake_volume + &
                              working_lake%unprocessed_water + &
                              working_lake%lake_volume
      if (working_lake%lake_type == overflowing_lake_type) then
        new_total_lake_volume = new_total_lake_volume + &
                                working_lake%excess_water
      end if
    end do
    change_in_total_lake_volume = new_total_lake_volume - &
                                  lake_prognostics%total_lake_volume
    total_water_to_lakes = sum(lake_fields%water_to_lakes)
    total_inflow_minus_outflow =  total_water_to_lakes + &
                                 sum(lake_fields%lake_water_from_ocean) - &
                                 sum(lake_fields%water_to_hd) - &
                                 sum(lake_fields%evaporation_from_lakes)
    if(present(initial_water_to_lakes)) then
      total_inflow_minus_outflow = total_inflow_minus_outflow + initial_water_to_lakes
    end if
    difference = change_in_total_lake_volume - total_inflow_minus_outflow
    tolerance = 5.0e-14_dp*max(abs(total_water_to_lakes),abs(new_total_lake_volume))
    if (abs(difference) > tolerance) then
      write(*,*) "*** Lake Water Budget ***"
      write(number_as_string,'(E21.14)') total_water_to_lakes
      write(*,*) "Total water to lakes" // trim(number_as_string)
      write(number_as_string,'(E21.14)') sum(lake_fields%water_to_hd)
      write(*,*) "Total water to hd" // trim(number_as_string)
      write(number_as_string,'(E21.14)') sum(lake_fields%lake_water_from_ocean)
      write(*,*) "Total lake water from ocean" // trim(number_as_string)
      write(number_as_string,'(E21.14)') lake_prognostics%total_lake_volume
      write(*,*) "Previous total lake volume: " // trim(number_as_string)
      write(number_as_string,'(E21.14)') new_total_lake_volume
      write(*,*) "Total lake volume: " // trim(number_as_string)
      write(number_as_string,'(E21.14)') total_inflow_minus_outflow
      write(*,*) "Total inflow - outflow: " // trim(number_as_string)
      write(number_as_string,'(E21.14)') change_in_total_lake_volume
      write(*,*) "Change in lake volume: " // trim(number_as_string)
      write(number_as_string,'(E21.14)') difference
      write(*,*) "Difference: " // trim(number_as_string)
    end if
    lake_prognostics%total_lake_volume = new_total_lake_volume
end subroutine check_water_budget

subroutine run_lake_number_retrieval(lake_prognostics)
  type(lakeprognostics), intent(inout) :: lake_prognostics
  type(lake), pointer :: working_lake
  integer :: i
    do i = 1,size(lake_prognostics%lakes)
      working_lake => lake_prognostics%lakes(i)%lake_pointer
      call working_lake%generate_lake_numbers()
    end do
end subroutine run_lake_number_retrieval

subroutine calculate_lake_fraction_on_surface_grid(lake_parameters,lake_fields, &
                                                   lake_fraction_on_surface_grid)
  real(dp), dimension(:,:), intent(inout) :: lake_fraction_on_surface_grid
  type(lakefields), pointer, intent(in) :: lake_fields
  type(lakeparameters), pointer, intent(in) :: lake_parameters
    lake_fraction_on_surface_grid(:,:) = &
      real(lake_fields%number_lake_cells(:,:))&
      /real(lake_parameters%number_fine_grid_cells(:,:))
end subroutine calculate_lake_fraction_on_surface_grid

subroutine calculate_effective_lake_height_on_surface_grid(lake_parameters, &
                                                           lake_fields)
  type(lakefields), pointer, intent(inout) :: lake_fields
  type(lakeparameters), pointer, intent(in) :: lake_parameters
    lake_fields%effective_lake_height_on_surface_grid_from_lakes(:,:) = &
      lake_fields%effective_volume_per_cell_on_surface_grid(:,:) / &
      lake_parameters%cell_areas_on_surface_model_grid(:,:)
end subroutine calculate_effective_lake_height_on_surface_grid

subroutine set_lake_evaporation_for_testing(lake_parameters, &
                                            lake_fields, &
                                            lake_evaporation)
  type(lakeparameters), pointer, intent(in) :: lake_parameters
  type(lakefields), pointer, intent(inout) :: lake_fields
  real(dp), allocatable, dimension(:,:), intent(in) :: lake_evaporation
    call calculate_effective_lake_height_on_surface_grid(lake_parameters,lake_fields)
    lake_fields%effective_lake_height_on_surface_grid_to_lakes(:,:) = &
      lake_fields%effective_lake_height_on_surface_grid_from_lakes(:,:) - &
      (lake_evaporation(:,:) / &
       lake_parameters%cell_areas_on_surface_model_grid(:,:))
end subroutine set_lake_evaporation_for_testing

subroutine set_lake_evaporation(lake_parameters, &
                                lake_fields, &
                                height_of_water_evaporated)
  type(lakeparameters), pointer, intent(in) :: lake_parameters
  type(lakefields), pointer, intent(inout) :: lake_fields
  real(dp), allocatable, dimension(:,:), intent(in) :: height_of_water_evaporated
  real(dp) :: working_effective_lake_height_on_surface_grid
  integer :: i,j
    call calculate_effective_lake_height_on_surface_grid(lake_parameters,lake_fields)
    lake_fields%effective_lake_height_on_surface_grid_to_lakes(:,:) = &
      lake_fields%effective_lake_height_on_surface_grid_from_lakes
    do j = 1,lake_parameters%nlon_surface_model
      do i = 1,lake_parameters%nlat_surface_model
        if (lake_fields%effective_lake_height_on_surface_grid_from_lakes(i,j) > 0.0) then
          working_effective_lake_height_on_surface_grid = &
            lake_fields%effective_lake_height_on_surface_grid_from_lakes(i,j) - &
            height_of_water_evaporated(i,j)
          if (working_effective_lake_height_on_surface_grid > 0.0) then
            lake_fields%effective_lake_height_on_surface_grid_to_lakes(i,j) = &
              working_effective_lake_height_on_surface_grid
          else
            lake_fields%effective_lake_height_on_surface_grid_to_lakes(i,j) = 0.0
          end if
        end if
      end do
    end do
end subroutine set_lake_evaporation

end module latlon_lake_model_mod
