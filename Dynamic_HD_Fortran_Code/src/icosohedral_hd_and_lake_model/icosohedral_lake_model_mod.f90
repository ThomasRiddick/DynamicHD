module icosohedral_lake_model_mod

! This module contains the main routines needed to run the lake model
! for a single timestep along with the objects used to store variables and
! the function used to initialise the lake model. The correct connection
! to the hd model is handled by the icosohedral_lake_model_interface_mod module
! and IO details are handled by the icosohedral_lake_model_mod module

! The lake model comprises of an array of individual lake sub-basins
! (and single basin lakes). Each lake has a current volume and a pre
! calculated list of cells is filled at that volume. The current cell
! in the process of being innundated is kept track of. Once it is
! innudated the next cell to target is looked up. Function are given
! to add or remove water from lakes. If two subbasins meet one becomes
! a subsumed lake (diverting its water to the other lake) and the other
! continues to fill. When a lake is full and begins to overflow water is
! returned to the HD system. If evaporation lowers the heigh beneath the
! rim of the outlet the lake becomes a filling lake again and no water goes
! to the HD system anymore till it overflows again. Also any sub-basin merges
! are reversed if necessary. At each timestep the cells containing of lakes is
! iterated over adding/removing any water as necessary and then any excess water
! is allowed to drain (at a rate controlled by the lake retention coefficient) back
! to the HD system (or to a very close by downstream lake)

! Can be used on the same grid as the HD model or a finer grid as desired.

! Can be used with the single height system - in this case just the flood
! heights and flood_only equal to true everywhere or seperate connect and
! flood heights

! Merge points represent both the points where lake sub-basins can merge and
! the points where lakes overflow

! A note on lake types

! A filling lake is one where the lake doesn't yet completely fill
! the depression containing it - adding water will increase the lake
! level

! A overflowing lake is one where the depression containing it is full
! and water is spilling out of the lowest possible outlet back into
! HD system (or to another downstream lake)

! A subsumed lake is a lake sub-basin that has merged with another lake
! where the other lake has been designated as primary and thus this lake
! now redirect water to that primary lake

implicit none

!Parameter for double precision floating point numbers
integer,parameter :: dp = selected_real_kind(12)

! Lake Type Labels
integer, parameter :: filling_lake_type = 1
integer, parameter :: overflowing_lake_type = 2
integer, parameter :: subsumed_lake_type = 3

! Simple Merge Type Labels
integer, parameter :: no_merge = 0
integer, parameter :: primary_merge   = 1
integer, parameter :: secondary_merge = 2
integer, parameter :: double_merge = 3
integer, parameter :: null_mtype = 4
! Conversion of complex merge types to simple merge types for connect cell
integer, parameter, dimension(17) :: convert_to_simple_merge_type_connect = (/no_merge,primary_merge,primary_merge, &
                                                                              primary_merge,primary_merge,secondary_merge, &
                                                                              secondary_merge,secondary_merge,secondary_merge, &
                                                                              no_merge,no_merge,no_merge,double_merge, &
                                                                              double_merge,double_merge,double_merge, &
                                                                              null_mtype/)
! Conversion of complex merge types to simple merge types for flood cell
integer, parameter, dimension(17) :: convert_to_simple_merge_type_flood = (/no_merge,primary_merge,secondary_merge, &
                                                                            no_merge,double_merge,primary_merge, &
                                                                            secondary_merge,no_merge,double_merge, &
                                                                            primary_merge,secondary_merge,double_merge, &
                                                                            primary_merge,secondary_merge,no_merge, &
                                                                            double_merge,null_mtype/)

type :: basinlist
  integer, pointer, dimension(:) :: basin_indices
end type basinlist

interface basinlist
  procedure :: basinlistconstructor
end interface basinlist

! Defines an object to store set of precalculated parameters for lakes
type :: lakeparameters
  logical :: instant_throughflow ! Flag to determine if sublakes recieving water from another sublake should
                                 ! recalculate immediately
  real(dp) :: lake_retention_coefficient ! Parameter determining the rate of flow from overflowing lakes
  ! Surface Spatial Variables
  logical, pointer, dimension(:) :: lake_centers ! Center points (deepest points) of lakes
  real(dp), pointer, dimension(:) :: connection_volume_thresholds ! Thresholds to connect to next cell
  real(dp), pointer, dimension(:) :: flood_volume_thresholds ! Threshold to flood next cell
  logical, pointer, dimension(:) :: flood_only ! Cells where the connection and flood height are the same
                                               ! thus the connection step can be skipped
  logical, pointer, dimension(:) :: flood_local_redirect ! If this cell is a flood outlet does water flood
                                                         ! to another nearby lake (local) or back the
                                                         ! HD model (non-local)
  logical, pointer, dimension(:) :: connect_local_redirect ! If this cell is connection outlet does water flood
                                                         ! to another nearby lake (local) or back the
                                                         ! HD model (non-local)
  logical, pointer, dimension(:) :: additional_flood_local_redirect ! Replicate variable for case of a double
                                                                    ! merge
  logical, pointer, dimension(:) :: additional_connect_local_redirect ! Replicate variable for case of a double
                                                                      ! merge
  integer, pointer, dimension(:) :: merge_points  ! Type of merge occuring at a point (or for most points
                                                  ! the no merge type)
  integer, pointer, dimension(:) :: basin_numbers ! Variable on the coarse grid:
                                                  ! Index of list of basins within a given coarse cell
  ! List Variable
  type(basinlist), pointer, dimension(:) :: basins ! A list of lakes/basins on the fine grid within a given coarse grid
                                                   ! cell - indexed by the basin number
  ! (Grid Specific) Surface Spatial Variables
  integer, pointer, dimension(:) :: flood_next_cell_index ! The next cell to flood
  integer, pointer, dimension(:) :: connect_next_cell_index ! The next cell to connect
  integer, pointer, dimension(:) :: flood_force_merge_index ! The location of sub-basin to force to merge
                                                            ! with this lake when this cell floods
  integer, pointer, dimension(:) :: connect_force_merge_index ! The location of sub-basin to force to merge
                                                              ! with this lake when this cell connects
  integer, pointer, dimension(:) :: flood_redirect_index ! Location of lake/HD model cell water spilling from this
                                                         ! point should flow to when this cell floods
  integer, pointer, dimension(:) :: connect_redirect_index ! Location of lake/HD model cell water spilling from this
                                                           ! point should flow to when this cell connects
  integer, pointer, dimension(:) :: additional_flood_redirect_index ! Replicate variable for case of a double
                                                                    ! merge
  integer, pointer, dimension(:) :: additional_connect_redirect_index ! Replicate variable for case of a double
                                                                      ! merge
  integer, pointer, dimension(:) :: coarse_cell_numbers_on_fine_grid !Mapping of coarse numbers to the fine grid
  integer :: ncells
  integer :: ncells_coarse
  contains
     procedure :: initialiselakeparameters
     procedure :: lakeparametersdestructor
end type lakeparameters

interface lakeparameters
  procedure :: lakeparametersconstructor
end interface lakeparameters

! Workaround for Fortran's lack of arrays of pointers
type :: lakepointer
  type(lake), pointer :: lake_pointer
end type lakepointer

! Object containing (spatial) prognostic variables shared by all lakes
type :: lakefields
  ! Surface Spatial Variables
  ! On the lake grid
  logical, pointer, dimension(:) :: completed_lake_cells ! Fully flood cells
  integer, pointer, dimension(:) :: lake_numbers ! ID number of lake in this cell (if any)
  ! On the HD grid
  real(dp), pointer, dimension(:) :: water_to_lakes ! Water from HD model sink points to distribute to lakes
  real(dp), pointer, dimension(:) :: water_to_hd ! Water from overflowing lakes to redistribute to the HD model
  real(dp), pointer, dimension(:) :: lake_water_from_ocean ! Water to remove from the ocean to prevent negative
                                                       ! lake volumes occuring
  ! Lists
  integer, pointer, dimension(:) :: cells_with_lakes_index ! Coordinates of cells containing at least one lake
                                                           ! on HD grid to speed up iteration
  type(lakepointer), pointer, dimension(:) :: other_lakes ! Access to the array of lake objects
  contains
    procedure :: initialiselakefields
    procedure :: lakefieldsdestructor
end type lakefields

interface lakefields
  procedure :: lakefieldsconstructor
end interface lakefields

! Non spatial prognostic variables for all lakes
type lakeprognostics
  type(lakepointer), pointer, dimension(:) :: lakes ! An array of lake objects representing each
                                                    ! lake basin/sub-basin
  real(dp) :: total_lake_volume
  contains
    procedure :: initialiselakeprognostics
    procedure :: lakeprognosticsdestructor
end type lakeprognostics

interface lakeprognostics
  procedure :: lakeprognosticsconstructor
end interface lakeprognostics

! The object containing the detials of particular lake (or lake sub-basin)
! Some variables cache information found in the lake parameters
type :: lake
  integer :: lake_type ! Is this a filling lake, an overflowing lake or a subsumed lake
  integer :: lake_number ! ID number of this lake - this is also the index of this lake
                         ! in the list of lakes
  integer :: center_cell_index ! Coordinates of this lakes deepest point
  real(dp)    :: lake_volume ! Current volume of this lake itself
  real(dp)    :: secondary_lake_volume ! Volume of any subsumed sub-basins of this lake
  real(dp)    :: unprocessed_water ! Water added to this lake that is yet to be added to its
                               ! volume for computational reasons
                               ! This will be dealt with at the end of the time-step
  integer :: current_cell_to_fill_index ! The current cell being filled
  integer :: center_cell_coarse_index ! The position of this lakes center on the coarse HD
                                      ! grid
  integer :: previous_cell_to_fill_index ! The last lake cell that was filled - used if
                                         ! evaporation causes the lake to shrink
  integer, dimension(:), allocatable :: filled_lake_cells ! A list of previously filled cells
                                                          ! used to reverse the filling of the
                                                          ! lake if the lake is shrinking
  integer :: filled_lake_cell_index ! Index of the last filled position in the filled_lake_cells
                                    ! list

  type(lakeparameters), pointer :: lake_parameters ! Pointer to the lake parameters object
  type(lakefields), pointer :: lake_fields ! Pointer to the lake fields object
  ! Filling lake variables
  logical :: primary_merge_completed ! Used to signal that the first merge is complete in the
                                     ! case of a double merge
  logical :: use_additional_fields   ! Need to switch to additional fields due to a double merge
  ! Overflowing lake variables
  real(dp) :: excess_water ! Water volume above the height of the outlet that needs to be drained off
  integer :: outflow_redirect_index ! Where (the HD model cell or lake) to place water being
                                    ! drained off in
  integer :: next_merge_target_index ! Used to determine the correct merge order in complex multi
                                     ! sub-basin merges
  logical :: local_redirect ! Should water drain back to the HD model (non-local) or back to the HD
                            ! model
  real(dp)    :: lake_retention_coefficient ! Parameter determining how fast water above the outlet height
                                        ! is drained off from this lake
  ! Subsumed lake variables
  integer :: primary_lake_number ! The lake that this lake should redirect its water to
  contains
    !General procedures
    procedure :: add_water
    procedure :: remove_water
    procedure :: accept_merge
    procedure :: store_water
    procedure :: find_true_primary_lake
    procedure :: get_merge_type
    procedure :: get_primary_merge_coords
    procedure :: get_secondary_merge_coords
    procedure :: check_if_merge_is_possible
    procedure :: initialiselake
    procedure :: set_filling_lake_parameter_to_null_values
    procedure :: set_overflowing_lake_parameter_to_null_values
    procedure :: set_subsumed_lake_parameter_to_null_values
    procedure :: accept_split
    procedure :: drain_current_cell
    procedure :: release_negative_water
    !Filling lake procedures
    procedure :: initialisefillinglake
    procedure :: perform_primary_merge
    procedure :: perform_secondary_merge
    procedure :: rollback_primary_merge
    procedure :: fill_current_cell
    procedure :: change_to_overflowing_lake
    procedure :: update_filling_cell
    procedure :: rollback_filling_cell
    procedure :: get_outflow_redirect_coords
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

! Initialise the variables of a lake and quickly build it with dry-run
! set to true to determine how long the filled lake cells array should be
subroutine initialiselake(this,center_cell_index_in,&
                          current_cell_to_fill_index_in, &
                          center_cell_coarse_index_in, &
                          lake_number_in,lake_volume_in,&
                          secondary_lake_volume_in,&
                          unprocessed_water_in, &
                          lake_parameters_in,lake_fields_in)
  class(lake),intent(inout) :: this
  integer, intent(in) :: center_cell_index_in
  integer, intent(in) :: current_cell_to_fill_index_in
  integer, intent(in) :: center_cell_coarse_index_in
  integer, intent(in) :: lake_number_in
  real(dp), intent(in)    :: lake_volume_in
  real(dp), intent(in)    :: secondary_lake_volume_in
  real(dp), intent(in)    :: unprocessed_water_in
  type(lakeparameters), target, intent(in) :: lake_parameters_in
  type(lakefields), target, intent(inout) :: lake_fields_in
  integer :: i
  integer :: merge_type
    this%lake_parameters => lake_parameters_in
    this%lake_fields => lake_fields_in
    this%lake_number = lake_number_in
    this%lake_volume = lake_volume_in
    this%secondary_lake_volume = secondary_lake_volume_in
    this%center_cell_index = center_cell_index_in
    this%center_cell_coarse_index = center_cell_coarse_index_in
    this%current_cell_to_fill_index = current_cell_to_fill_index_in
    this%unprocessed_water = unprocessed_water_in
    if (.not. allocated(this%filled_lake_cells)) then
      i = 0
      do
        merge_type = this%get_merge_type()
        if (merge_type == secondary_merge .or. merge_type == double_merge) exit
        call this%update_filling_cell(.true.)
        i = i + 1
      end do
      allocate(this%filled_lake_cells(i))
      this%filled_lake_cell_index = 0
      this%current_cell_to_fill_index = current_cell_to_fill_index_in
      this%lake_fields%completed_lake_cells(:) = .false.
    end if
end subroutine initialiselake

! Construct a basin list object
function basinlistconstructor(basin_indices_in) &
    result(constructor)
  type(basinlist) :: constructor
  integer, pointer, dimension(:) :: basin_indices_in
    constructor%basin_indices => basin_indices_in
end function basinlistconstructor

! Initialise the set of lake parameters. Set the flood only flag. Find out the
! number of fine lake centers in each coarse cell and create a list of the basins
! in each coarse cell
subroutine initialiselakeparameters(this,lake_centers_in, &
                                    connection_volume_thresholds_in, &
                                    flood_volume_thresholds_in, &
                                    flood_local_redirect_in, &
                                    connect_local_redirect_in, &
                                    additional_flood_local_redirect_in, &
                                    additional_connect_local_redirect_in, &
                                    merge_points_in, &
                                    flood_next_cell_index_in, &
                                    connect_next_cell_index_in, &
                                    flood_force_merge_index_in, &
                                    connect_force_merge_index_in, &
                                    flood_redirect_index_in, &
                                    connect_redirect_index_in, &
                                    additional_flood_redirect_index_in, &
                                    additional_connect_redirect_index_in, &
                                    ncells_in,&
                                    ncells_coarse_in,&
                                    instant_throughflow_in, &
                                    lake_retention_coefficient_in, &
                                    coarse_cell_numbers_on_fine_grid_in)
  class(lakeparameters),intent(inout) :: this
  logical :: instant_throughflow_in
  real(dp) :: lake_retention_coefficient_in
  logical, pointer, dimension(:) :: lake_centers_in
  real(dp), pointer, dimension(:) :: connection_volume_thresholds_in
  real(dp), pointer, dimension(:) :: flood_volume_thresholds_in
  logical, pointer, dimension(:) :: flood_local_redirect_in
  logical, pointer, dimension(:) :: connect_local_redirect_in
  logical, pointer, dimension(:) :: additional_flood_local_redirect_in
  logical, pointer, dimension(:) :: additional_connect_local_redirect_in
  integer, pointer, dimension(:) :: merge_points_in
  integer, pointer, dimension(:) :: flood_next_cell_index_in
  integer, pointer, dimension(:) :: connect_next_cell_index_in
  integer, pointer, dimension(:) :: flood_force_merge_index_in
  integer, pointer, dimension(:) :: connect_force_merge_index_in
  integer, pointer, dimension(:) :: flood_redirect_index_in
  integer, pointer, dimension(:) :: connect_redirect_index_in
  integer, pointer, dimension(:) :: additional_flood_redirect_index_in
  integer, pointer, dimension(:) :: additional_connect_redirect_index_in
  integer, pointer, dimension(:) :: basins_in_coarse_cell_index_temp
  integer, pointer, dimension(:) :: basins_in_coarse_cell_index
  integer, pointer, dimension(:), optional :: coarse_cell_numbers_on_fine_grid_in
  type(basinlist), pointer, dimension(:) :: basins_temp
  integer :: ncells_in
  integer :: ncells_coarse_in
  integer :: basin_number
  integer :: number_of_basins_in_coarse_cell
  integer :: i,k,m
  integer :: scale_factor
    number_of_basins_in_coarse_cell = 0
    this%lake_centers => lake_centers_in
    this%connection_volume_thresholds => connection_volume_thresholds_in
    this%flood_volume_thresholds => flood_volume_thresholds_in
    this%flood_local_redirect => flood_local_redirect_in
    this%connect_local_redirect => connect_local_redirect_in
    this%additional_flood_local_redirect => additional_flood_local_redirect_in
    this%additional_connect_local_redirect => additional_connect_local_redirect_in
    this%merge_points => merge_points_in
    this%flood_next_cell_index => flood_next_cell_index_in
    this%connect_next_cell_index => connect_next_cell_index_in
    this%flood_force_merge_index => flood_force_merge_index_in
    this%connect_force_merge_index => connect_force_merge_index_in
    this%flood_redirect_index => flood_redirect_index_in
    this%connect_redirect_index => connect_redirect_index_in
    this%additional_flood_redirect_index => additional_flood_redirect_index_in
    this%additional_connect_redirect_index => additional_connect_redirect_index_in
    this%ncells = ncells_in
    this%ncells_coarse = ncells_coarse_in
    ! If not provided map assume a coarse cells numbers are the same as fine cells
    ! numbers (thus the grids are identical)
    if (.not. present(coarse_cell_numbers_on_fine_grid_in)) then
      allocate(this%coarse_cell_numbers_on_fine_grid(ncells_in))
      do i = 1,ncells_in
        this%coarse_cell_numbers_on_fine_grid(i) = i
      end do
    else
      this%coarse_cell_numbers_on_fine_grid => coarse_cell_numbers_on_fine_grid_in
    end if
    this%instant_throughflow = instant_throughflow_in
    this%lake_retention_coefficient = lake_retention_coefficient_in
    ! Set flood only flag
    allocate(this%flood_only(ncells_in))
    this%flood_only(:) = .false.
    where (this%connection_volume_thresholds == -1.0_dp)
      this%flood_only = .true.
    else where
      this%flood_only = .false.
    end where
    allocate(basins_in_coarse_cell_index_temp((this%ncells)&
                                            /(this%ncells_coarse)))
    allocate(this%basin_numbers(this%ncells_coarse))
    this%basin_numbers(:) = 0
    allocate(basins_temp(this%ncells_coarse))
    basin_number = 0
    scale_factor = ncells_in/ncells_coarse_in
    !Enumerate and create list of the fine basin(lake) centers in each coarse cell
    do i=1,ncells_coarse_in
      number_of_basins_in_coarse_cell = 0
      do k = 1,this%ncells
        if (this%coarse_cell_numbers_on_fine_grid(k) == i) then
          if (this%lake_centers(k)) then
            number_of_basins_in_coarse_cell = number_of_basins_in_coarse_cell + 1
            basins_in_coarse_cell_index_temp(number_of_basins_in_coarse_cell) = k
          end if
        end if
      end do
      if(number_of_basins_in_coarse_cell > 0) then
        allocate(basins_in_coarse_cell_index(number_of_basins_in_coarse_cell))
        do m=1,number_of_basins_in_coarse_cell
          basins_in_coarse_cell_index(m) = basins_in_coarse_cell_index_temp(m)
        end do
        basin_number = basin_number + 1
        basins_temp(basin_number) = &
          basinlist(basins_in_coarse_cell_index)
        this%basin_numbers(i) = basin_number
      end if
    end do
    allocate(this%basins(basin_number))
    do m=1,basin_number
      this%basins(m) = basins_temp(m)
    end do
    deallocate(basins_temp)
    deallocate(basins_in_coarse_cell_index_temp)
end subroutine initialiselakeparameters

! Construct a lake parameters object
function lakeparametersconstructor(lake_centers_in, &
                                   connection_volume_thresholds_in, &
                                   flood_volume_thresholds_in, &
                                   flood_local_redirect_in, &
                                   connect_local_redirect_in, &
                                   additional_flood_local_redirect_in, &
                                   additional_connect_local_redirect_in, &
                                   merge_points_in, &
                                   flood_next_cell_index_in, &
                                   connect_next_cell_index_in, &
                                   flood_force_merge_index_in, &
                                   connect_force_merge_index_in, &
                                   flood_redirect_index_in, &
                                   connect_redirect_index_in, &
                                   additional_flood_redirect_index_in, &
                                   additional_connect_redirect_index_in, &
                                   ncells_in,&
                                   ncells_coarse_in, &
                                   instant_throughflow_in, &
                                   lake_retention_coefficient_in, &
                                   coarse_cell_numbers_on_fine_grid_in) result(constructor)
  type(lakeparameters), pointer :: constructor
  logical, pointer, dimension(:), intent(in) :: lake_centers_in
  real(dp), pointer, dimension(:), intent(in) :: connection_volume_thresholds_in
  real(dp), pointer, dimension(:), intent(in) :: flood_volume_thresholds_in
  logical, pointer, dimension(:), intent(in) :: flood_local_redirect_in
  logical, pointer, dimension(:), intent(in) :: connect_local_redirect_in
  logical, pointer, dimension(:), intent(in) :: additional_flood_local_redirect_in
  logical, pointer, dimension(:), intent(in) :: additional_connect_local_redirect_in
  integer, pointer, dimension(:), intent(in) :: merge_points_in
  integer, pointer, dimension(:), intent(in) :: flood_next_cell_index_in
  integer, pointer, dimension(:), intent(in) :: connect_next_cell_index_in
  integer, pointer, dimension(:), intent(in) :: flood_force_merge_index_in
  integer, pointer, dimension(:), intent(in) :: connect_force_merge_index_in
  integer, pointer, dimension(:), intent(in) :: flood_redirect_index_in
  integer, pointer, dimension(:), intent(in) :: connect_redirect_index_in
  integer, pointer, dimension(:), intent(in) :: additional_flood_redirect_index_in
  integer, pointer, dimension(:), intent(in) :: additional_connect_redirect_index_in
  integer, intent(in) :: ncells_in
  integer, intent(in) :: ncells_coarse_in
  logical, intent(in) :: instant_throughflow_in
  real(dp), intent(in) :: lake_retention_coefficient_in
  integer, pointer, dimension(:), intent(in), optional :: &
    coarse_cell_numbers_on_fine_grid_in
    allocate(constructor)
    if (present(coarse_cell_numbers_on_fine_grid_in)) then
      call constructor%initialiselakeparameters(lake_centers_in, &
                                                connection_volume_thresholds_in, &
                                                flood_volume_thresholds_in, &
                                                flood_local_redirect_in, &
                                                connect_local_redirect_in, &
                                                additional_flood_local_redirect_in, &
                                                additional_connect_local_redirect_in, &
                                                merge_points_in, &
                                                flood_next_cell_index_in, &
                                                connect_next_cell_index_in, &
                                                flood_force_merge_index_in, &
                                                connect_force_merge_index_in, &
                                                flood_redirect_index_in, &
                                                connect_redirect_index_in, &
                                                additional_flood_redirect_index_in, &
                                                additional_connect_redirect_index_in, &
                                                ncells_in, &
                                                ncells_coarse_in, &
                                                instant_throughflow_in, &
                                                lake_retention_coefficient_in, &
                                                coarse_cell_numbers_on_fine_grid_in)
    else
      call constructor%initialiselakeparameters(lake_centers_in, &
                                                connection_volume_thresholds_in, &
                                                flood_volume_thresholds_in, &
                                                flood_local_redirect_in, &
                                                connect_local_redirect_in, &
                                                additional_flood_local_redirect_in, &
                                                additional_connect_local_redirect_in, &
                                                merge_points_in, &
                                                flood_next_cell_index_in, &
                                                connect_next_cell_index_in, &
                                                flood_force_merge_index_in, &
                                                connect_force_merge_index_in, &
                                                flood_redirect_index_in, &
                                                connect_redirect_index_in, &
                                                additional_flood_redirect_index_in, &
                                                additional_connect_redirect_index_in, &
                                                ncells_in, &
                                                ncells_coarse_in, &
                                                instant_throughflow_in, &
                                                lake_retention_coefficient_in)
    end if
end function lakeparametersconstructor

! Free memory from a lake parameters object
subroutine lakeparametersdestructor(this)
  class(lakeparameters),intent(inout) :: this
  integer :: i
    deallocate(this%flood_only)
    deallocate(this%basin_numbers)
    do i = 1,size(this%basins)
      deallocate(this%basins(i)%basin_indices)
    end do
    deallocate(this%basins)
    deallocate(this%lake_centers)
    deallocate(this%connection_volume_thresholds)
    deallocate(this%flood_volume_thresholds)
    deallocate(this%flood_local_redirect)
    deallocate(this%connect_local_redirect)
    deallocate(this%additional_flood_local_redirect)
    deallocate(this%additional_connect_local_redirect)
    deallocate(this%merge_points)
    deallocate(this%flood_next_cell_index)
    deallocate(this%connect_next_cell_index)
    deallocate(this%flood_force_merge_index)
    deallocate(this%connect_force_merge_index)
    deallocate(this%flood_redirect_index)
    deallocate(this%connect_redirect_index)
    deallocate(this%additional_flood_redirect_index)
    deallocate(this%additional_connect_redirect_index)
end subroutine lakeparametersdestructor

! Initialise the set of fields pertaining to the lake model. Calculate
! which coarse cells contain lakes and produce a list of them to accelerate
! iteration over them
subroutine initialiselakefields(this,lake_parameters)
    class(lakefields), intent(inout) :: this
    type(lakeparameters) :: lake_parameters
    integer :: scale_factor
    integer, pointer, dimension(:) :: cells_with_lakes_index_temp
    integer :: i,k
    integer :: number_of_cells_containing_lakes
    logical :: contains_lake
      allocate(this%completed_lake_cells(lake_parameters%ncells))
      this%completed_lake_cells(:) = .false.
      allocate(this%lake_numbers(lake_parameters%ncells))
      this%lake_numbers(:) = 0
      allocate(this%water_to_lakes(lake_parameters%ncells_coarse))
      this%water_to_lakes(:) = 0.0_dp
      allocate(this%lake_water_from_ocean(lake_parameters%ncells_coarse))
      this%lake_water_from_ocean(:) = 0.0_dp
      allocate(this%water_to_hd(lake_parameters%ncells_coarse))
      this%water_to_hd(:) = 0.0_dp
      allocate(cells_with_lakes_index_temp(lake_parameters%ncells_coarse))
      number_of_cells_containing_lakes = 0
      scale_factor = lake_parameters%ncells/lake_parameters%ncells_coarse
      do i=1,lake_parameters%ncells_coarse
        contains_lake = .false.
        do k = 1,lake_parameters%ncells
          if (lake_parameters%coarse_cell_numbers_on_fine_grid(k) == i) then
            if (lake_parameters%lake_centers(k)) contains_lake = .true.
          end if
        end do
        if(contains_lake) then
          number_of_cells_containing_lakes = number_of_cells_containing_lakes + 1
          cells_with_lakes_index_temp(number_of_cells_containing_lakes) = i
        end if
      end do
      allocate(this%cells_with_lakes_index(number_of_cells_containing_lakes))
      do i = 1,number_of_cells_containing_lakes
        this%cells_with_lakes_index(i) = &
          & cells_with_lakes_index_temp(i)
      end do
      deallocate(cells_with_lakes_index_temp)
end subroutine initialiselakefields

! Construct lake fields object
function lakefieldsconstructor(lake_parameters) result(constructor)
  type(lakeparameters), intent(inout) :: lake_parameters
  type(lakefields), pointer :: constructor
    allocate(constructor)
    call constructor%initialiselakefields(lake_parameters)
end function lakefieldsconstructor

! Free memory from a lakefields object
subroutine lakefieldsdestructor(this)
  class(lakefields), intent(inout) :: this
    deallocate(this%completed_lake_cells)
    deallocate(this%lake_numbers)
    deallocate(this%lake_water_from_ocean)
    deallocate(this%water_to_lakes)
    deallocate(this%water_to_hd)
    deallocate(this%cells_with_lakes_index)
end subroutine lakefieldsdestructor

! Initialise a lake prognostics object by constructing the master list of lake objects
subroutine initialiselakeprognostics(this,lake_parameters_in,lake_fields_in)
  class(lakeprognostics), intent(inout) :: this
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), intent(inout) :: lake_fields_in
  type(lakepointer), pointer, dimension(:) :: lakes_temp
  type(lake), pointer :: lake_temp
  integer :: center_cell_coarse_index
  integer :: lake_number
  integer :: i
    this%total_lake_volume = 0.0_dp
    allocate(lakes_temp(lake_parameters_in%ncells))
    lake_number = 0
    do i=1,lake_parameters_in%ncells
      if(lake_parameters_in%lake_centers(i)) then
        center_cell_coarse_index = lake_parameters_in%coarse_cell_numbers_on_fine_grid(i)
        lake_number = lake_number + 1
        lake_temp => lake(lake_parameters_in,lake_fields_in,&
                          i,i,center_cell_coarse_index, &
                          lake_number,0.0_dp,0.0_dp,0.0_dp)
        lakes_temp(lake_number) = lakepointer(lake_temp)
        lake_fields_in%lake_numbers(i) = lake_number
      end if
    end do
    allocate(this%lakes(lake_number))
    do i=1,lake_number
      this%lakes(i) = lakes_temp(i)
    end do
    lake_fields_in%other_lakes => this%lakes
    deallocate(lakes_temp)
end subroutine initialiselakeprognostics

! Construct a lakeprognostic object
function lakeprognosticsconstructor(lake_parameters_in,lake_fields_in) result(constructor)
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), intent(inout) :: lake_fields_in
  type(lakeprognostics), pointer :: constructor
    allocate(constructor)
    call constructor%initialiselakeprognostics(lake_parameters_in,lake_fields_in)
end function lakeprognosticsconstructor

! Free memory from a lakeprognostics object
subroutine lakeprognosticsdestructor(this)
  class(lakeprognostics), intent(inout) :: this
  integer :: i
    do i = 1,size(this%lakes)
      deallocate(this%lakes(i)%lake_pointer)
    end do
    deallocate(this%lakes)
end subroutine lakeprognosticsdestructor

! Initialise a filling lake object
subroutine initialisefillinglake(this,lake_parameters_in,lake_fields_in,center_cell_index_in, &
                                 current_cell_to_fill_index_in, &
                                 center_cell_coarse_index_in, &
                                 lake_number_in, &
                                 lake_volume_in, &
                                 secondary_lake_volume_in,&
                                 unprocessed_water_in)
  class(lake) :: this
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), intent(inout) :: lake_fields_in
  integer, intent(in) :: center_cell_index_in
  integer, intent(in) :: current_cell_to_fill_index_in
  integer, intent(in) :: center_cell_coarse_index_in
  integer, intent(in) :: lake_number_in
  real(dp),    intent(in) :: lake_volume_in
  real(dp),    intent(in) :: secondary_lake_volume_in
  real(dp),    intent(in) :: unprocessed_water_in
    call this%initialiselake(center_cell_index_in, &
                             current_cell_to_fill_index_in, &
                             center_cell_coarse_index_in, &
                             lake_number_in, &
                             lake_volume_in, &
                             secondary_lake_volume_in,&
                             unprocessed_water_in, &
                             lake_parameters_in, &
                             lake_fields_in)
    this%lake_type = filling_lake_type
    this%primary_merge_completed = .false.
    this%use_additional_fields = .false.
    call this%set_overflowing_lake_parameter_to_null_values()
    call this%set_subsumed_lake_parameter_to_null_values()
end subroutine initialisefillinglake

!Construct a filling lake object - all lakes start as filling lakes hence this is
!the constructor for all lake objects
function lakeconstructor(lake_parameters_in,lake_fields_in,center_cell_index_in, &
                         current_cell_to_fill_index_in, &
                         center_cell_coarse_index_in, &
                         lake_number_in, &
                         lake_volume_in,secondary_lake_volume_in,&
                         unprocessed_water_in) result(constructor)

  type(lake), pointer :: constructor
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), intent(inout) :: lake_fields_in
  integer, intent(in) :: center_cell_index_in
  integer, intent(in) :: current_cell_to_fill_index_in
  integer, intent(in) :: center_cell_coarse_index_in
  integer, intent(in) :: lake_number_in
  real(dp),    intent(in) :: lake_volume_in
  real(dp),    intent(in) :: secondary_lake_volume_in
  real(dp),    intent(in) :: unprocessed_water_in
  allocate(constructor)
  call constructor%initialisefillinglake(lake_parameters_in, &
                                         lake_fields_in, &
                                         center_cell_index_in, &
                                         current_cell_to_fill_index_in, &
                                         center_cell_coarse_index_in, &
                                         lake_number_in, &
                                         lake_volume_in, &
                                         secondary_lake_volume_in,&
                                         unprocessed_water_in)
end function lakeconstructor

! Disable filling lake values when using another type of lake
subroutine set_filling_lake_parameter_to_null_values(this)
  class(lake) :: this
    this%primary_merge_completed = .false.
    this%use_additional_fields = .false.
end subroutine set_filling_lake_parameter_to_null_values

! Reinitialise this object as an overflowing lake
subroutine initialiseoverflowinglake(this,outflow_redirect_index_in, &
                                     local_redirect_in, &
                                     lake_retention_coefficient_in, &
                                     lake_parameters_in,lake_fields_in, &
                                     center_cell_index_in, &
                                     current_cell_to_fill_index_in, &
                                     center_cell_coarse_index_in, &
                                     next_merge_target_index_in, &
                                     lake_number_in,lake_volume_in, &
                                     secondary_lake_volume_in,&
                                     unprocessed_water_in)
  class(lake) :: this
  integer, intent(in) :: outflow_redirect_index_in
  logical, intent(in) :: local_redirect_in
  real(dp), intent(in)    :: lake_retention_coefficient_in
  integer, intent(in) :: center_cell_index_in
  integer, intent(in) :: current_cell_to_fill_index_in
  integer, intent(in) :: center_cell_coarse_index_in
  integer, intent(in) :: next_merge_target_index_in
  integer, intent(in) :: lake_number_in
  real(dp),    intent(in) :: lake_volume_in
  real(dp), intent(in)    :: secondary_lake_volume_in
  real(dp),    intent(in) :: unprocessed_water_in
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), intent(inout) :: lake_fields_in
    call this%initialiselake(center_cell_index_in, &
                             current_cell_to_fill_index_in, &
                             center_cell_coarse_index_in, &
                             lake_number_in,lake_volume_in, &
                             secondary_lake_volume_in,&
                             unprocessed_water_in, &
                             lake_parameters_in, &
                             lake_fields_in)
    this%lake_type = overflowing_lake_type
    this%outflow_redirect_index = outflow_redirect_index_in
    this%next_merge_target_index = next_merge_target_index_in
    this%local_redirect = local_redirect_in
    this%lake_retention_coefficient = lake_retention_coefficient_in
    this%excess_water = 0.0_dp
    call this%set_filling_lake_parameter_to_null_values()
    call this%set_subsumed_lake_parameter_to_null_values()
end subroutine initialiseoverflowinglake

! Set overflowing lake only variables to zero when using a subsumed or filling lake
subroutine set_overflowing_lake_parameter_to_null_values(this)
  class(lake) :: this
    this%outflow_redirect_index = 0
    this%next_merge_target_index = 0
    this%local_redirect = .false.
    this%lake_retention_coefficient = 0
    this%excess_water = 0.0_dp
end subroutine set_overflowing_lake_parameter_to_null_values

! Reinitialise this lake object as a subsumed lake
subroutine initialisesubsumedlake(this,lake_parameters_in,lake_fields_in, &
                                  primary_lake_number_in,center_cell_index_in, &
                                  current_cell_to_fill_index_in, &
                                  center_cell_coarse_index_in, &
                                  lake_number_in,&
                                  lake_volume_in,&
                                  secondary_lake_volume_in,&
                                  unprocessed_water_in)
  class(lake) :: this
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), intent(inout) :: lake_fields_in
  integer, intent(in) :: primary_lake_number_in
  integer, intent(in) :: center_cell_index_in
  integer, intent(in) :: current_cell_to_fill_index_in
  integer, intent(in) :: center_cell_coarse_index_in
  integer, intent(in) :: lake_number_in
  real(dp),    intent(in) :: lake_volume_in
  real(dp), intent(in)    :: secondary_lake_volume_in
  real(dp),    intent(in) :: unprocessed_water_in
    call this%initialiselake(center_cell_index_in, &
                             current_cell_to_fill_index_in, &
                             center_cell_coarse_index_in, &
                             lake_number_in,lake_volume_in, &
                             secondary_lake_volume_in,&
                             unprocessed_water_in, &
                             lake_parameters_in,lake_fields_in)
    this%lake_type = subsumed_lake_type
    this%primary_lake_number = primary_lake_number_in
    call this%set_filling_lake_parameter_to_null_values()
    call this%set_overflowing_lake_parameter_to_null_values()
end subroutine initialisesubsumedlake

! Set subsumed lake only variables to zero when using a overflowing or filling lake
subroutine set_subsumed_lake_parameter_to_null_values(this)
  class(lake) :: this
    this%primary_lake_number = 0
end subroutine set_subsumed_lake_parameter_to_null_values

! Fill lakes with their initial water content
subroutine setup_lakes(lake_parameters,lake_prognostics,lake_fields,initial_water_to_lake_centers)
  type(lakeparameters), intent(in) :: lake_parameters
  type(lakeprognostics), intent(inout) :: lake_prognostics
  type(lakefields), intent(inout) :: lake_fields
  real(dp), pointer, dimension(:), intent(in) :: initial_water_to_lake_centers
  type(lake), pointer :: working_lake
  real(dp) :: initial_water_to_lake_center
  integer :: lake_index
  integer :: i
    do i=1,lake_parameters%ncells
      if(lake_parameters%lake_centers(i)) then
        initial_water_to_lake_center = initial_water_to_lake_centers(i)
        if(initial_water_to_lake_center > 0.0_dp) then
          lake_index = lake_fields%lake_numbers(i)
          working_lake => lake_prognostics%lakes(lake_index)%lake_pointer
          call working_lake%add_water(initial_water_to_lake_center)
        end if
      end if
    end do
end subroutine setup_lakes

! The main routine called by the HD model. Adds water from sink points to
! lakes (if net inflow is postive) or (if net evaporation is postive) removes
! evaporated water. Processes any unprocessed water, rebalances any lakes with
! negative volumes and drains any water above the height of a lakes outlet
! according to the lakes retention coeffecient
subroutine run_lakes(lake_parameters,lake_prognostics,lake_fields)
  type(lakeparameters), intent(in) :: lake_parameters
  type(lakeprognostics), intent(inout) :: lake_prognostics
  type(lakefields), intent(inout) :: lake_fields
  integer :: index
  integer :: basin_center_index
  integer :: lake_index
  integer :: i,j
  integer :: num_basins_in_cell
  integer :: basin_number
  real(dp) :: share_to_each_lake
  type(lake), pointer :: working_lake
    lake_fields%water_to_hd(:) = 0.0_dp
    lake_fields%lake_water_from_ocean(:) = 0.0_dp
    do i = 1,size(lake_fields%cells_with_lakes_index)
      index = lake_fields%cells_with_lakes_index(i)
      if (lake_fields%water_to_lakes(index) > 0.0_dp) then
        basin_number = lake_parameters%basin_numbers(index)
        num_basins_in_cell = &
          size(lake_parameters%basins(basin_number)%basin_indices)
        share_to_each_lake = lake_fields%water_to_lakes(index)/ &
                             real(num_basins_in_cell,dp)
        do j = 1,num_basins_in_cell
          basin_center_index = lake_parameters%basins(basin_number)%basin_indices(j)
          lake_index = lake_fields%lake_numbers(basin_center_index)
          working_lake => lake_prognostics%lakes(lake_index)%lake_pointer
          call working_lake%add_water(share_to_each_lake)
        end do
      else if (lake_fields%water_to_lakes(index) < 0.0_dp) then
        basin_number = lake_parameters%basin_numbers(index)
        num_basins_in_cell = \
          size(lake_parameters%basins(basin_number)%basin_indices)
        share_to_each_lake = -1.0_dp*lake_fields%water_to_lakes(index)/ &
                             real(num_basins_in_cell)
        do j = 1,num_basins_in_cell
          basin_center_index = lake_parameters%basins(basin_number)%basin_indices(j)
          lake_index = lake_fields%lake_numbers(basin_center_index)
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
      if (working_lake%lake_volume < 0.0_dp) then
        call working_lake%release_negative_water()
      end if
      if (working_lake%lake_type == overflowing_lake_type) then
          call working_lake%drain_excess_water()
      end if
    end do
end subroutine run_lakes

! Add an amount of water to a lake.
! For a filling lake - check if this connects or flood any new cells
! and update this lake accordingly. When this happends then check if
! this in turn cause any lake merges and update this lake and other
! lakes accordingly. Also check if this lake has become an overflowing
! lake in which case add the remaining water to its excess water
! For a overflowing lake - add this water to the lakes excess water
! For a subsumed lake - add this water (plus an unprocessed water
! already allocted to this lake) to this lakes primary lake
recursive subroutine add_water(this,inflow)
  class(lake), target, intent(inout) :: this
  real(dp), intent(in) :: inflow
  type(lake),  pointer :: other_lake
  integer :: target_cell_index
  integer :: merge_type
  integer :: other_lake_number
  logical :: filled
  logical :: merge_possible
  logical :: already_merged
  real(dp) :: inflow_local
    if (this%lake_type == filling_lake_type) then
      inflow_local = inflow + this%unprocessed_water
      this%unprocessed_water = 0.0_dp
      do while (inflow_local > 0.0_dp)
        filled = this%fill_current_cell(inflow_local)
        if(filled) then
          merge_type = this%get_merge_type()
          if(merge_type /= no_merge) then
            do
              if (.not. (merge_type == primary_merge .and. &
                         this%primary_merge_completed)) then
                if (merge_type == double_merge .and. &
                    this%primary_merge_completed) then
                  merge_type = secondary_merge
                  this%use_additional_fields = .true.
                end if
                merge_possible = &
                  this%check_if_merge_is_possible(merge_type,already_merged)
                if (merge_possible) then
                  if (merge_type == secondary_merge) then
                    call this%perform_secondary_merge()
                    call this%store_water(inflow_local)
                    return
                  else
                    call this%perform_primary_merge()
                  end if
                else if (.not. already_merged) then
                  call this%change_to_overflowing_lake(merge_type)
                  call this%store_water(inflow_local)
                  return
                else if (merge_type == secondary_merge) then
                  call this%get_secondary_merge_coords(target_cell_index)
                  other_lake_number = &
                    this%lake_fields%lake_numbers(target_cell_index)
                  other_lake => &
                    this%lake_fields%other_lakes(other_lake_number)%lake_pointer
                  if (other_lake%lake_type == subsumed_lake_type) then
                    call other_lake%change_subsumed_lake_to_filling_lake()
                  end if
                  call other_lake%change_to_overflowing_lake(primary_merge)
                  call this%perform_secondary_merge()
                  call this%store_water(inflow_local)
                  return
                else if (merge_type == double_merge) then
                  this%primary_merge_completed = .true.
                end if
              end if
              if (merge_type /= double_merge) exit
            end do
          end if
          call this%update_filling_cell()
        end if
      end do
    else if (this%lake_type == overflowing_lake_type) then
      inflow_local = inflow + this%unprocessed_water
      this%unprocessed_water = 0.0_dp
      if (this%local_redirect) then
        other_lake_number  = this%lake_fields%lake_numbers(this%outflow_redirect_index)
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

! Remove an amount of water from a lake (via either evaporation or the addition of
! negative water from the HD model).
! For a filling lake - Remove water first from unprocessed water. Then drain the
! current cell. If it is completely drained roll back to the previous filled cell
! and undo any merges then make the previous cell the current cell and loop back
! until draining the current cell until the specified quantity of water has been
! removed
! For an overflowing lake - First remove any unprocessed then excess water. If this
! doesn't meet the specified quantity to remove then change back to a filling lake
! and remove the remaining water from that
! For a subsumed lake - First remove any unprocessed water then remove water from
! this lakes primary lake
subroutine remove_water(this,outflow)
  class(lake), target, intent(inout) :: this
  class(lake), pointer :: other_lake
  real(dp) :: outflow
  real(dp) :: outflow_local
  real(dp) :: new_outflow
  logical :: drained
  integer :: merge_type
    if (this%lake_type == filling_lake_type) then
      if (outflow <= this%unprocessed_water) then
        this%unprocessed_water = this%unprocessed_water - outflow
        return
      end if
      outflow_local = outflow - this%unprocessed_water
      this%unprocessed_water = 0.0_dp
      do while (outflow_local > 0.0_dp)
        outflow_local = this%drain_current_cell(outflow_local,drained)
        if (drained) then
          call this%rollback_filling_cell()
          merge_type = get_merge_type(this)
          if (merge_type == primary_merge .or. merge_type == double_merge) then
            new_outflow = outflow_local/2.0_dp
            outflow_local = outflow_local - new_outflow
            call this%rollback_primary_merge(new_outflow)
            this%primary_merge_completed = .false.
            this%use_additional_fields = .false.
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
      outflow_local =  outflow_local - this%excess_water
      this%excess_water = 0
      if (this%lake_parameters%flood_only(this%current_cell_to_fill_index) .or. &
          this%lake_volume <= &
          this%lake_parameters%connection_volume_thresholds(this%current_cell_to_fill_index)) then
        this%lake_fields%completed_lake_cells(this%current_cell_to_fill_index) = .false.
      end if
      call this%change_overflowing_lake_to_filling_lake()
      merge_type = this%get_merge_type()
      if (merge_type == double_merge) then
        if (this%primary_merge_completed) then
            new_outflow = outflow_local/2.0_dp
            outflow_local = outflow_local - new_outflow
            call this%rollback_primary_merge(new_outflow)
        end if
      end if
      this%primary_merge_completed = .false.
      this%use_additional_fields = .false.
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

! Store water for latter processing (adding) - this is sometimes necessary to prevent
! infinite loops
subroutine store_water(this,inflow)
  class(lake), target, intent(inout) :: this
  real(dp), intent(in) :: inflow
    this%unprocessed_water = this%unprocessed_water + inflow
end subroutine store_water

! Remove any negative water and allocate it the field of water to remove from the ocean
subroutine release_negative_water(this)
  class(lake), target, intent(inout) :: this
    this%lake_fields%lake_water_from_ocean(this%center_cell_coarse_index) = &
      this%lake_fields%lake_water_from_ocean(this%center_cell_coarse_index) - &
      this%lake_volume
    this%lake_volume = 0.0_dp
end subroutine

! Perform a merge as initiated by another lake. Change this lake to a subsumed lake,
! If this was originally an overflowing lake store the excess water to then be added
! back to this lake as a subsumed lake and redirected to the primary lake.
subroutine accept_merge(this,redirect_coords_index)
  class(lake), intent(inout) :: this
  integer, intent(in) :: redirect_coords_index
  integer :: lake_type
  real(dp) :: excess_water
    lake_type = this%lake_type
    excess_water = this%excess_water
    this%lake_fields%completed_lake_cells(this%current_cell_to_fill_index) = .true.
    call this%initialisesubsumedlake(this%lake_parameters, &
                                     this%lake_fields, &
                                     this%lake_fields%lake_numbers(redirect_coords_index), &
                                     this%center_cell_index, &
                                     this%current_cell_to_fill_index, &
                                     this%center_cell_coarse_index, &
                                     this%lake_number,this%lake_volume, &
                                     this%secondary_lake_volume,&
                                     this%unprocessed_water)
    if (lake_type == overflowing_lake_type) then
        call this%store_water(excess_water)
    end if
end subroutine accept_merge

! Split this subsumed lake from another lake. Change this lake back into a filling lake
subroutine accept_split(this)
  class(lake), intent(inout) :: this
    if (this%lake_parameters%flood_only(this%current_cell_to_fill_index) .or. &
        this%lake_volume <= &
        this%lake_parameters%connection_volume_thresholds(this%current_cell_to_fill_index)) then
      this%lake_fields%completed_lake_cells(this%current_cell_to_fill_index) = .false.
    end if
    call this%initialisefillinglake(this%lake_parameters,this%lake_fields, &
                                    this%center_cell_index, &
                                    this%current_cell_to_fill_index, &
                                    this%center_cell_coarse_index, &
                                    this%lake_number,this%lake_volume, &
                                    this%secondary_lake_volume,&
                                    this%unprocessed_water)
end subroutine

! Change this filling lake to an overflowing lake
subroutine change_to_overflowing_lake(this,merge_type)
  class(lake), intent(inout) :: this
  integer :: outflow_redirect_index
  integer :: merge_type
  integer :: target_cell_index
  logical :: local_redirect
    this%lake_fields%completed_lake_cells(this%current_cell_to_fill_index) = .true.
    call this%get_outflow_redirect_coords(outflow_redirect_index, &
                                          local_redirect)
    if (merge_type == secondary_merge) then
      call this%get_secondary_merge_coords(target_cell_index)
    else
      call this%get_primary_merge_coords(target_cell_index)
    end if
    call this%initialiseoverflowinglake(outflow_redirect_index, &
                                        local_redirect, &
                                        this%lake_parameters%lake_retention_coefficient, &
                                        this%lake_parameters,this%lake_fields, &
                                        this%center_cell_index, &
                                        this%current_cell_to_fill_index, &
                                        this%center_cell_coarse_index, &
                                        target_cell_index, &
                                        this%lake_number, &
                                        this%lake_volume, &
                                        this%secondary_lake_volume, &
                                        this%unprocessed_water)
end subroutine change_to_overflowing_lake

! Drain water above the height of the outlet from this overflowing lake
! Water is drained at a rate determined by the lake retention coefficient
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
      this%lake_fields%water_to_hd(this%outflow_redirect_index) = &
           this%lake_fields%water_to_hd(this%outflow_redirect_index)+flow
      this%excess_water = this%excess_water - flow
    end if
end subroutine drain_excess_water

! Fill the current cell of a filling lake with water with the
!given quantity of water. Return a flag indicating if the cell is full
!and if it is then also return the leftover water after filling it
function fill_current_cell(this,inflow) result(filled)
  class(lake), intent(inout) :: this
  real(dp), intent(inout) :: inflow
  logical :: filled
  real(dp) :: new_lake_volume
  real(dp) :: maximum_new_lake_volume
    new_lake_volume = inflow + this%lake_volume
    if (this%lake_fields%completed_lake_cells(this%current_cell_to_fill_index) .or. &
        this%lake_parameters%flood_only(this%current_cell_to_fill_index)) then
      maximum_new_lake_volume = &
        this%lake_parameters%flood_volume_thresholds(this%current_cell_to_fill_index)
    else
      maximum_new_lake_volume = &
        this%lake_parameters%connection_volume_thresholds(this%current_cell_to_fill_index)
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

! Drain the current cell of a filling lake. If this is the center cell remove all
! the water from this cell. Otherwise test if the new volume of the lake after
! removing the specified amount of water is enough to complete empty this cell
! If so indicate it this with the drained flag and return any leftover outflow
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
    if (this%lake_fields%completed_lake_cells(this%previous_cell_to_fill_index) .or. &
        this%lake_parameters%flood_only(this%previous_cell_to_fill_index)) then
      minimum_new_lake_volume = &
        this%lake_parameters%flood_volume_thresholds(this%previous_cell_to_fill_index)
    else
      minimum_new_lake_volume = &
        this%lake_parameters%connection_volume_thresholds(this%previous_cell_to_fill_index)
    end if
    if (new_lake_volume >= minimum_new_lake_volume) then
      this%lake_volume = new_lake_volume
      drained = .false.
      outflow_out = 0.0_dp
    else
      this%lake_volume = minimum_new_lake_volume
      drained = .true.
      outflow_out = minimum_new_lake_volume - new_lake_volume
    end if
end function

! Get the complex merge type for this cell and convert it to a simple merge type
function get_merge_type(this) result(simple_merge_type)
  class(lake), intent(inout) :: this
  integer :: simple_merge_type
  integer :: extended_merge_type
    extended_merge_type = this%lake_parameters%merge_points(this%current_cell_to_fill_index)
    if (this%lake_fields%completed_lake_cells(this%current_cell_to_fill_index) .or. &
        this%lake_parameters%flood_only(this%current_cell_to_fill_index)) then
      simple_merge_type = convert_to_simple_merge_type_flood(Int(extended_merge_type)+1)
    else
      simple_merge_type = convert_to_simple_merge_type_connect(Int(extended_merge_type)+1)
    end if
end function get_merge_type

! Check if a merge is possible (after changing to a new cell to fill) by checking if
! the target lake of the merge is ready to merge or not
function check_if_merge_is_possible(this,simple_merge_type,already_merged) &
   result(merge_possible)
  class(lake), intent(in) :: this
  integer, intent(in) :: simple_merge_type
  logical, intent(out) :: already_merged
  logical :: merge_possible
  type(lake), pointer :: other_lake
  type(lake), pointer :: other_lake_target_lake
  type(lake), pointer :: other_lake_target_lake_true_primary
  integer :: target_cell_index
  integer :: other_lake_target_lake_number
    if (simple_merge_type == secondary_merge) then
      call this%get_secondary_merge_coords(target_cell_index)
    else
      call this%get_primary_merge_coords(target_cell_index)
    end if
    if (.not. this%lake_fields%completed_lake_cells(target_cell_index)) then
      merge_possible = .false.
      already_merged = .false.
      return
    end if
    other_lake => this%lake_fields%&
                  &other_lakes(this%lake_fields%lake_numbers(target_cell_index))%lake_pointer
    other_lake => other_lake%find_true_primary_lake()
    if (other_lake%lake_number == this%lake_number) then
      merge_possible = .false.
      already_merged = .true.
      return
    end if
    if (other_lake%lake_type == overflowing_lake_type) then
      other_lake_target_lake_number = &
        this%lake_fields%lake_numbers(other_lake%next_merge_target_index)
      do
        if (other_lake_target_lake_number == 0) then
          merge_possible = .false.
          already_merged = .false.
          return
        end if
        other_lake_target_lake => &
          this%lake_fields%other_lakes(other_lake_target_lake_number)%lake_pointer
        if (other_lake_target_lake%lake_number == this%lake_number) then
          merge_possible = .true.
          already_merged = .false.
          return
        else
          if (other_lake_target_lake%lake_type == overflowing_lake_type) then
            other_lake_target_lake_number = &
              this%lake_fields%lake_numbers(other_lake_target_lake%next_merge_target_index)
          else if (other_lake_target_lake%lake_type == subsumed_lake_type) then
            other_lake_target_lake_true_primary => &
              other_lake_target_lake%find_true_primary_lake()
            other_lake_target_lake_number = &
              other_lake_target_lake_true_primary%lake_number
          else
             merge_possible = .false.
             already_merged = .false.
             return
          end if
        end if
      end do
    else
      merge_possible = .false.
      already_merged = .false.
    end if
end function check_if_merge_is_possible

! Perform a primary merge by signalling to the the other lake to
! change to a subsumed lake
subroutine perform_primary_merge(this)
  class(lake), intent(inout) :: this
  type(lake), pointer :: other_lake
  integer :: target_cell_index
  integer :: other_lake_number
    call this%get_primary_merge_coords(target_cell_index)
    other_lake_number = this%lake_fields%lake_numbers(target_cell_index)
    other_lake => this%lake_fields%other_lakes(other_lake_number)%lake_pointer
    other_lake => other_lake%find_true_primary_lake()
    this%secondary_lake_volume = this%secondary_lake_volume + other_lake%lake_volume + &
      other_lake%secondary_lake_volume
    this%primary_merge_completed = .true.
    call other_lake%accept_merge(this%center_cell_index)
end subroutine perform_primary_merge

! Perform a secondary merge by finding the true primary lake of the other lake
! and then changing this lake to a subsumed lake with that lake as its primary lake
subroutine perform_secondary_merge(this)
  class(lake), intent(inout) :: this
  type(lake), pointer :: other_lake
  integer :: target_cell_index
  integer :: other_lake_number
    this%lake_fields%completed_lake_cells(this%current_cell_to_fill_index) = .true.
    call this%get_secondary_merge_coords(target_cell_index)
    other_lake_number = this%lake_fields%lake_numbers(target_cell_index)
    other_lake => this%lake_fields%other_lakes(other_lake_number)%lake_pointer
    other_lake => other_lake%find_true_primary_lake()
    other_lake%secondary_lake_volume = other_lake%secondary_lake_volume + &
      this%lake_volume + this%secondary_lake_volume
    if (other_lake%lake_type == overflowing_lake_type) then
        call other_lake%change_overflowing_lake_to_filling_lake()
    end if
    call this%accept_merge(other_lake%center_cell_index)
end subroutine perform_secondary_merge

! Undo a primary merge by finding the subsumed lake that was merge in
! and finding it true primary before this lake and then having that
! lake accept the split. Finally remove the specified quantity of
! water from that other lake
subroutine rollback_primary_merge(this,other_lake_outflow)
  class(lake), intent(inout) :: this
  class(lake), pointer :: other_lake
  integer :: other_lake_number
  real(dp) :: other_lake_outflow
  integer :: target_cell_index
    call this%get_primary_merge_coords(target_cell_index)
    other_lake_number = this%lake_fields%lake_numbers(target_cell_index)
    other_lake => this%lake_fields%other_lakes(other_lake_number)%lake_pointer
    other_lake => find_true_rolledback_primary_lake(other_lake)
    this%secondary_lake_volume = this%secondary_lake_volume - other_lake%lake_volume - &
      other_lake%secondary_lake_volume
    call other_lake%accept_split()
    call other_lake%remove_water(other_lake_outflow)
end subroutine rollback_primary_merge

! Change an overflowing lake back to a filling lake
subroutine change_overflowing_lake_to_filling_lake(this)
  class(lake), intent(inout) :: this
  logical :: use_additional_fields
    this%unprocessed_water = this%unprocessed_water + this%excess_water
    use_additional_fields = (this%get_merge_type() == double_merge)
    call this%initialisefillinglake(this%lake_parameters,this%lake_fields, &
                                    this%center_cell_index, &
                                    this%current_cell_to_fill_index, &
                                    this%center_cell_coarse_index, &
                                    this%lake_number,this%lake_volume, &
                                    this%secondary_lake_volume,&
                                    this%unprocessed_water)
    this%primary_merge_completed = .true.
    this%use_additional_fields = use_additional_fields
end subroutine change_overflowing_lake_to_filling_lake

! Change a subsumed lake back to a filling lake
subroutine change_subsumed_lake_to_filling_lake(this)
  class(lake), intent(inout) :: this
    call this%initialisefillinglake(this%lake_parameters,this%lake_fields, &
                                    this%center_cell_index, &
                                    this%current_cell_to_fill_index, &
                                    this%center_cell_coarse_index, &
                                    this%lake_number,this%lake_volume, &
                                    this%secondary_lake_volume,&
                                    this%unprocessed_water)
    this%primary_merge_completed = .false.
    this%use_additional_fields = .false.
end subroutine change_subsumed_lake_to_filling_lake

! Move to a new cell to fill when previous cell is full
subroutine update_filling_cell(this,dry_run)
  class(lake), intent(inout) :: this
  logical, optional :: dry_run
  integer :: coords_index
    logical :: dry_run_local
    if (present(dry_run)) then
      dry_run_local = dry_run
    else
      dry_run_local = .false.
    end if
    coords_index = this%current_cell_to_fill_index
    if (this%lake_fields%completed_lake_cells(coords_index) .or. &
        this%lake_parameters%flood_only(coords_index)) then
      this%current_cell_to_fill_index = &
        this%lake_parameters%flood_next_cell_index(coords_index)
    else
      this%current_cell_to_fill_index = &
        this%lake_parameters%connect_next_cell_index(coords_index)
    end if
    this%lake_fields%completed_lake_cells(coords_index) = .true.
    if ( .not. dry_run_local) then
      this%lake_fields%lake_numbers(this%current_cell_to_fill_index) = this%lake_number
      this%previous_cell_to_fill_index = coords_index
      this%filled_lake_cell_index = this%filled_lake_cell_index + 1
      this%filled_lake_cells(this%filled_lake_cell_index) = coords_index
      this%primary_merge_completed = .false.
    end if
end subroutine update_filling_cell

! Return filling cell to previous cell to fill when the filling cell is empty
subroutine rollback_filling_cell(this)
  class(lake), intent(inout) :: this
    this%primary_merge_completed = .false.
    if (this%lake_parameters%flood_only(this%current_cell_to_fill_index) .or. &
        this%lake_volume <= &
        this%lake_parameters%connection_volume_thresholds(this%current_cell_to_fill_index)) then
      this%lake_fields%lake_numbers(this%current_cell_to_fill_index) = 0
    end if
    if (this%lake_parameters%flood_only(this%previous_cell_to_fill_index) .or. &
       (this%lake_volume <= &
        this%lake_parameters%connection_volume_thresholds(this%previous_cell_to_fill_index))) then
      this%lake_fields%completed_lake_cells(this%previous_cell_to_fill_index) = .false.
    end if
    this%current_cell_to_fill_index = this%previous_cell_to_fill_index
    this%filled_lake_cells(this%filled_lake_cell_index) = 0
    this%filled_lake_cell_index = this%filled_lake_cell_index - 1
    if (this%filled_lake_cell_index /= 0 ) then
      this%previous_cell_to_fill_index = this%filled_lake_cells(this%filled_lake_cell_index)
    end if
end subroutine

! Calculate the primary merge coords for a given cell
subroutine get_primary_merge_coords(this,index)
  class(lake), intent(in) :: this
  integer, intent(out) :: index
  logical :: completed_lake_cell
  completed_lake_cell = ((this%lake_fields%completed_lake_cells(this%current_cell_to_fill_index)) .or. &
                         (this%lake_parameters%flood_only(this%current_cell_to_fill_index)))
  if (completed_lake_cell) then
    index = &
      this%lake_parameters%flood_force_merge_index(this%current_cell_to_fill_index)
  else
    index = &
      this%lake_parameters%connect_force_merge_index(this%current_cell_to_fill_index)
  end if
end subroutine get_primary_merge_coords

! Calculate the secondary merge coords for a given cell
subroutine get_secondary_merge_coords(this,index)
  class(lake), intent(in) :: this
  integer, intent(out) :: index
  logical :: completed_lake_cell
  completed_lake_cell = (this%lake_fields%completed_lake_cells(this%current_cell_to_fill_index) .or. &
                         this%lake_parameters%flood_only(this%current_cell_to_fill_index))
  if (completed_lake_cell) then
    index = &
      this%lake_parameters%flood_next_cell_index(this%current_cell_to_fill_index)
  else
    index = &
      this%lake_parameters%connect_next_cell_index(this%current_cell_to_fill_index)
  end if
end subroutine get_secondary_merge_coords

! Get the outflow redirect coords for a given cell
subroutine get_outflow_redirect_coords(this,index,local_redirect)
  class(lake), intent(in) :: this
  integer, intent(out) :: index
  logical, intent(out) :: local_redirect
  logical :: completed_lake_cell
  completed_lake_cell = (this%lake_fields%completed_lake_cells(this%current_cell_to_fill_index) .or. &
                                                               this%lake_parameters%flood_only(this%&
                                                                         &current_cell_to_fill_index))
  if (this%use_additional_fields) then
    if (completed_lake_cell) then
      local_redirect = &
        this%lake_parameters%additional_flood_local_redirect(this%current_cell_to_fill_index)
    else
      local_redirect = &
        this%lake_parameters%additional_connect_local_redirect(this%current_cell_to_fill_index)
    end if
  else
    if (completed_lake_cell) then
      local_redirect = &
        this%lake_parameters%flood_local_redirect(this%current_cell_to_fill_index)
    else
      local_redirect = &
        this%lake_parameters%connect_local_redirect(this%current_cell_to_fill_index)
    end if
  end if
  if (this%use_additional_fields) then
    if (completed_lake_cell) then
      index = &
        this%lake_parameters%additional_flood_redirect_index(this%current_cell_to_fill_index)
    else
      index = &
        this%lake_parameters%additional_connect_redirect_index(this%current_cell_to_fill_index)
    end if
  else
    if (completed_lake_cell) then
      index = &
        this%lake_parameters%flood_redirect_index(this%current_cell_to_fill_index)
    else
      index = &
        this%lake_parameters%connect_redirect_index(this%current_cell_to_fill_index)
    end if
  end if
end subroutine get_outflow_redirect_coords

! Find the true primary cell of a lake before it was merged into this lake by following
! the redirects till the next redirect would be something other than a subsumed lake
recursive function find_true_rolledback_primary_lake(this) result(true_primary_lake)
  class(lake), target, intent(in) :: this
  type(lake), pointer :: true_primary_lake
  type(lake), pointer :: next_lake
    next_lake => this%lake_fields%other_lakes(this%primary_lake_number)%lake_pointer
    if (next_lake%lake_type == subsumed_lake_type) then
      true_primary_lake => next_lake%find_true_primary_lake()
    else
      true_primary_lake => this
    end if
end function

! Find the true primary lake of a given lake by following the redirects till a
! non-subsumed lake is found
recursive function find_true_primary_lake(this) result(true_primary_lake)
  class(lake), target, intent(in) :: this
  type(lake), pointer :: true_primary_lake
    if (this%lake_type == subsumed_lake_type) then
        true_primary_lake => find_true_primary_lake(this%lake_fields%other_lakes(this%primary_lake_number)%lake_pointer)
    else
        true_primary_lake => this
    end if
end function find_true_primary_lake

! Calculate the  volumes of all lakes excluding any sublakes and write to the center of the
! lake - this is usually less useful than  calculate_diagnostic_lake_volumes below
function calculate_lake_volumes(lake_parameters,&
                                lake_prognostics) result(lake_volumes)
  type(lakeparameters) :: lake_parameters
  type(lakeprognostics) :: lake_prognostics
  real(dp), dimension(:), pointer :: lake_volumes
  type(lake), pointer :: working_lake
  real(dp) :: lake_overflow
  integer :: i
      allocate(lake_volumes(lake_parameters%ncells))
      lake_volumes(:) = 0.0_dp
      do i = 1,size(lake_prognostics%lakes)
        working_lake => lake_prognostics%lakes(i)%lake_pointer
        if(working_lake%lake_type == overflowing_lake_type) then
          lake_overflow = working_lake%excess_water
        else
          lake_overflow = 0.0_dp
        end if
        lake_volumes(working_lake%center_cell_index) = working_lake%lake_volume + &
                                                       working_lake%unprocessed_water + &
                                                       lake_overflow
      end do
end function calculate_lake_volumes

! Calculate the total volumes of all lakes and write them to a field for all points in
! each lake. This is used for diagnostic output only.
function calculate_diagnostic_lake_volumes(lake_parameters,&
                                           lake_prognostics,&
                                           lake_fields) result(diagnostic_lake_volumes)
  type(lakeparameters), intent(in) :: lake_parameters
  type(lakeprognostics), intent(inout) :: lake_prognostics
  type(lakefields), intent(inout) :: lake_fields
  real(dp), dimension(:), pointer :: diagnostic_lake_volumes
  real(dp), dimension(:), allocatable :: lake_volumes_by_lake_number
  type(lake), pointer :: working_lake
  integer :: i
  integer :: original_lake_index,lake_number
    allocate(diagnostic_lake_volumes(lake_parameters%ncells))
    diagnostic_lake_volumes(:) = 0.0_dp
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
    do i=1,lake_parameters%ncells
      lake_number = lake_fields%lake_numbers(i)
      if (lake_number > 0) then
        diagnostic_lake_volumes(i) = lake_volumes_by_lake_number(lake_number)
      end if
    end do
    deallocate(lake_volumes_by_lake_number)
end function calculate_diagnostic_lake_volumes

!Check the lake models internal water budget is balance to within a floating
!point error - if not print out information
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
    total_inflow_minus_outflow = total_water_to_lakes + &
                                 sum(lake_fields%lake_water_from_ocean) - &
                                 sum(lake_fields%water_to_hd)
    if(present(initial_water_to_lakes)) then
      total_inflow_minus_outflow = total_inflow_minus_outflow + initial_water_to_lakes
    end if
    difference = change_in_total_lake_volume - total_inflow_minus_outflow
    tolerance = 5.0e-16_dp*max(abs(total_water_to_lakes),abs(new_total_lake_volume))
    if (abs(difference) > tolerance) then
      write(*,*) "*** Lake Water Budget ***"
      write(number_as_string,*) sum(lake_fields%water_to_lakes)
      write(*,*) "Total water to lakes" // trim(number_as_string)
      write(number_as_string,*) new_total_lake_volume
      write(*,*) "Total lake volume: " // trim(number_as_string)
      write(number_as_string,*) total_inflow_minus_outflow
      write(*,*) "Total inflow - outflow: " // trim(number_as_string)
      write(number_as_string,*) change_in_total_lake_volume
      write(*,*) "Change in lake volume: " // trim(number_as_string)
      write(number_as_string,*) difference
      write(*,*) "Difference: " // trim(number_as_string)
    end if
    lake_prognostics%total_lake_volume = new_total_lake_volume
end subroutine check_water_budget

end module icosohedral_lake_model_mod
