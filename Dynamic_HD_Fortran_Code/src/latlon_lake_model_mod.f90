module latlon_lake_model_mod

implicit none

integer, parameter :: no_merge = 0
integer, parameter :: primary_merge   = 1
integer, parameter :: secondary_merge = 2
integer, parameter :: double_merge = 3
integer, parameter :: null_mtype = 4
integer, parameter, dimension(17) :: convert_to_simple_merge_type_connect = (/no_merge,primary_merge,primary_merge, &
                                                                              primary_merge,primary_merge,secondary_merge, &
                                                                              secondary_merge,secondary_merge,secondary_merge, &
                                                                              no_merge,no_merge,no_merge,double_merge, &
                                                                              double_merge,double_merge,double_merge, &
                                                                              null_mtype/)
integer, parameter, dimension(17) :: convert_to_simple_merge_type_flood = (/no_merge,primary_merge,secondary_merge, &
                                                                            no_merge,double_merge,primary_merge, &
                                                                            secondary_merge,no_merge,double_merge, &
                                                                            primary_merge,secondary_merge,double_merge, &
                                                                            primary_merge,secondary_merge,no_merge, &
                                                                            double_merge,null_mtype/)

type :: basinlist
  integer, pointer, dimension(:) :: basin_lats
  integer, pointer, dimension(:) :: basin_lons
end type basinlist

interface basinlist
  procedure basinlistconstructor
end interface basinlist

type :: lakeparameters
  logical :: instant_throughflow
  real :: lake_retention_coefficient
  logical, pointer, dimension(:,:) :: lake_centers
  real, pointer, dimension(:,:) :: connection_volume_thresholds
  real, pointer, dimension(:,:) :: flood_volume_thresholds
  logical, pointer, dimension(:,:) :: flood_only
  logical, pointer, dimension(:,:) :: flood_local_redirect
  logical, pointer, dimension(:,:) :: connect_local_redirect
  logical, pointer, dimension(:,:) :: additional_flood_local_redirect
  logical, pointer, dimension(:,:) :: additional_connect_local_redirect
  integer, pointer, dimension(:,:) :: merge_points
  integer, pointer, dimension(:,:) :: basin_numbers
  type(basinlist), pointer, dimension(:) :: basins
  integer, pointer, dimension(:,:) :: flood_next_cell_lat_index
  integer, pointer, dimension(:,:) :: flood_next_cell_lon_index
  integer, pointer, dimension(:,:) :: connect_next_cell_lat_index
  integer, pointer, dimension(:,:) :: connect_next_cell_lon_index
  integer, pointer, dimension(:,:) :: flood_force_merge_lat_index
  integer, pointer, dimension(:,:) :: flood_force_merge_lon_index
  integer, pointer, dimension(:,:) :: connect_force_merge_lat_index
  integer, pointer, dimension(:,:) :: connect_force_merge_lon_index
  integer, pointer, dimension(:,:) :: flood_redirect_lat_index
  integer, pointer, dimension(:,:) :: flood_redirect_lon_index
  integer, pointer, dimension(:,:) :: connect_redirect_lat_index
  integer, pointer, dimension(:,:) :: connect_redirect_lon_index
  integer, pointer, dimension(:,:) :: additional_flood_redirect_lat_index
  integer, pointer, dimension(:,:) :: additional_flood_redirect_lon_index
  integer, pointer, dimension(:,:) :: additional_connect_redirect_lat_index
  integer, pointer, dimension(:,:) :: additional_connect_redirect_lon_index
  integer :: nlat,nlon
  integer :: nlat_coarse,nlon_coarse
  contains
     procedure :: initialiselakeparameters
     procedure :: lakeparametersdestructor
end type lakeparameters

interface lakeparameters
  procedure lakeparametersconstructor
end interface lakeparameters

type :: lakepointer
  class(lake), pointer :: lake_pointer
end type lakepointer

type :: lakefields
  logical, pointer, dimension(:,:) :: completed_lake_cells
  integer, pointer, dimension(:,:) :: lake_numbers
  real, pointer, dimension(:,:) :: water_to_lakes
  real, pointer, dimension(:,:) :: water_to_hd
  integer, pointer, dimension(:) :: cells_with_lakes_lat
  integer, pointer, dimension(:) :: cells_with_lakes_lon
  type(lakepointer), pointer, dimension(:) :: other_lakes
  logical, pointer, dimension(:) :: lock_lake_memory
  contains
    procedure :: initialiselakefields
    procedure :: lakefieldsdestructor
end type lakefields

interface lakefields
  procedure lakefieldsconstructor
end interface lakefields

type lakeprognostics
  type(lakepointer), pointer, dimension(:) :: lakes
  logical, pointer, dimension(:) :: lock_lake_memory
  contains
    procedure :: initialiselakeprognostics
    procedure :: lakeprognosticsdestructor
end type lakeprognostics

interface lakeprognostics
  procedure :: lakeprognosticsconstructor
end interface lakeprognostics

type, abstract :: lake
  integer :: lake_number
  integer :: center_cell_lat
  integer :: center_cell_lon
  real    :: lake_volume
  real    :: unprocessed_water
  integer :: current_cell_to_fill_lat
  integer :: current_cell_to_fill_lon
  type(lakeparameters), pointer :: lake_parameters
  type(lakefields), pointer :: lake_fields
  contains
    procedure(add_water), deferred :: add_water
    procedure :: accept_merge
    procedure :: store_water
    procedure :: find_true_primary_lake
    procedure :: get_merge_type
    procedure :: get_primary_merge_coords
    procedure :: get_secondary_merge_coords
    procedure :: check_if_merge_is_possible
    procedure :: initialiselake
end type lake

abstract interface
  function add_water(this,inflow) result(updated_lake)
    import lake
    class(lake), intent(inout), target :: this
    real, intent(in) :: inflow
    class(lake), pointer :: updated_lake
  end function add_water
end interface

type, extends(lake) :: fillinglake
  logical :: primary_merge_completed
  logical :: use_additional_fields
  contains
    procedure :: perform_primary_merge
    procedure :: perform_secondary_merge
    procedure :: fill_current_cell
    procedure :: change_to_overflowing_lake
    procedure :: update_filling_cell
    procedure :: get_outflow_redirect_coords
    procedure :: add_water => fillinglake_add_water
    procedure :: initialisefillinglake
end type fillinglake

interface fillinglake
  procedure :: fillinglakeconstructor
end interface fillinglake

type, extends(lake) :: overflowinglake
  real :: excess_water
  integer :: outflow_redirect_lat
  integer :: outflow_redirect_lon
  integer :: next_merge_target_lat
  integer :: next_merge_target_lon
  logical :: local_redirect
  real    :: lake_retention_coefficient
  contains
    procedure :: change_overflowing_lake_to_filling_lake
    procedure :: drain_excess_water
    procedure :: add_water => overflowinglake_add_water
    procedure :: initialiseoverflowinglake
end type overflowinglake


interface overflowinglake
  procedure :: overflowinglakeconstructor
end interface overflowinglake

type, extends(lake) :: subsumedlake
  integer :: primary_lake_number
  contains
    procedure :: add_water => subsumedlake_add_water
    procedure :: initialisesubsumedlake
    procedure :: change_subsumed_lake_to_filling_lake
end type subsumedlake

interface subsumedlake
  procedure :: subsumedlakeconstructor
end interface subsumedlake

contains

subroutine initialiselake(this,center_cell_lat_in,center_cell_lon_in, &
                          current_cell_to_fill_lat_in, &
                          current_cell_to_fill_lon_in, &
                          lake_number_in,lake_volume_in,&
                          lake_parameters_in,lake_fields_in)
  class(lake),intent(inout) :: this
  integer, intent(in) :: center_cell_lat_in
  integer, intent(in) :: center_cell_lon_in
  integer, intent(in) :: current_cell_to_fill_lat_in
  integer, intent(in) :: current_cell_to_fill_lon_in
  integer, intent(in) :: lake_number_in
  real, intent(in)    :: lake_volume_in
  type(lakeparameters), target, intent(in) :: lake_parameters_in
  type(lakefields), target, intent(in) :: lake_fields_in
    this%lake_parameters => lake_parameters_in
    this%lake_fields => lake_fields_in
    this%lake_number = lake_number_in
    this%lake_volume = lake_volume_in
    this%center_cell_lat = center_cell_lat_in
    this%center_cell_lon = center_cell_lon_in
    this%current_cell_to_fill_lat = current_cell_to_fill_lat_in
    this%current_cell_to_fill_lon = current_cell_to_fill_lon_in
    this%unprocessed_water = 0.0
end subroutine initialiselake

function basinlistconstructor(basin_lats_in,basin_lons_in) &
    result(constructor)
  type(basinlist) :: constructor
  integer, pointer, dimension(:) :: basin_lats_in
  integer, pointer, dimension(:) :: basin_lons_in
    constructor%basin_lats => basin_lats_in
    constructor%basin_lons => basin_lons_in
end function basinlistconstructor

subroutine initialiselakeparameters(this,lake_centers_in, &
                                    connection_volume_thresholds_in, &
                                    flood_volume_thresholds_in, &
                                    flood_local_redirect_in, &
                                    connect_local_redirect_in, &
                                    additional_flood_local_redirect_in, &
                                    additional_connect_local_redirect_in, &
                                    merge_points_in, &
                                    flood_next_cell_lat_index_in, &
                                    flood_next_cell_lon_index_in, &
                                    connect_next_cell_lat_index_in, &
                                    connect_next_cell_lon_index_in, &
                                    flood_force_merge_lat_index_in, &
                                    flood_force_merge_lon_index_in, &
                                    connect_force_merge_lat_index_in, &
                                    connect_force_merge_lon_index_in, &
                                    flood_redirect_lat_index_in, &
                                    flood_redirect_lon_index_in, &
                                    connect_redirect_lat_index_in, &
                                    connect_redirect_lon_index_in, &
                                    additional_flood_redirect_lat_index_in, &
                                    additional_flood_redirect_lon_index_in, &
                                    additional_connect_redirect_lat_index_in, &
                                    additional_connect_redirect_lon_index_in, &
                                    nlat_in,nlon_in, &
                                    nlat_coarse_in,nlon_coarse_in, &
                                    instant_throughflow_in, &
                                    lake_retention_coefficient_in)
  class(lakeparameters),intent(inout) :: this
  logical :: instant_throughflow_in
  real :: lake_retention_coefficient_in
  logical, pointer, dimension(:,:) :: lake_centers_in
  real, pointer, dimension(:,:) :: connection_volume_thresholds_in
  real, pointer, dimension(:,:) :: flood_volume_thresholds_in
  logical, pointer, dimension(:,:) :: flood_local_redirect_in
  logical, pointer, dimension(:,:) :: connect_local_redirect_in
  logical, pointer, dimension(:,:) :: additional_flood_local_redirect_in
  logical, pointer, dimension(:,:) :: additional_connect_local_redirect_in
  integer, pointer, dimension(:,:) :: merge_points_in
  integer, pointer, dimension(:,:) :: flood_next_cell_lat_index_in
  integer, pointer, dimension(:,:) :: flood_next_cell_lon_index_in
  integer, pointer, dimension(:,:) :: connect_next_cell_lat_index_in
  integer, pointer, dimension(:,:) :: connect_next_cell_lon_index_in
  integer, pointer, dimension(:,:) :: flood_force_merge_lat_index_in
  integer, pointer, dimension(:,:) :: flood_force_merge_lon_index_in
  integer, pointer, dimension(:,:) :: connect_force_merge_lat_index_in
  integer, pointer, dimension(:,:) :: connect_force_merge_lon_index_in
  integer, pointer, dimension(:,:) :: flood_redirect_lat_index_in
  integer, pointer, dimension(:,:) :: flood_redirect_lon_index_in
  integer, pointer, dimension(:,:) :: connect_redirect_lat_index_in
  integer, pointer, dimension(:,:) :: connect_redirect_lon_index_in
  integer, pointer, dimension(:,:) :: additional_flood_redirect_lat_index_in
  integer, pointer, dimension(:,:) :: additional_flood_redirect_lon_index_in
  integer, pointer, dimension(:,:) :: additional_connect_redirect_lat_index_in
  integer, pointer, dimension(:,:) :: additional_connect_redirect_lon_index_in
  integer, pointer, dimension(:) :: basins_in_coarse_cell_lat_temp
  integer, pointer, dimension(:) :: basins_in_coarse_cell_lon_temp
  integer, pointer, dimension(:) :: basins_in_coarse_cell_lat
  integer, pointer, dimension(:) :: basins_in_coarse_cell_lon
  type(basinlist), pointer, dimension(:) :: basins_temp
  integer :: nlat_in,nlon_in
  integer :: nlat_coarse_in,nlon_coarse_in
  integer :: basin_number
  integer :: number_of_basins_in_coarse_cell
  integer :: i,j,k,l,m
  integer :: lat_scale_factor,lon_scale_factor
    number_of_basins_in_coarse_cell = 0
    this%lake_centers => lake_centers_in
    this%connection_volume_thresholds => connection_volume_thresholds_in
    this%flood_volume_thresholds => flood_volume_thresholds_in
    this%flood_local_redirect => flood_local_redirect_in
    this%connect_local_redirect => connect_local_redirect_in
    this%additional_flood_local_redirect => additional_flood_local_redirect_in
    this%additional_connect_local_redirect => additional_connect_local_redirect_in
    this%merge_points => merge_points_in
    this%flood_next_cell_lat_index => flood_next_cell_lat_index_in
    this%flood_next_cell_lon_index => flood_next_cell_lon_index_in
    this%connect_next_cell_lat_index => connect_next_cell_lat_index_in
    this%connect_next_cell_lon_index => connect_next_cell_lon_index_in
    this%flood_force_merge_lat_index => flood_force_merge_lat_index_in
    this%flood_force_merge_lon_index => flood_force_merge_lon_index_in
    this%connect_force_merge_lat_index => connect_force_merge_lat_index_in
    this%connect_force_merge_lon_index => connect_force_merge_lon_index_in
    this%flood_redirect_lat_index => flood_redirect_lat_index_in
    this%flood_redirect_lon_index => flood_redirect_lon_index_in
    this%connect_redirect_lat_index => connect_redirect_lat_index_in
    this%connect_redirect_lon_index => connect_redirect_lon_index_in
    this%additional_flood_redirect_lat_index => additional_flood_redirect_lat_index_in
    this%additional_flood_redirect_lon_index => additional_flood_redirect_lon_index_in
    this%additional_connect_redirect_lat_index => additional_connect_redirect_lat_index_in
    this%additional_connect_redirect_lon_index => additional_connect_redirect_lon_index_in
    this%nlat = nlat_in
    this%nlon = nlon_in
    this%nlat_coarse = nlat_coarse_in
    this%nlon_coarse = nlon_coarse_in
    this%instant_throughflow = instant_throughflow_in
    this%lake_retention_coefficient = lake_retention_coefficient_in
    allocate(this%flood_only(nlat_in,nlon_in))
    this%flood_only(:,:) = .false.
    where (this%connection_volume_thresholds == -1.0)
      this%flood_only = .true.
    else where
      this%flood_only = .false.
    end where
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
            basinlist(basins_in_coarse_cell_lat,basins_in_coarse_cell_lon)
          this%basin_numbers(i,j) = basin_number
        end if
      end do
    end do
    allocate(this%basins(basin_number))
    do m=1,basin_number
      this%basins(m) = basins_temp(m)
    end do
    deallocate(basins_temp)
    deallocate(basins_in_coarse_cell_lat_temp)
    deallocate(basins_in_coarse_cell_lon_temp)
end subroutine initialiselakeparameters

function lakeparametersconstructor(lake_centers_in, &
                                   connection_volume_thresholds_in, &
                                   flood_volume_thresholds_in, &
                                   flood_local_redirect_in, &
                                   connect_local_redirect_in, &
                                   additional_flood_local_redirect_in, &
                                   additional_connect_local_redirect_in, &
                                   merge_points_in, &
                                   flood_next_cell_lat_index_in, &
                                   flood_next_cell_lon_index_in, &
                                   connect_next_cell_lat_index_in, &
                                   connect_next_cell_lon_index_in, &
                                   flood_force_merge_lat_index_in, &
                                   flood_force_merge_lon_index_in, &
                                   connect_force_merge_lat_index_in, &
                                   connect_force_merge_lon_index_in, &
                                   flood_redirect_lat_index_in, &
                                   flood_redirect_lon_index_in, &
                                   connect_redirect_lat_index_in, &
                                   connect_redirect_lon_index_in, &
                                   additional_flood_redirect_lat_index_in, &
                                   additional_flood_redirect_lon_index_in, &
                                   additional_connect_redirect_lat_index_in, &
                                   additional_connect_redirect_lon_index_in, &
                                   nlat_in,nlon_in, &
                                   nlat_coarse_in,nlon_coarse_in, &
                                   instant_throughflow_in, &
                                   lake_retention_coefficient_in) result(constructor)
  type(lakeparameters), pointer :: constructor
  logical, pointer, dimension(:,:), intent(in) :: lake_centers_in
  real, pointer, dimension(:,:), intent(in) :: connection_volume_thresholds_in
  real, pointer, dimension(:,:), intent(in) :: flood_volume_thresholds_in
  logical, pointer, dimension(:,:), intent(in) :: flood_local_redirect_in
  logical, pointer, dimension(:,:), intent(in) :: connect_local_redirect_in
  logical, pointer, dimension(:,:), intent(in) :: additional_flood_local_redirect_in
  logical, pointer, dimension(:,:), intent(in) :: additional_connect_local_redirect_in
  integer, pointer, dimension(:,:), intent(in) :: merge_points_in
  integer, pointer, dimension(:,:), intent(in) :: flood_next_cell_lat_index_in
  integer, pointer, dimension(:,:), intent(in) :: flood_next_cell_lon_index_in
  integer, pointer, dimension(:,:), intent(in) :: connect_next_cell_lat_index_in
  integer, pointer, dimension(:,:), intent(in) :: connect_next_cell_lon_index_in
  integer, pointer, dimension(:,:), intent(in) :: flood_force_merge_lat_index_in
  integer, pointer, dimension(:,:), intent(in) :: flood_force_merge_lon_index_in
  integer, pointer, dimension(:,:), intent(in) :: connect_force_merge_lat_index_in
  integer, pointer, dimension(:,:), intent(in) :: connect_force_merge_lon_index_in
  integer, pointer, dimension(:,:), intent(in) :: flood_redirect_lat_index_in
  integer, pointer, dimension(:,:), intent(in) :: flood_redirect_lon_index_in
  integer, pointer, dimension(:,:), intent(in) :: connect_redirect_lat_index_in
  integer, pointer, dimension(:,:), intent(in) :: connect_redirect_lon_index_in
  integer, pointer, dimension(:,:), intent(in) :: additional_flood_redirect_lat_index_in
  integer, pointer, dimension(:,:), intent(in) :: additional_flood_redirect_lon_index_in
  integer, pointer, dimension(:,:), intent(in) :: additional_connect_redirect_lat_index_in
  integer, pointer, dimension(:,:), intent(in) :: additional_connect_redirect_lon_index_in
  integer, intent(in) :: nlat_in,nlon_in
  integer, intent(in) :: nlat_coarse_in,nlon_coarse_in
  logical, intent(in) :: instant_throughflow_in
  real, intent(in) :: lake_retention_coefficient_in
    allocate(constructor)
    call constructor%initialiselakeparameters(lake_centers_in, &
                                              connection_volume_thresholds_in, &
                                              flood_volume_thresholds_in, &
                                              flood_local_redirect_in, &
                                              connect_local_redirect_in, &
                                              additional_flood_local_redirect_in, &
                                              additional_connect_local_redirect_in, &
                                              merge_points_in, &
                                              flood_next_cell_lat_index_in, &
                                              flood_next_cell_lon_index_in, &
                                              connect_next_cell_lat_index_in, &
                                              connect_next_cell_lon_index_in, &
                                              flood_force_merge_lat_index_in, &
                                              flood_force_merge_lon_index_in, &
                                              connect_force_merge_lat_index_in, &
                                              connect_force_merge_lon_index_in, &
                                              flood_redirect_lat_index_in, &
                                              flood_redirect_lon_index_in, &
                                              connect_redirect_lat_index_in, &
                                              connect_redirect_lon_index_in, &
                                              additional_flood_redirect_lat_index_in, &
                                              additional_flood_redirect_lon_index_in, &
                                              additional_connect_redirect_lat_index_in, &
                                              additional_connect_redirect_lon_index_in, &
                                              nlat_in,nlon_in, &
                                              nlat_coarse_in,nlon_coarse_in, &
                                              instant_throughflow_in, &
                                              lake_retention_coefficient_in)
end function lakeparametersconstructor

subroutine lakeparametersdestructor(this)
  class(lakeparameters),intent(inout) :: this
  integer :: i
    deallocate(this%flood_only)
    deallocate(this%basin_numbers)
    do i = 1,size(this%basins)
      deallocate(this%basins(i)%basin_lats)
      deallocate(this%basins(i)%basin_lons)
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
    deallocate(this%flood_next_cell_lat_index)
    deallocate(this%flood_next_cell_lon_index)
    deallocate(this%connect_next_cell_lat_index)
    deallocate(this%connect_next_cell_lon_index)
    deallocate(this%flood_force_merge_lat_index)
    deallocate(this%flood_force_merge_lon_index)
    deallocate(this%connect_force_merge_lat_index)
    deallocate(this%connect_force_merge_lon_index)
    deallocate(this%flood_redirect_lat_index)
    deallocate(this%flood_redirect_lon_index)
    deallocate(this%connect_redirect_lat_index)
    deallocate(this%connect_redirect_lon_index)
    deallocate(this%additional_flood_redirect_lat_index)
    deallocate(this%additional_flood_redirect_lon_index)
    deallocate(this%additional_connect_redirect_lat_index)
    deallocate(this%additional_connect_redirect_lon_index)
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
      allocate(this%completed_lake_cells(lake_parameters%nlat,lake_parameters%nlon))
      this%completed_lake_cells(:,:) = .false.
      allocate(this%lake_numbers(lake_parameters%nlat,lake_parameters%nlon))
      this%lake_numbers(:,:) = 0
      allocate(this%water_to_lakes(lake_parameters%nlat_coarse,lake_parameters%nlon_coarse))
      this%water_to_lakes(:,:) = 0.0
      allocate(this%water_to_hd(lake_parameters%nlat_coarse,lake_parameters%nlon_coarse))
      this%water_to_hd(:,:) = 0.0
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
    deallocate(this%completed_lake_cells)
    deallocate(this%lake_numbers)
    deallocate(this%water_to_lakes)
    deallocate(this%water_to_hd)
    deallocate(this%cells_with_lakes_lat)
    deallocate(this%cells_with_lakes_lon)
end subroutine lakefieldsdestructor

subroutine initialiselakeprognostics(this,lake_parameters_in,lake_fields_in)
  class(lakeprognostics), intent(inout) :: this
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), intent(inout) :: lake_fields_in
  type(lakepointer), pointer, dimension(:) :: lakes_temp
  class(lake), pointer :: lake_temp
  integer :: lake_number
  integer :: i,j
    allocate(lakes_temp(lake_parameters_in%nlat*lake_parameters_in%nlon))
    lake_number = 0
    do j=1,lake_parameters_in%nlon
      do i=1,lake_parameters_in%nlat
        if(lake_parameters_in%lake_centers(i,j)) then
          lake_number = lake_number + 1
          lake_temp => FillingLake(lake_parameters_in,lake_fields_in,&
                                   i,j,i,j,lake_number,0.0)
          lakes_temp(lake_number) = lakepointer(lake_temp)
          lake_fields_in%lake_numbers(i,j) = lake_number
        end if
      end do
    end do
    allocate(this%lakes(lake_number))
    do i=1,lake_number
      this%lakes(i) = lakes_temp(i)
    end do
    allocate(this%lock_lake_memory(lake_number))
    this%lock_lake_memory(:) = .false.
    lake_fields_in%other_lakes => this%lakes
    lake_fields_in%lock_lake_memory => this%lock_lake_memory
    deallocate(lakes_temp)
end subroutine initialiselakeprognostics

function lakeprognosticsconstructor(lake_parameters_in,lake_fields_in) result(constructor)
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), intent(inout) :: lake_fields_in
  type(lakeprognostics), pointer :: constructor
    allocate(constructor)
    call constructor%initialiselakeprognostics(lake_parameters_in,lake_fields_in)
end function lakeprognosticsconstructor

subroutine lakeprognosticsdestructor(this)
  class(lakeprognostics), intent(inout) :: this
  integer :: i
    do i = 1,size(this%lakes)
      deallocate(this%lakes(i)%lake_pointer)
    end do
    deallocate(this%lakes)
    deallocate(this%lock_lake_memory)
end subroutine lakeprognosticsdestructor

subroutine initialisefillinglake(this,lake_parameters_in,lake_fields_in,center_cell_lat_in, &
                                 center_cell_lon_in,current_cell_to_fill_lat_in, &
                                 current_cell_to_fill_lon_in,lake_number_in, &
                                 lake_volume_in)
  class(fillinglake) :: this
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), intent(inout) :: lake_fields_in
  integer, intent(in) :: center_cell_lat_in
  integer, intent(in) :: center_cell_lon_in
  integer, intent(in) :: current_cell_to_fill_lat_in
  integer, intent(in) :: current_cell_to_fill_lon_in
  integer, intent(in) :: lake_number_in
  real,    intent(in) :: lake_volume_in
    call this%initialiselake(center_cell_lat_in,center_cell_lon_in, &
                             current_cell_to_fill_lat_in, &
                             current_cell_to_fill_lon_in, &
                             lake_number_in, &
                             lake_volume_in, &
                             lake_parameters_in, &
                             lake_fields_in)
    this%primary_merge_completed = .false.
    this%use_additional_fields = .false.
end subroutine initialisefillinglake

function fillinglakeconstructor(lake_parameters_in,lake_fields_in,center_cell_lat_in, &
                                center_cell_lon_in,current_cell_to_fill_lat_in, &
                                current_cell_to_fill_lon_in,&
                                lake_number_in,lake_volume_in) result(constructor)
  type(fillinglake), pointer :: constructor
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), intent(inout) :: lake_fields_in
  integer, intent(in) :: center_cell_lat_in
  integer, intent(in) :: center_cell_lon_in
  integer, intent(in) :: current_cell_to_fill_lat_in
  integer, intent(in) :: current_cell_to_fill_lon_in
  integer, intent(in) :: lake_number_in
  real,    intent(in) :: lake_volume_in
    allocate(constructor)
    call constructor%initialisefillinglake(lake_parameters_in, &
                                           lake_fields_in, &
                                           center_cell_lat_in, &
                                           center_cell_lon_in, &
                                           current_cell_to_fill_lat_in, &
                                           current_cell_to_fill_lon_in, &
                                           lake_number_in,lake_volume_in)
end function fillinglakeconstructor

subroutine initialiseoverflowinglake(this,outflow_redirect_lat_in, &
                                     outflow_redirect_lon_in, &
                                     local_redirect_in, &
                                     lake_retention_coefficient_in, &
                                     lake_parameters_in,lake_fields_in, &
                                     center_cell_lat_in, &
                                     center_cell_lon_in,&
                                     current_cell_to_fill_lat_in, &
                                     current_cell_to_fill_lon_in, &
                                     next_merge_target_lat_in, &
                                     next_merge_target_lon_in, &
                                     lake_number_in,lake_volume_in)
  class(overflowinglake) :: this
  integer, intent(in) :: outflow_redirect_lat_in
  integer, intent(in) :: outflow_redirect_lon_in
  logical, intent(in) :: local_redirect_in
  real, intent(in)    :: lake_retention_coefficient_in
  integer, intent(in) :: center_cell_lat_in
  integer, intent(in) :: center_cell_lon_in
  integer, intent(in) :: current_cell_to_fill_lat_in
  integer, intent(in) :: current_cell_to_fill_lon_in
  integer, intent(in) :: next_merge_target_lat_in
  integer, intent(in) :: next_merge_target_lon_in
  integer, intent(in) :: lake_number_in
  real,    intent(in) :: lake_volume_in
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), intent(inout) :: lake_fields_in
    call this%initialiselake(center_cell_lat_in,center_cell_lon_in, &
                             current_cell_to_fill_lat_in, &
                             current_cell_to_fill_lon_in, &
                             lake_number_in,lake_volume_in, &
                             lake_parameters_in, &
                             lake_fields_in)
    this%outflow_redirect_lat = outflow_redirect_lat_in
    this%outflow_redirect_lon = outflow_redirect_lon_in
    this%next_merge_target_lat = next_merge_target_lat_in
    this%next_merge_target_lon = next_merge_target_lon_in
    this%local_redirect = local_redirect_in
    this%lake_retention_coefficient = lake_retention_coefficient_in
    this%excess_water = 0.0
end subroutine initialiseoverflowinglake

function overflowinglakeconstructor(outflow_redirect_lat_in, &
                                    outflow_redirect_lon_in, &
                                    local_redirect_in, &
                                    lake_retention_coefficient_in, &
                                    lake_parameters_in,lake_fields_in, &
                                    center_cell_lat_in, &
                                    center_cell_lon_in, &
                                    current_cell_to_fill_lat_in, &
                                    current_cell_to_fill_lon_in, &
                                    next_merge_target_lat_in, &
                                    next_merge_target_lon_in, &
                                    lake_number_in,lake_volume_in) result(constructor)
  class(overflowinglake), pointer :: constructor
  integer, intent(in) :: outflow_redirect_lat_in
  integer, intent(in) :: outflow_redirect_lon_in
  logical, intent(in) :: local_redirect_in
  real, intent(in)    :: lake_retention_coefficient_in
  integer, intent(in) :: center_cell_lat_in
  integer, intent(in) :: center_cell_lon_in
  integer, intent(in) :: current_cell_to_fill_lat_in
  integer, intent(in) :: current_cell_to_fill_lon_in
  integer, intent(in) :: next_merge_target_lat_in
  integer, intent(in) :: next_merge_target_lon_in
  integer, intent(in) :: lake_number_in
  real,    intent(in) :: lake_volume_in
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), intent(inout) :: lake_fields_in
    allocate(constructor)
    call constructor%initialiseoverflowinglake(outflow_redirect_lat_in, &
                                               outflow_redirect_lon_in, &
                                               local_redirect_in, &
                                               lake_retention_coefficient_in, &
                                               lake_parameters_in,lake_fields_in, &
                                               center_cell_lat_in, &
                                               center_cell_lon_in, &
                                               current_cell_to_fill_lat_in, &
                                               current_cell_to_fill_lon_in, &
                                               next_merge_target_lat_in, &
                                               next_merge_target_lon_in, &
                                               lake_number_in,lake_volume_in)
end function overflowinglakeconstructor

subroutine initialisesubsumedlake(this,lake_parameters_in,lake_fields_in, &
                                  primary_lake_number_in,center_cell_lat_in, &
                                  center_cell_lon_in,current_cell_to_fill_lat_in, &
                                  current_cell_to_fill_lon_in,lake_number_in,&
                                  lake_volume_in)
  class(subsumedlake) :: this
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), intent(inout) :: lake_fields_in
  integer, intent(in) :: primary_lake_number_in
  integer, intent(in) :: center_cell_lat_in
  integer, intent(in) :: center_cell_lon_in
  integer, intent(in) :: current_cell_to_fill_lat_in
  integer, intent(in) :: current_cell_to_fill_lon_in
  integer, intent(in) :: lake_number_in
  real,    intent(in) :: lake_volume_in
    this%primary_lake_number = primary_lake_number_in
    call this%initialiselake(center_cell_lat_in,center_cell_lon_in, &
                             current_cell_to_fill_lat_in, &
                             current_cell_to_fill_lon_in, &
                             lake_number_in,lake_volume_in, &
                             lake_parameters_in,lake_fields_in)
end subroutine initialisesubsumedlake

function subsumedlakeconstructor(lake_parameters_in,lake_fields_in, &
                                 primary_lake_number_in, &
                                 center_cell_lat_in, &
                                 center_cell_lon_in, &
                                 current_cell_to_fill_lat_in, &
                                 current_cell_to_fill_lon_in, &
                                 lake_number_in,lake_volume_in) result(constructor)
  class(subsumedlake), pointer :: constructor
  type(lakeparameters), intent(in) :: lake_parameters_in
  type(lakefields), intent(inout) :: lake_fields_in
  integer, intent(in) :: primary_lake_number_in
  integer, intent(in) :: center_cell_lat_in
  integer, intent(in) :: center_cell_lon_in
  integer, intent(in) :: current_cell_to_fill_lat_in
  integer, intent(in) :: current_cell_to_fill_lon_in
  integer, intent(in) :: lake_number_in
  real,    intent(in) :: lake_volume_in
    allocate(constructor)
    call constructor%initialisesubsumedlake(lake_parameters_in,lake_fields_in, &
                                            primary_lake_number_in, &
                                            center_cell_lat_in, &
                                            center_cell_lon_in, &
                                            current_cell_to_fill_lat_in, &
                                            current_cell_to_fill_lon_in, &
                                            lake_number_in,lake_volume_in)
end function subsumedlakeconstructor

subroutine setup_lakes(lake_parameters,lake_prognostics,lake_fields,initial_water_to_lake_centers)
  type(lakeparameters), intent(in) :: lake_parameters
  type(lakeprognostics), intent(inout) :: lake_prognostics
  type(lakefields), intent(inout) :: lake_fields
  real, pointer, dimension(:,:), intent(in) :: initial_water_to_lake_centers
  class(lake), pointer :: working_lake
  class(lake), pointer :: updated_lake
  real :: initial_water_to_lake_center
  integer :: lake_index
  integer :: i,j
    do j=1,lake_parameters%nlon
      do i=1,lake_parameters%nlat
        if(lake_parameters%lake_centers(i,j)) then
          initial_water_to_lake_center = initial_water_to_lake_centers(i,j)
          if(initial_water_to_lake_center > 0.0) then
            lake_index = lake_fields%lake_numbers(i,j)
            working_lake => lake_prognostics%lakes(lake_index)%lake_pointer
            lake_prognostics%lock_lake_memory(lake_index) = .true.
            updated_lake => working_lake%add_water(initial_water_to_lake_center)
            lake_prognostics%lakes(lake_index)%lake_pointer => updated_lake
            lake_prognostics%lock_lake_memory(lake_index) = .false.
            if (.not. associated(working_lake,updated_lake)) then
              deallocate(working_lake)
            end if
          end if
        end if
      end do
    end do
end subroutine setup_lakes

subroutine run_lakes(lake_parameters,lake_prognostics,lake_fields)
  type(lakeparameters), intent(in) :: lake_parameters
  type(lakeprognostics), intent(inout) :: lake_prognostics
  type(lakefields), intent(inout) :: lake_fields
  integer :: lat,lon
  integer :: basin_center_lat,basin_center_lon
  integer :: lake_index
  integer :: i,j
  integer :: num_basins_in_cell
  integer :: basin_number
  real :: share_to_each_lake
  class(lake), pointer :: working_lake
  class(lake), pointer :: updated_lake
    lake_fields%water_to_hd = 0.0
    do i = 1,size(lake_fields%cells_with_lakes_lat)
      lat = lake_fields%cells_with_lakes_lat(i)
      lon = lake_fields%cells_with_lakes_lon(i)
      if (lake_fields%water_to_lakes(lat,lon) > 0.0) then
        basin_number = lake_parameters%basin_numbers(lat,lon)
        num_basins_in_cell = &
          size(lake_parameters%basins(basin_number)%basin_lats)
        share_to_each_lake = lake_fields%water_to_lakes(lat,lon)/num_basins_in_cell
        do j = 1,num_basins_in_cell
          basin_center_lat = lake_parameters%basins(basin_number)%basin_lats(j)
          basin_center_lon = lake_parameters%basins(basin_number)%basin_lons(j)
          lake_index = lake_fields%lake_numbers(basin_center_lat,basin_center_lon)
          working_lake => lake_prognostics%lakes(lake_index)%lake_pointer
          lake_prognostics%lock_lake_memory(lake_index) = .true.
          updated_lake => working_lake%add_water(share_to_each_lake)
          lake_prognostics%lakes(lake_index)%lake_pointer => updated_lake
          lake_prognostics%lock_lake_memory(lake_index) = .false.
          if (.not. associated(working_lake,updated_lake)) then
            deallocate(working_lake)
          end if
        end do
      end if
    end do
    do i = 1,size(lake_prognostics%lakes)
      working_lake => lake_prognostics%lakes(i)%lake_pointer
      if (working_lake%unprocessed_water > 0.0) then
          lake_prognostics%lock_lake_memory(i) = .true.
          updated_lake => working_lake%add_water(0.0)
          lake_prognostics%lakes(i)%lake_pointer => updated_lake
          lake_prognostics%lock_lake_memory(i) = .false.
          if (.not. associated(working_lake,updated_lake)) then
            deallocate(working_lake)
            working_lake => lake_prognostics%lakes(i)%lake_pointer
          end if
      end if
      select type(working_lake)
        type is (overflowinglake)
          call working_lake%drain_excess_water()
      end select
    end do
end subroutine run_lakes

function fillinglake_add_water(this,inflow) result(updated_lake)
  class(fillinglake), target, intent(inout) :: this
  real, intent(in) :: inflow
  class(lake), pointer :: updated_lake
  class(lake),  pointer :: other_lake
  type(fillinglake), pointer :: other_lake_as_filling_lake
  integer :: target_cell_lat,target_cell_lon
  integer :: merge_type
  integer :: other_lake_number
  logical :: filled
  logical :: merge_possible
  logical :: already_merged
  real :: inflow_local
    inflow_local = inflow + this%unprocessed_water
    this%unprocessed_water = 0.0
    do while (inflow_local > 0.0)
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
                  updated_lake => this%perform_secondary_merge()
                  updated_lake => updated_lake%store_water(inflow_local)
                  return
                else
                  call this%perform_primary_merge()
                end if
              else if (.not. already_merged) then
                updated_lake => this%change_to_overflowing_lake(merge_type)
                updated_lake => updated_lake%store_water(inflow_local)
                return
              else if (merge_type == secondary_merge) then
                call this%get_secondary_merge_coords(target_cell_lat,target_cell_lon)
                other_lake_number = &
                  this%lake_fields%lake_numbers(target_cell_lat,target_cell_lon)
                other_lake => &
                  this%lake_fields%other_lakes(other_lake_number)%lake_pointer
                select type (other_lake)
                type is (subsumedlake)
                  other_lake_as_filling_lake => &
                    other_lake%change_subsumed_lake_to_filling_lake()
                end select
                this%lake_fields%other_lakes(other_lake_number)%lake_pointer => &
                  other_lake_as_filling_lake%change_to_overflowing_lake(primary_merge)
                if (.not. this%lake_fields%lock_lake_memory(other_lake_number)) deallocate(other_lake)
                deallocate(other_lake_as_filling_lake)
                updated_lake => this%perform_secondary_merge()
                updated_lake => updated_lake%store_water(inflow_local)
                return
              end if
            end if
            if (merge_type /= double_merge) exit
          end do
        end if
        call this%update_filling_cell()
      end if
    end do
    updated_lake => this
end function fillinglake_add_water

function overflowinglake_add_water(this,inflow) result(updated_lake)
  class(overflowinglake), intent(inout), target :: this
  real, intent(in) :: inflow
  class(lake), pointer :: updated_lake
  class(lake), pointer :: other_lake
  integer :: other_lake_number
  real :: inflow_local
  logical :: locking_other_lake_memory
    inflow_local = inflow + this%unprocessed_water
    this%unprocessed_water = 0.0
    if (this%local_redirect) then
      other_lake_number  = this%lake_fields%lake_numbers(this%outflow_redirect_lat, &
                                                         this%outflow_redirect_lon)
      other_lake => &
        this%lake_fields%other_lakes(other_lake_number)%lake_pointer
      if (this%lake_parameters%instant_throughflow) then
        if (.not. this%lake_fields%lock_lake_memory(other_lake_number)) then
          this%lake_fields%lock_lake_memory(other_lake_number) = .true.
          locking_other_lake_memory = .true.
        else
          locking_other_lake_memory = .false.
        end if
        updated_lake => other_lake%add_water(inflow_local)
        this%lake_fields%other_lakes(other_lake_number)%lake_pointer => &
          updated_lake
        if (locking_other_lake_memory) then
         this%lake_fields%lock_lake_memory(other_lake_number) = .false.
          if (.not. associated(other_lake,updated_lake)) then
            deallocate(other_lake)
          end if
        end if
      else
        this%lake_fields%other_lakes(other_lake_number)%lake_pointer => &
          other_lake%store_water(inflow_local)
      end if
    else
      this%excess_water = this%excess_water + inflow_local
    end if
    updated_lake => this%lake_fields%other_lakes(this%lake_number)%lake_pointer
end function overflowinglake_add_water

function store_water(this,inflow) result(lake_after_storage)
  class(lake), target, intent(inout) :: this
  real, intent(in) :: inflow
  class(lake), pointer :: lake_after_storage
    this%unprocessed_water = this%unprocessed_water + inflow
    lake_after_storage => this
end function store_water

function subsumedlake_add_water(this,inflow) result(updated_lake)
  class(subsumedlake), intent(inout), target :: this
  real, intent(in) :: inflow
  class(lake), pointer :: updated_lake
  class(lake), pointer :: other_lake
  real :: inflow_local
  logical :: locking_other_lake_memory
    inflow_local = inflow + this%unprocessed_water
    this%unprocessed_water = 0.0
    other_lake => this%lake_fields%other_lakes(this%primary_lake_number)%lake_pointer
    if (.not. this%lake_fields%lock_lake_memory(this%primary_lake_number)) then
      this%lake_fields%lock_lake_memory(this%primary_lake_number) = .true.
      locking_other_lake_memory = .true.
    else
      locking_other_lake_memory = .false.
    end if
    updated_lake => &
      other_lake%add_water(inflow_local)
    this%lake_fields%other_lakes(this%primary_lake_number)%lake_pointer => &
      updated_lake
    if (locking_other_lake_memory) then
      this%lake_fields%lock_lake_memory(this%primary_lake_number) = .false.
      if (.not. associated(other_lake,updated_lake)) then
        deallocate(other_lake)
      end if
    end if
    updated_lake => this
end function subsumedlake_add_water

function accept_merge(this,redirect_coords_lat,redirect_coords_lon) &
    result(subsumed_lake)
  class(lake), intent(inout) :: this
  integer, intent(in) :: redirect_coords_lat
  integer, intent(in) :: redirect_coords_lon
  class(lake), pointer :: subsumed_lake
    this%lake_fields%completed_lake_cells(this%current_cell_to_fill_lat,&
                                          this%current_cell_to_fill_lon) = .true.
    subsumed_lake => subsumedlake(this%lake_parameters, &
                                  this%lake_fields, &
                                  this%lake_fields%lake_numbers(redirect_coords_lat, &
                                                                redirect_coords_lon), &
                                  this%center_cell_lat, &
                                  this%center_cell_lon, &
                                  this%current_cell_to_fill_lat, &
                                  this%current_cell_to_fill_lon, &
                                  this%lake_number,this%lake_volume)
    select type(this)
      type is (overflowinglake)
        subsumed_lake => subsumed_lake%store_water(this%excess_water)
    end select
end function accept_merge

function change_to_overflowing_lake(this,merge_type) result(overflowing_lake)
  class(fillinglake), intent(inout) :: this
  type(overflowinglake), pointer :: overflowing_lake
  integer :: outflow_redirect_lat
  integer :: outflow_redirect_lon
  integer :: merge_type
  integer :: target_cell_lat,target_cell_lon
  logical :: local_redirect
    this%lake_fields%completed_lake_cells(this%current_cell_to_fill_lat, &
                                          this%current_cell_to_fill_lon) = .true.
    call this%get_outflow_redirect_coords(outflow_redirect_lat, &
                                          outflow_redirect_lon, &
                                          local_redirect)
    if (merge_type == secondary_merge) then
      call this%get_secondary_merge_coords(target_cell_lat,target_cell_lon)
    else
      call this%get_primary_merge_coords(target_cell_lat,target_cell_lon)
    end if
    overflowing_lake => overflowinglake(outflow_redirect_lat, &
                                        outflow_redirect_lon, &
                                        local_redirect, &
                                        this%lake_parameters%lake_retention_coefficient, &
                                        this%lake_parameters,this%lake_fields, &
                                        this%center_cell_lat, &
                                        this%center_cell_lon, &
                                        this%current_cell_to_fill_lat, &
                                        this%current_cell_to_fill_lon, &
                                        target_cell_lat, &
                                        target_cell_lon, &
                                        this%lake_number, &
                                        this%lake_volume)
end function change_to_overflowing_lake

subroutine drain_excess_water(this)
  class(overflowinglake), intent(inout) :: this
  real :: flow
    if (this%excess_water > 0.0) then
      this%excess_water = this%excess_water + this%unprocessed_water
      this%unprocessed_water = 0.0
      flow = this%excess_water*this%lake_retention_coefficient
      this%lake_fields%water_to_hd(this%outflow_redirect_lat,this%outflow_redirect_lon) = &
           this%lake_fields%water_to_hd(this%outflow_redirect_lat, &
                                        this%outflow_redirect_lon)+flow
      this%excess_water = this%excess_water - flow
    end if
end subroutine drain_excess_water

function fill_current_cell(this,inflow) result(filled)
  class(fillinglake), intent(inout) :: this
  real, intent(inout) :: inflow
  logical :: filled
  real :: new_lake_volume
  real :: maximum_new_lake_volume
    new_lake_volume = inflow + this%lake_volume
    if (this%lake_fields%completed_lake_cells(this%current_cell_to_fill_lat, &
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
      inflow = 0.0
      filled = .false.
    else
      inflow = new_lake_volume - maximum_new_lake_volume
      this%lake_volume = maximum_new_lake_volume
      filled = .true.
    end if
end function fill_current_cell

function get_merge_type(this) result(simple_merge_type)
  class(lake), intent(inout) :: this
  integer :: simple_merge_type
  integer :: extended_merge_type
    extended_merge_type = this%lake_parameters%merge_points(this%current_cell_to_fill_lat, &
                                                            this%current_cell_to_fill_lon)
    if (this%lake_fields%completed_lake_cells(this%current_cell_to_fill_lat, &
                                              this%current_cell_to_fill_lon) .or. &
        this%lake_parameters%flood_only(this%current_cell_to_fill_lat, &
                                        this%current_cell_to_fill_lon)) then
      simple_merge_type = convert_to_simple_merge_type_flood(Int(extended_merge_type)+1)
    else
      simple_merge_type = convert_to_simple_merge_type_connect(Int(extended_merge_type)+1)
    end if
end function get_merge_type

function check_if_merge_is_possible(this,simple_merge_type,already_merged) &
   result(merge_possible)
  class(lake), intent(in) :: this
  integer, intent(in) :: simple_merge_type
  logical, intent(out) :: already_merged
  logical :: merge_possible
  class(lake), pointer :: other_lake
  class(lake), pointer :: other_lake_target_lake
  class(lake), pointer :: other_lake_target_lake_true_primary
  integer :: target_cell_lat
  integer :: target_cell_lon
  integer :: other_lake_target_lake_number
    if (simple_merge_type == secondary_merge) then
      call this%get_secondary_merge_coords(target_cell_lat,target_cell_lon)
    else
      call this%get_primary_merge_coords(target_cell_lat,target_cell_lon)
    end if
    if (.not. this%lake_fields%completed_lake_cells(target_cell_lat,target_cell_lon)) then
      merge_possible = .false.
      already_merged = .false.
      return
    end if
    other_lake => this%lake_fields%&
                  &other_lakes(this%lake_fields%lake_numbers(target_cell_lat, &
                                                             target_cell_lon))%lake_pointer
    other_lake => other_lake%find_true_primary_lake()
    if (other_lake%lake_number == this%lake_number) then
      merge_possible = .false.
      already_merged = .true.
      return
    end if
    select type(other_lake)
    type is (overflowinglake)
      other_lake_target_lake_number = &
        this%lake_fields%lake_numbers(other_lake%next_merge_target_lat, &
                                      other_lake%next_merge_target_lon)
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
          select type (other_lake_target_lake)
          type is (overflowinglake)
            other_lake_target_lake_number = &
              this%lake_fields%lake_numbers(other_lake_target_lake%next_merge_target_lat,&
                                            other_lake_target_lake%next_merge_target_lon)
          type is (subsumedlake)
            other_lake_target_lake_true_primary => &
              other_lake_target_lake%find_true_primary_lake()
            other_lake_target_lake_number = &
              other_lake_target_lake_true_primary%lake_number
          class default
             merge_possible = .false.
             already_merged = .false.
             return
          end select
        end if
      end do
    class default
      merge_possible = .false.
      already_merged = .false.
    end select
end function check_if_merge_is_possible

subroutine perform_primary_merge(this)
  class(fillinglake), intent(inout) :: this
  class(lake), pointer :: other_lake
  integer :: target_cell_lat,target_cell_lon
  integer :: other_lake_number
    call this%get_primary_merge_coords(target_cell_lat,target_cell_lon)
    other_lake_number = this%lake_fields%lake_numbers(target_cell_lat,target_cell_lon)
    other_lake => this%lake_fields%other_lakes(other_lake_number)%lake_pointer
    other_lake => other_lake%find_true_primary_lake()
    other_lake_number = other_lake%lake_number
    this%primary_merge_completed = .true.
    this%lake_fields%other_lakes(other_lake_number)%lake_pointer => &
      other_lake%accept_merge(this%center_cell_lat,this%center_cell_lon)
    if(.not. this%lake_fields%lock_lake_memory(other_lake_number)) deallocate(other_lake)
end subroutine perform_primary_merge

function perform_secondary_merge(this) result(merged_lake)
  class(fillinglake), intent(inout) :: this
  class(lake), pointer :: merged_lake
  class(lake), pointer :: other_lake
  type(fillinglake), pointer :: other_lake_as_filling_lake
  integer :: target_cell_lat,target_cell_lon
  integer :: other_lake_number
    this%lake_fields%completed_lake_cells(this%current_cell_to_fill_lat, &
                                          this%current_cell_to_fill_lon) = .true.
    call this%get_secondary_merge_coords(target_cell_lat,target_cell_lon)
    other_lake_number = this%lake_fields%lake_numbers(target_cell_lat,target_cell_lon)
    other_lake => this%lake_fields%other_lakes(other_lake_number)%lake_pointer
    other_lake => other_lake%find_true_primary_lake()
    other_lake_number = other_lake%lake_number
    select type (other_lake)
    type is (overflowinglake)
      other_lake_as_filling_lake => other_lake%change_overflowing_lake_to_filling_lake()
    end select
    this%lake_fields%other_lakes(other_lake_number)%lake_pointer => &
                                            other_lake_as_filling_lake
    if(.not. this%lake_fields%lock_lake_memory(other_lake_number)) deallocate(other_lake)
    merged_lake => this%accept_merge(other_lake_as_filling_lake%center_cell_lat, &
                                     other_lake_as_filling_lake%center_cell_lon)
end function perform_secondary_merge

function change_overflowing_lake_to_filling_lake(this) result(filling_lake)
  class(overflowinglake), intent(inout) :: this
  type(fillinglake), pointer :: filling_lake
  logical :: use_additional_fields
    this%unprocessed_water = this%unprocessed_water + this%excess_water
    use_additional_fields = (this%get_merge_type() == double_merge)
    filling_lake => FillingLake(this%lake_parameters,this%lake_fields, &
                                this%center_cell_lat, &
                                this%center_cell_lon, &
                                this%current_cell_to_fill_lat, &
                                this%current_cell_to_fill_lon,&
                                this%lake_number,this%lake_volume)
    filling_lake%primary_merge_completed = .true.
    filling_lake%use_additional_fields = use_additional_fields
end function change_overflowing_lake_to_filling_lake

function change_subsumed_lake_to_filling_lake(this) result(filling_lake)
  class(subsumedlake), intent(inout) :: this
  type(fillinglake), pointer :: filling_lake
    filling_lake => FillingLake(this%lake_parameters,this%lake_fields, &
                                this%center_cell_lat, &
                                this%center_cell_lon, &
                                this%current_cell_to_fill_lat, &
                                this%current_cell_to_fill_lon,&
                                this%lake_number,this%lake_volume)
    filling_lake%primary_merge_completed = .false.
    filling_lake%use_additional_fields = .false.
end function change_subsumed_lake_to_filling_lake

subroutine update_filling_cell(this)
  class(fillinglake), intent(inout) :: this
  integer :: coords_lat, coords_lon
    this%primary_merge_completed = .false.
    coords_lat = this%current_cell_to_fill_lat
    coords_lon = this%current_cell_to_fill_lon
    if (this%lake_fields%completed_lake_cells(coords_lat,coords_lon) .or. &
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
    this%lake_fields%lake_numbers(this%current_cell_to_fill_lat, &
                            this%current_cell_to_fill_lon) = this%lake_number
    this%lake_fields%completed_lake_cells(coords_lat,coords_lon) = .true.
end subroutine update_filling_cell

subroutine get_primary_merge_coords(this,lat,lon)
  class(lake), intent(in) :: this
  integer, intent(out) :: lat
  integer, intent(out) :: lon
  logical :: completed_lake_cell
  completed_lake_cell = ((this%lake_fields%completed_lake_cells(this%current_cell_to_fill_lat, &
                                                                this%current_cell_to_fill_lon)) .or. &
                         (this%lake_parameters%flood_only(this%current_cell_to_fill_lat, &
                                                          this%current_cell_to_fill_lon)))
  if (completed_lake_cell) then
    lat = &
      this%lake_parameters%flood_force_merge_lat_index(this%current_cell_to_fill_lat, &
                                                       this%current_cell_to_fill_lon)
    lon = &
      this%lake_parameters%flood_force_merge_lon_index(this%current_cell_to_fill_lat, &
                                                  this%current_cell_to_fill_lon)
  else
    lat = &
      this%lake_parameters%connect_force_merge_lat_index(this%current_cell_to_fill_lat, &
                                                    this%current_cell_to_fill_lon)
    lon = &
      this%lake_parameters%connect_force_merge_lon_index(this%current_cell_to_fill_lat, &
                                                    this%current_cell_to_fill_lon)
  end if
end subroutine get_primary_merge_coords

subroutine get_secondary_merge_coords(this,lat,lon)
  class(lake), intent(in) :: this
  integer, intent(out) :: lat
  integer, intent(out) :: lon
  logical :: completed_lake_cell
  completed_lake_cell = ((this%lake_fields%completed_lake_cells(this%current_cell_to_fill_lat, &
                                                                this%current_cell_to_fill_lon)) .or. &
                         (this%lake_parameters%flood_only(this%current_cell_to_fill_lat, &
                                                          this%current_cell_to_fill_lon)))
  if (completed_lake_cell) then
    lat = &
      this%lake_parameters%flood_next_cell_lat_index(this%current_cell_to_fill_lat, &
                                                     this%current_cell_to_fill_lon)
    lon = &
      this%lake_parameters%flood_next_cell_lon_index(this%current_cell_to_fill_lat, &
                                                     this%current_cell_to_fill_lon)
  else
    lat = &
      this%lake_parameters%connect_next_cell_lat_index(this%current_cell_to_fill_lat, &
                                                       this%current_cell_to_fill_lon)
    lon = &
      this%lake_parameters%connect_next_cell_lon_index(this%current_cell_to_fill_lat, &
                                                       this%current_cell_to_fill_lon)
  end if
end subroutine get_secondary_merge_coords

subroutine get_outflow_redirect_coords(this,lat,lon,local_redirect)
  class(fillinglake), intent(in) :: this
  integer, intent(out) :: lat
  integer, intent(out) :: lon
  logical, intent(out) :: local_redirect
  logical :: completed_lake_cell
  completed_lake_cell = (this%lake_fields%completed_lake_cells(this%current_cell_to_fill_lat, &
                                                               this%current_cell_to_fill_lon) .or. &
                                                               this%lake_parameters%flood_only(this%&
                                                                         &current_cell_to_fill_lat, &
                                                               this%current_cell_to_fill_lon))
  if (this%use_additional_fields) then
    if (completed_lake_cell) then
      local_redirect = &
        this%lake_parameters%additional_flood_local_redirect(this%current_cell_to_fill_lat, &
                                                             this%current_cell_to_fill_lon)
    else
      local_redirect = &
        this%lake_parameters%additional_connect_local_redirect(this%current_cell_to_fill_lat, &
                                                               this%current_cell_to_fill_lon)
    end if
  else
    if (completed_lake_cell) then
      local_redirect = &
        this%lake_parameters%flood_local_redirect(this%current_cell_to_fill_lat, &
                                                  this%current_cell_to_fill_lon)
    else
      local_redirect = &
        this%lake_parameters%connect_local_redirect(this%current_cell_to_fill_lat, &
                                                    this%current_cell_to_fill_lon)
    end if
  end if
  if (this%use_additional_fields) then
    if (completed_lake_cell) then
      lat = &
        this%lake_parameters%additional_flood_redirect_lat_index(this%current_cell_to_fill_lat, &
                                                                 this%current_cell_to_fill_lon)
      lon = &
        this%lake_parameters%additional_flood_redirect_lon_index(this%current_cell_to_fill_lat, &
                                                                 this%current_cell_to_fill_lon)
    else
      lat = &
        this%lake_parameters%additional_connect_redirect_lat_index(this%current_cell_to_fill_lat, &
                                                                   this%current_cell_to_fill_lon)
      lon = &
        this%lake_parameters%additional_connect_redirect_lon_index(this%current_cell_to_fill_lat, &
                                                                   this%current_cell_to_fill_lon)
    end if
  else
    if (completed_lake_cell) then
      lat = &
        this%lake_parameters%flood_redirect_lat_index(this%current_cell_to_fill_lat, &
                                                      this%current_cell_to_fill_lon)
      lon = &
        this%lake_parameters%flood_redirect_lon_index(this%current_cell_to_fill_lat, &
                                                      this%current_cell_to_fill_lon)
    else
      lat = &
        this%lake_parameters%connect_redirect_lat_index(this%current_cell_to_fill_lat, &
                                                        this%current_cell_to_fill_lon)
      lon = &
        this%lake_parameters%connect_redirect_lon_index(this%current_cell_to_fill_lat, &
                                                        this%current_cell_to_fill_lon)
    end if
  end if
end subroutine get_outflow_redirect_coords

recursive function find_true_primary_lake(this) result(true_primary_lake)
  class(lake), target, intent(in) :: this
  class(lake), pointer :: true_primary_lake
    select type (this)
      type is (subsumedlake)
        true_primary_lake => find_true_primary_lake(this%lake_fields%other_lakes(this%primary_lake_number)%lake_pointer)
      class default
        true_primary_lake => this
    end select
end function find_true_primary_lake

end module latlon_lake_model_mod
