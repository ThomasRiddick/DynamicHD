module latlon_lake_model_mod

type, abstract :: Lake
  integer :: lake_number
  integer :: center_cell_lat
  integer :: center_cell_lon
  real    :: lake_volume
  real    :: unprocessed_water
  integer :: current_cell_to_fill_lat
  integer :: current_cell_to_fill_lon
  type(LakeParameters) :: lake_parameters
  type(LakesFields) :: lake_fields
  contains
    procedure(add_water), deferred :: add_water
    procedure :: accept_merge
    procedure :: store_water
    procedure :: find_true_primary_lake
    procedure :: get_merge_type
    procedure :: get_primary_merge_coords
    procedure :: get_secondary_merge_coords
    procedure :: InitialiseLake
end type Lake

abstract interface
  function add_water result(updated_lake)
    import Lake
    class(Lake), intent(in) :: this
    real, intent(in) :: inflow
    class(Lake) :: updated_lake
  end function add_water
end interface

type :: BasinList
  integer, allocatable, dimension(:) basin_lats
  integer, allocatable, dimension(:) basin_lons
end type BasinList

type :: LakeParameters
  logical :: instant_throughflow
  real :: lake_retention_coefficient
  logical, allocatable, dimension(:,:) :: lake_centers
  real, allocatable, dimension(:,:) :: connection_volume_thresholds
  real, allocatable, dimension(:,:) :: flood_volume_thresholds
  logical, allocatable, dimension(:,:) :: flood_only
  logical, allocatable, dimension(:,:) :: flood_local_redirect
  logical, allocatable, dimension(:,:) :: connect_local_redirect
  logical, allocatable, dimension(:,:) :: additional_flood_local_redirect
  logical, allocatable, dimension(:,:) :: additional_connect_local_redirect
  integer, allocatable, dimension(:,:) :: merge_points
  integer, allocatable, dimension(:,:) :: basin_numbers
  type(BasinList), allocatable, dimension(:) :: basins
  integer, allocatable, dimension(:,:) :: flood_next_cell_lat_index
  integer, allocatable, dimension(:,:) :: flood_next_cell_lon_index
  integer, allocatable, dimension(:,:) :: connect_next_cell_lat_index
  integer, allocatable, dimension(:,:) :: connect_next_cell_lon_index
  integer, allocatable, dimension(:,:) :: flood_force_merge_lat_index
  integer, allocatable, dimension(:,:) :: flood_force_merge_lon_index
  integer, allocatable, dimension(:,:) :: connect_force_merge_lat_index
  integer, allocatable, dimension(:,:) :: connect_force_merge_lon_index
  integer, allocatable, dimension(:,:) :: flood_redirect_lat_index
  integer, allocatable, dimension(:,:) :: flood_redirect_lon_index
  integer, allocatable, dimension(:,:) :: connect_redirect_lat_index
  integer, allocatable, dimension(:,:) :: connect_redirect_lon_index
  integer, allocatable, dimension(:,:) :: additional_flood_redirect_lat_index
  integer, allocatable, dimension(:,:) :: additional_flood_redirect_lon_index
  integer, allocatable, dimension(:,:) :: additional_connect_redirect_lat_index
  integer, allocatable, dimension(:,:) :: additional_connect_redirect_lon_index
  integer :: nlat,nlon
  integer :: nlat_coarse,nlon_coarse
  contains
     procedure :: InitialiseLakeParameters
end type LakeParameters

interface LakeParameters
  procedure LakeParametersConstructor
end interface LakeParameters

type :: LakePointer
  type(Lake), pointer :: lake_pointer
end type LakePointer

type :: LakeFields
  logical, allocatable, dimension(:,:) :: completed_lake_cells
  integer, allocatable, dimension(:,:) :: lake_numbers
  integer, allocatable, dimension(:,:) :: water_to_lakes
  integer, allocatable, dimension(:,:) :: water_to_hd
  integer, allocatable, dimension(:) :: cells_with_lakes_lat
  integer, allocatable, dimension(:) :: cells_with_lakes_lon
  type(LakePointer), allocatable, dimension(:) :: other_lakes
  contains
    procedure :: InitialiseLakeFields
end type LakeFields

interface LakeFields
  procedure LakesConstructor
end interface LakeFields

type LakePrognostics
  type(LakePointer), allocatable, dimension(:) :: lakes
  contains
    procedure :: InitialiseLakePrognostics
end type LakePrognostics

interface LakePrognostics
  procedure LakePrognosticsConstructor
end interface LakePrognostics

type, extends(Lake) :: FillingLake
  logical :: primary_merge_completed
  logical :: use_additional_fields
  contains
    procedure :: perform_primary_merge
    procedure :: perform_secondary_merge
    procedure :: fill_current_cell
    procedure :: change_to_overflowing_lake
    procedure :: update_filling_cell
    procedure :: get_outflow_redirect_coords
    procedure :: add_water => FillingLake_add_water
    procedure :: FillingLake
end type FillingLake

interface FillingLake
  procedure FillingLakeConstructor
end interface FillingLake

type, extends(Lake) :: OverflowingLake
  real :: excess_water
  integer :: outflow_redirect_lat
  integer :: outflow_redirect_lon
  logical :: local_redirect
  real    :: lake_retention_coefficient
  contains
    procedure :: change_to_filling_lake
    procedure :: drain_excess_water
    procedure :: add_water => OverflowingLake_add_water
    procedure :: InitialiseOverflowingLake
end type OverflowingLake


interface OverflowingLake
  procedure OverflowingLakeConstructor
end interface OverflowingLake

type, extends(Lake) :: SubsumedLake
  integer :: primary_lake_number
  contains
    procedure :: add_water => SubsumedLake_add_water
    procedure :: InitialiseSubsumedLake
end type SubsumedLake

interface SubsumedLake
  procedure SubsumedLakeConstructor
end interface SubsumedLake

subroutine InitialiseLake(this,center_cell_lat_in,center_cell_lon_in,
                          lake_number_in,lake_parameters_in,lake_fields_in)
  class(Lake),intent(inout) :: this
  integer, intent(in) :: center_cell_lat_in
  integer, intent(in) :: center_cell_lon_in
  integer, intent(in) :: lake_number_in
    this%lake_number = lake_number_in
    this%center_cell_lat = center_cell_lat_in
    this%center_cell_lon = center_cell_lon_in
    this%lake_volume = 0.0
    this%unprocessed_water = 0.0
    this%current_cell_to_fill_lat = center_cell_lat_in
    this%current_cell_to_fill_lon = center_cell_lon_in
    this%lake_parameters = lake_parameters_in
    this%lake_fields = lake_fields_in
end subroutine InitialiseLake

subroutine InitialiseLakeParameters(this,lake_centers_in, &
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
    class(LakeParameters),intent(inout) :: this
    logical, allocatable, dimension(:,:), intent(in) :: lake_centers_in
    real, allocatable, dimension(:,:), intent(in) :: connection_volume_thresholds_in
    real, allocatable, dimension(:,:), intent(in) :: flood_volume_thresholds_in
    logical, allocatable, dimension(:,:), intent(in) :: flood_local_redirect_in
    logical, allocatable, dimension(:,:), intent(in) :: connect_local_redirect_in
    logical, allocatable, dimension(:,:), intent(in) :: additional_flood_local_redirect_in
    logical, allocatable, dimension(:,:), intent(in) :: additional_connect_local_redirect_in
    integer, allocatable, dimension(:,:), intent(in) :: merge_points_in
    integer, allocatable, dimension(:,:), intent(in) :: flood_next_cell_lat_index_in
    integer, allocatable, dimension(:,:), intent(in) :: flood_next_cell_lon_index_in
    integer, allocatable, dimension(:,:), intent(in) :: connect_next_cell_lat_index_in
    integer, allocatable, dimension(:,:), intent(in) :: connect_next_cell_lon_index_in
    integer, allocatable, dimension(:,:), intent(in) :: flood_force_merge_lat_index_in
    integer, allocatable, dimension(:,:), intent(in) :: flood_force_merge_lon_index_in
    integer, allocatable, dimension(:,:), intent(in) :: connect_force_merge_lat_index_in
    integer, allocatable, dimension(:,:), intent(in) :: connect_force_merge_lon_index_in
    integer, allocatable, dimension(:,:), intent(in) :: flood_redirect_lat_index_in
    integer, allocatable, dimension(:,:), intent(in) :: flood_redirect_lon_index_in
    integer, allocatable, dimension(:,:), intent(in) :: connect_redirect_lat_index_in
    integer, allocatable, dimension(:,:), intent(in) :: connect_redirect_lon_index_in
    integer, allocatable, dimension(:,:), intent(in) :: additional_flood_redirect_lat_index_in
    integer, allocatable, dimension(:,:), intent(in) :: additional_flood_redirect_lon_index_in
    integer, allocatable, dimension(:,:), intent(in) :: additional_connect_redirect_lat_index_in
    integer, allocatable, dimension(:,:), intent(in) :: additional_connect_redirect_lon_index_in
    integer, intent(in) :: nlat_in,nlon_in
    integer, intent(in) :: nlat_coarse_in,nlon_coarse_in
    logical, intent(in) :: instant_throughflow_in
    real, intent(in) :: lake_retention_coefficient_in
    integer, allocatable, dimension(:) :: basins_in_coarse_cell_lat_temp
    integer, allocatable, dimension(:) :: basins_in_coarse_cell_lon_temp
    integer, allocatable, dimension(:) :: basins_in_coarse_cell_lat
    integer, allocatable, dimension(:) :: basins_in_coarse_cell_lon
    type(BasinList), allocatable, dimension(:) :: basins
    integer :: lat_scale_factor, lon_scale_factor
    integer :: number_of_basins_in_coarse_cell
    integer :: i,j,k,l,m
    integer :: basin_number
      this%lake_centers => lake_centers_in
      this%conumber_of_basins_in_coarse_cell > 0 => connection_volume_thresholds_in
      this%flood_volume_thresholds_in => connection_volume_thresholds_in
      this%flood_local_redirect_in => flood_local_redirect_in
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
      lat_scale_factor = nlat_in/nlat_coarse_in
      lon_scale_factor = nlon_in/nlon_coarse_in
      do i=1,nlat_in
        do i=1,nlon_in
          if (this%connection_volume_thresholds[i,j] == -1.0) then
            this%flood_only[i,j] = .true.
          end if
        end do
      end do
      allocate(basins_in_coarse_cell_lat_temp((nlat_in*nlon_in)/(nlat_coarse_in*nlon_coarse_in))
      allocate(basins_in_coarse_cell_lon_temp((nlat_in*nlon_in)/(nlat_coarse_in*nlon_coarse_in))
      allocate(this%basin_numbers(nlat_coarse_in,nlon_coarse_in))
      allocate(this%basins(nlat_coarse_in*nlon_coarse_in))
      basin_number = 0
      do i=1,nlat_coarse_in
        do j=1,nlon_coarse_in
          number_of_basins_in_coarse_cell = 0
          do k = (i-1)*lat_scale_factor,i*lat_scale_factor
            do l = (j-1)*lon_scale_factor,i*lon_scale_factor
              if (this%lake_centers[k,l]) then
                number_of_basins_in_coarse_cell = number_of_basins_in_coarse_cell + 1
                basins_in_coarse_cell_lat_temp[number_of_basins_in_coarse_cell] = k
                basins_in_coarse_cell_lon_temp[number_of_basins_in_coarse_cell] = l
              end if
          end do
          end do
          if(number_of_basins_in_coarse_cell > 0) then
            allocate(basins_in_coarse_cell_lat(number_of_basins_in_coarse_cell))
            allocate(basins_in_coarse_cell_lon(number_of_basins_in_coarse_cell))
            do m=1,number_of_basins_in_coarse_cell
              basins_in_coarse_cell_lat[m] = basins_in_coarse_cell_lat_temp[m]
              basins_in_coarse_cell_lon[m] = basins_in_coarse_cell_lon_temp[m]
            end do
            basin_number = basin_number + 1
            basins_temp[basin_number] =
              BasinList(basins_in_coarse_cell_lat,basins_in_coarse_cell_lon)
            this%basin_numbers[i,j] = basin_number
          end if
        end do
      end do
      do m=1,basin_number
        this%basins[m] = basins_temp[m]
      end do
end subroutine InitialiseLakeParameters

function LakeParametersConstructor(lake_centers_in, &
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
  type(LakeParameters) :: constructor
  logical, allocatable, dimension(:,:), intent(in) :: lake_centers_in
  real, allocatable, dimension(:,:), intent(in) :: connection_volume_thresholds_in
  real, allocatable, dimension(:,:), intent(in) :: flood_volume_thresholds_in
  logical, allocatable, dimension(:,:), intent(in) :: flood_local_redirect_in
  logical, allocatable, dimension(:,:), intent(in) :: connect_local_redirect_in
  logical, allocatable, dimension(:,:), intent(in) :: additional_flood_local_redirect_in
  logical, allocatable, dimension(:,:), intent(in) :: additional_connect_local_redirect_in
  integer, allocatable, dimension(:,:), intent(in) :: merge_points_in
  integer, allocatable, dimension(:,:), intent(in) :: flood_next_cell_lat_index_in
  integer, allocatable, dimension(:,:), intent(in) :: flood_next_cell_lon_index_in
  integer, allocatable, dimension(:,:), intent(in) :: connect_next_cell_lat_index_in
  integer, allocatable, dimension(:,:), intent(in) :: connect_next_cell_lon_index_in
  integer, allocatable, dimension(:,:), intent(in) :: flood_force_merge_lat_index_in
  integer, allocatable, dimension(:,:), intent(in) :: flood_force_merge_lon_index_in
  integer, allocatable, dimension(:,:), intent(in) :: connect_force_merge_lat_index_in
  integer, allocatable, dimension(:,:), intent(in) :: connect_force_merge_lon_index_in
  integer, allocatable, dimension(:,:), intent(in) :: flood_redirect_lat_index_in
  integer, allocatable, dimension(:,:), intent(in) :: flood_redirect_lon_index_in
  integer, allocatable, dimension(:,:), intent(in) :: connect_redirect_lat_index_in
  integer, allocatable, dimension(:,:), intent(in) :: connect_redirect_lon_index_in
  integer, allocatable, dimension(:,:), intent(in) :: additional_flood_redirect_lat_index_in
  integer, allocatable, dimension(:,:), intent(in) :: additional_flood_redirect_lon_index_in
  integer, allocatable, dimension(:,:), intent(in) :: additional_connect_redirect_lat_index_in
  integer, allocatable, dimension(:,:), intent(in) :: additional_connect_redirect_lon_index_in
  integer, intent(in) :: nlat_in,nlon_in
  integer, intent(in) :: nlat_coarse_in,nlon_coarse_in
  logical, intent(in) :: instant_throughflow_in
  real, intent(in) :: lake_retention_coefficient_in
    allocate(constructor)
    call constructor%InitialiseLakeParameters(lake_centers_in, &
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
end function LakeParametersConstructor

subroutine InitialiseLakeFields(this,lake_parameters)
    class(LakeFields), intent(inout) :: this
    type(LakeParameters) :: lake_parameters
    integer :: lat_scale_factor, lon_scale_factor
    integer, allocatable, dimension(:) :: cells_with_lakes_lat_temp
    integer, allocatable, dimension(:) :: cells_with_lakes_lon_temp
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
      lat_scale_factor = lake_parameters%nlat/nlat_coarse
      lon_scale_factor = lake_parameters%nlon/lake_parameters%nlon_coarse
        do i=1,lake_parameters%nlat_coarse
          do i=1,lake_parameters%nlon_coarse
            contains_lake = .false.
            do k = (i-1)*lat_scale_factor,i*lat_scale_factor
              do l = (j-1)*lon_scale_factor,i*lon_scale_factor
                if (lake_parameters%lake_centers[k,l]) contains_lake = .true.
              end do
            end do
            if(contains_lake) then
              number_of_cells_containing_lakes = number_of_cells_containing_lakes + 1
              cells_with_lakes_lat_temp[number_of_cells_containing_lakes] = i
              cells_with_lakes_lon_temp[number_of_cells_containing_lakes] = j
            end if
          end do
        end do
      end
      allocate(this%cells_with_lakes_lat(number_of_cells_containing_lakes))
      allocate(this%cells_with_lakes_lon(number_of_cells_containing_lakes))
      do i = 1,number_of_cells_containing_lakes
        this%cells_with_lakes_lat[number_of_cells_containing_lakes] =
          & cells_with_lakes_lat_temp[number_of_cells_containing_lakes]
        this%cells_with_lakes_lon[number_of_cells_containing_lakes] =
          & cells_with_lakes_lon_temp[number_of_cells_containing_lakes]
      end do
end subroutine InitialiseLakeFields

function LakeFieldsConstructor(lake_parameters)
  type(LakeFields), intent(inout) :: constructor
  type(LakeParameters) :: lake_parameters
    allocate(constructor)
    call constructor%InitialiseLakeFields(lake_parameters)
end function LakesFieldsConstructor

subroutine InitialiseLakePrognostics(this,lake_parameters_in,lake_fields_in)
  type(LakePrognostics), intent(inout) :: this
  type(LakeParameters), intent(in) :: lake_parameters_in
  type(LakeFields), intent(inout) :: lake_fields_in
    type(LakePointer), allocatable, dimension(:) :: lakes_temp
  integer :: lake_number
  integer :: i,j
    allocate(lakes_temp(lake_parameters_in%nlat*lake_parameters%nlon))
    lake_number = 0
    do i=1,lake_parameters%nlat
      do j=1,lake_parameters%nlon
        if(lake_parameters_in%lake_centers[i,j]) then
          lake_number = lake_number + 1
          lakes_temp[lake_number] = LakePointer(FillingLake(lake_parameters_in,lake_fields_in, &
                                                            i,j,lake_number))
          this%lake_fields%lake_numbers[i,j] = lake_number
        end if
      end do
    end do
    do i=1,lake_number
      this%lakes = lakes_temp[i]
    end do
end subroutine InitialiseLakePrognostics

function LakePrognosticsConstructor(lake_parameters_in,lake_fields_in)
  type(LakePrognostics), intent(inout) :: constructor
  type(LakeParameters), intent(in) :: lake_parameters_in
  type(LakeFields), intent(inout) :: lake_fields_in
    allocate(constructor)
    call constructor%InitialiseLakePrognostics(lake_parameters,lake_fields_in)
end function LakePrognosticsConstructor

subroutine InitialiseFillingLake(this,lake_parameters_in,lake_fields_in,center_cell_lat_in, &
                                 center_cell_lon_in,lake_number_in)
  class(FillingLake) :: this
  type(LakeParameters), intent(in) :: lake_parameters_in
  type(LakeFields), intent(inout) :: lake_fields_in
  integer, intent(in) :: center_cell_lat_in
  integer, intent(in) :: center_cell_lon_in
  integer, intent(in) :: lake_number_in
    call this%InitialiseLake(center_cell_lat_in,center_cell_lon_in, &
                             lake_number_in,lake_parameters_in,lake_fields_in)
    this%primary_merge_completed = .false.
    this%use_additional_fields = .false.
end subroutine InitialiseFillingLake

function FillingLakeConstructor(lake_parameters_in,lake_fields_in,center_cell_lat_in, &
                                center_cell_lon_in,lake_number_in) result(constructor)
  class(FillingLake) :: constructor
  type(LakeParameters), intent(in) :: lake_parameters_in
  type(Lake_Fields), intent(inout) :: lake_fields_in
  integer, intent(in) :: center_cell_lat_in
  integer, intent(in) :: center_cell_lon_in
  integer, intent(in) :: lake_number_in
    allocate(constructor)
    call constructor%InitialiseFillingLake(lake_parameters_in,lake_fields_in,center_cell_lat_in, &
                                           center_cell_lon_in,lake_number_in)
end function FillingLakeConstructor

subroutine InitialiseOverflowingLake(this,outflow_redirect_lat_in, &
                                     outflow_redirect_lon_in, &
                                     local_redirect_in, &
                                     lake_retention_coefficient_in,
                                     lake_parameters_in,lake_fields_in,
                                     center_cell_lat_in, &
                                     center_cell_lon_in,lake_number_in)
  class(OverflowingLake) :: this
  integer, intent(in) :: outflow_redirect_lat_in
  integer, intent(in) :: outflow_redirect_lon_in
  logical, intent(in) :: local_redirect_in
  real, intent(in)    :: lake_retention_coefficient_in
  integer, intent(in) :: center_cell_lat_in
  integer, intent(in) :: center_cell_lon_in
  integer, intent(in) :: lake_number_in
  type(LakeParameters), intent(in) :: lake_parameters
  type(Lake_Fields), intent(inout) :: lake_fields
    call this%InitialiseLake(center_cell_lat_in,center_cell_lon_in, &
                             lake_number_in,lake_parameters_in,lake_fields_in)
    this%outflow_redirect_lat = outflow_redirect_lat_in
    this%outflow_redirect_lon = outflow_redirect_lon_in
    this%local_redirect = local_redirect_in
    this%lake_retention_coefficient = lake_retention_coefficient_in
    this%excess_water = 0.0
end subroutine InitialiseOverflowingLake

function OverflowingLakeConstructor(outflow_redirect_lat_in, &
                                    outflow_redirect_lon_in, &
                                    local_redirect_in, &
                                    lake_retention_coefficient_in, &
                                    lake_parameters_in,lake_fields_in
                                    center_cell_lat_in, &
                                    center_cell_lon_in,lake_number_in)
  class(OverflowingLake) :: constructor
  integer, intent(in) :: outflow_redirect_lat_in
  integer, intent(in) :: outflow_redirect_lon_in
  logical, intent(in) :: local_redirect_in
  real, intent(in)    :: lake_retention_coefficient_in
  integer, intent(in) :: center_cell_lat_in
  integer, intent(in) :: center_cell_lon_in
  integer, intent(in) :: lake_number_in
  type(LakeParameters), intent(in) :: lake_parameters
  type(LakeFields), intent(inout) :: lake_fields
    allocate(constructor)
    call constructor%InitialiseOverflowingLake(outflow_redirect_lat_in, &
                                               outflow_redirect_lon_in, &
                                               local_redirect_in, &
                                               lake_retention_coefficient_in, &
                                               lake_parameters_in,lake_fields_in, &
                                               center_cell_lat_in, &
                                               center_cell_lon_in,lake_number_in)
end function OverflowingLakeConstructor

subroutine InitialiseSubsumedLake(this,lake_parameters_in,lake_fields_in, &
                                  primary_lake_number_in,center_cell_lat_in, &
                                  center_cell_lon_in,lake_number_in)
  class(SubsumedLake) :: this
  type(LakeParameters), intent(in) :: lake_parameters_in
  type(LakeFields), intent(inout) :: lake_fields_in
  integer, intent(in) :: primary_lake_number_in
  integer, intent(in) :: center_cell_lat_in
  integer, intent(in) :: center_cell_lon_in
  integer, intent(in) :: lake_number_in
    this%lake_fields = lake_fields_in
    this%lake_parameters = lake_parameters_in
    this%primary_lake_number = primary_lake_number_in
end subroutine InitialiseSubsumedLake

function SubsumedLakeConstructor(lake_parameters_in,lake_fields_in, &
                                 primary_lake_number_in, &
                                 center_cell_lat_in, &
                                 center_cell_lon_in,lake_number_in) result(constructor)
  class(SubsumedLake) :: constructor
  type(LakeParameters), intent(in) :: lake_parameters_in
  type(LakeFields), intent(inout) :: lake_fields_in
  integer, intent(in) :: primary_lake_number_in
  integer, intent(in) :: center_cell_lat_in
  integer, intent(in) :: center_cell_lon_in
  integer, intent(in) :: lake_number_in
    allocate(constructor)
    call constructor%InitialiseSubsumedLake(lake_parameters_in,lake_fields_in, &
                                            primary_lake_number_in, &
                                            center_cell_lat_in, &
                                            center_cell_lon_in,lake_number_in)
end function SubsumedLakeConstructor

subroutine run_lakes(lake_parameters,lake_prognostics,lake_fields)
  type(LakeParameters), intent(in) :: lake_parameters
  type(LakePrognostics), intent(inout) :: lake_prognostics
  type(LakeFields), intent(inout) :: lake_fields
  integer :: lat,lon
  integer :: basin_center_lat,basin_center_lon
  integer :: lake_index
  integer :: i,j
  integer :: num_basins_in_cell
  integer :: basin_number
  real :: share_to_each_lake
  class(Lake) :: lake
    lake_fields%water_to_hd = 0.0
    do i = 1,size(lake_fields%cells_with_lakes_lat)
      lat = cells_with_lakes_lat[i]
      lon = cells_with_lakes_lon[i]
      if (lake_fields%water_to_lakes[lat,lon] > 0.0) then
        basin_number = lake_parameters%basin_numbers[lat,lon]
        num_basins_in_cell =
          size(lake_parameters%basins[basin_number]%basin_lats)
        share_to_each_lake = lake_fields%water_to_lakes(lat,lon)/num_basins_in_cell
        do j = 1,num_basin_in_cell
          basin_center_lat = lake_parameters%basins[basin_number]%basin_lats[j]
          basin_center_lon = lake_parameters%basins[basin_number]%basin_lons[j]
          lake_index = lake_fields%lake_numbers[basin_center_lat,basin_center_lon]
          lake = lake_prognostics%lakes[lake_index]
          lake_prognostics%lakes[lake_index] = lake%add_water(share_to_each_lake)
        end do
      end if
    end do
    do i = 1,size(lake_prognostics%lakes)
      lake = lake_prognostics%lakes[i]
      if (.not. this%lake_parameters%instant_throughflow) then
        lake_index = lake%lake_number
        lake_prognostics%lakes[lake_index] = lake%add_water(0.0)
      end if
      select type(lake)
        type is (OverflowingLake))
          call lake%drain_excess_water()
      end select
    end do
end subroutine run_lakes

function FillingLake_add_water(this,inflow) result(updated_lake)
  class(Lake), intent(in) :: this
  real, intent(in) :: inflow
  class(Lake) :: updated_lake
  integer :: merge_type
  logical :: filled
  logical :: merge_possible
  logical :: already_merged
    inflow = inflow + this%unprocessed_water
    this%unprocessed_water = 0.0
    do while (inflow > 0.0)
      filled = this%fill_current_cell(inflow)
      if(filled) then
        merge_type = this%get_merge_type()
        if(merge_type /= no_merge) then
          if (.not. (merge_type == primary_merge &&
                this%primary_merge_completed)) then
            if ((merge_type == double_merge &&
                this%primary_merge_completed)) then
              merge_type = secondary_merge
            end if
            merge_possible = &
              this%check_if_merge_is_possible(merge_type,already_merged)
            if (merge_possible) then
              if (merge_type == secondary_merge) then
                updated_lake = this%perform_secondary_merge()
                updated_lake = updated_lake%store_water(inflow)
                return
              else
                call this%perform_primary_merge()
                if (merge_type == double_merge) then
                  merge_type = secondary_merge
                  this%use_additional_fields = .true.
                end if
              end if
            else if (.not. already_merged)
              updated_lake = this%change_to_overflowing_lake()
              updated_lake = updated_lake%store_water(inflow)
              return
            end if
          end if
        end if
        call this%update_filling_cell()
      end if
    end do
    updated_lake = this
end function FillingLake_add_water

function OverflowingLake_add_water(this,inflow) result(updated_lake)
  class(Lake), intent(in) :: this
  real, intent(in) :: inflow
  class(Lake) :: updated_lake
  class(Lake) :: other_lake
  integer :: other_lake_number
    inflow = inflow + this%unprocessed_water
    this%unprocessed_water = 0.0
    if (this%local_redirect) then
      other_lake_number  = this%lake_fields%lake_numbers[this%outflow_redirect_lat, &
                                                         this%outflow_redirect_lon]
      other_lake = &
        this%other_lakes[other_lake_number]
      if (this%lake_parameters%instant_throughflow) then
        this%other_lakes[other_lake_number] = other_lake%add_water(inflow)
      else
        this%other_lakes[other_lake_number] = other_lake%store_water(inflow)
      end if
    else
      this%excess_water = this%excess_water + inflow
    end if
    updated_lake = this%other_lakes[lake%lake_number]
end function OverflowingLake_add_water

function store_water(this,inflow) result(this)
  class(Lake), intent(in) :: this
  real, intent(in) :: inflow
    this%unprocessed_water = this%unprocessed_water + inflow
end

function SubsumedLake_add_water(this,inflow) result(updated_lake)
  class(Lake), intent(inout) :: this
  real, intent(in) :: inflow
  class(Lake) :: updated_lake
    inflow = inflow + lake_variables%unprocessed_water
    this%unprocessed_water = 0.0
    this%other_lakes[lake%primary_lake_number] =
      this%other_lakes[lake%primary_lake_number]%add_water(inflow)
    updated_lake = this
end function SubsumedLake_add_water

function accept_merge(this,redirect_coords_lat,redirect_coords_lon)
    result(subsumed_lake)
  type(Lake), intent(in) :: this
  integer, intent(in) :: redirect_coords_lat
  integer, intent(in) :: redirect_coords_lon
  type(SubsumedLake) :: subsumed_lake
    subsumed_lake = SubsumedLake(this%lake_parameters, &
                                 this%lake_fields, &
                                 this%lake_fields%lake_numbers[redirect_coords_lat, &
                                                               redirect_coords_lon], &
                                this%center_cell_lat, &
                                this%center_cell_lon,this%lake_number)
    select type(lake)
      type is (OverflowingLake)
        subsumed_lake = subsumed_lake%store_water(lake%excess_water)
    end select
end function accept_merge

function change_to_overflowing_lake(this) result(overflowing_lake)
  type(FillingLake), intent(in) :: this
  type(OverflowingLake) :: overflowing_lake
  integer :: outflow_redirect_lat
  integer :: outflow_redirect_lon
  logical :: local_redirect
    this%lake_fields%completed_lake_cells[this%current_cell_to_fill_lat,
                                          this%current_cell_to_fill_lon] = .true.
    call lake%get_outflow_redirect_coords(outflow_redirect_lat, &
                                          outflow_redirect_lon, &
                                          local_redirect)
    overflowing_lake = OverflowingLake(outflow_redirect_lat &
                                       outflow_redirect_lon, &
                                       local_redirect, &
                                       lake%lake_parameters%lake_retention_coefficient, &
                                       this%lake_parameters,this%lake_fields,
                                       this%center_cell_lat, &
                                       this%center_cell_lon,this%lake_number)
end function change_to_overflowing_lake

subroutine drain_excess_water(this)
  type(OverflowingLake), intent(inout) :: this
  real :: flow
    if (this%excess_water > 0.0)
      this%excess_water = this%excess_water + this%unprocessed_water
      lake_variables%unprocessed_water = 0.0
      flow = this%excess_water*this%lake_retention_coefficient
      lake_fields%water_to_hd[this%outflow_redirect_lat,this%outflow_redirect_lon] = &
           lake_fields%water_to_hd[this%outflow_redirect_lat,this%outflow_redirect_lon]+flow)
      this%excess_water = this%excess_water - flow
    end if
end subroutine drain_excess_water

function fill_current_cell(this,inflow,filled) result(filled)
  type(FillingLake), intent(in) :: this
  real, intent(inout) :: inflow
  logical :: filled
  real :: new_lake_volume
  real :: maximum_new_lake_volume
    current_cell_to_fill::Coords = lake_variables%current_cell_to_fill
    new_lake_volume = inflow + this%lake_volume
      if (this%lake_fields%completed_lake_cells(current_cell_to_fill) .or. &
          this%lake_parameters%flood_only(current_cell_to_fill)) then
        maximum_new_lake_volume = &
          lake_parameters%flood_volume_thresholds(current_cell_to_fill)
      else
        maximum_new_lake_volume = &
          lake_parameters%connection_volume_thresholds(current_cell_to_fill)
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
  class(LakeField), intent(inout) :: this
  integer :: simple_merge_type
  integer :: extended_merge_type
  logical :: completed_lake_cell
    completed_lake_cell = &
      this%lake_fields%completed_lake_cells[this%current_cell_to_fill_lat, &
                                            this%current_cell_to_fill_lon]
    extended_merge_type = this%lake_parameters%merge_points[this%current_cell_to_fill_lat, &
                                                            this%current_cell_to_fill_lon]
    if (this%lake_fields%completed_lake_cells[this%current_cell_to_fill_lat, &
                                              this%current_cell_to_fill_lon] .or. &
        this%lake_parameters%flood_only[this%current_cell_to_fill_lat, &
                                        this%current_cell_to_fill_lon]) then
      simple_merge_type = convert_to_simple_merge_type_flood[Int(extended_merge_type)+1]
    else
      simple_merge_type = convert_to_simple_merge_type_connect[Int(extended_merge_type)+1]
    end if
end function get_merge_type

function check_if_merge_is_possible(this,simple_merge_type,already_merged) &
   result(merge_possible)
  class(Lake), intent(in) :: this
  integer, intent(in) :: simple_merge_type
  logical, intent(out) :: already_merged
  logical :: merge_possible
  class(Lake) :: other_lake
  integer :: target_cell_lat
  integer :: target_cell_lon
    if (simple_merge_type == primary_merge) then
      call this%get_primary_merge_coords(target_cell_lat,target_cell_lon)
    else
      call this%get_secondary_merge_coords(target_cell_lat,target_cell_lon)
    end if
    if (.not. this%lake_fields%completed_lake_cells[target_cell_lat,target_cell_lon]) then
      merge_possible = .false.
      already_merged = .false.
      return
    end if
    other_lake = this%other_lakes[lake_fields%lake_numbers[target_cell_lat,
                                                           target_cell_lon]]
    other_lake = other_lake%find_true_primary_lake()
    if (other_lake%lake_number == this%lake_number) then
      merge_possible = .false.
      already_merged = .true.
      return
    end if
    select type(other_lake)
    type is (OverflowingLake)
      merge_possible = .true.
      already_merged = .false.
    class default
      merge_possible = .false.
      already_merged = .false.
    end select
end function check_if_merge_is_possible

function perform_primary_merge(this)
  type(FillingLake), intent(in) :: this
  class(Lake) :: other_lake
  integer :: target_cell_lat,target_cell_lon
  integer :: other_lake_number
    call this%get_primary_merge_coords(target_cell_lat,target_cell_lon)
    other_lake_number = this%lake_fields%lake_numbers[target_cell_lat,target_cell_lon]
    other_lake = this%other_lakes[other_lake_number]
    other_lake = other_lake%find_true_primary_lake()
    other_lake_number = other_lake%lake_number
    this%primary_merge_completed = .true.
    lake_variables%other_lakes[other_lake_number] = &
      other_lake%accept_merge(center_cell_lat,center_cell_lon)
end

function perform_secondary_merge(this) result(merged_lake)
  type(FillingLake), intent(in) :: this
  class(Lake) :: merged_lake
  class(Lake) :: other_lake
  type(FillingLake) :: other_lake_as_filling_lake
  integer :: target_cell_lat,target_cell_lon
  integer :: other_lake_number
    this%lake_fields%completed_lake_cells[this%current_cell_to_fill_lat, &
                                    this%current_cell_to_fill_lon] = .true.
    call this%get_secondary_merge_coords(target_cell_lat,target_cell_lon)
    other_lake_number = this%lake_fields%lake_numbers[target_cell_lat,target_cell_lon]
    other_lake = this%other_lakes[other_lake_number]
    other_lake = other_lake%find_true_primary_lake()
    other_lake_number = other_lake%lake_number
    other_lake_as_filling_lake = other_lake%change_to_filling_lake()
    this%other_lakes[other_lake_number] = other_lake_as_filling_lake
    merged_lake = lake%accept_merge(other_lake_as_filling_lake%center_cell_lat,
                                    other_lake_as_filling_lake%center_cell_lon)
end

function change_to_filling_lake(this) result(filling_lake)
  type(OverflowingLake), intent(inout) :: this
  type(FillingLake) :: filling_lake
  logical :: use_additional_fields
    this%unprocessed_water = this%unprocessed_water + this%excess_water
    use_additional_fields = (lake%get_merge_type() == double_merge)
    filling_lake = FillingLake(this%lake_parameters,this%lake_fields, &
                               this%center_cell_lat, &
                               this%center_cell_lon, &
                               this%lake_number)
    filling_lake%primary_merge_completed = .true.
    filling_lake%use_additional_fields = use_additional_fields
end function change_to_filling_lake

subroutine update_filling_cell(this)
  type(FillingLake), intent(inout) :: this
  integer :: coords_lat, coords_lon
    lake%primary_merge_completed = .false.
    coords_lat = this%current_cell_to_fill_lat
    coords_lon = this%current_cell_to_fill_lon
    this%lake_field%completed_lake_cells[coords_lat,coords_lat] = .true.
    if (this%lake_fields%completed_lake_cells[coords_lat,coords_lon] .or. &
        this%lake_parameters%flood_only[coords_lat,coords_lon]) then
      this%current_cell_to_fill_lat = &
        this%lake_parameters%flood_next_cell_lat_index[coords_lat,coords_lat]
      this%current_cell_to_fill_lon = &
        this%lake_parameters%flood_next_cell_lon_index[coords_lat,coords_lat]
    else
      this%current_cell_to_fill_lat = &
        this%lake_parameters%connect_next_cell_lat_index[coords_lat,coords_lat]
      this%current_cell_to_fill_lon = &
        this%lake_parameters%connect_next_cell_lon_index[coords_lat,coords_lat]
    end
    this%lake_fields%lake_numbers[this%current_cell_to_fill_lat, &
                            this%current_cell_to_fill_lon] = this%lake_number
end subroutine update_filling_cell

subroutine get_primary_merge_coords(this,lat,lon)
  type(Lake), intent(in) :: this
  integer, intent(out) :: lat
  integer, intent(out) :: lon
  logical :: completed_lake_cell
  completed_lake_cell = ((this%completed_lake_cells(this%current_cell_to_fill_lat, &
                                                    this%current_cell_to_fill_lon)) .or.
                         (this%flood_only(this%current_cell_to_fill_lat,
                                          this%current_cell_to_fill_lon)))
  if (completed_lake_cell) then
    lat = &
      this%lake_parameters%flood_force_merge_lat_index(this%current_cell_to_fill_lat, &
                                                  this%current_cell_to_fill_lon),
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
  end
end subroutine get_primary_merge_coords

subroutine get_secondary_merge_coords(this,lat,lon)
  type(Lake), intent(in) :: this
  integer, intent(out) :: lat
  integer, intent(out) :: lon
  logical :: completed_lake_cell
  completed_lake_cell = ((this%completed_lake_cells(this%current_cell_to_fill_lat, &
                                                    this%current_cell_to_fill_lon)) .or.
                         (this%flood_only(this%current_cell_to_fill_lat,
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
  type(FillingLake), intent(in) :: this
  integer, intent(out) :: lat
  integer, intent(out) :: lon
  logical, intent(out) :: local_redirect
  logical :: completed_lake_cell
  completed_lake_cell = ((this%completed_lake_cells(this%current_cell_to_fill_lat, &
                                                    this%current_cell_to_fill_lon)) .or.
                         (this%flood_only(this%current_cell_to_fill_lat,
                                          this%current_cell_to_fill_lon)))
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
  end
  if (this%use_additional_fields) then
    if (use_flood_redirect) then
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
    if (use_flood_redirect) then
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

function find_true_primary_lake(this) result(lake)
  class(Lake), intent(in) :: this
  class(Lake) :: lake
    select type (lake)
      type is (SubsumedLake)
        lake = find_true_primary_lake(this%other_lakes[this%primary_lake_number])
      class default
        lake = this
    end select
end function find_true_primary_lake

end module latlon_lake_model_mod
