module latlon_lake_model_interface_mod

use latlon_lake_model_mod
use latlon_lake_model_io_mod

implicit none

type lakeinterfaceprognosticfields
  real, allocatable, dimension(:,:)   :: water_from_lakes
  real, allocatable, dimension(:,:)   :: water_to_lakes
  !Lake water from ocean means negative water from lakes to ocean
  real, allocatable, dimension(:,:)   :: lake_water_from_ocean
  contains
    procedure :: initialiselakeinterfaceprognosticfields
    procedure :: lakeinterfaceprognosticfieldsdestructor
end type lakeinterfaceprognosticfields

interface lakeinterfaceprognosticfields
  procedure :: lakeinterfaceprognosticfieldsconstructor
end interface

type(lakeparameters), pointer ::  global_lake_parameters
type(lakeprognostics), pointer :: global_lake_prognostics
type(lakefields), pointer ::      global_lake_fields

contains

function lakeinterfaceprognosticfieldsconstructor(nlat_coarse, &
                                                  nlon_coarse) result(constructor)
  type(lakeinterfaceprognosticfields), pointer :: constructor
  integer :: nlat_coarse
  integer :: nlon_coarse
    allocate(constructor)
    call constructor%initialiselakeinterfaceprognosticfields(nlat_coarse, &
                                                             nlon_coarse)
end function lakeinterfaceprognosticfieldsconstructor

subroutine  lakeinterfaceprognosticfieldsdestructor(this)
  class(lakeinterfaceprognosticfields) :: this
    deallocate(this%water_from_lakes)
    deallocate(this%water_to_lakes)
    deallocate(this%lake_water_from_ocean)
end subroutine  lakeinterfaceprognosticfieldsdestructor

subroutine initialiselakeinterfaceprognosticfields(this, &
                                                   nlat_coarse, &
                                                   nlon_coarse)
  class(lakeinterfaceprognosticfields) :: this
  integer :: nlat_coarse
  integer :: nlon_coarse
    allocate(this%water_from_lakes(nlat_coarse,nlon_coarse))
    allocate(this%water_to_lakes(nlat_coarse,nlon_coarse))
    allocate(this%lake_water_from_ocean(nlat_coarse,nlon_coarse))
    this%water_from_lakes = 0.0
    this%water_to_lakes = 0.0
    this%lake_water_from_ocean = 0.0
end subroutine initialiselakeinterfaceprognosticfields

subroutine init_lake_model(lake_model_ctl_filename,initial_spillover_to_rivers)
  character(len = *) :: lake_model_ctl_filename
  real, pointer, dimension(:,:),intent(out) :: initial_spillover_to_rivers
  real, pointer, dimension(:,:) :: initial_water_to_lake_centers
    call config_lakes(lake_model_ctl_filename)
    global_lake_parameters => read_lake_parameters(.true.)
    global_lake_fields => lakefields(global_lake_parameters)
    global_lake_prognostics => lakeprognostics(global_lake_parameters, &
                                               global_lake_fields)
    call load_lake_initial_values(initial_water_to_lake_centers,&
                                    initial_spillover_to_rivers)
    call setup_lakes(global_lake_parameters,global_lake_prognostics, &
                     global_lake_fields, initial_water_to_lake_centers)
    deallocate(initial_water_to_lake_centers)
end subroutine init_lake_model

subroutine init_lake_model_test(lake_parameters,initial_water_to_lake_centers)
  type(lakeparameters), pointer :: lake_parameters
  real, pointer, dimension(:,:), intent(in) :: initial_water_to_lake_centers
    global_lake_parameters => lake_parameters
    global_lake_fields => lakefields(global_lake_parameters)
    global_lake_prognostics => lakeprognostics(global_lake_parameters, &
                                               global_lake_fields)
    call setup_lakes(global_lake_parameters,global_lake_prognostics, &
                     global_lake_fields, initial_water_to_lake_centers)
end subroutine init_lake_model_test

subroutine clean_lake_model()
  call global_lake_fields%lakefieldsdestructor()
  call global_lake_prognostics%lakeprognosticsdestructor()
  call global_lake_parameters%lakeparametersdestructor()
  deallocate(global_lake_parameters)
  deallocate(global_lake_fields)
  deallocate(global_lake_prognostics)
end subroutine clean_lake_model

subroutine run_lake_model(lake_interface_fields)
  type(lakeinterfaceprognosticfields), intent(inout) :: lake_interface_fields
    global_lake_fields%water_to_lakes(:,:) = lake_interface_fields%water_to_lakes(:,:)
    call run_lakes(global_lake_parameters,global_lake_prognostics,global_lake_fields)
    lake_interface_fields%water_from_lakes(:,:) = global_lake_fields%water_to_hd(:,:)
    lake_interface_fields%lake_water_from_ocean(:,:) = global_lake_fields%lake_water_from_ocean(:,:)
end subroutine run_lake_model

subroutine run_lake_model_jsbach(water_to_lakes_in,water_to_hd_out)
  real, allocatable, dimension(:,:), intent(in)   :: water_to_lakes_in
  real, allocatable, dimension(:,:), intent(out)  :: water_to_hd_out
    global_lake_fields%water_to_lakes(:,:) = water_to_lakes_in(:,:)
    call run_lakes(global_lake_parameters,global_lake_prognostics,global_lake_fields)
    water_to_hd_out(:,:) = global_lake_fields%water_to_hd(:,:)
end subroutine run_lake_model_jsbach

subroutine write_lake_numbers_field_interface(timestep)
  integer :: timestep
    call write_lake_numbers_field(global_lake_parameters,global_lake_fields,timestep)
end subroutine write_lake_numbers_field_interface

function get_lake_prognostics() result(value)
  type(lakeprognostics), pointer :: value
    value => global_lake_prognostics
end function get_lake_prognostics

function get_lake_fields() result(value)
  type(lakefields), pointer :: value
    value => global_lake_fields
end function get_lake_fields

end module latlon_lake_model_interface_mod
