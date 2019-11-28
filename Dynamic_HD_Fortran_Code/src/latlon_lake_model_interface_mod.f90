module latlon_lake_model_interface_mod

use latlon_lake_model_mod

implicit none

type lakeinterfaceprognosticfields
  real, allocatable, dimension(:,:)   :: water_from_lakes
  real, allocatable, dimension(:,:)   :: water_to_lakes
  contains
    procedure :: initialiselakeinterfaceprognosticfields
end type lakeinterfaceprognosticfields

interface lakeinterfaceprognosticfields
  procedure :: lakeinterfaceprognosticfieldsconstructor
end interface

type(lakeparameters) ::  global_lake_parameters
type(lakeprognostics) :: global_lake_prognostics
type(lakefields) ::      global_lake_fields

contains

function lakeinterfaceprognosticfieldsconstructor() result(constructor)
  type(lakeinterfaceprognosticfields), allocatable :: constructor
    allocate(constructor)
    call constructor%initialiselakeinterfaceprognosticfields()
end function lakeinterfaceprognosticfieldsconstructor

subroutine initialiselakeinterfaceprognosticfields(this)
  class(lakeinterfaceprognosticfields) :: this
    this%water_from_lakes = 0.0
    this%water_to_lakes = 0.0
end subroutine initialiselakeinterfaceprognosticfields

subroutine run_lake_model(lake_interface_fields)
  type(lakeinterfaceprognosticfields), intent(inout) :: lake_interface_fields
    global_lake_fields%water_to_lakes(:,:) = lake_interface_fields%water_to_lakes(:,:)
    call run_lakes(global_lake_parameters,global_lake_prognostics,global_lake_fields)
    lake_interface_fields%water_from_lakes(:,:) = global_lake_fields%water_to_hd(:,:)
end subroutine run_lake_model

subroutine run_lake_model_jsbach(water_to_lakes_in,water_to_hd_out)
  real, allocatable, dimension(:,:), intent(in)   :: water_to_lakes_in
  real, allocatable, dimension(:,:), intent(out)  :: water_to_hd_out
    global_lake_fields%water_to_lakes(:,:) = water_to_lakes_in(:,:)
    call run_lakes(global_lake_parameters,global_lake_prognostics,global_lake_fields)
    water_to_hd_out(:,:) = global_lake_fields%water_to_hd(:,:)
end subroutine

end module latlon_lake_model_interface_mod
