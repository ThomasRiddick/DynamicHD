module icosohedral_lake_model_interface_mod

! A module that interfaces between the lake, HD modules and IO modules
! and contains the concrete instances of the main lake objects

use icosohedral_lake_model_mod
use icosohedral_lake_model_io_mod
use grid_information_mod

implicit none

! An object to store fields to be transferred between the HD model and the lake model
type lakeinterfaceprognosticfields
  real, allocatable, dimension(:)   :: water_from_lakes ! Water flowing out of lakes
  real, allocatable, dimension(:)   :: water_to_lakes ! Water flowing into lakes, can also include
                                                      ! evaporation as a negative contribution and
                                                      ! thus can be overall negative or positive
  !Lake water from ocean means negative water from lakes to ocean
  real, allocatable, dimension(:)   :: lake_water_from_ocean ! Negative water to add to the ocean
                                                             ! to close water budget should a lake
                                                             ! ever achieve a negative volume
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
real :: global_step_length

contains

! Constructs an lakeinterfaceprognosticfields object
function lakeinterfaceprognosticfieldsconstructor(ncells_coarse) result(constructor)
  type(lakeinterfaceprognosticfields), pointer :: constructor
  integer :: ncells_coarse
    allocate(constructor)
    call constructor%initialiselakeinterfaceprognosticfields(ncells_coarse)
end function lakeinterfaceprognosticfieldsconstructor

! Free memory from a lakeinterfaceprognosticfields object
subroutine  lakeinterfaceprognosticfieldsdestructor(this)
  class(lakeinterfaceprognosticfields) :: this
    deallocate(this%water_from_lakes)
    deallocate(this%water_to_lakes)
    deallocate(this%lake_water_from_ocean)
end subroutine  lakeinterfaceprognosticfieldsdestructor

! Initialise a lakeinterfaceprogonsticfields object - set all initial values to zero
subroutine initialiselakeinterfaceprognosticfields(this, &
                                                   ncells_coarse)
  class(lakeinterfaceprognosticfields) :: this
  integer :: ncells_coarse
    allocate(this%water_from_lakes(ncells_coarse))
    allocate(this%water_to_lakes(ncells_coarse))
    allocate(this%lake_water_from_ocean(ncells_coarse))
    this%water_from_lakes = 0.0
    this%water_to_lakes = 0.0
    this%lake_water_from_ocean = 0.0
end subroutine initialiselakeinterfaceprognosticfields

! Initialise the lake model using input data in the provided files
subroutine init_lake_model(lake_model_ctl_filename,initial_spillover_to_rivers, &
                           step_length)
  character(len = *) :: lake_model_ctl_filename
  real, pointer, dimension(:),intent(out) :: initial_spillover_to_rivers
  real, pointer, dimension(:) :: initial_water_to_lake_centers
  real :: step_length
    call config_lakes(lake_model_ctl_filename)
    global_lake_parameters => read_lake_parameters(.true.)
    global_lake_fields => lakefields(global_lake_parameters)
    global_lake_prognostics => lakeprognostics(global_lake_parameters, &
                                               global_lake_fields)
    global_step_length = step_length
    call load_lake_initial_values(initial_water_to_lake_centers,&
                                    initial_spillover_to_rivers)
    call setup_lakes(global_lake_parameters,global_lake_prognostics, &
                     global_lake_fields, initial_water_to_lake_centers)
    deallocate(initial_water_to_lake_centers)
end subroutine init_lake_model

! Initialise the lake model for testing using specified input data
subroutine init_lake_model_test(lake_parameters,initial_water_to_lake_centers, &
                                step_length)
  type(lakeparameters), pointer :: lake_parameters
  real, pointer, dimension(:), intent(in) :: initial_water_to_lake_centers
  real :: step_length
    global_lake_parameters => lake_parameters
    global_lake_fields => lakefields(global_lake_parameters)
    global_lake_prognostics => lakeprognostics(global_lake_parameters, &
                                               global_lake_fields)
    global_step_length = step_length
    call setup_lakes(global_lake_parameters,global_lake_prognostics, &
                     global_lake_fields, initial_water_to_lake_centers)
end subroutine init_lake_model_test

! Free memory at the end of a run
subroutine clean_lake_model()
  call global_lake_fields%lakefieldsdestructor()
  call global_lake_prognostics%lakeprognosticsdestructor()
  call global_lake_parameters%lakeparametersdestructor()
  deallocate(global_lake_parameters)
  deallocate(global_lake_fields)
  deallocate(global_lake_prognostics)
end subroutine clean_lake_model

! Interface for the lake model to be used by the HD model code in this repo
subroutine run_lake_model(lake_interface_fields)
  type(lakeinterfaceprognosticfields), intent(inout) :: lake_interface_fields
    global_lake_fields%water_to_lakes(:) = &
      lake_interface_fields%water_to_lakes(:)*global_step_length
    call run_lakes(global_lake_parameters,global_lake_prognostics,global_lake_fields)
    global_lake_fields%water_to_hd(:) = &
      global_lake_fields%water_to_hd(:)/global_step_length
    lake_interface_fields%water_from_lakes(:) = global_lake_fields%water_to_hd(:)
    global_lake_fields%lake_water_from_ocean(:) = &
      global_lake_fields%lake_water_from_ocean(:)/global_step_length
    lake_interface_fields%lake_water_from_ocean(:) = &
      global_lake_fields%lake_water_from_ocean(:)
end subroutine run_lake_model

! Interface for the lake model without using the lakeinterfaceprognosticfields derived
! type
! Multiplication by step length is to be applied outside of this routine in the
! main HD model
subroutine run_lake_model_jsbach(water_to_lakes_in,water_to_hd_out,water_from_ocean_out)
  real, allocatable, dimension(:), intent(in)   :: water_to_lakes_in
  real, allocatable, dimension(:), intent(out)  :: water_to_hd_out
  real, allocatable, dimension(:), intent(out)  :: water_from_ocean_out
    global_lake_fields%water_to_lakes(:) = water_to_lakes_in(:)
    call run_lakes(global_lake_parameters,global_lake_prognostics,global_lake_fields)
    water_to_hd_out(:) = global_lake_fields%water_to_hd(:)
    water_from_ocean_out(:) = global_lake_fields%lake_water_from_ocean(:)
end subroutine run_lake_model_jsbach

! Write a field of lake numbers out
subroutine write_lake_numbers_field_interface(timestep,grid_information,working_directory)
  integer :: timestep
  type(gridinformation) :: grid_information
  character(len = *), intent(in),optional :: working_directory
    call write_lake_numbers_field(working_directory,global_lake_parameters, &
                                  global_lake_fields,timestep,grid_information)
end subroutine write_lake_numbers_field_interface

! Write a field of diagnostic lake volumes out - the same volume is written for every
! point in the lake
subroutine write_diagnostic_lake_volumes_interface(timestep,grid_information)
  integer :: timestep
  type(gridinformation) :: grid_information
    call write_diagnostic_lake_volumes(global_lake_parameters, &
                                       global_lake_prognostics, &
                                       global_lake_fields, &
                                       timestep, &
                                       grid_information)
end subroutine write_diagnostic_lake_volumes_interface

! Get the lake prognostics object
function get_lake_prognostics() result(value)
  type(lakeprognostics), pointer :: value
    value => global_lake_prognostics
end function get_lake_prognostics

! Get the lake fields object
function get_lake_fields() result(value)
  type(lakefields), pointer :: value
    value => global_lake_fields
end function get_lake_fields

end module icosohedral_lake_model_interface_mod
