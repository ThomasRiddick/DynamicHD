module latlon_hd_model_mod

use latlon_lake_model_interface_mod

implicit none

type, public :: riverparameters
  real, allocatable, dimension(:,:) :: rdirs
  integer, allocatable, dimension(:,:) :: river_reservoir_nums
  integer, allocatable, dimension(:,:) :: overland_reservoir_nums
  integer, allocatable, dimension(:,:) :: base_reservoir_nums
  real, allocatable, dimension(:,:) :: river_retention_coefficients
  real, allocatable, dimension(:,:) :: overland_retention_coefficients
  real, allocatable, dimension(:,:) :: base_retention_coefficients
  logical, allocatable, dimension(:,:) :: landsea_mask
  logical, allocatable, dimension(:,:) :: cascade_flag
  integer :: nlat,nlon
  contains
    procedure :: initialiseriverparameters
end type riverparameters

interface riverparameters
  procedure :: riverparametersconstructor
end interface

type, public :: riverprognosticfields
  real, allocatable, dimension(:,:)   :: runoff
  real, allocatable, dimension(:,:)   :: drainage
  real, allocatable, dimension(:,:)   :: river_inflow
  real, allocatable, dimension(:,:,:)   :: base_flow_reservoirs
  real, allocatable, dimension(:,:,:)   :: overland_flow_reservoirs
  real, allocatable, dimension(:,:,:) :: river_flow_reservoirs
  real, allocatable, dimension(:,:)   :: water_to_ocean
  contains
    procedure :: initialiseriverprognosticfields
end type riverprognosticfields

interface riverprognosticfields
  procedure :: riverprognosticfieldsconstructor
end interface

type riverdiagnosticfields
  real, allocatable, dimension(:,:) :: runoff_to_rivers
  real, allocatable, dimension(:,:) :: drainage_to_rivers
  real, allocatable, dimension(:,:) :: river_outflow
  contains
    procedure :: initialiseriverdiagnosticfields
end type riverdiagnosticfields

interface riverdiagnosticfields
  procedure :: riverdiagnosticfieldsconstructor
end interface

type riverdiagnosticoutputfields
  real, allocatable, dimension(:,:) :: cumulative_river_flow
  contains
    procedure :: initialiseriverdiagnosticoutputfields
end type riverdiagnosticoutputfields

interface riverdiagnosticoutputfields
  procedure :: riverdiagnosticoutputfieldsconstructor
end interface

type prognostics
  type(riverparameters) :: river_parameters
  type(riverprognosticfields) :: river_fields
  type(lakeinterfaceprognosticfields) :: lake_interface_fields
  type(riverdiagnosticfields) :: river_diagnostic_fields
  type(riverdiagnosticoutputfields) :: river_diagnostic_output_fields
  logical :: using_lakes
  contains
    procedure :: initialiseprognostics
end type prognostics

interface prognostics
  procedure :: prognosticsconstructor
end interface

contains

subroutine initialiseriverparameters(this,rdirs_in,river_reservoir_nums_in, &
                                     overland_reservoir_nums_in, &
                                     base_reservoir_nums_in, &
                                     river_retention_coefficients_in, &
                                     overland_retention_coefficients_in, &
                                     base_retention_coefficients_in, &
                                     landsea_mask_in)
  class(riverparameters) :: this
  real, allocatable, dimension(:,:) :: rdirs_in
  integer, allocatable, dimension(:,:) :: river_reservoir_nums_in
  integer, allocatable, dimension(:,:) :: overland_reservoir_nums_in
  integer, allocatable, dimension(:,:) :: base_reservoir_nums_in
  real, allocatable, dimension(:,:) :: river_retention_coefficients_in
  real, allocatable, dimension(:,:) :: overland_retention_coefficients_in
  real, allocatable, dimension(:,:) :: base_retention_coefficients_in
  logical, allocatable, dimension(:,:) :: landsea_mask_in
    this%rdirs = rdirs_in
    this%river_reservoir_nums = river_reservoir_nums_in
    this%overland_reservoir_nums = overland_reservoir_nums_in
    this%base_reservoir_nums = base_reservoir_nums_in
    this%river_retention_coefficients = river_retention_coefficients_in
    this%overland_retention_coefficients = overland_retention_coefficients_in
    this%base_retention_coefficients = base_retention_coefficients_in
    this%landsea_mask = landsea_mask_in
    this%cascade_flag(:,:) = .not. landsea_mask_in(:,:)
end subroutine

function riverparametersconstructor(rdirs_in,river_reservoir_nums_in, &
                                    overland_reservoir_nums_in, &
                                    base_reservoir_nums_in, &
                                    river_retention_coefficients_in, &
                                    overland_retention_coefficients_in, &
                                    base_retention_coefficients_in, &
                                    landsea_mask_in) result(constructor)
  type(riverparameters), allocatable :: constructor
  real, allocatable, dimension(:,:), intent(in) :: rdirs_in
  integer, allocatable, dimension(:,:), intent(in) :: river_reservoir_nums_in
  integer, allocatable, dimension(:,:), intent(in) :: overland_reservoir_nums_in
  integer, allocatable, dimension(:,:), intent(in) :: base_reservoir_nums_in
  real, allocatable, dimension(:,:), intent(in) :: river_retention_coefficients_in
  real, allocatable, dimension(:,:), intent(in) :: overland_retention_coefficients_in
  real, allocatable, dimension(:,:), intent(in) :: base_retention_coefficients_in
  logical, allocatable, dimension(:,:), intent(in) :: landsea_mask_in
    allocate(constructor)
    call constructor%initialiseriverparameters(rdirs_in,river_reservoir_nums_in, &
                                               overland_reservoir_nums_in, &
                                               base_reservoir_nums_in, &
                                               river_retention_coefficients_in, &
                                               overland_retention_coefficients_in, &
                                               base_retention_coefficients_in, &
                                               landsea_mask_in)
end function riverparametersconstructor

subroutine initialiseriverprognosticfields(this,river_inflow_in, &
                                           base_flow_reservoirs_in, &
                                           overland_flow_reservoirs_in, &
                                           river_flow_reservoirs_in)
  class(riverprognosticfields) :: this
  real, allocatable, dimension(:,:)   :: river_inflow_in
  real, allocatable, dimension(:,:,:)   :: base_flow_reservoirs_in
  real, allocatable, dimension(:,:,:)   :: overland_flow_reservoirs_in
  real, allocatable, dimension(:,:,:) :: river_flow_reservoirs_in
    this%runoff(:,:) = 0
    this%drainage(:,:) = 0
    this%water_to_ocean(:,:) = 0
    this%river_inflow = river_inflow_in
    this%base_flow_reservoirs = base_flow_reservoirs_in
    this%overland_flow_reservoirs = overland_flow_reservoirs_in
    this%river_flow_reservoirs = river_flow_reservoirs_in
end subroutine initialiseriverprognosticfields

function riverprognosticfieldsconstructor(river_inflow_in, &
                                          base_flow_reservoirs_in, &
                                          overland_flow_reservoirs_in, &
                                          river_flow_reservoirs_in) &
    result(constructor)
  real, allocatable, dimension(:,:)   :: river_inflow_in
  real, allocatable, dimension(:,:,:)   :: base_flow_reservoirs_in
  real, allocatable, dimension(:,:,:)   :: overland_flow_reservoirs_in
  real, allocatable, dimension(:,:,:) :: river_flow_reservoirs_in
  type(riverprognosticfields), allocatable :: constructor
    allocate(constructor)
    call constructor%initialiseriverprognosticfields(river_inflow_in, &
                                                     base_flow_reservoirs_in, &
                                                     overland_flow_reservoirs_in, &
                                                     river_flow_reservoirs_in)
end function riverprognosticfieldsconstructor

subroutine initialiseriverdiagnosticfields(this)
  class(riverdiagnosticfields) :: this
    this%runoff_to_rivers = 0.0
    this%drainage_to_rivers = 0.0
    this%river_outflow = 0.0
end subroutine initialiseriverdiagnosticfields

function riverdiagnosticfieldsconstructor() result(constructor)
  type(riverdiagnosticfields), allocatable :: constructor
    allocate(constructor)
    call constructor%initialiseriverdiagnosticfields()
end function riverdiagnosticfieldsconstructor

subroutine initialiseriverdiagnosticoutputfields(this)
  class(riverdiagnosticoutputfields) :: this
    this%cumulative_river_flow = 0.0
end subroutine initialiseriverdiagnosticoutputfields

function riverdiagnosticoutputfieldsconstructor() result(constructor)
  type(riverdiagnosticoutputfields), allocatable :: constructor
    allocate(constructor)
    call constructor%initialiseriverdiagnosticoutputfields()
end function riverdiagnosticoutputfieldsconstructor

subroutine initialiseprognostics(this,using_lakes,river_parameters,river_fields)
  class(prognostics) :: this
  type(riverparameters) :: river_parameters
  type(riverprognosticfields) :: river_fields
  logical :: using_lakes
    this%river_parameters = river_parameters
    this%river_fields = river_fields
    this%lake_interface_fields = lakeinterfaceprognosticfields()
    this%river_diagnostic_fields = riverdiagnosticfields()
    this%river_diagnostic_output_fields = riverdiagnosticoutputfields()
    this%using_lakes = using_lakes
end subroutine initialiseprognostics

function prognosticsconstructor(using_lakes,river_parameters,river_fields) result(constructor)
  type(prognostics), allocatable :: constructor
  type(riverparameters) :: river_parameters
  type(riverprognosticfields) :: river_fields
  logical :: using_lakes
    allocate(constructor)
    call constructor%initialiseprognostics(using_lakes,river_parameters,river_fields)
end function prognosticsconstructor

subroutine run_hd(prognostic_fields)
  type(prognostics), intent(inout) :: prognostic_fields
  real, allocatable, dimension(:,:) :: flow_in
    if (prognostic_fields%using_lakes) then
      where (prognostic_fields%river_parameters%cascade_flag)
        prognostic_fields%river_fields%river_inflow =  &
          prognostic_fields%river_fields%river_inflow + &
            prognostic_fields%lake_interface_fields%water_from_lakes
      elsewhere
        prognostic_fields%river_diagnostic_fields%river_outflow = &
            prognostic_fields%lake_interface_fields%water_from_lakes
      end where
    end if
    call cascade(prognostic_fields%river_fields%overland_flow_reservoirs, &
                 prognostic_fields%river_fields%runoff, &
                 prognostic_fields%river_diagnostic_fields%runoff_to_rivers, &
                 prognostic_fields%river_parameters%overland_retention_coefficients, &
                 prognostic_fields%river_parameters%base_reservoir_nums, &
                 prognostic_fields%river_parameters%cascade_flag, &
                 prognostic_fields%river_parameters%nlat, &
                 prognostic_fields%river_parameters%nlon)
    call cascade(prognostic_fields%river_fields%base_flow_reservoirs, &
                 prognostic_fields%river_fields%drainage, &
                 prognostic_fields%river_diagnostic_fields%drainage_to_rivers, &
                 prognostic_fields%river_parameters%base_retention_coefficients, &
                 prognostic_fields%river_parameters%base_reservoir_nums, &
                 prognostic_fields%river_parameters%cascade_flag, &
                 prognostic_fields%river_parameters%nlat, &
                 prognostic_fields%river_parameters%nlon)
    call cascade(prognostic_fields%river_fields%river_flow_reservoirs, &
                 prognostic_fields%river_fields%river_inflow, &
                 prognostic_fields%river_diagnostic_fields%river_outflow, &
                 prognostic_fields%river_parameters%river_retention_coefficients, &
                 prognostic_fields%river_parameters%river_reservoir_nums, &
                 prognostic_fields%river_parameters%cascade_flag, &
                 prognostic_fields%river_parameters%nlat, &
                 prognostic_fields%river_parameters%nlon)
    prognostic_fields%river_fields%river_inflow(:,:) = 0.0
    flow_in =  prognostic_fields%river_diagnostic_fields%river_outflow(:,:) + &
               prognostic_fields%river_diagnostic_fields%runoff_to_rivers(:,:) + &
               prognostic_fields%river_diagnostic_fields%drainage_to_rivers(:,:)
    call route(prognostic_fields%river_parameters%rdirs, &
               flow_in, &
               prognostic_fields%river_fields%river_inflow, &
               prognostic_fields%river_parameters%nlat, &
               prognostic_fields%river_parameters%nlon)
    prognostic_fields%river_diagnostic_fields%river_outflow(:,:) = 0.0
    prognostic_fields%river_diagnostic_fields%runoff_to_rivers(:,:) = 0.0
    prognostic_fields%river_diagnostic_fields%drainage_to_rivers(:,:) = 0.0
    where (prognostic_fields%river_parameters%rdirs == -1.0 .or. &
           prognostic_fields%river_parameters%rdirs ==  0.0 .or. &
           prognostic_fields%river_parameters%rdirs ==  5.0)
        prognostic_fields%river_fields%water_to_ocean = &
              prognostic_fields%river_fields%river_inflow + &
              prognostic_fields%river_fields%runoff + &
              prognostic_fields%river_fields%drainage
        prognostic_fields%river_fields%river_inflow = 0.0
    end where
    if (prognostic_fields%using_lakes) then
      where(prognostic_fields%river_parameters%rdirs == -2.0)
        prognostic_fields%lake_interface_fields%water_to_lakes = &
            prognostic_fields%river_fields%river_inflow + &
            prognostic_fields%river_fields%runoff + &
            prognostic_fields%river_fields%drainage
        prognostic_fields%river_fields%river_inflow = 0.0
      end where
    end if
    prognostic_fields%river_fields%runoff = 0.0
    prognostic_fields%river_fields%drainage = 0.0
    if (prognostic_fields%using_lakes) then
      call run_lake_model(prognostic_fields%lake_interface_fields)
    end if
end subroutine run_hd

subroutine cascade(reservoirs,inflow,outflow,retention_coefficients, &
                   reservoir_nums,cascade_flag,nlat,nlon)
  real,    allocatable, dimension(:,:), intent(in) :: inflow
  real,    allocatable, dimension(:,:), intent(in) :: retention_coefficients
  integer, allocatable, dimension(:,:), intent(in) :: reservoir_nums
  logical, allocatable, dimension(:,:), intent(in) :: cascade_flag
  real,    allocatable, dimension(:,:,:), intent(inout) :: reservoirs
  real,    allocatable, dimension(:,:), intent(out) :: outflow
  integer, intent(in) :: nlat,nlon
  integer :: i,j,n
  real :: flow
  real :: new_reservoir_value
    do i=1,nlat
      do j=1,nlon
        if (cascade_flag(i,j)) then
          flow = inflow(i,j)
          do n = 1,reservoir_nums(i,j)
            new_reservoir_value = reservoirs(i,j,n) + flow
            flow = new_reservoir_value/(retention_coefficients(i,j)+1.0)
            reservoirs(i,j,n) = new_reservoir_value - flow
          end do
          outflow(i,j) = flow
        end if
      end do
    end do
end subroutine cascade

subroutine route(flow_directions,flow_in,flow_out,nlat,nlon)
  real, allocatable, dimension(:,:) :: flow_directions
  real, allocatable, dimension(:,:) :: flow_in
  real, allocatable, dimension(:,:) :: flow_out
  integer, intent(in) :: nlat,nlon
  real :: flow_in_local
  real :: flow_direction
  integer :: i,j
  integer :: new_i,new_j
    do i=1,nlat
      do j=1,nlon
        flow_in_local = flow_in(i,j)
        if (flow_in_local /= 0.0) then
          flow_direction = flow_directions(i,j)
          if (flow_in_local == 5.0 .or. flow_in_local == -2.0 .or. &
              flow_in_local == -1.0 .or. flow_in_local == 0.0) then
            flow_in_local = flow_out(i,j) + flow_in_local
            flow_out(i,j) = flow_in_local
          else
            if (flow_direction == 1.0 .or. &
                flow_direction == 4.0 .or. &
                flow_direction == 7.0) then
              new_i = i - 1
              if (new_i == 0) new_i = nlon
            else if(flow_direction == 3.0 .or. &
                    flow_direction == 6.0 .or. &
                    flow_direction == 9.0) then
              new_i = i
            else
              new_i = i + 1
              if (new_i == nlon + 1) new_i = 1
            end if
            if (flow_direction <= 3.0) then
              new_j = j - 1
            else if (flow_direction <= 6.0) then
              new_j = j
            else
              new_j = j + 1
            end if
            flow_out(new_i,new_j) = flow_in_local + flow_out(new_i,new_j)
          end if
        end if
      end do
    end do
end subroutine route

subroutine set_drainage(prognostic_fields,drainage)
  type(prognostics), intent(inout) :: prognostic_fields
  real, allocatable, dimension(:,:) :: drainage
    prognostic_fields%river_fields%drainage = drainage
end subroutine set_drainage

subroutine set_runoff(prognostic_fields,runoff)
  type(prognostics), intent(inout) :: prognostic_fields
  real, allocatable, dimension(:,:) :: runoff
    prognostic_fields%river_fields%runoff = runoff
end subroutine set_runoff

! function handle_event(prognostic_fields::prognosticfields,
!                       print_results::PrintResults)
!   println("Timestep: $(print_results.timestep)")
!   print_river_results(prognostic_fields)
!   return prognostic_fields
! end

! function print_river_results(prognostic_fields::prognosticfields)
!   river_fields::riverprognosticfields = get_river_fields(prognostic_fields)
!   println("")
!   println("river Flow")
!   println(river_fields.river_inflow)
!   println("")
!   println("Water to Ocean")
!   println(river_fields.water_to_ocean)
! end

! function handle_event(prognostic_fields::prognosticfields,
!                       print_results::PrintSection)
!    print_river_results_section(prognostic_fields)
!   return prognostic_fields
! end

! function print_river_results_section(prognostic_fields::prognosticfields)
!   section_coords::LatLonSectionCoords = LatLonSectionCoords(65,75,125,165)
!   river_fields::riverprognosticfields = get_river_fields(prognostic_fields)
!   river_parameters::riverparameters = get_river_parameters(prognostic_fields)
!   for_section_with_line_breaks(river_parameters.grid,section_coords) do coords::Coords
!     @printf("%.2f ",river_fields.river_inflow(coords))
!     flush(stdout)
!   end
!   println()
! end

! struct WriteriverInitialValues <: Event end

! function handle_event(prognostic_fields::prognosticfields,
!                       print_results::WriteriverInitialValues)
!   river_fields::riverprognosticfields = get_river_fields(prognostic_fields)
!   river_parameters::riverparameters = get_river_parameters(prognostic_fields)
!   hd_start_filepath::AbstractString = "/Users/thomasriddick/Documents/data/temp/transient_sim_1/river_model_out.nc"
!   write_river_initial_values(hd_start_filepath,river_parameters,river_fields)
!   return prognostic_fields
! end

! function write_river_initial_values(hd_start_filepath::AbstractString,
!                                     river_parameters::riverparameters,
!                                     river_prognostic_fields::riverprognosticfields)
!   throw(UserError())
! end

! struct WriteriverFlow <: Event
!   timestep::Int64
! end

! function handle_event(prognostic_fields::prognosticfields,
!                       write_river_flow::WriteriverFlow)
!   river_fields::riverprognosticfields = get_river_fields(prognostic_fields)
!   river_parameters::riverparameters = get_river_parameters(prognostic_fields)
!   write_river_flow_field(river_parameters,river_fields.river_inflow,
!                          timestep=write_river_flow.timestep)
!   return prognostic_fields
! end

! function write_river_flow_field(river_parameters::riverparameters,
!                                 river_flow_field::Field{Float64};
!                                 timestep::Int64=-1)
!   throw(UserError())
! end

! struct AccumulateriverFlow <: Event end

! function handle_event(prognostic_fields::prognosticfields,
!                       accumulate_river_flow::AccumulateriverFlow)
!   river_fields::riverprognosticfields = get_river_fields(prognostic_fields)
!   river_diagnostic_output_fields::riverdiagnosticoutputfields =
!     get_river_diagnostic_output_fields(prognostic_fields)
!   river_diagnostic_output_fields.cumulative_river_flow += river_fields.river_inflow
!   return prognostic_fields
! end

! struct ResetCumulativeriverFlow <: Event end

! function handle_event(prognostic_fields::prognosticfields,
!                       reset_cumulative_river_flow::ResetCumulativeriverFlow)
!   river_diagnostic_output_fields::riverdiagnosticoutputfields =
!     get_river_diagnostic_output_fields(prognostic_fields)
!   fill!(river_diagnostic_output_fields.cumulative_river_flow,0.0)
!   return prognostic_fields
! end

! struct WriteMeanriverFlow <: Event
!   timestep::Int64
!   number_of_timesteps::Int64
! end

! function handle_event(prognostic_fields::prognosticfields,
!                       write_mean_river_flow::WriteMeanriverFlow)
!   river_diagnostic_output_fields::riverdiagnosticoutputfields =
!     get_river_diagnostic_output_fields(prognostic_fields)
!   river_parameters::riverparameters = get_river_parameters(prognostic_fields)
!   mean_river_flow::Field{Float64} =
!       Field{Float64}(river_parameters.grid,0.0)
!   for_all(river_parameters.grid) do coords::Coords
!     set!(mean_river_flow,coords,
!          river_diagnostic_output_fields.cumulative_river_flow(coords)/
!          convert(Float64,write_mean_river_flow.number_of_timesteps))
!   end
!   write_river_flow_field(river_parameters,mean_river_flow,
!                          timestep=write_mean_river_flow.timestep)
!   return prognostic_fields
! end

end module latlon_hd_model_mod
