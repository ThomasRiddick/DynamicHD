module icosohedral_hd_model_mod

use icosohedral_lake_model_interface_mod

implicit none

type, public :: riverparameters
  integer, pointer, dimension(:) :: next_cell_index
  integer, pointer, dimension(:) :: river_reservoir_nums
  integer, pointer, dimension(:) :: overland_reservoir_nums
  integer, pointer, dimension(:) :: base_reservoir_nums
  real, pointer, dimension(:) :: river_retention_coefficients
  real, pointer, dimension(:) :: overland_retention_coefficients
  real, pointer, dimension(:) :: base_retention_coefficients
  logical, pointer, dimension(:) :: landsea_mask
  logical, pointer, dimension(:) :: cascade_flag
  integer :: ncells
  contains
    procedure :: initialiseriverparameters
    procedure :: riverparametersdestructor
end type riverparameters

interface riverparameters
  procedure :: riverparametersconstructor
end interface

type, public :: riverprognosticfields
  real, pointer,     dimension(:)   :: runoff
  real, pointer,     dimension(:)   :: drainage
  real, pointer,     dimension(:) :: lake_evaporation
  real, pointer,     dimension(:)   :: river_inflow
  real, pointer,     dimension(:,:)   :: base_flow_reservoirs
  real, pointer,     dimension(:,:)   :: overland_flow_reservoirs
  real, pointer,     dimension(:,:) :: river_flow_reservoirs
  real, allocatable, dimension(:)   :: water_to_ocean
  contains
    procedure :: initialiseriverprognosticfields
    procedure :: riverprognosticfieldsdestructor
end type riverprognosticfields

interface riverprognosticfields
  procedure :: riverprognosticfieldsconstructor,&
               riverprognosticfieldsgridonlyconstructor
end interface

type riverdiagnosticfields
  real, pointer, dimension(:) :: runoff_to_rivers
  real, pointer, dimension(:) :: drainage_to_rivers
  real, pointer, dimension(:) :: river_outflow
  real, pointer, dimension(:) :: flow_in
  contains
    procedure :: initialiseriverdiagnosticfields
    procedure :: riverdiagnosticfieldsdestructor
end type riverdiagnosticfields

interface riverdiagnosticfields
  procedure :: riverdiagnosticfieldsconstructor
end interface

type riverdiagnosticoutputfields
  real, allocatable, dimension(:) :: cumulative_river_flow
  contains
    procedure :: initialiseriverdiagnosticoutputfields
    procedure :: riverdiagnosticoutputfieldsdestructor
end type riverdiagnosticoutputfields

interface riverdiagnosticoutputfields
  procedure :: riverdiagnosticoutputfieldsconstructor
end interface

type prognostics
  type(riverparameters), pointer :: river_parameters
  type(riverprognosticfields), pointer :: river_fields
  type(lakeinterfaceprognosticfields), pointer :: lake_interface_fields
  type(riverdiagnosticfields), pointer :: river_diagnostic_fields
  type(riverdiagnosticoutputfields), pointer :: river_diagnostic_output_fields
  logical :: using_lakes
  contains
    procedure :: initialiseprognostics
    procedure :: prognosticsdestructor
end type prognostics

interface prognostics
  procedure :: prognosticsconstructor
end interface

contains

subroutine initialiseriverparameters(this,next_cell_index_in,river_reservoir_nums_in, &
                                     overland_reservoir_nums_in, &
                                     base_reservoir_nums_in, &
                                     river_retention_coefficients_in, &
                                     overland_retention_coefficients_in, &
                                     base_retention_coefficients_in, &
                                     landsea_mask_in)
  class(riverparameters) :: this
  integer, pointer, dimension(:) :: next_cell_index_in
  integer, pointer, dimension(:) :: river_reservoir_nums_in
  integer, pointer, dimension(:) :: overland_reservoir_nums_in
  integer, pointer, dimension(:) :: base_reservoir_nums_in
  real, pointer, dimension(:) :: river_retention_coefficients_in
  real, pointer, dimension(:) :: overland_retention_coefficients_in
  real, pointer, dimension(:) :: base_retention_coefficients_in
  logical, pointer, dimension(:) :: landsea_mask_in
    this%next_cell_index => next_cell_index_in
    this%river_reservoir_nums => river_reservoir_nums_in
    this%overland_reservoir_nums => overland_reservoir_nums_in
    this%base_reservoir_nums => base_reservoir_nums_in
    this%river_retention_coefficients => river_retention_coefficients_in
    this%overland_retention_coefficients => overland_retention_coefficients_in
    this%base_retention_coefficients => base_retention_coefficients_in
    this%landsea_mask => landsea_mask_in
    allocate(this%cascade_flag,mold=landsea_mask_in)
    this%cascade_flag = .not. landsea_mask_in
    this%ncells = size(this%next_cell_index,1)
    where(next_cell_index_in == -2)
      this%cascade_flag = .False.
    end where
end subroutine initialiseriverparameters

function riverparametersconstructor(next_cell_index_in,river_reservoir_nums_in, &
                                    overland_reservoir_nums_in, &
                                    base_reservoir_nums_in, &
                                    river_retention_coefficients_in, &
                                    overland_retention_coefficients_in, &
                                    base_retention_coefficients_in, &
                                    landsea_mask_in) result(constructor)
  type(riverparameters), pointer :: constructor
  integer, pointer, dimension(:), intent(in) :: next_cell_index_in
  integer, pointer, dimension(:), intent(in) :: river_reservoir_nums_in
  integer, pointer, dimension(:), intent(in) :: overland_reservoir_nums_in
  integer, pointer, dimension(:), intent(in) :: base_reservoir_nums_in
  real, pointer, dimension(:), intent(in) :: river_retention_coefficients_in
  real, pointer, dimension(:), intent(in) :: overland_retention_coefficients_in
  real, pointer, dimension(:), intent(in) :: base_retention_coefficients_in
  logical, pointer, dimension(:), intent(in) :: landsea_mask_in
    allocate(constructor)
    call constructor%initialiseriverparameters(next_cell_index_in,river_reservoir_nums_in, &
                                               overland_reservoir_nums_in, &
                                               base_reservoir_nums_in, &
                                               river_retention_coefficients_in, &
                                               overland_retention_coefficients_in, &
                                               base_retention_coefficients_in, &
                                               landsea_mask_in)
end function riverparametersconstructor

subroutine riverparametersdestructor(this)
  class(riverparameters) :: this
    deallocate(this%next_cell_index)
    deallocate(this%river_reservoir_nums)
    deallocate(this%overland_reservoir_nums)
    deallocate(this%base_reservoir_nums)
    deallocate(this%river_retention_coefficients)
    deallocate(this%overland_retention_coefficients)
    deallocate(this%base_retention_coefficients)
    deallocate(this%landsea_mask)
    deallocate(this%cascade_flag)
end subroutine riverparametersdestructor

subroutine initialiseriverprognosticfields(this,river_inflow_in, &
                                           base_flow_reservoirs_in, &
                                           overland_flow_reservoirs_in, &
                                           river_flow_reservoirs_in)
  class(riverprognosticfields) :: this
  real, pointer, dimension(:)   :: river_inflow_in
  real, pointer, dimension(:,:)   :: base_flow_reservoirs_in
  real, pointer, dimension(:,:)   :: overland_flow_reservoirs_in
  real, pointer, dimension(:,:) :: river_flow_reservoirs_in
    allocate(this%runoff,mold=river_inflow_in)
    allocate(this%drainage,mold=river_inflow_in)
    allocate(this%lake_evaporation,mold=river_inflow_in)
    allocate(this%water_to_ocean,mold=river_inflow_in)
    this%runoff(:) = 0
    this%drainage(:) = 0
    this%water_to_ocean(:) = 0
    this%lake_evaporation(:) = 0
    this%river_inflow => river_inflow_in
    this%base_flow_reservoirs => base_flow_reservoirs_in
    this%overland_flow_reservoirs => overland_flow_reservoirs_in
    this%river_flow_reservoirs => river_flow_reservoirs_in
end subroutine initialiseriverprognosticfields

subroutine riverprognosticfieldsdestructor(this)
  class(riverprognosticfields) :: this
    deallocate(this%runoff)
    deallocate(this%drainage)
    deallocate(this%lake_evaporation)
    deallocate(this%water_to_ocean)
    deallocate(this%river_inflow)
    deallocate(this%base_flow_reservoirs)
    deallocate(this%overland_flow_reservoirs)
    deallocate(this%river_flow_reservoirs)
end subroutine riverprognosticfieldsdestructor

function riverprognosticfieldsconstructor(river_inflow_in, &
                                          base_flow_reservoirs_in, &
                                          overland_flow_reservoirs_in, &
                                          river_flow_reservoirs_in) &
    result(constructor)
  real, pointer, dimension(:)   :: river_inflow_in
  real, pointer, dimension(:,:)   :: base_flow_reservoirs_in
  real, pointer, dimension(:,:)   :: overland_flow_reservoirs_in
  real, pointer, dimension(:,:) :: river_flow_reservoirs_in
  type(riverprognosticfields), pointer :: constructor
    allocate(constructor)
    call constructor%initialiseriverprognosticfields(river_inflow_in, &
                                                     base_flow_reservoirs_in, &
                                                     overland_flow_reservoirs_in, &
                                                     river_flow_reservoirs_in)
end function riverprognosticfieldsconstructor

function riverprognosticfieldsgridonlyconstructor(ncells,nres_b,nres_o,nres_r) &
    result(constructor)
  integer, intent(in) :: ncells,nres_b,nres_o,nres_r
  real, pointer, dimension(:)   :: river_inflow_in
  real, pointer, dimension(:,:)   :: base_flow_reservoirs_in
  real, pointer, dimension(:,:)   :: overland_flow_reservoirs_in
  real, pointer, dimension(:,:) :: river_flow_reservoirs_in
  type(riverprognosticfields), pointer :: constructor
    allocate(river_inflow_in(ncells))
    allocate(base_flow_reservoirs_in(ncells,nres_b))
    allocate(overland_flow_reservoirs_in(ncells,nres_o))
    allocate(river_flow_reservoirs_in(ncells,nres_r))
    river_inflow_in(:) = 0
    base_flow_reservoirs_in(:,:) = 0
    overland_flow_reservoirs_in(:,:) = 0
    river_flow_reservoirs_in(:,:) = 0
    allocate(constructor)
    call constructor%initialiseriverprognosticfields(river_inflow_in, &
                                                     base_flow_reservoirs_in, &
                                                     overland_flow_reservoirs_in, &
                                                     river_flow_reservoirs_in)
end function riverprognosticfieldsgridonlyconstructor

subroutine initialiseriverdiagnosticfields(this,river_parameters)
  class(riverdiagnosticfields), intent(inout) :: this
  type(riverparameters), intent(in) :: river_parameters
    allocate(this%runoff_to_rivers(river_parameters%ncells))
    allocate(this%drainage_to_rivers(river_parameters%ncells))
    allocate(this%river_outflow(river_parameters%ncells))
    allocate(this%flow_in(river_parameters%ncells))
    this%runoff_to_rivers = 0.0
    this%drainage_to_rivers = 0.0
    this%river_outflow = 0.0
    this%flow_in = 0.0
end subroutine initialiseriverdiagnosticfields

function riverdiagnosticfieldsconstructor(river_parameters) result(constructor)
  type(riverparameters), intent(in) :: river_parameters
  type(riverdiagnosticfields), pointer :: constructor
    allocate(constructor)
    call constructor%initialiseriverdiagnosticfields(river_parameters)
end function riverdiagnosticfieldsconstructor

subroutine riverdiagnosticfieldsdestructor(this)
  class(riverdiagnosticfields), intent(inout) :: this
        deallocate(this%runoff_to_rivers)
        deallocate(this%drainage_to_rivers)
        deallocate(this%river_outflow)
        deallocate(this%flow_in)
end subroutine riverdiagnosticfieldsdestructor
subroutine initialiseriverdiagnosticoutputfields(this,river_parameters)
  class(riverdiagnosticoutputfields) :: this
  type(riverparameters), intent(in) :: river_parameters
    allocate(this%cumulative_river_flow(river_parameters%ncells))
    this%cumulative_river_flow = 0.0
end subroutine initialiseriverdiagnosticoutputfields

function riverdiagnosticoutputfieldsconstructor(river_parameters) result(constructor)
  type(riverdiagnosticoutputfields), pointer :: constructor
  type(riverparameters), intent(in) :: river_parameters
    allocate(constructor)
    call constructor%initialiseriverdiagnosticoutputfields(river_parameters)
end function riverdiagnosticoutputfieldsconstructor

subroutine riverdiagnosticoutputfieldsdestructor(this)
  class(riverdiagnosticoutputfields) :: this
    deallocate(this%cumulative_river_flow)
end subroutine riverdiagnosticoutputfieldsdestructor

subroutine initialiseprognostics(this,using_lakes,river_parameters,river_fields)
  class(prognostics) :: this
  type(riverparameters),target :: river_parameters
  type(riverprognosticfields),target,intent(inout) :: river_fields
  logical :: using_lakes
    this%river_parameters => river_parameters
    this%river_fields => river_fields
    if(using_lakes) then
      this%lake_interface_fields => lakeinterfaceprognosticfields(river_parameters%ncells)
    end if
    this%river_diagnostic_fields => riverdiagnosticfields(river_parameters)
    this%river_diagnostic_output_fields => riverdiagnosticoutputfields(river_parameters)
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

subroutine prognosticsdestructor(this)
  class(prognostics) :: this
    if(this%using_lakes) then
      call this%lake_interface_fields%&
           &lakeinterfaceprognosticfieldsdestructor()
      deallocate(this%lake_interface_fields)
    end if
    call this%river_diagnostic_fields%riverdiagnosticfieldsdestructor()
    call this%river_diagnostic_output_fields%&
              &riverdiagnosticoutputfieldsdestructor()
    call this%river_fields%riverprognosticfieldsdestructor()
    call this%river_parameters%riverparametersdestructor()
    deallocate(this%river_fields)
    deallocate(this%river_parameters)
    deallocate(this%river_diagnostic_output_fields)
    deallocate(this%river_diagnostic_fields)
end subroutine prognosticsdestructor

subroutine set_runoff_and_drainage(prognostic_fields,runoff,drainage)
  type(prognostics), intent(inout) :: prognostic_fields
  real   ,dimension(:) :: runoff
  real   ,dimension(:) :: drainage
    prognostic_fields%river_fields%runoff(:) = runoff(:)
    prognostic_fields%river_fields%drainage(:) = drainage(:)
end subroutine set_runoff_and_drainage

subroutine run_hd(prognostic_fields)
  type(prognostics), intent(inout) :: prognostic_fields
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
                 prognostic_fields%river_parameters%overland_reservoir_nums, &
                 prognostic_fields%river_parameters%cascade_flag, &
                 prognostic_fields%river_parameters%ncells)
    call cascade(prognostic_fields%river_fields%base_flow_reservoirs, &
                 prognostic_fields%river_fields%drainage, &
                 prognostic_fields%river_diagnostic_fields%drainage_to_rivers, &
                 prognostic_fields%river_parameters%base_retention_coefficients, &
                 prognostic_fields%river_parameters%base_reservoir_nums, &
                 prognostic_fields%river_parameters%cascade_flag, &
                 prognostic_fields%river_parameters%ncells)
    call cascade(prognostic_fields%river_fields%river_flow_reservoirs, &
                 prognostic_fields%river_fields%river_inflow, &
                 prognostic_fields%river_diagnostic_fields%river_outflow, &
                 prognostic_fields%river_parameters%river_retention_coefficients, &
                 prognostic_fields%river_parameters%river_reservoir_nums, &
                 prognostic_fields%river_parameters%cascade_flag, &
                 prognostic_fields%river_parameters%ncells)
    prognostic_fields%river_fields%river_inflow(:) = 0.0
    prognostic_fields%river_diagnostic_fields%flow_in(:) =  &
        prognostic_fields%river_diagnostic_fields%river_outflow(:) + &
        prognostic_fields%river_diagnostic_fields%runoff_to_rivers(:) + &
        prognostic_fields%river_diagnostic_fields%drainage_to_rivers(:)
    call route(prognostic_fields%river_parameters%next_cell_index, &
               prognostic_fields%river_diagnostic_fields%flow_in, &
               prognostic_fields%river_fields%river_inflow, &
               prognostic_fields%river_parameters%ncells)
    prognostic_fields%river_diagnostic_fields%river_outflow(:) = 0.0
    prognostic_fields%river_diagnostic_fields%runoff_to_rivers(:) = 0.0
    prognostic_fields%river_diagnostic_fields%drainage_to_rivers(:) = 0.0
    where (prognostic_fields%river_parameters%next_cell_index == -1 .or. &
           prognostic_fields%river_parameters%next_cell_index ==  0 .or. &
           prognostic_fields%river_parameters%next_cell_index ==  -5)
        prognostic_fields%river_fields%water_to_ocean = &
              prognostic_fields%river_fields%river_inflow + &
              prognostic_fields%river_fields%runoff + &
              prognostic_fields%river_fields%drainage
        prognostic_fields%river_fields%river_inflow = 0.0
    else where
        prognostic_fields%river_fields%water_to_ocean = 0.0
    end where
    if (prognostic_fields%using_lakes) then
      where(prognostic_fields%river_parameters%next_cell_index == -2)
        prognostic_fields%lake_interface_fields%water_to_lakes = &
            prognostic_fields%river_fields%river_inflow + &
            prognostic_fields%river_fields%runoff + &
            prognostic_fields%river_fields%drainage - &
            prognostic_fields%river_fields%lake_evaporation
        prognostic_fields%river_fields%river_inflow = 0.0
        prognostic_fields%river_fields%water_to_ocean = &
          -1.0*prognostic_fields%lake_interface_fields%lake_water_from_ocean
      end where
    end if
    prognostic_fields%river_fields%runoff(:) = 0.0
    prognostic_fields%river_fields%drainage(:) = 0.0
    if (prognostic_fields%using_lakes) then
      call run_lake_model(prognostic_fields%lake_interface_fields)
    end if
end subroutine run_hd

subroutine cascade(reservoirs,inflow,outflow,retention_coefficients, &
                   reservoir_nums,cascade_flag,ncells)
  real,    pointer, dimension(:), intent(in) :: inflow
  real,    pointer, dimension(:), intent(in) :: retention_coefficients
  integer, pointer, dimension(:), intent(in) :: reservoir_nums
  logical, pointer, dimension(:), intent(in) :: cascade_flag
  real,    pointer, dimension(:,:), intent(inout) :: reservoirs
  real,    pointer, dimension(:), intent(inout) :: outflow
  integer, intent(in) :: ncells
  integer :: i,n
  real :: flow
  real :: new_reservoir_value
      do i=1,ncells
        if (cascade_flag(i)) then
          flow = inflow(i)
          do n = 1,reservoir_nums(i)
            new_reservoir_value = reservoirs(i,n) + flow
            flow = new_reservoir_value/(retention_coefficients(i)+1.0)
            reservoirs(i,n) = new_reservoir_value - flow
          end do
          outflow(i) = flow
        end if
    end do
end subroutine cascade

subroutine route(next_cell_index,flow_in,flow_out,ncells)
  integer, pointer, dimension(:) :: next_cell_index
  real, pointer, dimension(:) :: flow_in
  real, pointer, dimension(:) :: flow_out
  integer, intent(in) :: ncells
  real :: flow_in_local
  integer :: i
  integer :: new_i
    do i=1,ncells
      flow_in_local = flow_in(i)
      if (flow_in_local /= 0.0) then
        new_i = next_cell_index(i)
        if (new_i == -5 .or. new_i == -2 .or. &
            new_i == -1 .or. new_i == 0) then
          flow_in_local = flow_out(i) + flow_in_local
          flow_out(i) = flow_in_local
        else
          flow_out(new_i) = flow_in_local + flow_out(new_i)
        end if
      end if
    end do
end subroutine route

subroutine set_drainage(prognostic_fields,drainage)
  type(prognostics), intent(inout) :: prognostic_fields
  real, allocatable, dimension(:) :: drainage
    prognostic_fields%river_fields%drainage = drainage
end subroutine set_drainage

subroutine set_runoff(prognostic_fields,runoff)
  type(prognostics), intent(inout) :: prognostic_fields
  real, allocatable, dimension(:) :: runoff
    prognostic_fields%river_fields%runoff = runoff
end subroutine set_runoff

subroutine set_lake_evaporation(prognostic_fields,lake_evaporation)
  type(prognostics), intent(inout) :: prognostic_fields
  real, dimension(:) :: lake_evaporation
    prognostic_fields%river_fields%lake_evaporation = lake_evaporation
end subroutine set_lake_evaporation

subroutine distribute_spillover(prognostic_fields, &
                                initial_spillover_to_rivers)
  type(prognostics), intent(inout) :: prognostic_fields
  real, pointer, dimension(:), intent(in) :: initial_spillover_to_rivers
    where (prognostic_fields%river_parameters%cascade_flag &
           .and. initial_spillover_to_rivers > 0)
      prognostic_fields%river_fields%river_inflow =  &
        prognostic_fields%river_fields%river_inflow + initial_spillover_to_rivers
    elsewhere (initial_spillover_to_rivers > 0)
      prognostic_fields%river_diagnostic_fields%river_outflow = &
        initial_spillover_to_rivers
    end where
end subroutine distribute_spillover

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

end module icosohedral_hd_model_mod
