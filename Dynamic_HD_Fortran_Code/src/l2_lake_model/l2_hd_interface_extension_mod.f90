module l2_hd_interface_extension_mod

use latlon_hd_model_mod, only: riverparameters, &
                               riverprognosticfields, &
                               prognostics
use latlon_hd_model_interface_mod, only: global_prognostics, &
                                         global_step_length, &
                                         write_output
!use latlon_hd_model_io_mod
use l2_lake_model_mod, only: dp, lakemodelparameters, &
                             lakeparameters

implicit none

contains

subroutine l2_init_hd_model_for_testing(river_parameters,river_fields,using_lakes, &
                                        lake_model_parameters, lake_parameters_as_array, &
                                        initial_water_to_lake_centers, &
                                        initial_spillover_to_rivers)
  logical, intent(in) :: using_lakes
  type(riverparameters), intent(in) :: river_parameters
  type(riverprognosticfields), pointer, optional, intent(inout) :: river_fields
  type(lakemodelparameters), pointer, optional, intent(in) :: lake_model_parameters
  real(dp), allocatable, dimension(:), optional, intent(in) :: lake_parameters_as_array
  real(dp), pointer, dimension(:,:), optional, intent(in) :: initial_spillover_to_rivers
  real(dp), pointer, dimension(:,:), optional, intent(in) :: initial_water_to_lake_centers
  real(dp), pointer, dimension(:,:) :: initial_spillover_to_rivers_local
  real(dp), pointer, dimension(:,:) :: initial_water_to_lake_centers_local
    if(associated(global_prognostics)) then
      if (using_lakes) then
        call global_prognostics%lake_interface_fields%&
             &lakeinterfaceprognosticfieldsdestructor()
        deallocate(global_prognostics%lake_interface_fields)
      end if
      call global_prognostics%river_diagnostic_fields%riverdiagnosticfieldsdestructor()
      deallocate(global_prognostics%river_diagnostic_fields)
      call global_prognostics%river_diagnostic_output_fields%&
                              &riverdiagnosticoutputfieldsdestructor()
      deallocate(global_prognostics%river_diagnostic_output_fields)
      deallocate(global_prognostics)
    end if
    global_prognostics => prognostics(using_lakes,river_parameters,river_fields)
    if (using_lakes) then
      allocate(initial_water_to_lake_centers_local(lake_model_parameters%nlat_lake, &
                                                   lake_model_parameters%nlon_lake))
      if (present(initial_water_to_lake_centers)) then
        initial_water_to_lake_centers_local(:,:) = &
          initial_water_to_lake_centers(:,:)
      else
        initial_water_to_lake_centers_local(:,:) = 0.0_dp
      end if
      call init_lake_model_test(lake_model_parameters,lake_parameters_as_array, &
                                initial_water_to_lake_centers_local, &
                                global_prognostics%lake_interface_fields, &
                                global_step_length)
      deallocate(initial_water_to_lake_centers_local)
      allocate(initial_spillover_to_rivers_local(river_parameters%nlat, &
                                                 river_parameters%nlon))
      if (present(initial_spillover_to_rivers)) then
        initial_spillover_to_rivers_local(:,:) = &
          initial_spillover_to_rivers(:,:)/global_step_length
      else
        initial_spillover_to_rivers_local(:,:) = 0.0_dp
      end if
      initial_spillover_to_rivers_local(:,:) = initial_spillover_to_rivers_local(:,:) + &
        global_prognostics%lake_interface_fields%water_from_lakes(:,:)
      global_prognostics%lake_interface_fields%water_from_lakes(:,:) = 0.0_dp
      call distribute_spillover(global_prognostics,initial_spillover_to_rivers_local)
      deallocate(initial_spillover_to_rivers_local)
    end if
    write_output = .false.
end subroutine l2_init_hd_model_for_testing

end module l2_hd_interface_extension_mod
