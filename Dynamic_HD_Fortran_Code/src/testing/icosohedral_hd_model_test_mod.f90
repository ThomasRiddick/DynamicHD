module icosohedral_hd_model_test_mod
use fruit
implicit none

contains

    subroutine setup
    end subroutine setup

    subroutine teardown
    end subroutine teardown

subroutine testHdModel
   use icosohedral_hd_model_interface_mod
   use icosohedral_hd_model_mod

   type(riverparameters), pointer :: river_parameters
   type(riverprognosticfields), pointer :: river_fields
   integer,dimension(:), pointer :: river_reservoir_nums
   integer,dimension(:), pointer :: overland_reservoir_nums
   integer,dimension(:), pointer :: base_reservoir_nums
   real(dp)   ,dimension(:), pointer :: river_retention_coefficients
   real(dp)   ,dimension(:), pointer :: overland_retention_coefficients
   real(dp)   ,dimension(:), pointer :: base_retention_coefficients
   logical,dimension(:), pointer :: landsea_mask
   real(dp)   ,dimension(:), allocatable :: drainage
   real(dp)   ,dimension(:,:), allocatable :: drainages
   real(dp)   ,dimension(:,:), allocatable :: runoffs
   real(dp)   ,dimension(:), allocatable :: expected_water_to_ocean
   real(dp)   ,dimension(:), allocatable :: expected_river_inflow
   integer   ,dimension(:), pointer :: flow_directions
   type(prognostics), pointer :: global_prognostics_ptr
   integer :: timesteps, i
      timesteps = 200
      allocate(flow_directions(16))
      flow_directions = (/ 5, 6, 7, 8, &
                           8, 7, 11, 12, &
                          10,11, -1, 11, &
                          10,10, 11, 11 /)
      allocate(river_reservoir_nums(16))
      river_reservoir_nums(:) = 5
      allocate(overland_reservoir_nums(16))
      overland_reservoir_nums(:) = 1
      allocate(base_reservoir_nums(16))
      base_reservoir_nums(:) = 1
      allocate(river_retention_coefficients(16))
      river_retention_coefficients(:) = 0.7
      allocate(overland_retention_coefficients(16))
      overland_retention_coefficients(:) = 0.5
      allocate(base_retention_coefficients(16))
      base_retention_coefficients(:) = 0.1
      allocate(landsea_mask(16))
      landsea_mask(:) = .False.
      river_reservoir_nums(11) = 0
      overland_reservoir_nums(11) = 0
      base_reservoir_nums(11) = 0
      landsea_mask(11) = .True.
      river_parameters => riverparameters(flow_directions, &
                                          river_reservoir_nums, &
                                          overland_reservoir_nums, &
                                          base_reservoir_nums, &
                                          river_retention_coefficients, &
                                          overland_retention_coefficients, &
                                          base_retention_coefficients, &
                                          landsea_mask,1.0_dp,1.0_dp)
      river_fields => riverprognosticfields(16,1,1,5)
      allocate(drainage(16))
      drainage = (/ &
         1.0, 1.0, 1.0, 1.0, &
         1.0, 1.0, 1.0, 1.0, &
         1.0, 1.0, 0.0, 1.0, &
         1.0, 1.0, 1.0, 1.0 /)
      allocate(drainages(16,timesteps))
      allocate(runoffs(16,timesteps))
      do i = 1,timesteps
        drainages(:,i) = drainage(:)
        runoffs(:,i) = drainage(:)
      end do
      allocate(expected_water_to_ocean(16))
      expected_water_to_ocean = (/ &
          0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 30.0, 0.0, &
         0.0, 0.0, 0.0, 0.0 /)
      allocate(expected_river_inflow(16))
      expected_river_inflow = (/ &
          0.0, 0.0, 0.0, 0.0, &
         2.0, 2.0, 6.0, 6.0, &
         0.0, 6.0, 0.0, 8.0, &
         0.0, 0.0, 0.0, 0.0 /)
      call init_hd_model_for_testing(river_parameters,river_fields,using_lakes=.False.)
      call run_hd_model(timesteps,runoffs,drainages)
      global_prognostics_ptr => get_global_prognostics()
      call assert_equals(global_prognostics_ptr%river_fields%water_to_ocean, &
                         expected_water_to_ocean,16)
      call assert_equals(global_prognostics_ptr%river_fields%river_inflow,&
                         expected_river_inflow,16)
      deallocate(drainage)
      deallocate(drainages)
      deallocate(runoffs)
      deallocate(expected_water_to_ocean)
      deallocate(expected_river_inflow)
      call clean_hd_model()
end subroutine testHdModel

end module icosohedral_hd_model_test_mod
