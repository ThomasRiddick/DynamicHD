module latlon_hd_model_test_mod
use fruit
implicit none

contains

    subroutine setup
    end subroutine setup

    subroutine teardown
    end subroutine teardown

subroutine testHdModel
   use latlon_hd_model_interface_mod
   use latlon_hd_model_mod

   type(riverparameters), pointer :: river_parameters
   type(riverprognosticfields), pointer :: river_fields
   integer,dimension(:,:), pointer :: river_reservoir_nums
   integer,dimension(:,:), pointer :: overland_reservoir_nums
   integer,dimension(:,:), pointer :: base_reservoir_nums
   real(dp)   ,dimension(:,:), pointer :: river_retention_coefficients
   real(dp)   ,dimension(:,:), pointer :: overland_retention_coefficients
   real(dp)   ,dimension(:,:), pointer :: base_retention_coefficients
   logical,dimension(:,:), pointer :: landsea_mask
   real(dp)   ,dimension(:,:), pointer :: drainage
   real(dp)   ,dimension(:,:,:), pointer :: drainages
   real(dp)   ,dimension(:,:,:), pointer :: runoffs
   real(dp)   ,dimension(:,:), pointer :: expected_water_to_ocean
   real(dp)   ,dimension(:,:), pointer :: expected_river_inflow
   real(dp)   ,dimension(:,:), pointer :: flow_directions
   type(prognostics), pointer :: global_prognostics_ptr
   integer :: timesteps, i
      timesteps = 200
      allocate(flow_directions(4,4))
      flow_directions = transpose(reshape((/ 2.0, 2.0, 2.0, 2.0, &
                                   4.0, 6.0, 2.0, 2.0, &
                                   6.0, 6.0, 0.0, 4.0, &
                                   9.0, 8.0, 8.0, 7.0 /), &
                                  (/4,4/)))
      allocate(river_reservoir_nums(4,4))
      river_reservoir_nums(:,:) = 5
      allocate(overland_reservoir_nums(4,4))
      overland_reservoir_nums(:,:) = 1
      allocate(base_reservoir_nums(4,4))
      base_reservoir_nums(:,:) = 1
      allocate(river_retention_coefficients(4,4))
      river_retention_coefficients(:,:) = 0.7
      allocate(overland_retention_coefficients(4,4))
      overland_retention_coefficients(:,:) = 0.5
      allocate(base_retention_coefficients(4,4))
      base_retention_coefficients(:,:) = 0.1
      allocate(landsea_mask(4,4))
      landsea_mask(:,:) = .False.
      river_reservoir_nums(3,3) = 0
      overland_reservoir_nums(3,3) = 0
      base_reservoir_nums(3,3) = 0
      landsea_mask(3,3) = .True.
      river_parameters => riverparameters(flow_directions, &
                                          river_reservoir_nums, &
                                          overland_reservoir_nums, &
                                          base_reservoir_nums, &
                                          river_retention_coefficients, &
                                          overland_retention_coefficients, &
                                          base_retention_coefficients, &
                                          landsea_mask)
      river_fields => riverprognosticfields(4,4,1,1,5)
      allocate(drainage(4,4))
      drainage = transpose(reshape((/ &
         1.0, 1.0, 1.0, 1.0, &
         1.0, 1.0, 1.0, 1.0, &
         1.0, 1.0, 0.0, 1.0, &
         1.0, 1.0, 1.0, 1.0 /), &
         (/4,4/)))
      allocate(drainages(4,4,timesteps))
      allocate(runoffs(4,4,timesteps))
      do i = 1,timesteps
        drainages(:,:,i) = drainage(:,:)
        runoffs(:,:,i) = drainage(:,:)
      end do
      allocate(expected_water_to_ocean(4,4))
      expected_water_to_ocean = transpose(reshape((/ &
          0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 30.0, 0.0, &
         0.0, 0.0, 0.0, 0.0 /), &
         (/4,4/)))
      allocate(expected_river_inflow(4,4))
      expected_river_inflow = transpose(reshape((/ &
          0.0, 0.0, 0.0, 0.0, &
         2.0, 2.0, 6.0, 6.0, &
         0.0, 6.0, 0.0, 8.0, &
         0.0, 0.0, 0.0, 0.0 /), &
         (/4,4/)))
      call init_hd_model_for_testing(river_parameters,river_fields,using_lakes=.False.)
      call run_hd_model(timesteps,runoffs,drainages)
      global_prognostics_ptr => get_global_prognostics()
      call assert_equals(global_prognostics_ptr%river_fields%water_to_ocean, &
                         expected_water_to_ocean,4,4)
      call assert_equals(global_prognostics_ptr%river_fields%river_inflow,&
                         expected_river_inflow,4,4)
      deallocate(drainage)
      deallocate(drainages)
      deallocate(runoffs)
      deallocate(expected_water_to_ocean)
      deallocate(expected_river_inflow)
      call clean_hd_model()
end subroutine testHdModel

end module latlon_hd_model_test_mod
