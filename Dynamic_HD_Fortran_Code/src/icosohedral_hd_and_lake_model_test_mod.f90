module icosohedral_hd_and_lake_model_test_mod
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
   real   ,dimension(:), pointer :: river_retention_coefficients
   real   ,dimension(:), pointer :: overland_retention_coefficients
   real   ,dimension(:), pointer :: base_retention_coefficients
   logical,dimension(:), pointer :: landsea_mask
   real   ,dimension(:), allocatable :: drainage
   real   ,dimension(:,:), allocatable :: drainages
   real   ,dimension(:,:), allocatable :: runoffs
   real   ,dimension(:), allocatable :: expected_water_to_ocean
   real   ,dimension(:), allocatable :: expected_river_inflow
   integer   ,dimension(:), pointer :: flow_directions
   type(prognostics), pointer :: global_prognostics_ptr
   integer :: timesteps, i
      timesteps = 200
      allocate(flow_directions(16))
      flow_directions = (/ 5, 6, 7, 8, &
                           8, 7, 11, 12, &
                          10,11,  0, 11, &
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
                                          landsea_mask)
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

subroutine testLakeModel1
   use icosohedral_hd_model_interface_mod
   use icosohedral_hd_model_mod
   use icosohedral_lake_model_io_mod
   type(riverparameters), pointer :: river_parameters
   type(riverprognosticfields), pointer:: river_fields
   type(lakeparameters),pointer :: lake_parameters
   type(lakefields), pointer :: lake_fields_out
   type(lakeprognostics), pointer :: lake_prognostics_out
   type(lakepointer) :: working_lake_ptr
   integer,dimension(:), pointer :: flow_directions
   integer,dimension(:), pointer :: river_reservoir_nums
   integer,dimension(:), pointer :: overland_reservoir_nums
   integer,dimension(:), pointer :: base_reservoir_nums
   real,dimension(:), pointer :: river_retention_coefficients
   real,dimension(:), pointer :: overland_retention_coefficients
   real,dimension(:), pointer :: base_retention_coefficients
   logical,dimension(:), pointer :: landsea_mask
   logical,dimension(:), pointer :: lake_centers
   real,dimension(:), pointer :: connection_volume_thresholds
   real,dimension(:), pointer :: flood_volume_thresholds
   logical,dimension(:), pointer :: flood_local_redirect
   logical,dimension(:), pointer :: connect_local_redirect
   logical,dimension(:), pointer :: additional_flood_local_redirect
   logical,dimension(:), pointer :: additional_connect_local_redirect
   integer,dimension(:), pointer :: merge_points
   integer,dimension(:), pointer :: flood_next_cell_index
   integer,dimension(:), pointer :: connect_next_cell_index
   integer,dimension(:), pointer :: flood_force_merge_index
   integer,dimension(:), pointer :: connect_force_merge_index
   integer,dimension(:), pointer :: flood_redirect_index
   integer,dimension(:), pointer :: connect_redirect_index
   integer,dimension(:), pointer :: additional_flood_redirect_index
   integer,dimension(:), pointer :: additional_connect_redirect_index
   integer,dimension(:), pointer :: coarse_cell_numbers_on_fine_grid
   real,dimension(:), pointer :: drainage
   real,dimension(:,:), pointer :: drainages
   real,dimension(:), pointer :: runoff
   real,dimension(:,:), pointer :: runoffs
   real, pointer, dimension(:) :: initial_spillover_to_rivers
   real, pointer, dimension(:) :: initial_water_to_lake_centers
   real,dimension(:), pointer :: expected_river_inflow
   real,dimension(:), pointer :: expected_water_to_ocean
   real,dimension(:), pointer :: expected_water_to_hd
   integer,dimension(:), pointer :: expected_lake_numbers
   integer,dimension(:), pointer :: expected_lake_types
   real,dimension(:), pointer :: expected_lake_volumes
   integer,dimension(:), pointer :: lake_types
   integer :: no_merge_mtype,connection_merge_not_set_flood_merge_as_secondary
   real,dimension(:), pointer :: lake_volumes
   integer :: ncells
   integer :: ncells_coarse
   integer :: timesteps
   integer :: lake_number
   integer :: i
   logical :: instant_throughflow_local
   real :: lake_retention_coefficient_local
   integer :: lake_type
      no_merge_mtype = 0
      connection_merge_not_set_flood_merge_as_secondary = 10
      timesteps = 1000
      ncells = 81
      ncells_coarse = 9
      allocate(flow_directions(ncells_coarse))
      flow_directions = (/ &
                          -2,3,6, &
                           1,1,9, &
                           4,4,0 /)
      allocate(river_reservoir_nums(ncells_coarse))
      river_reservoir_nums(:) = 5
      allocate(overland_reservoir_nums(ncells_coarse))
      overland_reservoir_nums(:) = 1
      allocate(base_reservoir_nums(ncells_coarse))
      base_reservoir_nums(:) = 1
      allocate(river_retention_coefficients(ncells_coarse))
      river_retention_coefficients(:) = 0.7
      allocate(overland_retention_coefficients(ncells_coarse))
      overland_retention_coefficients(:) = 0.5
      allocate(base_retention_coefficients(ncells_coarse))
      base_retention_coefficients(:) = 0.1
      allocate(landsea_mask(ncells_coarse))
      river_reservoir_nums(9) = 0
      overland_reservoir_nums(9) = 0
      base_reservoir_nums(9) = 0
      landsea_mask(:) = .False.
      landsea_mask(9) = .True.
      river_parameters => RiverParameters(flow_directions, &
                                          river_reservoir_nums, &
                                          overland_reservoir_nums, &
                                          base_reservoir_nums, &
                                          river_retention_coefficients, &
                                          overland_retention_coefficients, &
                                          base_retention_coefficients, &
                                          landsea_mask)
      river_fields => riverprognosticfields(9,1,1,5)
      allocate(lake_centers(ncells))
      lake_centers = (/ &
          .False., .False., .False., .False., .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., &
         .False., .False.,  .True., .False., .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False. /)
      allocate(connection_volume_thresholds(ncells))
      connection_volume_thresholds(:) = -1.0
      allocate(flood_volume_thresholds(ncells))
      flood_volume_thresholds = (/ &
          -1.0, -1.0, -1.0, 80.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, 80.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0,  1.0, 80.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, 10.0, 10.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, 10.0, 10.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 /)
      allocate(flood_local_redirect(ncells))
      flood_local_redirect(:) = .False.
      allocate(connect_local_redirect(ncells))
      connect_local_redirect(:) = .False.
      allocate(additional_flood_local_redirect(ncells))
      additional_flood_local_redirect(:) = .False.
      allocate(additional_connect_local_redirect(ncells))
      additional_connect_local_redirect(:) = .False.
      allocate(merge_points(ncells))
      merge_points = (/ &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,connection_merge_not_set_flood_merge_as_secondary,&
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, no_merge_mtype, &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,&
        no_merge_mtype,no_merge_mtype,no_merge_mtype,&
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,&
        no_merge_mtype,no_merge_mtype,no_merge_mtype,&
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,&
        no_merge_mtype,no_merge_mtype,no_merge_mtype,&
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,&
        no_merge_mtype,no_merge_mtype,no_merge_mtype,&
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,&
        no_merge_mtype,no_merge_mtype,no_merge_mtype,&
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,&
        no_merge_mtype,no_merge_mtype,no_merge_mtype,&
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,&
        no_merge_mtype,no_merge_mtype,no_merge_mtype,&
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,&
        no_merge_mtype,no_merge_mtype,no_merge_mtype /)
      allocate(flood_next_cell_index(ncells))
      flood_next_cell_index = (/ &
          -1,-1,-1,5,-1,-1,-1,-1,-1,-1,-1,-1,4,-1,-1,-1,-1,-1,-1,-1,30,13,-1,-1,-1,-1,-1,-1,-1,39,&
          40,-1,-1,-1,-1,-1,-1,-1,31,22,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
          -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 /)
      allocate(connect_next_cell_index(ncells))
      connect_next_cell_index = (/ &
          -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
          -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
          -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 /)
      allocate(flood_force_merge_index(ncells))
      flood_force_merge_index = (/ &
          -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
          -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
          -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 /)
      allocate(connect_force_merge_index(ncells))
      connect_force_merge_index = (/ &
          -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
          -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
          -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 /)
      allocate(flood_redirect_index(ncells))
      flood_redirect_index = (/ &
           -1,-1,-1,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
           -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
           -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 /)
      allocate(connect_redirect_index(ncells))
      connect_redirect_index = (/ &
          -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
          -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
          -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 /)
      allocate(additional_flood_redirect_index(ncells))
      additional_flood_redirect_index(:) = 0
      allocate(additional_connect_redirect_index(ncells))
      additional_connect_redirect_index(:) = 0
      instant_throughflow_local=.True.
      lake_retention_coefficient_local=0.1
      allocate(coarse_cell_numbers_on_fine_grid(ncells))
      coarse_cell_numbers_on_fine_grid = (/ 1,1,1,2,2,2,3,3,3, &
                                            1,1,1,2,2,2,3,3,3, &
                                            1,1,1,2,2,2,3,3,3, &
                                            4,4,4,5,5,5,6,6,6, &
                                            4,4,4,5,5,5,6,6,6, &
                                            4,4,4,5,5,5,6,6,6, &
                                            7,7,7,8,8,8,9,9,9, &
                                            7,7,7,8,8,8,9,9,9, &
                                            7,7,7,8,8,8,9,9,9 /)
      lake_parameters => lakeparameters(lake_centers, &
                                        connection_volume_thresholds, &
                                        flood_volume_thresholds, &
                                        flood_local_redirect, &
                                        connect_local_redirect, &
                                        additional_flood_local_redirect, &
                                        additional_connect_local_redirect, &
                                        merge_points, &
                                        flood_next_cell_index, &
                                        connect_next_cell_index, &
                                        flood_force_merge_index, &
                                        connect_force_merge_index, &
                                        flood_redirect_index, &
                                        connect_redirect_index, &
                                        additional_flood_redirect_index, &
                                        additional_connect_redirect_index, &
                                        ncells, &
                                        ncells_coarse, &
                                        instant_throughflow_local, &
                                        lake_retention_coefficient_local, &
                                        coarse_cell_numbers_on_fine_grid)
      allocate(drainage(ncells_coarse))
      drainage = (/ &
         1.0, 1.0, 1.0, &
         1.0, 1.0, 1.0, &
         1.0, 1.0, 0.0 /)
      allocate(runoff(ncells_coarse))
      runoff = (/ &
         1.0, 1.0, 1.0, &
         1.0, 1.0, 1.0, &
         1.0, 1.0, 0.0 /)
      allocate(drainages(ncells_coarse,timesteps))
      allocate(runoffs(ncells_coarse,timesteps))
      do i = 1,timesteps
        drainages(:,i) = drainage(:)
        runoffs(:,i) = runoff(:)
      end do
      allocate(initial_spillover_to_rivers(ncells_coarse))
      initial_spillover_to_rivers(:) = 0.0
      allocate(initial_water_to_lake_centers(ncells))
      initial_water_to_lake_centers(:) = 0.0
      allocate(expected_river_inflow(ncells_coarse))
      expected_river_inflow = (/ &
          0.0, 0.0,  2.0, &
         4.0, 0.0, 14.0, &
         0.0, 0.0,  0.0 /)
      allocate(expected_water_to_ocean(ncells_coarse))
      expected_water_to_ocean = (/ &
          0.0, 0.0,  0.0, &
         0.0, 0.0,  0.0, &
         0.0, 0.0, 16.0 /)
      allocate(expected_water_to_hd(ncells_coarse))
      expected_water_to_hd = (/ &
          0.0, 0.0, 10.0, &
         0.0, 0.0,  0.0, &
         0.0, 0.0,  0.0 /)
      allocate(expected_lake_numbers(ncells))
      expected_lake_numbers = (/ &
             0,    0,    0,    1,    0,    0,    0,    0,    0, &
         0,    0,    0,    1,    0,    0,    0,    0,    0, &
         0,    0,    1,    1,    0,    0,    0,    0,    0, &
         0,    0,    1,    1,    0,    0,    0,    0,    0, &
         0,    0,    1,    1,    0,    0,    0,    0,    0, &
         0,    0,    0,    0,    0,    0,    0,    0,    0, &
         0,    0,    0,    0,    0,    0,    0,    0,    0, &
         0,    0,    0,    0,    0,    0,    0,    0,    0, &
         0,    0,    0,    0,    0,    0,    0,    0,    0 /)
      allocate(expected_lake_types(ncells))
      expected_lake_types = (/ &
             0,    0,    0,    2,    0,    0,    0,    0,    0, &
         0,    0,    0,    2,    0,    0,    0,    0,    0, &
         0,    0,    2,    2,    0,    0,    0,    0,    0, &
         0,    0,    2,    2,    0,    0,    0,    0,    0, &
         0,    0,    2,    2,    0,    0,    0,    0,    0, &
         0,    0,    0,    0,    0,    0,    0,    0,    0, &
         0,    0,    0,    0,    0,    0,    0,    0,    0, &
         0,    0,    0,    0,    0,    0,    0,    0,    0, &
         0,    0,    0,    0,    0,    0,    0,    0,    0 /)
      allocate(expected_lake_volumes(1))
      expected_lake_volumes(1) = 80.0
      call init_hd_model_for_testing(river_parameters,river_fields,.True., &
                                     lake_parameters, &
                                     initial_water_to_lake_centers, &
                                     initial_spillover_to_rivers)
      call run_hd_model(timesteps,runoffs,drainages)
      lake_prognostics_out => get_lake_prognostics()
      lake_fields_out => get_lake_fields()
      allocate(lake_types(ncells))
      lake_types(:) = 0
      do i = 1,81
        lake_number = lake_fields_out%lake_numbers(i)
        if (lake_number > 0) then
          working_lake_ptr = lake_prognostics_out%lakes(lake_number)
          lake_type = working_lake_ptr%lake_pointer%lake_type
          if (lake_type == filling_lake_type) then
            lake_types(i) = 1
          else if (lake_type == overflowing_lake_type) then
            lake_types(i) = 2
          else if (lake_type == subsumed_lake_type) then
            lake_types(i) = 3
          else
            lake_types(i) = 4
          end if
        end if
      end do
      allocate(lake_volumes(1))
      do i = 1,size(lake_volumes)
        working_lake_ptr = lake_prognostics_out%lakes(i)
        lake_volumes(i) = working_lake_ptr%lake_pointer%lake_volume
      end do
      call assert_equals(expected_river_inflow,river_fields%river_inflow,9,0.00001)
      call assert_equals(expected_water_to_ocean,river_fields%water_to_ocean,9,0.00001)
      call assert_equals(expected_water_to_hd,lake_fields_out%water_to_hd,9,0.00001)
      call assert_equals(expected_lake_numbers,lake_fields_out%lake_numbers,81)
      call assert_equals(expected_lake_types,lake_types,81)
      call assert_equals(expected_lake_volumes,lake_volumes,1)
      deallocate(lake_volumes)
      deallocate(drainage)
      deallocate(drainages)
      deallocate(runoff)
      deallocate(runoffs)
      deallocate(initial_spillover_to_rivers)
      deallocate(initial_water_to_lake_centers)
      deallocate(expected_river_inflow)
      deallocate(expected_water_to_ocean)
      deallocate(expected_water_to_hd)
      deallocate(expected_lake_numbers)
      deallocate(expected_lake_types)
      deallocate(expected_lake_volumes)
      deallocate(lake_types)
      call clean_lake_model()
      call clean_hd_model()
end subroutine testLakeModel1

subroutine testLakeModel2
  use icosohedral_hd_model_interface_mod
  use icosohedral_hd_model_mod
  use icosohedral_lake_model_io_mod
  type(riverparameters), pointer :: river_parameters
  type(riverprognosticfields), pointer :: river_fields
  type(lakeparameters),pointer :: lake_parameters
  type(lakefields), pointer :: lake_fields_out
  type(lakeprognostics), pointer :: lake_prognostics_out
  type(lakepointer) :: working_lake_ptr
  integer,dimension(:), pointer :: flow_directions
  integer,dimension(:), pointer :: river_reservoir_nums
  integer,dimension(:), pointer :: overland_reservoir_nums
  integer,dimension(:), pointer :: base_reservoir_nums
  real,dimension(:), pointer :: river_retention_coefficients
  real,dimension(:), pointer :: overland_retention_coefficients
  real,dimension(:), pointer :: base_retention_coefficients
  logical,dimension(:), pointer :: landsea_mask
  logical,dimension(:), pointer :: lake_centers
  real,dimension(:), pointer :: connection_volume_thresholds
  real,dimension(:), pointer :: flood_volume_thresholds
  logical,dimension(:), pointer :: flood_local_redirect
  logical,dimension(:), pointer :: connect_local_redirect
  logical,dimension(:), pointer :: additional_flood_local_redirect
  logical,dimension(:), pointer :: additional_connect_local_redirect
  integer,dimension(:), pointer :: merge_points
  integer,dimension(:), pointer :: flood_next_cell_index
  integer,dimension(:), pointer :: connect_next_cell_index
  integer,dimension(:), pointer :: flood_force_merge_index
  integer,dimension(:), pointer :: connect_force_merge_index
  integer,dimension(:), pointer :: flood_redirect_index
  integer,dimension(:), pointer :: connect_redirect_index
  integer,dimension(:), pointer :: additional_flood_redirect_index
  integer,dimension(:), pointer :: additional_connect_redirect_index
  integer,dimension(:), pointer :: coarse_cell_numbers_on_fine_grid
  real,dimension(:), pointer :: drainage
  real,dimension(:,:), pointer :: drainages
  real,dimension(:), pointer :: runoff
  real,dimension(:,:), pointer :: runoffs
  real, pointer, dimension(:) :: initial_spillover_to_rivers
  real, pointer, dimension(:) :: initial_water_to_lake_centers
  real,dimension(:), pointer :: expected_river_inflow
  real,dimension(:), pointer :: expected_water_to_ocean
  real,dimension(:), pointer :: expected_water_to_hd
  integer,dimension(:), pointer :: expected_lake_numbers
  integer,dimension(:), pointer :: expected_lake_types
  real,dimension(:), pointer :: expected_lake_volumes
  integer,dimension(:), pointer :: lake_types
  integer :: no_merge_mtype
  integer :: connection_merge_not_set_flood_merge_as_primary
  integer :: connection_merge_not_set_flood_merge_as_secondary
  real,dimension(:), pointer :: lake_volumes
  integer :: ncells
  integer :: ncells_coarse
  integer :: timesteps
  integer :: lake_number
  integer :: i
  logical :: instant_throughflow_local
  real :: lake_retention_coefficient_local
  integer :: lake_type
    no_merge_mtype = 0
    connection_merge_not_set_flood_merge_as_primary = 9
    connection_merge_not_set_flood_merge_as_secondary = 10
    timesteps = 10000
    ncells = 400
    ncells_coarse = 16
    allocate(flow_directions(ncells_coarse))
    flow_directions = (/ -2, 1, 7, 8, &
                          6,-2,-2,12, &
                         10,14,16,-2, &
                         -2, 0,16, 0 /)
    allocate(river_reservoir_nums(ncells_coarse))
    river_reservoir_nums(:) = 5
    allocate(overland_reservoir_nums(ncells_coarse))
    overland_reservoir_nums(:) = 1
    allocate(base_reservoir_nums(ncells_coarse))
    base_reservoir_nums(:) = 1
    allocate(river_retention_coefficients(ncells_coarse))
    river_retention_coefficients(:) = 0.7
    allocate(overland_retention_coefficients(ncells_coarse))
    overland_retention_coefficients(:) = 0.5
    allocate(base_retention_coefficients(ncells_coarse))
    base_retention_coefficients(:) = 0.1
    allocate(landsea_mask(ncells_coarse))
    landsea_mask(:) = .False.
    river_reservoir_nums(16) = 0
    overland_reservoir_nums(16) = 0
    base_reservoir_nums(16) = 0
    landsea_mask(16) = .True.
    river_reservoir_nums(14) = 0
    overland_reservoir_nums(14) = 0
    base_reservoir_nums(14) = 0
    landsea_mask(14) = .True.
    river_parameters => RiverParameters(flow_directions, &
                                        river_reservoir_nums, &
                                        overland_reservoir_nums, &
                                        base_reservoir_nums, &
                                        river_retention_coefficients, &
                                        overland_retention_coefficients, &
                                        base_retention_coefficients, &
                                        landsea_mask)
    river_fields => riverprognosticfields(16,1,1,5)
    allocate(coarse_cell_numbers_on_fine_grid(ncells))
    coarse_cell_numbers_on_fine_grid = (/ &
              1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4, &
              1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4, &
              1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4, &
              1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4, &
              1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4, &
              5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8, &
              5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8, &
              5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8, &
              5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8, &
              5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8, &
              9,9,9,9,9,10,10,10,10,10,11,11,11,11,11,12,12,12,12,12, &
              9,9,9,9,9,10,10,10,10,10,11,11,11,11,11,12,12,12,12,12, &
              9,9,9,9,9,10,10,10,10,10,11,11,11,11,11,12,12,12,12,12, &
              9,9,9,9,9,10,10,10,10,10,11,11,11,11,11,12,12,12,12,12, &
              9,9,9,9,9,10,10,10,10,10,11,11,11,11,11,12,12,12,12,12, &
              13,13,13,13,13,14,14,14,14,14,15,15,15,15,15,16,16,16,16,16, &
              13,13,13,13,13,14,14,14,14,14,15,15,15,15,15,16,16,16,16,16, &
              13,13,13,13,13,14,14,14,14,14,15,15,15,15,15,16,16,16,16,16, &
              13,13,13,13,13,14,14,14,14,14,15,15,15,15,15,16,16,16,16,16, &
              13,13,13,13,13,14,14,14,14,14,15,15,15,15,15,16,16,16,16,16 /)
    allocate(lake_centers(ncells))
    lake_centers = (/ &
        .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                .False., .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .True.,  .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .True.,  .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .True.,  .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .True.,  .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .True.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .True.,  .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False. /)
    allocate(connection_volume_thresholds(ncells))
    connection_volume_thresholds = (/ &
       -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0, -1.0, -1.0, -1.0, -1.0, 186.0, 23.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, 56.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
             -1.0, -1.0, -1.0, -1.0, -1.0 &
       /)
    allocate(flood_volume_thresholds(ncells))
    flood_volume_thresholds = (/ &
       -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
              -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0, 5.0, &
       0.0, 262.0,  5.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0, 111.0, 111.0, 56.0, &
            111.0,   -1.0,    -1.0,   -1.0, 2.0, &
       -1.0,   5.0,  5.0,    -1.0,   -1.0, 340.0, 262.0,   -1.0,   -1.0,  -1.0,  -1.0, 111.0,   1.0,   1.0, 56.0, &
            56.0,   -1.0,    -1.0,   -1.0,  -1.0, &
       -1.0,   5.0,  5.0,    -1.0,   -1.0,  10.0,  10.0, 38.0, 10.0,  -1.0,  -1.0,    -1.0,   0.0,   1.0,  1.0, &
            26.0, 56.0,    -1.0,   -1.0,  -1.0, &
       -1.0,   5.0,  5.0, 186.0,  2.0,   2.0,    -1.0, 10.0, 10.0,  -1.0, 1.0,   6.0,   1.0,   0.0,  1.0,  &
            26.0, 26.0, 111.0,   -1.0,  -1.0, &
       16.0,  16.0, 16.0,    -1.0,  2.0,   0.0,   2.0,  2.0, 10.0,  -1.0, 1.0,   0.0,   0.0,   1.0,  1.0, &
             1.0, 26.0,  56.0,   -1.0,  -1.0, &
       -1.0,  46.0, 16.0,    -1.0,   -1.0,   2.0,    -1.0, 23.0,   -1.0,  -1.0,  -1.0,   1.0,   0.0,   1.0,  1.0, &
             1.0, 56.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,  56.0,   1.0,   1.0,  1.0, &
            26.0, 56.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,  56.0,  56.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
             0.0,  3.0,   3.0, 10.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
             0.0,  3.0,   3.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,   1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0,  -1.0 /)
    allocate(flood_local_redirect(ncells))
    flood_local_redirect = (/ &
        .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                 .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False.,  .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                 .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .True., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                 .True., .False., .False.,  .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False.,  .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False.,  .False.,    .True., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .True., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False.,  .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False. /)
    allocate(connect_local_redirect(ncells))
    connect_local_redirect = (/ &
        .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                 .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False.,  .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                 .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                .False., .False., .False.,  .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False.,  .False., .False., .False., .False., .False., &
                 .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                .False., .False., .False., .False.,  .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False.,  .False., .False., .False., .False., .False., .False., .False., &
                 .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False. /)
    allocate(additional_flood_local_redirect(ncells))
    additional_flood_local_redirect(:) = .False.
    allocate(additional_connect_local_redirect(ncells))
    additional_connect_local_redirect(:) = .False.
    allocate(merge_points(ncells))
    merge_points = (/ &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,connection_merge_not_set_flood_merge_as_primary,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype, &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,&
       connection_merge_not_set_flood_merge_as_secondary, &
       no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
       no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        connection_merge_not_set_flood_merge_as_secondary,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,&
        no_merge_mtype,no_merge_mtype, &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,connection_merge_not_set_flood_merge_as_secondary,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
      no_merge_mtype,connection_merge_not_set_flood_merge_as_secondary,no_merge_mtype,no_merge_mtype,no_merge_mtype,&
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,connection_merge_not_set_flood_merge_as_primary, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype, &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        connection_merge_not_set_flood_merge_as_primary,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype, &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,connection_merge_not_set_flood_merge_as_secondary,no_merge_mtype, &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,connection_merge_not_set_flood_merge_as_secondary,no_merge_mtype,&
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
      no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
       no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
       no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype /)
    allocate(flood_next_cell_index(ncells))
    flood_next_cell_index =  (/ &
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,61,59,65,102,-1,-1,-1,-1,-1,-1,-1,-1,-1,55,52,96,103,-1,-1,-1,39,&
    -1,82,42,-1,-1,178,41,-1,-1,-1,-1,53,115,72,193,54,-1,-1,-1,-1,-1,62,81,-1,-1,128,85,130,66,&
    -1,-1,-1,152,73,93,74,137,-1,-1,-1,-1,122,101,65,127,145,-1,86,88,-1,114,154,130,132,94,175,&
    95,71,-1,-1,142,120,121,-1,104,105,124,107,108,-1,110,111,92,172,133,134,116,117,-1,-1,-1,124,&
    141,-1,-1,126,-1,87,-1,-1,-1,112,131,135,174,153,109,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    176,151,155,173,136,156,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,171,192,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,295,296,278,335,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    276,297,277,-1,-1,-1,-1,-1,343,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1 /)
    allocate(connect_next_cell_index(ncells))
    connect_next_cell_index = (/ &
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,66,147,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,75,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 /)
    allocate(flood_force_merge_index(ncells))
    flood_force_merge_index = (/ &
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,122,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,128,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,152,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 /)
    allocate(connect_force_merge_index(ncells))
    connect_force_merge_index = (/ &
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 /)
    allocate(flood_redirect_index(ncells))
    flood_redirect_index = (/ &
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,154,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,154,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,125,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,113,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,13,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1 /)
    allocate(connect_redirect_index(ncells))
    connect_redirect_index = (/ &
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1 /)
    call add_offset(flood_next_cell_index,1,(/-1/))
    call add_offset(connect_next_cell_index,1,(/-1/))
    call add_offset(flood_force_merge_index,1,(/-1/))
    call add_offset(connect_force_merge_index,1,(/-1/))
    call add_offset(flood_redirect_index,1,(/-1/))
    call add_offset(connect_redirect_index,1,(/-1/))
    allocate(additional_flood_redirect_index(ncells))
    additional_flood_redirect_index(:) = 0
    allocate(additional_connect_redirect_index(ncells))
    additional_connect_redirect_index(:) = 0
    instant_throughflow_local=.True.
    lake_retention_coefficient_local=0.1
    lake_parameters => LakeParameters(lake_centers, &
                                      connection_volume_thresholds, &
                                      flood_volume_thresholds, &
                                      flood_local_redirect, &
                                      connect_local_redirect, &
                                      additional_flood_local_redirect, &
                                      additional_connect_local_redirect, &
                                      merge_points, &
                                      flood_next_cell_index, &
                                      connect_next_cell_index, &
                                      flood_force_merge_index, &
                                      connect_force_merge_index, &
                                      flood_redirect_index, &
                                      connect_redirect_index, &
                                      additional_flood_redirect_index, &
                                      additional_connect_redirect_index, &
                                      ncells, &
                                      ncells_coarse, &
                                      instant_throughflow_local, &
                                      lake_retention_coefficient_local, &
                                      coarse_cell_numbers_on_fine_grid)
    allocate(drainage(ncells_coarse))
    drainage = (/ &
       1.0, 1.0, 1.0, 1.0, &
       1.0, 1.0, 1.0, 1.0, &
       1.0, 1.0, 1.0, 1.0, &
       1.0, 0.0, 1.0, 0.0 /)
    allocate(runoff(ncells_coarse))
    runoff = (/ &
       1.0, 1.0, 1.0, 1.0, &
       1.0, 1.0, 1.0, 1.0, &
       1.0, 1.0, 1.0, 1.0, &
       1.0, 0.0, 1.0, 0.0  /)
    allocate(drainages(ncells_coarse,timesteps))
    allocate(runoffs(ncells_coarse,timesteps))
    do i = 1,timesteps
      drainages(:,i) = drainage(:)
      runoffs(:,i) = runoff(:)
    end do
    allocate(initial_spillover_to_rivers(ncells_coarse))
    initial_spillover_to_rivers(:) = 0.0
    allocate(initial_water_to_lake_centers(ncells))
    initial_water_to_lake_centers(:) = 0.0
    allocate(expected_river_inflow(ncells_coarse))
    expected_river_inflow = (/ &
        0.0, 0.0, 0.0, 0.0, &
       0.0, 0.0, 0.0, 2.0, &
       0.0, 2.0, 0.0, 0.0, &
       0.0, 0.0, 0.0, 0.0 /)
    allocate(expected_water_to_ocean(ncells_coarse))
    expected_water_to_ocean = (/ &
        0.0, 0.0, 0.0, 0.0, &
       0.0, 0.0, 0.0, 0.0, &
       0.0, 0.0, 0.0, 0.0, &
       0.0, 6.0, 0.0, 22.0 /)
    allocate(expected_water_to_hd(ncells_coarse))
    expected_water_to_hd = (/ &
        0.0, 0.0, 0.0,  0.0, &
       0.0, 0.0, 0.0, 12.0, &
       0.0, 0.0, 0.0,  0.0, &
       0.0, 2.0, 0.0, 18.0 /)
    allocate(expected_lake_numbers(ncells))
    expected_lake_numbers = (/ &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, &
       1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 1, &
       0, 1, 1, 0, 0, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0, &
       0, 1, 1, 0, 0, 3, 3, 3, 3, 0, 0, 0, 2, 4, 4, 4, 4, 0, 0, 0, &
       0, 1, 1, 4, 3, 3, 0, 3, 3, 4, 4, 2, 4, 2, 4, 4, 4, 4, 0, 0, &
       1, 1, 1, 0, 3, 3, 3, 3, 3, 0, 4, 2, 2, 4, 4, 4, 4, 4, 0, 0, &
       0, 1, 1, 0, 0, 3, 0, 3, 0, 0, 0, 4, 2, 4, 4, 4, 4, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, &
       0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 /)
    allocate(expected_lake_types(ncells))
    expected_lake_types = (/ &
   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, &
   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    3, &
   3,    2,    3,    0,    0,    0,    0,    0,    0,    0,    0,    0,    2,    2,    2,    2,    0,    0,    0,    3, &
   0,    3,    3,    0,    0,    2,    2,    0,    0,    0,    0,    2,    2,    2,    2,    2,    0,    0,    0,    0, &
   0,    3,    3,    0,    0,    3,    3,    3,    3,    0,    0,    0,    3,    2,    2,    2,    2,    0,    0,    0, &
   0,    3,    3,    2,    3,    3,    0,    3,    3,    2,    2,    3,    2,    3,    2,    2,    2,    2,    0,    0, &
   3,    3,    3,    0,    3,    3,    3,    3,    3,    0,    2,    3,    3,    2,    2,    2,    2,    2,    0,    0, &
   0,    3,    3,    0,    0,    3,    0,    3,    0,    0,    0,    2,    3,    2,    2,    2,    2,    0,    0,    0, &
   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    2,    2,    2,    2,    2,    2,    0,    0,    0, &
   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    2,    2,    0,    0,    0,    0,    0,    0, &
   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, &
   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, &
   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, &
   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    2,    2,    2,    2,    0, &
   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    2,    2,    2,    0,    0, &
   0,    0,    0,    2,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, &
   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, &
   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, &
   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, &
   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 /)
    allocate(expected_lake_volumes(6))
    expected_lake_volumes = (/46.0, 6.0, 38.0,  340.0, 10.0, 1.0/)
    call init_hd_model_for_testing(river_parameters,river_fields,.True., &
                                   lake_parameters, &
                                   initial_water_to_lake_centers, &
                                   initial_spillover_to_rivers)
    call run_hd_model(timesteps,runoffs,drainages)
    lake_prognostics_out => get_lake_prognostics()
    lake_fields_out => get_lake_fields()
    allocate(lake_types(ncells))
    lake_types(:) = 0
    do i = 1,400
      lake_number = lake_fields_out%lake_numbers(i)
      if (lake_number > 0) then
        working_lake_ptr = lake_prognostics_out%lakes(lake_number)
        lake_type = working_lake_ptr%lake_pointer%lake_type
          if (lake_type == filling_lake_type) then
            lake_types(i) = 1
          else if (lake_type == overflowing_lake_type) then
            lake_types(i) = 2
          else if (lake_type == subsumed_lake_type) then
            lake_types(i) = 3
          else
            lake_types(i) = 4
          end if
      end if
    end do
    allocate(lake_volumes(6))
    do i = 1,size(lake_volumes)
      working_lake_ptr = lake_prognostics_out%lakes(i)
      lake_volumes(i) = working_lake_ptr%lake_pointer%lake_volume
    end do
    call assert_equals(expected_river_inflow,river_fields%river_inflow,&
                       ncells_coarse)
    call assert_equals(expected_water_to_ocean,river_fields%water_to_ocean,16,0.00001)
    call assert_equals(expected_water_to_hd,lake_fields_out%water_to_hd,16,0.00001)
    call assert_equals(expected_lake_numbers,lake_fields_out%lake_numbers,400)
    call assert_equals(expected_lake_types,lake_types,400)
    call assert_equals(expected_lake_volumes,lake_volumes,6)
    deallocate(lake_volumes)
    deallocate(drainage)
    deallocate(drainages)
    deallocate(runoff)
    deallocate(runoffs)
    deallocate(initial_spillover_to_rivers)
    deallocate(initial_water_to_lake_centers)
    deallocate(expected_river_inflow)
    deallocate(expected_water_to_ocean)
    deallocate(expected_water_to_hd)
    deallocate(expected_lake_numbers)
    deallocate(expected_lake_types)
    deallocate(expected_lake_volumes)
    deallocate(lake_types)
    call clean_lake_model()
    call clean_hd_model()
end subroutine testLakeModel2

subroutine testLakeModel3
  use icosohedral_hd_model_interface_mod
  use icosohedral_hd_model_mod
  use icosohedral_lake_model_io_mod
  type(riverparameters), pointer :: river_parameters
  type(riverprognosticfields), pointer :: river_fields
  type(lakeparameters),pointer :: lake_parameters
  type(lakefields), pointer :: lake_fields_out
  type(lakeprognostics), pointer :: lake_prognostics_out
  type(lakepointer) :: working_lake_ptr
  integer,dimension(:), pointer :: flow_directions
  integer,dimension(:), pointer :: river_reservoir_nums
  integer,dimension(:), pointer :: overland_reservoir_nums
  integer,dimension(:), pointer :: base_reservoir_nums
  real,dimension(:), pointer :: river_retention_coefficients
  real,dimension(:), pointer :: overland_retention_coefficients
  real,dimension(:), pointer :: base_retention_coefficients
  logical,dimension(:), pointer :: landsea_mask
  logical,dimension(:), pointer :: lake_centers
  real,dimension(:), pointer :: connection_volume_thresholds
  real,dimension(:), pointer :: flood_volume_thresholds
  logical,dimension(:), pointer :: flood_local_redirect
  logical,dimension(:), pointer :: connect_local_redirect
  logical,dimension(:), pointer :: additional_flood_local_redirect
  logical,dimension(:), pointer :: additional_connect_local_redirect
  integer,dimension(:), pointer :: merge_points
  integer,dimension(:), pointer :: flood_next_cell_index
  integer,dimension(:), pointer :: connect_next_cell_index
  integer,dimension(:), pointer :: flood_force_merge_index
  integer,dimension(:), pointer :: connect_force_merge_index
  integer,dimension(:), pointer :: flood_redirect_index
  integer,dimension(:), pointer :: connect_redirect_index
  integer,dimension(:), pointer :: additional_flood_redirect_index
  integer,dimension(:), pointer :: additional_connect_redirect_index
  real,dimension(:), pointer :: drainage
  real,dimension(:,:), pointer :: drainages
  real,dimension(:,:), pointer :: drainages_copy
  real,dimension(:), pointer :: runoff
  real,dimension(:,:), pointer :: runoffs
  real,dimension(:,:), pointer :: runoffs_copy
  real,dimension(:), pointer :: evaporation
  real,dimension(:,:), pointer :: evaporations
  real,dimension(:,:), pointer :: evaporations_set_two
  real, pointer, dimension(:) :: initial_spillover_to_rivers
  real, pointer, dimension(:) :: initial_water_to_lake_centers
  real,dimension(:), pointer :: expected_river_inflow
  real,dimension(:), pointer :: expected_water_to_ocean
  real,dimension(:), pointer :: expected_water_to_hd
  integer,dimension(:), pointer :: expected_lake_numbers
  integer,dimension(:), pointer :: expected_lake_types
  real,dimension(:), pointer :: expected_lake_volumes
  real,dimension(:), pointer :: expected_intermediate_river_inflow
  real,dimension(:), pointer :: expected_intermediate_water_to_ocean
  real,dimension(:), pointer :: expected_intermediate_water_to_hd
  integer,dimension(:), pointer :: expected_intermediate_lake_numbers
  integer,dimension(:), pointer :: expected_intermediate_lake_types
  real,dimension(:), pointer :: expected_intermediate_lake_volumes
  integer,dimension(:), pointer :: coarse_cell_numbers_on_fine_grid
  integer,dimension(:), pointer :: lake_types
  integer :: no_merge_mtype
  integer :: connection_merge_not_set_flood_merge_as_primary
  integer :: connection_merge_not_set_flood_merge_as_secondary
  real,dimension(:), pointer :: lake_volumes
  integer :: ncells
  integer :: ncells_coarse
  integer :: lake_number
  integer :: i
  logical :: instant_throughflow_local
  real :: lake_retention_coefficient_local
  integer :: lake_type
      no_merge_mtype = 0
      connection_merge_not_set_flood_merge_as_primary = 9
      connection_merge_not_set_flood_merge_as_secondary = 10
      ncells = 400
      ncells_coarse = 16
      allocate(flow_directions(ncells_coarse))
      flow_directions = (/ -2, 1, 7, 8, &
                           6,-2,-2,12, &
                          10,14,16,-2, &
                          -2, 0,16, 0 /)
      allocate(river_reservoir_nums(ncells_coarse))
      river_reservoir_nums(:) = 5
      allocate(overland_reservoir_nums(ncells_coarse))
      overland_reservoir_nums(:) = 1
      allocate(base_reservoir_nums(ncells_coarse))
      base_reservoir_nums(:) = 1
      allocate(river_retention_coefficients(ncells_coarse))
      river_retention_coefficients(:) = 0.7
      allocate(overland_retention_coefficients(ncells_coarse))
      overland_retention_coefficients(:) = 0.5
      allocate(base_retention_coefficients(ncells_coarse))
      base_retention_coefficients(:) = 0.1
      allocate(landsea_mask(ncells_coarse))
      landsea_mask(:) = .False.
      river_reservoir_nums(16) = 0
      overland_reservoir_nums(16) = 0
      base_reservoir_nums(16) = 0
      landsea_mask(16) = .True.
      river_reservoir_nums(14) = 0
      overland_reservoir_nums(14) = 0
      base_reservoir_nums(14) = 0
      landsea_mask(14) = .True.
      river_parameters => RiverParameters(flow_directions, &
                                          river_reservoir_nums, &
                                          overland_reservoir_nums, &
                                          base_reservoir_nums, &
                                          river_retention_coefficients, &
                                          overland_retention_coefficients, &
                                          base_retention_coefficients, &
                                          landsea_mask)
      river_fields => riverprognosticfields(16,1,1,5)
      allocate(lake_centers(ncells))
      lake_centers = (/ &
        .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                .False., .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .True.,  .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .True.,  .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .True.,  .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .True.,  .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .True.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .True.,  .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False., &
       .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False. /)
      allocate(connection_volume_thresholds(ncells))
      connection_volume_thresholds = (/ &
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0, 186.0, 23.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, 56.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
               -1.0, -1.0, -1.0, -1.0, -1.0 &
         /)
      allocate(flood_volume_thresholds(ncells))
      flood_volume_thresholds = (/ &
       -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
              -1.0, -1.0, -1.0, -1.0, -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0, 5.0, &
       0.0, 262.0,  5.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0, 111.0, 111.0, 56.0, &
            111.0,   -1.0,    -1.0,   -1.0, 2.0, &
       -1.0,   5.0,  5.0,    -1.0,   -1.0, 340.0, 262.0,   -1.0,   -1.0,  -1.0,  -1.0, 111.0,   1.0,   1.0, 56.0, &
            56.0,   -1.0,    -1.0,   -1.0,  -1.0, &
       -1.0,   5.0,  5.0,    -1.0,   -1.0,  10.0,  10.0, 38.0, 10.0,  -1.0,  -1.0,    -1.0,   0.0,   1.0,  1.0, &
            26.0, 56.0,    -1.0,   -1.0,  -1.0, &
       -1.0,   5.0,  5.0, 186.0,  2.0,   2.0,    -1.0, 10.0, 10.0,  -1.0, 1.0,   6.0,   1.0,   0.0,  1.0,  &
            26.0, 26.0, 111.0,   -1.0,  -1.0, &
       16.0,  16.0, 16.0,    -1.0,  2.0,   0.0,   2.0,  2.0, 10.0,  -1.0, 1.0,   0.0,   0.0,   1.0,  1.0, &
             1.0, 26.0,  56.0,   -1.0,  -1.0, &
       -1.0,  46.0, 16.0,    -1.0,   -1.0,   2.0,    -1.0, 23.0,   -1.0,  -1.0,  -1.0,   1.0,   0.0,   1.0,  1.0, &
             1.0, 56.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,  56.0,   1.0,   1.0,  1.0, &
            26.0, 56.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,  56.0,  56.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
             0.0,  3.0,   3.0, 10.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
             0.0,  3.0,   3.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,   1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0,  -1.0, &
       -1.0,    -1.0,   -1.0,    -1.0,   -1.0,    -1.0,    -1.0,   -1.0,   -1.0,  -1.0,  -1.0,    -1.0,    -1.0,    -1.0,   -1.0, &
              -1.0,   -1.0,    -1.0,   -1.0,  -1.0 /)
      allocate(flood_local_redirect(ncells))
      flood_local_redirect = (/ &
          .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                   .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False.,  .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                   .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .True., .False., .False., .False., &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                   .True., .False., .False.,  .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False.,  .False., .False., .False., .False., .False., &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                  .False., .False., .False., .False.,  .False.,    .True., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                  .False., .False., .False., .True., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False.,  .False., .False., .False., .False., .False., .False., .False., &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False. /)
      allocate(connect_local_redirect(ncells))
      connect_local_redirect = (/ &
          .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                   .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False.,  .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., &
                   .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                  .False., .False., .False.,  .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False.,  .False., .False., .False., .False., .False., &
                   .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                  .False., .False., .False., .False.,  .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False.,  .False., .False., .False., .False., .False., .False., .False., &
                   .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., .False., .False.,  &
                  .False., .False., .False., .False., .False.,    .False., .False., .False., .False. /)
      allocate(additional_flood_local_redirect(ncells))
      additional_flood_local_redirect(:) = .False.
      allocate(additional_connect_local_redirect(ncells))
      additional_connect_local_redirect(:) = .False.
      allocate(merge_points(ncells))
      merge_points = (/ &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
            no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
            no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
            no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
            no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
            no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
            no_merge_mtype,connection_merge_not_set_flood_merge_as_primary,no_merge_mtype,no_merge_mtype, &
            no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,&
         connection_merge_not_set_flood_merge_as_secondary, &
         no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
         no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          connection_merge_not_set_flood_merge_as_secondary,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,&
          no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,connection_merge_not_set_flood_merge_as_secondary,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,connection_merge_not_set_flood_merge_as_secondary,no_merge_mtype,no_merge_mtype,no_merge_mtype,&
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,connection_merge_not_set_flood_merge_as_primary, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          connection_merge_not_set_flood_merge_as_primary,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
            no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,connection_merge_not_set_flood_merge_as_secondary,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,connection_merge_not_set_flood_merge_as_secondary,no_merge_mtype,&
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
          no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
         no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
         no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype /)
     allocate(flood_next_cell_index(ncells))
    flood_next_cell_index =  (/ &
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,61,59,65,102,-1,-1,-1,-1,-1,-1,-1,-1,-1,55,52,96,103,-1,-1,-1,39,&
    -1,82,42,-1,-1,178,41,-1,-1,-1,-1,53,115,72,193,54,-1,-1,-1,-1,-1,62,81,-1,-1,128,85,130,66,&
    -1,-1,-1,152,73,93,74,137,-1,-1,-1,-1,122,101,65,127,145,-1,86,88,-1,114,154,130,132,94,175,&
    95,71,-1,-1,142,120,121,-1,104,105,124,107,108,-1,110,111,92,172,133,134,116,117,-1,-1,-1,124,&
    141,-1,-1,126,-1,87,-1,-1,-1,112,131,135,174,153,109,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    176,151,155,173,136,156,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,171,192,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,295,296,278,335,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    276,297,277,-1,-1,-1,-1,-1,343,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1 /)
    allocate(connect_next_cell_index(ncells))
    connect_next_cell_index = (/ &
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,66,147,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,75,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 /)
    allocate(flood_force_merge_index(ncells))
    flood_force_merge_index = (/ &
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,122,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,128,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,152,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 /)
    allocate(connect_force_merge_index(ncells))
    connect_force_merge_index = (/ &
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1 /)
    allocate(flood_redirect_index(ncells))
    flood_redirect_index = (/ &
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,154,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,154,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,125,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,113,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,13,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1 /)
    allocate(connect_redirect_index(ncells))
    connect_redirect_index = (/ &
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
    -1,-1,-1,-1 /)
      call add_offset(flood_next_cell_index,1,(/-1/))
      call add_offset(connect_next_cell_index,1,(/-1/))
      call add_offset(flood_force_merge_index,1,(/-1/))
      call add_offset(connect_force_merge_index,1,(/-1/))
      call add_offset(flood_redirect_index,1,(/-1/))
      call add_offset(connect_redirect_index,1,(/-1/))
      allocate(additional_flood_redirect_index(ncells))
      additional_flood_redirect_index(:) = 0
      allocate(additional_connect_redirect_index(ncells))
      additional_connect_redirect_index(:) = 0
      instant_throughflow_local=.True.
      lake_retention_coefficient_local=0.1
      allocate(coarse_cell_numbers_on_fine_grid(ncells))
      coarse_cell_numbers_on_fine_grid = (/ &
              1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4, &
              1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4, &
              1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4, &
              1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4, &
              1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4, &
              5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8, &
              5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8, &
              5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8, &
              5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8, &
              5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8, &
              9,9,9,9,9,10,10,10,10,10,11,11,11,11,11,12,12,12,12,12, &
              9,9,9,9,9,10,10,10,10,10,11,11,11,11,11,12,12,12,12,12, &
              9,9,9,9,9,10,10,10,10,10,11,11,11,11,11,12,12,12,12,12, &
              9,9,9,9,9,10,10,10,10,10,11,11,11,11,11,12,12,12,12,12, &
              9,9,9,9,9,10,10,10,10,10,11,11,11,11,11,12,12,12,12,12, &
              13,13,13,13,13,14,14,14,14,14,15,15,15,15,15,16,16,16,16,16, &
              13,13,13,13,13,14,14,14,14,14,15,15,15,15,15,16,16,16,16,16, &
              13,13,13,13,13,14,14,14,14,14,15,15,15,15,15,16,16,16,16,16, &
              13,13,13,13,13,14,14,14,14,14,15,15,15,15,15,16,16,16,16,16, &
              13,13,13,13,13,14,14,14,14,14,15,15,15,15,15,16,16,16,16,16 /)
      lake_parameters => LakeParameters(lake_centers, &
                                        connection_volume_thresholds, &
                                        flood_volume_thresholds, &
                                        flood_local_redirect, &
                                        connect_local_redirect, &
                                        additional_flood_local_redirect, &
                                        additional_connect_local_redirect, &
                                        merge_points, &
                                        flood_next_cell_index, &
                                        connect_next_cell_index, &
                                        flood_force_merge_index, &
                                        connect_force_merge_index, &
                                        flood_redirect_index, &
                                        connect_redirect_index, &
                                        additional_flood_redirect_index, &
                                        additional_connect_redirect_index, &
                                        ncells, &
                                        ncells_coarse, &
                                        instant_throughflow_local, &
                                        lake_retention_coefficient_local, &
                                        coarse_cell_numbers_on_fine_grid)
      allocate(drainage(16))
      drainage = (/ &
         1.0, 1.0, 1.0, 1.0, &
         1.0, 1.0, 1.0, 1.0, &
         1.0, 1.0, 1.0, 1.0, &
         1.0, 0.0, 1.0, 0.0 /)
      allocate(runoff(16))
      runoff = (/ &
         1.0, 1.0, 1.0, 1.0, &
         1.0, 1.0, 1.0, 1.0, &
         1.0, 1.0, 1.0, 1.0, &
         1.0, 0.0, 1.0, 0.0  /)
      allocate(drainages(16,5000))
      allocate(runoffs(16,5000))
      do i = 1,5000
        drainages(:,i) = drainage(:)
        runoffs(:,i) = runoff(:)
      end do
      allocate(evaporation(16))
      evaporation(:) = 0.0
      allocate(evaporations(16,5000))
      do i = 1,5000
        evaporations(:,i) = evaporation(:)
      end do
      evaporation(:) = 100.0
      allocate(evaporations_set_two(16,5000))
      do i = 1,5000
        evaporations_set_two(:,i) = evaporation(:)
      end do
      allocate(initial_spillover_to_rivers(16))
      initial_spillover_to_rivers(:) = 0.0
      allocate(initial_water_to_lake_centers(400))
      initial_water_to_lake_centers(:) = 0.0
      allocate(expected_river_inflow(16))
      expected_river_inflow = (/ &
          0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 2.0, &
         0.0, 2.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0 /)
      allocate(expected_water_to_ocean(16))
      expected_water_to_ocean = (/ &
          -96.0,   0.0,   0.0,   0.0, &
         0.0, -96.0, -96.0,   0.0, &
         0.0,   0.0,   0.0, -94.0, &
         -98.0,   4.0,   0.0,   4.0 /)
      allocate(expected_water_to_hd(16))
      expected_water_to_hd = (/ &
          0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0 /)
      allocate(expected_lake_numbers(400))
      expected_lake_numbers = (/ &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 /)
      allocate(expected_lake_types(400))
      expected_lake_types = (/ &
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 /)
      allocate(expected_lake_volumes(6))
      expected_lake_volumes = (/0.0, 0.0, 0.0, 0.0, 0.0, 0.0/)
      allocate(expected_intermediate_river_inflow(16))
      expected_intermediate_river_inflow = (/ &
          0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 2.0, &
         0.0, 2.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0 /)
      allocate(expected_intermediate_water_to_ocean(16))
      expected_intermediate_water_to_ocean = (/ &
          0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, &
         0.0, 6.0, 0.0, 22.0 /)
      allocate(expected_intermediate_water_to_hd(16))
      expected_intermediate_water_to_hd = (/ &
         0.0, 0.0, 0.0,  0.0, &
         0.0, 0.0, 0.0, 12.0, &
         0.0, 0.0, 0.0,  0.0, &
         0.0, 2.0, 0.0, 18.0 /)
      allocate(expected_intermediate_lake_numbers(400))
      expected_intermediate_lake_numbers = (/ &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, &
         1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 1, &
         0, 1, 1, 0, 0, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0, &
         0, 1, 1, 0, 0, 3, 3, 3, 3, 0, 0, 0, 2, 4, 4, 4, 4, 0, 0, 0, &
         0, 1, 1, 4, 3, 3, 0, 3, 3, 4, 4, 2, 4, 2, 4, 4, 4, 4, 0, 0, &
         1, 1, 1, 0, 3, 3, 3, 3, 3, 0, 4, 2, 2, 4, 4, 4, 4, 4, 0, 0, &
         0, 1, 1, 0, 0, 3, 0, 3, 0, 0, 0, 4, 2, 4, 4, 4, 4, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, &
         0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 /)
      allocate(expected_intermediate_lake_types(400))
      expected_intermediate_lake_types = (/ &
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, &
         3, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 3, &
         0, 3, 3, 0, 0, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, &
         0, 3, 3, 0, 0, 3, 3, 3, 3, 0, 0, 0, 3, 2, 2, 2, 2, 0, 0, 0, &
         0, 3, 3, 2, 3, 3, 0, 3, 3, 2, 2, 3, 2, 3, 2, 2, 2, 2, 0, 0, &
         3, 3, 3, 0, 3, 3, 3, 3, 3, 0, 2, 3, 3, 2, 2, 2, 2, 2, 0, 0, &
         0, 3, 3, 0, 0, 3, 0, 3, 0, 0, 0, 2, 3, 2, 2, 2, 2, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, &
         0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 /)
      allocate(expected_intermediate_lake_volumes(6))
      expected_intermediate_lake_volumes = (/46.0, 6.0, 38.0,  340.0, 10.0, 1.0/)
      allocate(runoffs_copy(16,5000))
      allocate(drainages_copy(16,5000))
      drainages_copy(:,:) = drainages(:,:)
      runoffs_copy(:,:) = runoffs(:,:)
      call init_hd_model_for_testing(river_parameters,river_fields,.True., &
                                     lake_parameters, &
                                     initial_water_to_lake_centers, &
                                     initial_spillover_to_rivers)
      call run_hd_model(5000,runoffs,drainages,evaporations)
      lake_prognostics_out => get_lake_prognostics()
      lake_fields_out => get_lake_fields()
      allocate(lake_types(ncells))
      lake_types(:) = 0
      do i = 1,400
        lake_number = lake_fields_out%lake_numbers(i)
        if (lake_number > 0) then
          working_lake_ptr = lake_prognostics_out%lakes(lake_number)
          lake_type = working_lake_ptr%lake_pointer%lake_type
            if (lake_type == filling_lake_type) then
              lake_types(i) = 1
            else if (lake_type == overflowing_lake_type) then
              lake_types(i) = 2
            else if (lake_type == subsumed_lake_type) then
              lake_types(i) = 3
            else
              lake_types(i) = 4
            end if
        end if
      end do
      allocate(lake_volumes(6))
      do i = 1,size(lake_volumes)
        working_lake_ptr = lake_prognostics_out%lakes(i)
        lake_volumes(i) = working_lake_ptr%lake_pointer%lake_volume
      end do
      call assert_equals(expected_intermediate_river_inflow,river_fields%river_inflow,&
                         16)
      call assert_equals(expected_intermediate_water_to_ocean,river_fields%water_to_ocean,16,0.00001)
      call assert_equals(expected_intermediate_water_to_hd,lake_fields_out%water_to_hd,16,0.00001)
      call assert_equals(expected_intermediate_lake_numbers,lake_fields_out%lake_numbers,400)
      call assert_equals(expected_intermediate_lake_types,lake_types,400)
      call assert_equals(expected_intermediate_lake_volumes,lake_volumes,6)
      call run_hd_model(5000,runoffs_copy,drainages_copy,evaporations_set_two)
      lake_prognostics_out => get_lake_prognostics()
      lake_fields_out => get_lake_fields()
      lake_types(:) = 0
      do i = 1,400
        lake_number = lake_fields_out%lake_numbers(i)
        if (lake_number > 0) then
          working_lake_ptr = lake_prognostics_out%lakes(lake_number)
          lake_type = working_lake_ptr%lake_pointer%lake_type
            if (lake_type == filling_lake_type) then
              lake_types(i) = 1
            else if (lake_type == overflowing_lake_type) then
              lake_types(i) = 2
            else if (lake_type == subsumed_lake_type) then
              lake_types(i) = 3
            else
              lake_types(i) = 4
            end if
        end if
      end do
      do i = 1,size(lake_volumes)
        working_lake_ptr = lake_prognostics_out%lakes(i)
        lake_volumes(i) = working_lake_ptr%lake_pointer%lake_volume
      end do
      call assert_equals(expected_river_inflow,river_fields%river_inflow,&
                         16)
      call assert_equals(expected_water_to_ocean,river_fields%water_to_ocean,16,0.00001)
      call assert_equals(expected_water_to_hd,lake_fields_out%water_to_hd,16,0.00001)
      call assert_equals(expected_lake_numbers,lake_fields_out%lake_numbers,400)
      call assert_equals(expected_lake_types,lake_types,400)
      call assert_equals(expected_lake_volumes,lake_volumes,6)
      deallocate(lake_volumes)
      deallocate(drainage)
      deallocate(drainages)
      deallocate(runoff)
      deallocate(runoffs)
      deallocate(initial_spillover_to_rivers)
      deallocate(initial_water_to_lake_centers)
      deallocate(expected_river_inflow)
      deallocate(expected_water_to_ocean)
      deallocate(expected_water_to_hd)
      deallocate(expected_lake_numbers)
      deallocate(expected_lake_types)
      deallocate(expected_lake_volumes)
      deallocate(lake_types)
      call clean_lake_model()
      call clean_hd_model()
end subroutine testLakeModel3

subroutine testLakeModel4
  use icosohedral_hd_model_interface_mod
  use icosohedral_hd_model_mod
  use icosohedral_lake_model_io_mod
  type(riverparameters), pointer :: river_parameters
  type(riverprognosticfields), pointer :: river_fields
  type(lakeparameters),pointer :: lake_parameters
  type(lakefields), pointer :: lake_fields_out
  type(lakeprognostics), pointer :: lake_prognostics_out
  type(lakepointer) :: working_lake_ptr
  integer,dimension(:), pointer :: flow_directions
  integer,dimension(:), pointer :: river_reservoir_nums
  integer,dimension(:), pointer :: overland_reservoir_nums
  integer,dimension(:), pointer :: base_reservoir_nums
  real,dimension(:), pointer :: river_retention_coefficients
  real,dimension(:), pointer :: overland_retention_coefficients
  real,dimension(:), pointer :: base_retention_coefficients
  logical,dimension(:), pointer :: landsea_mask
  logical,dimension(:), pointer :: lake_centers
  real,dimension(:), pointer :: connection_volume_thresholds
  real,dimension(:), pointer :: flood_volume_thresholds
  logical,dimension(:), pointer :: flood_local_redirect
  logical,dimension(:), pointer :: connect_local_redirect
  logical,dimension(:), pointer :: additional_flood_local_redirect
  logical,dimension(:), pointer :: additional_connect_local_redirect
  integer,dimension(:), pointer :: merge_points
  integer,dimension(:), pointer :: flood_next_cell_index
  integer,dimension(:), pointer :: connect_next_cell_index
  integer,dimension(:), pointer :: flood_force_merge_index
  integer,dimension(:), pointer :: connect_force_merge_index
  integer,dimension(:), pointer :: flood_redirect_index
  integer,dimension(:), pointer :: connect_redirect_index
  integer,dimension(:), pointer :: additional_flood_redirect_index
  integer,dimension(:), pointer :: additional_connect_redirect_index
  real,dimension(:), pointer :: drainage
  real,dimension(:,:), pointer :: drainages
  real,dimension(:,:), pointer :: drainages_copy
  real,dimension(:), pointer :: runoff
  real,dimension(:,:), pointer :: runoffs
  real,dimension(:,:), pointer :: runoffs_copy
  real,dimension(:), pointer :: evaporation
  real,dimension(:,:), pointer :: evaporations
  real,dimension(:,:), pointer :: evaporations_set_two
  real, pointer, dimension(:) :: initial_spillover_to_rivers
  real, pointer, dimension(:) :: initial_water_to_lake_centers
  real,dimension(:), pointer :: expected_river_inflow
  real,dimension(:), pointer :: expected_water_to_ocean
  real,dimension(:), pointer :: expected_water_to_hd
  integer,dimension(:), pointer :: expected_lake_numbers
  integer,dimension(:), pointer :: expected_lake_types
  real,dimension(:), pointer :: expected_lake_volumes
  real,dimension(:), pointer :: expected_intermediate_river_inflow
  real,dimension(:), pointer :: expected_intermediate_water_to_ocean
  real,dimension(:), pointer :: expected_intermediate_water_to_hd
  integer,dimension(:), pointer :: expected_intermediate_lake_numbers
  integer,dimension(:), pointer :: expected_intermediate_lake_types
  real,dimension(:), pointer :: expected_intermediate_lake_volumes
  integer,dimension(:), pointer :: coarse_cell_numbers_on_fine_grid
  integer,dimension(:), pointer :: lake_types
  integer :: no_merge_mtype
  integer :: connection_merge_not_set_flood_merge_as_primary
  integer :: connection_merge_not_set_flood_merge_as_secondary
  real,dimension(:), pointer :: lake_volumes
  integer :: ncells
  integer :: ncells_coarse
  integer :: lake_number
  integer :: i
  logical :: instant_throughflow_local
  real :: lake_retention_coefficient_local
  integer :: lake_type
      no_merge_mtype = 0
      connection_merge_not_set_flood_merge_as_primary = 9
      connection_merge_not_set_flood_merge_as_secondary = 10
      ncells = 80
      ncells_coarse = 80
      allocate(flow_directions(ncells_coarse))
      flow_directions = (/  8,13,13,13,19, &
                            8,8,24,24,13, 13,13,-2,13,13, &
                            13,36,36,37,37, 8,24,24,64,45, &
                            45,45,49,49,13, 13,30,52,55,55, &
                            55,55,55,37,38, 61,61,64,64,64, &
                            64,64,64,-2,49, 30,54,55,55,0, &
                            55,55,38,38,59, 63,64,64,-2,64, &
                            38,49,52,55,55, 55,55,56,58,58, &
                            64,64,68,71,71 /)
      allocate(river_reservoir_nums(ncells_coarse))
      river_reservoir_nums(:) = 5
      allocate(overland_reservoir_nums(ncells_coarse))
      overland_reservoir_nums(:) = 1
      allocate(base_reservoir_nums(ncells_coarse))
      base_reservoir_nums(:) = 1
      allocate(river_retention_coefficients(ncells_coarse))
      river_retention_coefficients(:) = 0.7
      allocate(overland_retention_coefficients(ncells_coarse))
      overland_retention_coefficients(:) = 0.5
      allocate(base_retention_coefficients(ncells_coarse))
      base_retention_coefficients(:) = 0.1
      allocate(landsea_mask(ncells_coarse))
      landsea_mask(:) = .False.
      river_reservoir_nums(55) = 0
      overland_reservoir_nums(55) = 0
      base_reservoir_nums(55) = 0
      landsea_mask(55) = .True.
      river_parameters => RiverParameters(flow_directions, &
                                          river_reservoir_nums, &
                                          overland_reservoir_nums, &
                                          base_reservoir_nums, &
                                          river_retention_coefficients, &
                                          overland_retention_coefficients, &
                                          base_retention_coefficients, &
                                          landsea_mask)
      instant_throughflow_local=.True.
      lake_retention_coefficient_local=0.1
      allocate(coarse_cell_numbers_on_fine_grid(ncells))
      coarse_cell_numbers_on_fine_grid = (/  1,2,3,4,5, &
                                             6,7,8,9,10,     11,12,13,14,15, &
                                             16,17,18,19,20, 21,22,23,24,25, &
                                             26,27,28,29,30, 31,32,33,34,35, &
                                             36,37,38,39,40, 41,42,43,44,45, &
                                             46,47,48,49,50, 51,52,53,54,55, &
                                             56,57,58,59,60, 61,62,63,64,65, &
                                             66,67,68,69,70, 71,72,73,74,75, &
                                             76,77,78,79,80 /)
      river_fields => riverprognosticfields(80,1,1,5)
      allocate(lake_centers(ncells))
      lake_centers = (/ &
           .False.,.False.,.False.,.False.,.False., &
           .False.,.False.,.False.,.False.,.False., .False.,.False.,.True.,.False.,.False., &
           .False.,.False.,.False.,.False.,.False., .False.,.False.,.False.,.False.,.False., &
           .False.,.False.,.False.,.False.,.False., .False.,.False.,.False.,.False.,.False., &
           .False.,.False.,.False.,.False.,.False., .False.,.False.,.False.,.False.,.False., &
           .False.,.False.,.False.,.True., .False., .False.,.False.,.False.,.False.,.False., &
           .False.,.False.,.False.,.False.,.False., .False.,.False.,.False., .True.,.False., &
           .False.,.False.,.False.,.False.,.False., .False.,.False.,.False.,.False.,.False., &
           .False.,.False.,.False.,.False.,.False. /)
      allocate(connection_volume_thresholds(ncells))
      connection_volume_thresholds(:) = -1.0
      allocate(flood_volume_thresholds(ncells))
      flood_volume_thresholds = (/ -1.0,-1.0,-1.0,-1.0,-1.0, &
                                  -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,3.0,-1.0,-1.0, &
                                  -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, &
                                  -1.0,-1.0,-1.0,-1.0, 4.0, -1.0,-1.0,-1.0,-1.0,-1.0, &
                                  -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0, 5.0, &
                                  1.0,22.0,-1.0,1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, &
                                  -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,1.0,1.0,-1.0, &
                                  -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, &
                                  15.0,-1.0,-1.0,-1.0,-1.0 /)
      allocate(flood_local_redirect(ncells))
      flood_local_redirect = (/ &
        .True.,.True.,.True.,.True.,.True., &
        .True.,.True.,.True.,.True.,.True., .True.,.True.,.True.,.True.,.True., &
        .True.,.True.,.True.,.True.,.True., .True.,.True.,.True.,.True.,.True., &
        .True.,.True.,.True.,.True.,.True., .True.,.True.,.True.,.True.,.True., &
        .True.,.True.,.True.,.True.,.True., .True.,.True.,.True.,.True.,.True., &
        .True.,.False.,.True.,.True.,.True.,.True.,.True.,.True.,.True.,.True., &
        .True.,.True.,.True.,.True.,.True., .True.,.True.,.True.,.True.,.True., &
        .True.,.True.,.True.,.True.,.True., .True.,.True.,.True.,.True.,.True., &
        .True.,.True.,.True.,.True.,.True. /)
      allocate(connect_local_redirect(ncells))
      connect_local_redirect = .False.
      allocate(additional_flood_local_redirect(ncells))
      additional_flood_local_redirect = .False.
      allocate(additional_connect_local_redirect(ncells))
      additional_connect_local_redirect = .False.
      allocate(merge_points(ncells))
      merge_points = (/ &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,connection_merge_not_set_flood_merge_as_secondary, &
        no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        connection_merge_not_set_flood_merge_as_primary, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,connection_merge_not_set_flood_merge_as_secondary,no_merge_mtype, &
        connection_merge_not_set_flood_merge_as_primary,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype, &
        connection_merge_not_set_flood_merge_as_secondary,no_merge_mtype,no_merge_mtype, &
        no_merge_mtype,no_merge_mtype /)
      allocate(flood_next_cell_index(ncells))
      flood_next_cell_index(:) = (/ -1,-1,-1,-1,-1, &
                                    -1,-1,-1,-1,-1, -1,-1,49,-1,-1, &
                                    -1,-1,-1,-1,-1, -1,-1,-1,-1,-1, &
                                    -1,-1,-1,-1,47, -1,-1,-1,-1,-1, &
                                    -1,-1,-1,-1,-1, -1,-1,-1,-1,76, &
                                    45,52,-1,30,-1, -1,-1,-1,-1,-1, &
                                    -1,-1,-1,-1,-1, -1,-1,46,63,-1, &
                                    -1,-1,-1,-1,-1, -1,-1,-1,-1,-1, &
                                    49,-1,-1,-1,-1 /)
      allocate(connect_next_cell_index(ncells))
      connect_next_cell_index(:) = -1
      allocate(flood_force_merge_index(ncells))
      flood_force_merge_index(:) = (/ -1,-1,-1,-1,-1, &
                                      -1,-1,-1,-1,-1, -1,-1,-1,-1,-1, &
                                      -1,-1,-1,-1,-1, -1,-1,-1,-1,-1, &
                                      -1,-1,-1,-1,64, -1,-1,-1,-1,-1, &
                                      -1,-1,-1,-1,-1, -1,-1,-1,-1,-1, &
                                      -1,-1,-1,13,-1, -1,-1,-1,-1,-1, &
                                      -1,-1,-1,-1,-1, -1,-1,-1,-1,-1, &
                                      -1,-1,-1,-1,-1, -1,-1,-1,-1,-1, &
                                      -1,-1,-1,-1,-1 /)
      allocate(connect_force_merge_index(ncells))
      connect_force_merge_index(:) = -1
      allocate(flood_redirect_index(ncells))
      flood_redirect_index = (/ -1,-1,-1,-1,-1, &
                                -1,-1,-1,-1,-1, -1,-1,49,-1,-1, &
                                -1,-1,-1,-1,-1, -1,-1,-1,-1,-1, &
                                -1,-1,-1,-1,64, -1,-1,-1,-1,-1, &
                                -1,-1,-1,-1,-1, -1,-1,-1,-1,-1, &
                                -1,52,-1,13,-1, -1,-1,-1,-1,-1, &
                                -1,-1,-1,-1,-1, -1,-1,-1,-1,-1, &
                                -1,-1,-1,-1,-1, -1,-1,-1,-1,-1, &
                                49,-1,-1,-1,-1 /)
      allocate(connect_redirect_index(ncells))
      connect_redirect_index(:) = -1
      allocate(additional_flood_redirect_index(ncells))
      additional_flood_redirect_index(:) = 0
      allocate(additional_connect_redirect_index(ncells))
      additional_connect_redirect_index(:) = 0
      lake_parameters => LakeParameters(lake_centers, &
                                        connection_volume_thresholds, &
                                        flood_volume_thresholds, &
                                        flood_local_redirect, &
                                        connect_local_redirect, &
                                        additional_flood_local_redirect, &
                                        additional_connect_local_redirect, &
                                        merge_points, &
                                        flood_next_cell_index, &
                                        connect_next_cell_index, &
                                        flood_force_merge_index, &
                                        connect_force_merge_index, &
                                        flood_redirect_index, &
                                        connect_redirect_index, &
                                        additional_flood_redirect_index, &
                                        additional_connect_redirect_index, &
                                        ncells, &
                                        ncells_coarse, &
                                        instant_throughflow_local, &
                                        lake_retention_coefficient_local, &
                                        coarse_cell_numbers_on_fine_grid)
      allocate(drainage(ncells_coarse))
      drainage(:) = 1.0
      drainage(55) = 0.0
      allocate(runoff(ncells_coarse))
      runoff(:) = drainage(:)
      allocate(drainages(ncells_coarse,5000))
      allocate(runoffs(ncells_coarse,5000))
      do i = 1,5000
        drainages(:,i) = drainage(:)
        runoffs(:,i) = runoff(:)
      end do
      allocate(evaporation(ncells_coarse))
      evaporation(:) = 0.0
      allocate(evaporations(ncells_coarse,5000))
      do i = 1,5000
        evaporations(:,i) = evaporation(:)
      end do
      evaporation(:) = 100.0
      allocate(evaporations_set_two(ncells_coarse,5000))
      do i = 1,5000
        evaporations_set_two(:,i) = evaporation(:)
      end do
      allocate(initial_spillover_to_rivers(ncells_coarse))
      initial_spillover_to_rivers(:) = 0.0
      allocate(initial_water_to_lake_centers(ncells_coarse))
      initial_water_to_lake_centers(:) = 0.0
      allocate(expected_river_inflow(ncells_coarse))
      expected_river_inflow = (/ &
              0.0, 0.0, 0.0, 0.0, 0.0, &
              0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, &
              0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, &
              0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 8.0, 14.0, 0.0, 0.0, &
              0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
              0.0, 6.0, 0.0, 8.0, 0.0, 2.0, 0.0, 4.0, 2.0, 0.0, &
              4.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, &
              0.0, 0.0, 0.0, 0.0, 0.0  /)
      allocate(expected_water_to_ocean(ncells_coarse))
      expected_water_to_ocean = (/ &
              0.0, 0.0, 0.0, 0.0, 0.0, &
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-72.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, &
             0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, &
             0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-90.0,0.0,0.0,0.0,0.0,0.0, 66.0,0.0,0.0,0.0,0.0,0.0, &
             0.0,0.0,0.0,-46.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, &
             0.0,0.0,0.0,0.0, 0.0 /)
      allocate(expected_water_to_hd(ncells_coarse))
      expected_water_to_hd = (/ &
              0.0, 0.0, 0.0, 0.0, 0.0, &
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
              0.0, 0.0, 0.0, 0.0, 0.0 /)
      allocate(expected_lake_numbers(ncells))
      expected_lake_numbers = (/ &
                     0, 0, 0, 0, 0, &
                     0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, &
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                     0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                     0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                     0, 0, 0, 0, 0 /)
      allocate(expected_lake_types(ncells))
      expected_lake_types = (/ &
                     0, 0, 0, 0, 0, &
                     0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, &
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                     0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
                     0, 0, 0, 0, 0 /)
      allocate(expected_lake_volumes(3))
      expected_lake_volumes = (/ 0.0, 0.0, 0.0 /)
      allocate(expected_intermediate_river_inflow(ncells_coarse))
      expected_intermediate_river_inflow =  (/ &
                0.0, 0.0, 0.0, 0.0, 0.0, &
                0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, &
                0.0, 0.0, 0.0,16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0,  0.0, &
                4.0, 8.0, 14.0,0.0, 0.0,&
                0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 100.00,0.0, &
                2.0, 0.0, 4.0, 2.0, 0.0, &
                4.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, &
                0.0, 0.0, 0.0, 0.0, 0.0 /)
      allocate(expected_intermediate_water_to_ocean(ncells_coarse))
      expected_intermediate_water_to_ocean =  (/ &
              0.0,0.0,0.0,0.0,0.0, &
              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, &
              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, &
              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,158.0,0.0,0.0,0.0,0.0,0.0, &
              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, &
              0.0,0.0,0.0,0.0, 0.0 /)
      allocate(expected_intermediate_water_to_hd(ncells_coarse))
      expected_intermediate_water_to_hd =  (/ &
              0.0,0.0,0.0,0.0,0.0, &
              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, &
              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, &
              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,92.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, &
              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, &
              0.0,0.0,0.0,0.0, 0.0 /)
      allocate(expected_intermediate_lake_numbers(ncells_coarse))
      expected_intermediate_lake_numbers =  (/  &
               0, 0, 0, 0, 0, &
               0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, &
               0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
               0, 0, 0, 0, 3, 3, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
               0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
               3, 0, 0, 0, 0 /)
      allocate(expected_intermediate_lake_types(ncells))
      expected_intermediate_lake_types =  (/ &
               0, 0, 0, 0, 0, &
               0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, &
               0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
               0, 0, 0, 0, 3, 3, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
               0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
               3, 0, 0, 0, 0 /)
      allocate(expected_intermediate_lake_volumes(3))
      expected_intermediate_lake_volumes = (/ 3.0, 22.0, 15.0 /)
      allocate(runoffs_copy(80,5000))
      allocate(drainages_copy(80,5000))
      drainages_copy(:,:) = drainages(:,:)
      runoffs_copy(:,:) = runoffs(:,:)
      call init_hd_model_for_testing(river_parameters,river_fields,.True., &
                                     lake_parameters, &
                                     initial_water_to_lake_centers, &
                                     initial_spillover_to_rivers)
      call run_hd_model(5000,runoffs,drainages,evaporations)
      lake_prognostics_out => get_lake_prognostics()
      lake_fields_out => get_lake_fields()
      allocate(lake_types(ncells))
      lake_types(:) = 0
      do i = 1,ncells
        lake_number = lake_fields_out%lake_numbers(i)
        if (lake_number > 0) then
          working_lake_ptr = lake_prognostics_out%lakes(lake_number)
          lake_type = working_lake_ptr%lake_pointer%lake_type
            if (lake_type == filling_lake_type) then
              lake_types(i) = 1
            else if (lake_type == overflowing_lake_type) then
              lake_types(i) = 2
            else if (lake_type == subsumed_lake_type) then
              lake_types(i) = 3
            else
              lake_types(i) = 4
            end if
        end if
      end do
      allocate(lake_volumes(3))
      do i = 1,size(lake_volumes)
        working_lake_ptr = lake_prognostics_out%lakes(i)
        lake_volumes(i) = working_lake_ptr%lake_pointer%lake_volume
      end do
      call assert_equals(expected_intermediate_river_inflow,river_fields%river_inflow,&
                         ncells_coarse,0.0001)
      call assert_equals(expected_intermediate_water_to_ocean, &
                         river_fields%water_to_ocean,ncells_coarse,0.0001)
      call assert_equals(expected_intermediate_water_to_hd, &
                         lake_fields_out%water_to_hd,ncells_coarse,0.0001)
      call assert_equals(expected_intermediate_lake_numbers,lake_fields_out%lake_numbers,ncells)
      call assert_equals(expected_intermediate_lake_types,lake_types,ncells)
      call assert_equals(expected_intermediate_lake_volumes,lake_volumes,3)
      call run_hd_model(5000,runoffs_copy,drainages_copy,evaporations_set_two)
      lake_prognostics_out => get_lake_prognostics()
      lake_fields_out => get_lake_fields()
      lake_types(:) = 0
      do i = 1,ncells
        lake_number = lake_fields_out%lake_numbers(i)
        if (lake_number > 0) then
          working_lake_ptr = lake_prognostics_out%lakes(lake_number)
          lake_type = working_lake_ptr%lake_pointer%lake_type
            if (lake_type == filling_lake_type) then
              lake_types(i) = 1
            else if (lake_type == overflowing_lake_type) then
              lake_types(i) = 2
            else if (lake_type == subsumed_lake_type) then
              lake_types(i) = 3
            else
              lake_types(i) = 4
            end if
        end if
      end do
      do i = 1,size(lake_volumes)
        working_lake_ptr = lake_prognostics_out%lakes(i)
        lake_volumes(i) = working_lake_ptr%lake_pointer%lake_volume
      end do
      call assert_equals(expected_river_inflow,river_fields%river_inflow,&
                         ncells_coarse)
      call assert_equals(expected_water_to_ocean,river_fields%water_to_ocean,ncells_coarse,0.00001)
      call assert_equals(expected_water_to_hd,lake_fields_out%water_to_hd,ncells_coarse,0.00001)
      call assert_equals(expected_lake_numbers,lake_fields_out%lake_numbers,ncells)
      call assert_equals(expected_lake_types,lake_types,ncells)
      call assert_equals(expected_lake_volumes,lake_volumes,3)
      deallocate(lake_volumes)
      deallocate(drainage)
      deallocate(drainages)
      deallocate(runoff)
      deallocate(runoffs)
      deallocate(initial_spillover_to_rivers)
      deallocate(initial_water_to_lake_centers)
      deallocate(expected_river_inflow)
      deallocate(expected_water_to_ocean)
      deallocate(expected_water_to_hd)
      deallocate(expected_lake_numbers)
      deallocate(expected_lake_types)
      deallocate(expected_lake_volumes)
      deallocate(lake_types)
      call clean_lake_model()
      call clean_hd_model()
end subroutine testLakeModel4

end module icosohedral_hd_and_lake_model_test_mod
