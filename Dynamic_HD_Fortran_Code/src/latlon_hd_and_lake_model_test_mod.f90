module latlon_hd_and_lake_model_test_mod
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
   real   ,dimension(:,:), pointer :: river_retention_coefficients
   real   ,dimension(:,:), pointer :: overland_retention_coefficients
   real   ,dimension(:,:), pointer :: base_retention_coefficients
   logical,dimension(:,:), pointer :: landsea_mask
   real   ,dimension(:,:), allocatable :: drainage
   real   ,dimension(:,:,:), allocatable :: drainages
   real   ,dimension(:,:,:), allocatable :: runoffs
   real   ,dimension(:,:), allocatable :: expected_water_to_ocean
   real   ,dimension(:,:), allocatable :: expected_river_inflow
   real   ,dimension(:,:), pointer :: flow_directions
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

subroutine testLakeModel1
   use latlon_hd_model_interface_mod
   use latlon_hd_model_mod
   use latlon_lake_model_io_mod
   type(riverparameters), pointer :: river_parameters
   type(riverprognosticfields), pointer:: river_fields
   type(lakeparameters),pointer :: lake_parameters
   type(lakefields), pointer :: lake_fields_out
   type(lakeprognostics), pointer :: lake_prognostics_out
   type(lakepointer) :: working_lake_ptr
   real   ,dimension(:,:), pointer :: flow_directions
   integer,dimension(:,:), pointer :: river_reservoir_nums
   integer,dimension(:,:), pointer :: overland_reservoir_nums
   integer,dimension(:,:), pointer :: base_reservoir_nums
   real,dimension(:,:), pointer :: river_retention_coefficients
   real,dimension(:,:), pointer :: overland_retention_coefficients
   real,dimension(:,:), pointer :: base_retention_coefficients
   logical,dimension(:,:), pointer :: landsea_mask
   logical,dimension(:,:), pointer :: lake_centers
   real,dimension(:,:), pointer :: connection_volume_thresholds
   real,dimension(:,:), pointer :: flood_volume_thresholds
   logical,dimension(:,:), pointer :: flood_local_redirect
   logical,dimension(:,:), pointer :: connect_local_redirect
   logical,dimension(:,:), pointer :: additional_flood_local_redirect
   logical,dimension(:,:), pointer :: additional_connect_local_redirect
   integer,dimension(:,:), pointer :: merge_points
   integer,dimension(:,:), pointer :: flood_next_cell_lat_index
   integer,dimension(:,:), pointer :: flood_next_cell_lon_index
   integer,dimension(:,:), pointer :: connect_next_cell_lat_index
   integer,dimension(:,:), pointer :: connect_next_cell_lon_index
   integer,dimension(:,:), pointer :: flood_force_merge_lat_index
   integer,dimension(:,:), pointer :: flood_force_merge_lon_index
   integer,dimension(:,:), pointer :: connect_force_merge_lat_index
   integer,dimension(:,:), pointer :: connect_force_merge_lon_index
   integer,dimension(:,:), pointer :: flood_redirect_lat_index
   integer,dimension(:,:), pointer :: flood_redirect_lon_index
   integer,dimension(:,:), pointer :: connect_redirect_lat_index
   integer,dimension(:,:), pointer :: connect_redirect_lon_index
   integer,dimension(:,:), pointer :: additional_flood_redirect_lat_index
   integer,dimension(:,:), pointer :: additional_flood_redirect_lon_index
   integer,dimension(:,:), pointer :: additional_connect_redirect_lat_index
   integer,dimension(:,:), pointer :: additional_connect_redirect_lon_index
   real,dimension(:,:), pointer :: drainage
   real,dimension(:,:,:), pointer :: drainages
   real,dimension(:,:), pointer :: runoff
   real,dimension(:,:,:), pointer :: runoffs
   real, pointer, dimension(:,:) :: initial_spillover_to_rivers
   real, pointer, dimension(:,:) :: initial_water_to_lake_centers
   real,dimension(:,:), pointer :: expected_river_inflow
   real,dimension(:,:), pointer :: expected_water_to_ocean
   real,dimension(:,:), pointer :: expected_water_to_hd
   integer,dimension(:,:), pointer :: expected_lake_numbers
   integer,dimension(:,:), pointer :: expected_lake_types
   real,dimension(:), pointer :: expected_lake_volumes
   integer,dimension(:,:), pointer :: lake_types
   integer :: no_merge_mtype,connection_merge_not_set_flood_merge_as_secondary
   real,dimension(:), pointer :: lake_volumes
   integer :: nlat,nlon
   integer :: nlat_coarse,nlon_coarse
   integer :: timesteps
   integer :: lake_number
   integer :: i,j
   logical :: instant_throughflow_local
   real :: lake_retention_coefficient_local
   integer :: lake_type
      no_merge_mtype = 0
      connection_merge_not_set_flood_merge_as_secondary = 10
      timesteps = 1000
      nlat = 9
      nlon = 9
      nlat_coarse = 3
      nlon_coarse = 3
      allocate(flow_directions(nlat_coarse,nlon_coarse))
      flow_directions = transpose(reshape((/ &
                                  -2,6,2, &
                                   8,7,2, &
                                   8,7,0 /), &
                                    (/3,3/)))
      allocate(river_reservoir_nums(nlat_coarse,nlon_coarse))
      river_reservoir_nums(:,:) = 5
      allocate(overland_reservoir_nums(nlat_coarse,nlon_coarse))
      overland_reservoir_nums(:,:) = 1
      allocate(base_reservoir_nums(nlat_coarse,nlon_coarse))
      base_reservoir_nums(:,:) = 1
      allocate(river_retention_coefficients(nlat_coarse,nlon_coarse))
      river_retention_coefficients(:,:) = 0.7
      allocate(overland_retention_coefficients(nlat_coarse,nlon_coarse))
      overland_retention_coefficients(:,:) = 0.5
      allocate(base_retention_coefficients(nlat_coarse,nlon_coarse))
      base_retention_coefficients(:,:) = 0.1
      allocate(landsea_mask(nlat_coarse,nlon_coarse))
      river_reservoir_nums(3,3) = 0
      overland_reservoir_nums(3,3) = 0
      base_reservoir_nums(3,3) = 0
      landsea_mask(:,:) = .False.
      landsea_mask(3,3) = .True.
      river_parameters => RiverParameters(flow_directions, &
                                          river_reservoir_nums, &
                                          overland_reservoir_nums, &
                                          base_reservoir_nums, &
                                          river_retention_coefficients, &
                                          overland_retention_coefficients, &
                                          base_retention_coefficients, &
                                          landsea_mask)
      river_fields => riverprognosticfields(3,3,1,1,5)
      allocate(lake_centers(nlat,nlon))
      lake_centers = transpose(reshape((/ &
          .False., .False., .False., .False., .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., &
         .False., .False.,  .True., .False., .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., .False., .False., .False., .False. /), &
         (/nlon,nlat/)))
      allocate(connection_volume_thresholds(nlat,nlon))
      connection_volume_thresholds(:,:) = -1.0
      allocate(flood_volume_thresholds(nlat,nlon))
      flood_volume_thresholds = transpose(reshape((/ &
          -1.0, -1.0, -1.0, 80.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, 80.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0,  1.0, 80.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, 10.0, 10.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, 10.0, 10.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, &
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 /), &
         (/nlon,nlat/)))
      allocate(flood_local_redirect(nlat,nlon))
      flood_local_redirect(:,:) = .False.
      allocate(connect_local_redirect(nlat,nlon))
      connect_local_redirect(:,:) = .False.
      allocate(additional_flood_local_redirect(nlat,nlon))
      additional_flood_local_redirect(:,:) = .False.
      allocate(additional_connect_local_redirect(nlat,nlon))
      additional_connect_local_redirect(:,:) = .False.
      allocate(merge_points(nlat,nlon))
      merge_points = transpose(reshape((/ &
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
        no_merge_mtype,no_merge_mtype,no_merge_mtype /), &
         (/nlon,nlat/)))
      allocate(flood_next_cell_lat_index(nlat,nlon))
      flood_next_cell_lat_index = transpose(reshape((/ &
          0, 0, 0, 1, 0, 0, 0, 0, 0, &
         0, 0, 0, 1, 0, 0, 0, 0, 0, &
         0, 0, 4, 2, 0, 0, 0, 0, 0, &
         0, 0, 5, 5, 0, 0, 0, 0, 0, &
         0, 0, 4, 3, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0 /), &
         (/nlon,nlat/)))
      allocate(flood_next_cell_lon_index(nlat,nlon))
      flood_next_cell_lon_index = transpose(reshape((/ &
          0, 0, 0, 5, 0, 0, 0, 0, 0, &
         0, 0, 0, 4, 0, 0, 0, 0, 0, &
         0, 0, 3, 4, 0, 0, 0, 0, 0, &
         0, 0, 3, 4, 0, 0, 0, 0, 0, &
         0, 0, 4, 4, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0 /), &
         (/nlon,nlat/)))
      allocate(connect_next_cell_lat_index(nlat,nlon))
      connect_next_cell_lat_index = transpose(reshape((/ &
          0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0 /), &
         (/nlon,nlat/)))
      allocate(connect_next_cell_lon_index(nlat,nlon))
      connect_next_cell_lon_index = transpose(reshape((/ &
          0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0 /), &
         (/nlon,nlat/)))
      allocate(flood_force_merge_lat_index(nlat,nlon))
      flood_force_merge_lat_index = transpose(reshape((/ &
          0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0 /), &
         (/nlon,nlat/)))
      allocate(flood_force_merge_lon_index(nlat,nlon))
      flood_force_merge_lon_index = transpose(reshape((/ &
          0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0 /), &
         (/nlon,nlat/)))
      allocate(connect_force_merge_lat_index(nlat,nlon))
      connect_force_merge_lat_index = transpose(reshape((/ &
          0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0 /), &
         (/nlon,nlat/)))
      allocate(connect_force_merge_lon_index(nlat,nlon))
      connect_force_merge_lon_index = transpose(reshape((/ &
          0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0 /), &
         (/nlon,nlat/)))
      allocate(flood_redirect_lat_index(nlat,nlon))
      flood_redirect_lat_index = transpose(reshape((/ &
          0, 0, 0, 1, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0 /), &
         (/nlon,nlat/)))
      allocate(flood_redirect_lon_index(nlat,nlon))
      flood_redirect_lon_index = transpose(reshape((/ &
          0, 0, 0, 3, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0 /), &
         (/nlon,nlat/)))
      allocate(connect_redirect_lat_index(nlat,nlon))
      connect_redirect_lat_index = transpose(reshape((/ &
          0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0 /), &
         (/nlon,nlat/)))
      allocate(connect_redirect_lon_index(nlat,nlon))
      connect_redirect_lon_index = transpose(reshape((/ &
          0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, 0, 0, 0, 0 /), &
         (/nlon,nlat/)))
      allocate(additional_flood_redirect_lat_index(nlat,nlon))
      additional_flood_redirect_lat_index(:,:) = 0
      allocate(additional_flood_redirect_lon_index(nlat,nlon))
      additional_flood_redirect_lon_index(:,:) = 0
      allocate(additional_connect_redirect_lat_index(nlat,nlon))
      additional_connect_redirect_lat_index(:,:) = 0
      allocate(additional_connect_redirect_lon_index(nlat,nlon))
      additional_connect_redirect_lon_index(:,:) = 0
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
                                        flood_next_cell_lat_index, &
                                        flood_next_cell_lon_index, &
                                        connect_next_cell_lat_index, &
                                        connect_next_cell_lon_index, &
                                        flood_force_merge_lat_index, &
                                        flood_force_merge_lon_index, &
                                        connect_force_merge_lat_index, &
                                        connect_force_merge_lon_index, &
                                        flood_redirect_lat_index, &
                                        flood_redirect_lon_index, &
                                        connect_redirect_lat_index, &
                                        connect_redirect_lon_index, &
                                        additional_flood_redirect_lat_index, &
                                        additional_flood_redirect_lon_index, &
                                        additional_connect_redirect_lat_index, &
                                        additional_connect_redirect_lon_index, &
                                        nlat,nlon, &
                                        nlat_coarse,nlon_coarse, &
                                        instant_throughflow_local, &
                                        lake_retention_coefficient_local)
      allocate(drainage(nlat_coarse,nlon_coarse))
      drainage = transpose(reshape((/ &
         1.0, 1.0, 1.0, &
         1.0, 1.0, 1.0, &
         1.0, 1.0, 0.0 /), &
         (/nlon_coarse,nlat_coarse/)))
      allocate(runoff(nlat_coarse,nlon_coarse))
      runoff = transpose(reshape((/ &
         1.0, 1.0, 1.0, &
         1.0, 1.0, 1.0, &
         1.0, 1.0, 0.0 /), &
         (/nlon_coarse,nlat_coarse/)))
      allocate(drainages(nlat_coarse,nlon_coarse,timesteps))
      allocate(runoffs(nlat_coarse,nlon_coarse,timesteps))
      do i = 1,timesteps
        drainages(:,:,i) = drainage(:,:)
        runoffs(:,:,i) = runoff(:,:)
      end do
      allocate(initial_spillover_to_rivers(nlat_coarse,nlon_coarse))
      initial_spillover_to_rivers(:,:) = 0.0
      allocate(initial_water_to_lake_centers(nlat,nlon))
      initial_water_to_lake_centers(:,:) = 0.0
      allocate(expected_river_inflow(nlat_coarse,nlon_coarse))
      expected_river_inflow = transpose(reshape((/ &
          0.0, 0.0,  2.0, &
         4.0, 0.0, 14.0, &
         0.0, 0.0,  0.0 /), &
         (/nlon_coarse,nlat_coarse/)))
      allocate(expected_water_to_ocean(nlat_coarse,nlon_coarse))
      expected_water_to_ocean = transpose(reshape((/ &
          0.0, 0.0,  0.0, &
         0.0, 0.0,  0.0, &
         0.0, 0.0, 16.0 /), &
         (/nlon_coarse,nlat_coarse/)))
      allocate(expected_water_to_hd(nlat_coarse,nlon_coarse))
      expected_water_to_hd = transpose(reshape((/ &
          0.0, 0.0, 10.0, &
         0.0, 0.0,  0.0, &
         0.0, 0.0,  0.0 /), &
         (/nlon_coarse,nlat_coarse/)))
      allocate(expected_lake_numbers(nlat,nlon))
      expected_lake_numbers = transpose(reshape((/ &
             0,    0,    0,    1,    0,    0,    0,    0,    0, &
         0,    0,    0,    1,    0,    0,    0,    0,    0, &
         0,    0,    1,    1,    0,    0,    0,    0,    0, &
         0,    0,    1,    1,    0,    0,    0,    0,    0, &
         0,    0,    1,    1,    0,    0,    0,    0,    0, &
         0,    0,    0,    0,    0,    0,    0,    0,    0, &
         0,    0,    0,    0,    0,    0,    0,    0,    0, &
         0,    0,    0,    0,    0,    0,    0,    0,    0, &
         0,    0,    0,    0,    0,    0,    0,    0,    0 /), &
         (/nlon,nlat/)))
      allocate(expected_lake_types(nlat,nlon))
      expected_lake_types = transpose(reshape((/ &
             0,    0,    0,    2,    0,    0,    0,    0,    0, &
         0,    0,    0,    2,    0,    0,    0,    0,    0, &
         0,    0,    2,    2,    0,    0,    0,    0,    0, &
         0,    0,    2,    2,    0,    0,    0,    0,    0, &
         0,    0,    2,    2,    0,    0,    0,    0,    0, &
         0,    0,    0,    0,    0,    0,    0,    0,    0, &
         0,    0,    0,    0,    0,    0,    0,    0,    0, &
         0,    0,    0,    0,    0,    0,    0,    0,    0, &
         0,    0,    0,    0,    0,    0,    0,    0,    0 /), &
         (/nlon,nlat/)))
      allocate(expected_lake_volumes(1))
      expected_lake_volumes(1) = 80.0
      call init_hd_model_for_testing(river_parameters,river_fields,.True., &
                                     lake_parameters, &
                                     initial_water_to_lake_centers, &
                                     initial_spillover_to_rivers)
      call run_hd_model(timesteps,runoffs,drainages)
      lake_prognostics_out => get_lake_prognostics()
      lake_fields_out => get_lake_fields()
      allocate(lake_types(nlat,nlon))
      lake_types(:,:) = 0
      do i = 1,9
        do j = 1,9
          lake_number = lake_fields_out%lake_numbers(i,j)
          if (lake_number > 0) then
            working_lake_ptr = lake_prognostics_out%lakes(lake_number)
            lake_type = working_lake_ptr%lake_pointer%lake_type
            if (lake_type == filling_lake_type) then
              lake_types(i,j) = 1
            else if (lake_type == overflowing_lake_type) then
              lake_types(i,j) = 2
            else if (lake_type == subsumed_lake_type) then
              lake_types(i,j) = 3
            else
              lake_types(i,j) = 4
            end if
          end if
        end do
      end do
      allocate(lake_volumes(1))
      do i = 1,size(lake_volumes)
        working_lake_ptr = lake_prognostics_out%lakes(i)
        lake_volumes(i) = working_lake_ptr%lake_pointer%lake_volume
      end do
      call assert_equals(expected_river_inflow,river_fields%river_inflow,3,3,0.00001)
      call assert_equals(expected_water_to_ocean,river_fields%water_to_ocean,3,3,0.00001)
      call assert_equals(expected_water_to_hd,lake_fields_out%water_to_hd,3,3,0.00001)
      call assert_equals(expected_lake_numbers,lake_fields_out%lake_numbers,9,9)
      call assert_equals(expected_lake_types,lake_types,9,9)
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
  use latlon_hd_model_interface_mod
  use latlon_hd_model_mod
  use latlon_lake_model_io_mod
  type(riverparameters), pointer :: river_parameters
  type(riverprognosticfields), pointer :: river_fields
  type(lakeparameters),pointer :: lake_parameters
  type(lakefields), pointer :: lake_fields_out
  type(lakeprognostics), pointer :: lake_prognostics_out
  type(lakepointer) :: working_lake_ptr
  real   ,dimension(:,:), pointer :: flow_directions
  integer,dimension(:,:), pointer :: river_reservoir_nums
  integer,dimension(:,:), pointer :: overland_reservoir_nums
  integer,dimension(:,:), pointer :: base_reservoir_nums
  real,dimension(:,:), pointer :: river_retention_coefficients
  real,dimension(:,:), pointer :: overland_retention_coefficients
  real,dimension(:,:), pointer :: base_retention_coefficients
  logical,dimension(:,:), pointer :: landsea_mask
  logical,dimension(:,:), pointer :: lake_centers
  real,dimension(:,:), pointer :: connection_volume_thresholds
  real,dimension(:,:), pointer :: flood_volume_thresholds
  logical,dimension(:,:), pointer :: flood_local_redirect
  logical,dimension(:,:), pointer :: connect_local_redirect
  logical,dimension(:,:), pointer :: additional_flood_local_redirect
  logical,dimension(:,:), pointer :: additional_connect_local_redirect
  integer,dimension(:,:), pointer :: merge_points
  integer,dimension(:,:), pointer :: flood_next_cell_lat_index
  integer,dimension(:,:), pointer :: flood_next_cell_lon_index
  integer,dimension(:,:), pointer :: connect_next_cell_lat_index
  integer,dimension(:,:), pointer :: connect_next_cell_lon_index
  integer,dimension(:,:), pointer :: flood_force_merge_lat_index
  integer,dimension(:,:), pointer :: flood_force_merge_lon_index
  integer,dimension(:,:), pointer :: connect_force_merge_lat_index
  integer,dimension(:,:), pointer :: connect_force_merge_lon_index
  integer,dimension(:,:), pointer :: flood_redirect_lat_index
  integer,dimension(:,:), pointer :: flood_redirect_lon_index
  integer,dimension(:,:), pointer :: connect_redirect_lat_index
  integer,dimension(:,:), pointer :: connect_redirect_lon_index
  integer,dimension(:,:), pointer :: additional_flood_redirect_lat_index
  integer,dimension(:,:), pointer :: additional_flood_redirect_lon_index
  integer,dimension(:,:), pointer :: additional_connect_redirect_lat_index
  integer,dimension(:,:), pointer :: additional_connect_redirect_lon_index
  real,dimension(:,:), pointer :: drainage
  real,dimension(:,:,:), pointer :: drainages
  real,dimension(:,:), pointer :: runoff
  real,dimension(:,:,:), pointer :: runoffs
  real, pointer, dimension(:,:) :: initial_spillover_to_rivers
  real, pointer, dimension(:,:) :: initial_water_to_lake_centers
  real,dimension(:,:), pointer :: expected_river_inflow
  real,dimension(:,:), pointer :: expected_water_to_ocean
  real,dimension(:,:), pointer :: expected_water_to_hd
  integer,dimension(:,:), pointer :: expected_lake_numbers
  integer,dimension(:,:), pointer :: expected_lake_types
  real,dimension(:), pointer :: expected_lake_volumes
  integer,dimension(:,:), pointer :: lake_types
  integer :: no_merge_mtype
  integer :: connection_merge_not_set_flood_merge_as_primary
  integer :: connection_merge_not_set_flood_merge_as_secondary
  real,dimension(:), pointer :: lake_volumes
  integer :: nlat,nlon
  integer :: nlat_coarse,nlon_coarse
  integer :: timesteps
  integer :: lake_number
  integer :: i,j
  logical :: instant_throughflow_local
  real :: lake_retention_coefficient_local
  integer :: lake_type
    no_merge_mtype = 0
    connection_merge_not_set_flood_merge_as_primary = 9
    connection_merge_not_set_flood_merge_as_secondary = 10
    timesteps = 10000
    nlat = 20
    nlon = 20
    nlat_coarse = 4
    nlon_coarse = 4
    allocate(flow_directions(nlat_coarse,nlon_coarse))
    flow_directions = transpose(reshape((/ -2, 4, 2, 2, &
                                            6,-2,-2, 2, &
                                            6, 2, 3,-2, &
                                           -2, 0, 6, 0 /), &
                                          (/nlon_coarse, &
                                            nlat_coarse/)))
    allocate(river_reservoir_nums(nlat_coarse,nlon_coarse))
    river_reservoir_nums(:,:) = 5
    allocate(overland_reservoir_nums(nlat_coarse,nlon_coarse))
    overland_reservoir_nums(:,:) = 1
    allocate(base_reservoir_nums(nlat_coarse,nlon_coarse))
    base_reservoir_nums(:,:) = 1
    allocate(river_retention_coefficients(nlat_coarse,nlon_coarse))
    river_retention_coefficients(:,:) = 0.7
    allocate(overland_retention_coefficients(nlat_coarse,nlon_coarse))
    overland_retention_coefficients(:,:) = 0.5
    allocate(base_retention_coefficients(nlat_coarse,nlon_coarse))
    base_retention_coefficients(:,:) = 0.1
    allocate(landsea_mask(nlat_coarse,nlon_coarse))
    landsea_mask(:,:) = .False.
    river_reservoir_nums(4,4) = 0
    overland_reservoir_nums(4,4) = 0
    base_reservoir_nums(4,4) = 0
    landsea_mask(4,4) = .True.
    river_reservoir_nums(4,2) = 0
    overland_reservoir_nums(4,2) = 0
    base_reservoir_nums(4,2) = 0
    landsea_mask(4,2) = .True.
    river_parameters => RiverParameters(flow_directions, &
                                        river_reservoir_nums, &
                                        overland_reservoir_nums, &
                                        base_reservoir_nums, &
                                        river_retention_coefficients, &
                                        overland_retention_coefficients, &
                                        base_retention_coefficients, &
                                        landsea_mask)
    river_fields => riverprognosticfields(4,4,1,1,5)
    allocate(lake_centers(nlat,nlon))
    lake_centers = transpose(reshape((/ &
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
                .False., .False., .False., .False., .False.,   .False., .False., .False., .False. /), &
       (/nlon,nlat/)))
    allocate(connection_volume_thresholds(nlat,nlon))
    connection_volume_thresholds = transpose(reshape((/ &
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
       /), &
       (/nlon,nlat/)))
    allocate(flood_volume_thresholds(nlat,nlon))
    flood_volume_thresholds = transpose(reshape((/ &
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
              -1.0,   -1.0,    -1.0,   -1.0,  -1.0 /), &
       (/nlon,nlat/)))
    allocate(flood_local_redirect(nlat,nlon))
    flood_local_redirect = transpose(reshape((/ &
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
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False. /), &
       (/nlon,nlat/)))
    allocate(connect_local_redirect(nlat,nlon))
    connect_local_redirect = transpose(reshape((/ &
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
                .False., .False., .False., .False., .False.,    .False., .False., .False., .False. /), &
       (/nlon,nlat/)))
    allocate(additional_flood_local_redirect(nlat,nlon))
    additional_flood_local_redirect(:,:) = .False.
    allocate(additional_connect_local_redirect(nlat,nlon))
    additional_connect_local_redirect(:,:) = .False.
    allocate(merge_points(nlat,nlon))
    merge_points = transpose(reshape((/ &
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
       no_merge_mtype,no_merge_mtype,no_merge_mtype,no_merge_mtype /), &
       (/nlon,nlat/)))
    allocate(flood_next_cell_lat_index(nlat,nlon))
    flood_next_cell_lat_index = transpose(reshape((/ &
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  3, &
       2,  3,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1,  2,  2,  4,  5, -1, -1, -1,  1, &
       -1,  4,  2, -1, -1,  8,  2, -1, -1, -1, -1,  2,  5,  3,  9,  2, -1, -1, -1, -1, &
       -1,  3,  4, -1, -1,  6,  4,  6,  3, -1, -1, -1,  7,  3,  4,  3,  6, -1, -1, -1, &
       -1,  6,  5,  3,  6,  7, -1,  4,  4, -1,  5,  7,  6,  6,  4,  8,  4,  3, -1, -1, &
       7,  6,  6, -1,  5,  5,  6,  5,  5, -1,  5,  5,  4,  8,  6,  6,  5,  5, -1, -1, &
       -1,  6,  7, -1, -1,  6, -1,  4, -1, -1, -1,  5,  6,  6,  8,  7,  5, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  8,  7,  7,  8,  6,  7, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  8,  9, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 14, 13, 16, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 14, 13, -1, -1, &
       -1, -1, -1, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 /), &
       (/nlon,nlat/)))
    allocate(flood_next_cell_lon_index(nlat,nlon))
    flood_next_cell_lon_index = transpose(reshape((/ &
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, &
       19,  5,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 16,  3, -1, -1, -1, 19, &
       -1,  2,  2, -1, -1, 18,  1, -1, -1, -1, -1, 13, 15, 12, 13, 14, -1, -1, -1, -1, &
       -1,  2,  1, -1, -1,  8,  5, 10,  6, -1, -1, -1, 12, 13, 13, 14, 17, -1, -1, -1, &
       -1,  2,  1,  5,  7,  5, -1,  6,  8, -1, 14, 14, 10, 12, 14, 15, 15, 11, -1, -1, &
       2,  0,  1, -1,  4,  5,  4,  7,  8, -1, 10, 11, 12, 12, 13, 14, 16, 17, -1, -1, &
       -1,  4,  1, -1, -1,  6, -1,  7, -1, -1, -1, 12, 11, 15, 14, 13,  9, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 16, 11, 15, 13, 16, 16, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, 12, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 16, 18, 15, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 16, 17, 17, -1, -1, &
       -1, -1, -1,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 /), &
       (/nlon,nlat/)))
    allocate(connect_next_cell_lat_index(nlat,nlon))
    connect_next_cell_lat_index = transpose(reshape((/ &
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1,  3,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 /), &
       (/nlon,nlat/)))
    allocate(connect_next_cell_lon_index(nlat,nlon))
    connect_next_cell_lon_index = transpose(reshape((/ &
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1,  6,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 /), &
       (/nlon,nlat/)))
    allocate(flood_force_merge_lat_index(nlat,nlon))
    flood_force_merge_lat_index = transpose(reshape((/ &
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  6, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  6, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  7, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 /), &
       (/nlon,nlat/)))
    allocate(flood_force_merge_lon_index(nlat,nlon))
    flood_force_merge_lon_index = transpose(reshape((/ &
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  2, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  8, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 /), &
       (/nlon,nlat/)))
    allocate(connect_force_merge_lat_index(nlat,nlon))
    connect_force_merge_lat_index = transpose(reshape((/ &
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 /), &
       (/nlon,nlat/)))
    allocate(connect_force_merge_lon_index(nlat,nlon))
    connect_force_merge_lon_index = transpose(reshape((/ &
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 /), &
       (/nlon,nlat/)))
    allocate(flood_redirect_lat_index(nlat,nlon))
    flood_redirect_lat_index = transpose(reshape((/ &
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  7, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  6, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  5, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  3, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 /), &
       (/nlon,nlat/)))
    allocate(flood_redirect_lon_index(nlat,nlon))
    flood_redirect_lon_index = transpose(reshape((/ &
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  5, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  3, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 /), &
       (/nlon,nlat/)))
    allocate(connect_redirect_lat_index(nlat,nlon))
    connect_redirect_lat_index = transpose(reshape((/ &
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 /), &
       (/nlon,nlat/)))
    allocate(connect_redirect_lon_index(nlat,nlon))
    connect_redirect_lon_index = transpose(reshape((/ &
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, &
       -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 /), &
       (/nlon,nlat/)))
    call add_offset(flood_next_cell_lat_index,1,(/-1/))
    call add_offset(flood_next_cell_lon_index,1,(/-1/))
    call add_offset(connect_next_cell_lat_index,1,(/-1/))
    call add_offset(connect_next_cell_lon_index,1,(/-1/))
    call add_offset(flood_force_merge_lat_index,1,(/-1/))
    call add_offset(flood_force_merge_lon_index,1,(/-1/))
    call add_offset(connect_force_merge_lat_index,1,(/-1/))
    call add_offset(connect_force_merge_lon_index,1,(/-1/))
    call add_offset(flood_redirect_lat_index,1,(/-1/))
    call add_offset(flood_redirect_lon_index,1,(/-1/))
    call add_offset(connect_redirect_lat_index,1,(/-1/))
    call add_offset(connect_redirect_lon_index,1,(/-1/))
    allocate(additional_flood_redirect_lat_index(nlat,nlon))
    additional_flood_redirect_lat_index(:,:) = 0
    allocate(additional_flood_redirect_lon_index(nlat,nlon))
    additional_flood_redirect_lon_index(:,:) = 0
    allocate(additional_connect_redirect_lat_index(nlat,nlon))
    additional_connect_redirect_lat_index(:,:) = 0
    allocate(additional_connect_redirect_lon_index(nlat,nlon))
    additional_connect_redirect_lon_index(:,:) = 0
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
                                    flood_next_cell_lat_index, &
                                    flood_next_cell_lon_index, &
                                    connect_next_cell_lat_index, &
                                    connect_next_cell_lon_index, &
                                    flood_force_merge_lat_index, &
                                    flood_force_merge_lon_index, &
                                    connect_force_merge_lat_index, &
                                    connect_force_merge_lon_index, &
                                    flood_redirect_lat_index, &
                                    flood_redirect_lon_index, &
                                    connect_redirect_lat_index, &
                                    connect_redirect_lon_index, &
                                    additional_flood_redirect_lat_index, &
                                    additional_flood_redirect_lon_index, &
                                    additional_connect_redirect_lat_index, &
                                    additional_connect_redirect_lon_index, &
                                    nlat,nlon, &
                                    nlat_coarse,nlon_coarse, &
                                    instant_throughflow_local, &
                                    lake_retention_coefficient_local)
    allocate(drainage(nlat_coarse,nlon_coarse))
    drainage = transpose(reshape((/ &
       1.0, 1.0, 1.0, 1.0, &
       1.0, 1.0, 1.0, 1.0, &
       1.0, 1.0, 1.0, 1.0, &
       1.0, 0.0, 1.0, 0.0 /), &
       (/nlon_coarse,nlat_coarse/)))
    allocate(runoff(nlat_coarse,nlon_coarse))
    runoff = transpose(reshape((/ &
       1.0, 1.0, 1.0, 1.0, &
       1.0, 1.0, 1.0, 1.0, &
       1.0, 1.0, 1.0, 1.0, &
       1.0, 0.0, 1.0, 0.0  /), &
       (/nlon_coarse,nlat_coarse/)))
    allocate(drainages(nlat_coarse,nlon_coarse,timesteps))
    allocate(runoffs(nlat_coarse,nlon_coarse,timesteps))
    do i = 1,timesteps
      drainages(:,:,i) = drainage(:,:)
      runoffs(:,:,i) = runoff(:,:)
    end do
    allocate(initial_spillover_to_rivers(nlat_coarse,nlon_coarse))
    initial_spillover_to_rivers(:,:) = 0.0
    allocate(initial_water_to_lake_centers(nlat,nlon))
    initial_water_to_lake_centers(:,:) = 0.0
    allocate(expected_river_inflow(nlat_coarse,nlon_coarse))
    expected_river_inflow = transpose(reshape((/ &
        0.0, 0.0, 0.0, 0.0, &
       0.0, 0.0, 0.0, 2.0, &
       0.0, 2.0, 0.0, 0.0, &
       0.0, 0.0, 0.0, 0.0 /), &
       (/nlon_coarse,nlat_coarse/)))
    allocate(expected_water_to_ocean(nlat_coarse,nlon_coarse))
    expected_water_to_ocean = transpose(reshape((/ &
        0.0, 0.0, 0.0, 0.0, &
       0.0, 0.0, 0.0, 0.0, &
       0.0, 0.0, 0.0, 0.0, &
       0.0, 6.0, 0.0, 22.0 /), &
       (/nlon_coarse,nlat_coarse/)))
    allocate(expected_water_to_hd(nlat_coarse,nlon_coarse))
    expected_water_to_hd = transpose(reshape((/ &
        0.0, 0.0, 0.0,  0.0, &
       0.0, 0.0, 0.0, 12.0, &
       0.0, 0.0, 0.0,  0.0, &
       0.0, 2.0, 0.0, 18.0 /), &
       (/nlon_coarse,nlat_coarse/)))
    allocate(expected_lake_numbers(nlat,nlon))
    expected_lake_numbers = transpose(reshape((/ &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, &
       1, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 1, &
       0, 1, 1, 0, 0, 5, 5, 0, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0, 0, &
       0, 1, 1, 0, 0, 3, 3, 3, 3, 0, 0, 0, 4, 5, 5, 5, 5, 0, 0, 0, &
       0, 1, 1, 5, 3, 3, 0, 3, 3, 5, 5, 4, 5, 4, 5, 5, 5, 5, 0, 0, &
       1, 1, 1, 0, 3, 3, 3, 3, 3, 0, 5, 4, 4, 5, 5, 5, 5, 5, 0, 0, &
       0, 1, 1, 0, 0, 3, 0, 3, 0, 0, 0, 5, 4, 5, 5, 5, 5, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 0, 0, &
       0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 /), &
       (/nlon,nlat/)))
    allocate(expected_lake_types(nlat,nlon))
    expected_lake_types = transpose(reshape((/ &
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
   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0 /), &
       (/nlon,nlat/)))
    allocate(expected_lake_volumes(6))
    expected_lake_volumes = (/46.0, 1.0, 38.0, 6.0, 340.0, 10.0/)
    call init_hd_model_for_testing(river_parameters,river_fields,.True., &
                                   lake_parameters, &
                                   initial_water_to_lake_centers, &
                                   initial_spillover_to_rivers)
    call run_hd_model(timesteps,runoffs,drainages)
    lake_prognostics_out => get_lake_prognostics()
    lake_fields_out => get_lake_fields()
    allocate(lake_types(nlat,nlon))
    lake_types(:,:) = 0
    do i = 1,20
      do j = 1,20
        lake_number = lake_fields_out%lake_numbers(i,j)
        if (lake_number > 0) then
          working_lake_ptr = lake_prognostics_out%lakes(lake_number)
          lake_type = working_lake_ptr%lake_pointer%lake_type
            if (lake_type == filling_lake_type) then
              lake_types(i,j) = 1
            else if (lake_type == overflowing_lake_type) then
              lake_types(i,j) = 2
            else if (lake_type == subsumed_lake_type) then
              lake_types(i,j) = 3
            else
              lake_types(i,j) = 4
            end if
        end if
      end do
    end do
    allocate(lake_volumes(6))
    do i = 1,size(lake_volumes)
      working_lake_ptr = lake_prognostics_out%lakes(i)
      lake_volumes(i) = working_lake_ptr%lake_pointer%lake_volume
    end do
    call assert_equals(expected_river_inflow,river_fields%river_inflow,&
                       nlat_coarse,nlon_coarse)
    call assert_equals(expected_water_to_ocean,river_fields%water_to_ocean,4,4,0.00001)
    call assert_equals(expected_water_to_hd,lake_fields_out%water_to_hd,4,4,0.00001)
    call assert_equals(expected_lake_numbers,lake_fields_out%lake_numbers,20,20)
    call assert_equals(expected_lake_types,lake_types,20,20)
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

end module latlon_hd_and_lake_model_test_mod
