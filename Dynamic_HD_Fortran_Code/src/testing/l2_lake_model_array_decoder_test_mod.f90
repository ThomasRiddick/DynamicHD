module l2_lake_model_array_decoder_test_mod

use fruit
implicit none

contains

subroutine testArrayDecoderOneLake
  use l2_lake_model_mod
  use l2_lake_model_array_decoder_mod
  type(lakeparameterspointer), dimension(:), pointer :: lake_parameters
  real(dp), dimension(:), pointer :: lake_parameters_as_array
  integer, pointer, dimension(:) :: expected_outflow_keys
    allocate(expected_outflow_keys(1))
    expected_outflow_keys = (/-1/)
    !One lake
    allocate(lake_parameters_as_array(68))
    lake_parameters_as_array = &
      (/ 1.0,66.0,1.0,-1.0,0.0,4.0,3.0,11.0,4.0,3.0,1.0,0.0,5.0,4.0,4.0, &
         1.0,0.0,5.0,3.0,4.0,1.0,0.0,5.0,3.0,3.0,1.0,0.0,5.0,2.0,5.0,1.0, &
         5.0,6.0,4.0,5.0,1.0,5.0,6.0,3.0,5.0,1.0,12.0,7.0,3.0,2.0,1.0,12.0, &
         7.0,4.0,2.0,1.0,21.0,8.0,2.0,3.0,1.0,21.0,8.0,2.0,4.0,1.0,43.0, &
         10.0,1.0,-1.0,2.0,2.0,0.0 /)
    lake_parameters => &
        get_lake_parameters_from_array(lake_parameters_as_array, &
                                       6,6,3,3)
    call assert_equals(size(lake_parameters),1)
    call assert_equals(lake_parameters(1)%&
                      &lake_parameters_pointer%center_coords_lat,4)
    call assert_equals(lake_parameters(1)%&
                      &lake_parameters_pointer%center_coords_lon,3)
    call assert_equals(lake_parameters(1)%&
                      &lake_parameters_pointer%center_cell_coarse_coords_lat,2)
    call assert_equals(lake_parameters(1)%&
                      &lake_parameters_pointer%center_cell_coarse_coords_lon,2)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%lake_number,1)
    call assert_true(lake_parameters(1)%lake_parameters_pointer%is_primary)
    call assert_true(lake_parameters(1)%lake_parameters_pointer%is_leaf)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%primary_lake,-1)
    call assert_equals(size(lake_parameters(1)%lake_parameters_pointer%secondary_lakes),0)

    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%coords_lat,4)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%coords_lon,3)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%fill_threshold,0.0_dp)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%height,5.0_dp)

    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%coords_lat,4)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%coords_lon,4)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%fill_threshold,0.0_dp)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%height,5.0_dp)

    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(3)%&
                       &cell_pointer%coords_lat,3)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(3)%&
                       &cell_pointer%coords_lon,4)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(3)%&
                       &cell_pointer%fill_threshold,0.0_dp)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(3)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(3)%&
                       &cell_pointer%height,5.0_dp)

    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(4)%&
                       &cell_pointer%coords_lat,3)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(4)%&
                       &cell_pointer%coords_lon,3)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(4)%&
                       &cell_pointer%fill_threshold,0.0_dp)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(4)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(4)%&
                       &cell_pointer%height,5.0_dp)

    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(5)%&
                       &cell_pointer%coords_lat,2)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(5)%&
                       &cell_pointer%coords_lon,5)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(5)%&
                       &cell_pointer%fill_threshold,5.0_dp)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(5)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(5)%&
                       &cell_pointer%height,6.0_dp)

    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(6)%&
                       &cell_pointer%coords_lat,4)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(6)%&
                       &cell_pointer%coords_lon,5)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(6)%&
                       &cell_pointer%fill_threshold,5.0_dp)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(6)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(6)%&
                       &cell_pointer%height,6.0_dp)

    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(7)%&
                       &cell_pointer%coords_lat,3)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(7)%&
                       &cell_pointer%coords_lon,5)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(7)%&
                       &cell_pointer%fill_threshold,12.0_dp)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(7)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(7)%&
                       &cell_pointer%height,7.0_dp)

    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(8)%&
                       &cell_pointer%coords_lat,3)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(8)%&
                       &cell_pointer%coords_lon,2)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(8)%&
                       &cell_pointer%fill_threshold,12.0_dp)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(8)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(8)%&
                       &cell_pointer%height,7.0_dp)

    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(9)%&
                       &cell_pointer%coords_lat,4)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(9)%&
                       &cell_pointer%coords_lon,2)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(9)%&
                       &cell_pointer%fill_threshold,21.0_dp)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(9)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(9)%&
                       &cell_pointer%height,8.0_dp)

    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(10)%&
                       &cell_pointer%coords_lat,2)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(10)%&
                       &cell_pointer%coords_lon,3)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(10)%&
                       &cell_pointer%fill_threshold,21.0_dp)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(10)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(10)%&
                       &cell_pointer%height,8.0_dp)

    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(11)%&
                       &cell_pointer%coords_lat,2)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(11)%&
                       &cell_pointer%coords_lon,4)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(11)%&
                       &cell_pointer%fill_threshold,43.0_dp)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(11)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(11)%&
                       &cell_pointer%height,10.0_dp)

    call assert_equals(lake_parameters(1)%lake_parameters_pointer%outflow_points%keys,&
                       expected_outflow_keys,1)
    call assert_false(lake_parameters(1)%lake_parameters_pointer%outflow_points%values(1)%&
                       &redirect_pointer%use_local_redirect)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%outflow_points%values(1)%&
                       &redirect_pointer%local_redirect_target_lake_number,-1)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%outflow_points%values(1)%&
                       &redirect_pointer%non_local_redirect_target_lat,2)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%outflow_points%values(1)%&
                       &redirect_pointer%non_local_redirect_target_lon,2)
    deallocate(expected_outflow_keys)
    deallocate(lake_parameters_as_array)
    call clean_lake_parameters(lake_parameters(1)%lake_parameters_pointer)
    deallocate(lake_parameters(1)%lake_parameters_pointer)
    deallocate(lake_parameters)
end subroutine testArrayDecoderOneLake

subroutine testArrayDecoderTwoLakes
  use l2_lake_model_mod
  use l2_lake_model_array_decoder_mod
  type(lakeparameterspointer), dimension(:), pointer :: lake_parameters
  real(dp), dimension(:), pointer :: lake_parameters_as_array
  integer, pointer, dimension(:) :: expected_outflow_keys_lake_one
  integer, pointer, dimension(:) :: expected_outflow_keys_lake_two
  integer, pointer, dimension(:) :: expected_outflow_keys_lake_three
  integer, pointer, dimension(:) :: expected_secondary_lakes
  integer :: i
    allocate(expected_outflow_keys_lake_one(1))
    expected_outflow_keys_lake_one = (/ 2 /)
    allocate(expected_outflow_keys_lake_two(1))
    expected_outflow_keys_lake_two = (/ 1 /)
    allocate(expected_outflow_keys_lake_three(1))
    expected_outflow_keys_lake_three = (/ -1 /)
    allocate(expected_secondary_lakes(2))
    expected_secondary_lakes = (/ 1, 2 /)
    !Two lakes that join
    allocate(lake_parameters_as_array(94))
    lake_parameters_as_array = &
      (/ 3.0, 21.0, 1.0, 3.0, 0.0, 4.0, 5.0, 2.0, 4.0, 5.0, 1.0, 0.0, 5.0, 3.0, &
         5.0, 1.0, 6.0, 8.0, 1.0, 2.0, 2.0, 2.0, 0.0, 26.0, 2.0, 3.0, 0.0, 4.0, &
         2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 5.0, 3.0, 2.0, 1.0, 2.0, 6.0, 4.0, 3.0, &
         1.0, 8.0, 8.0, 1.0, 1.0, -1.0, -1.0, 1.0, 43.0, 3.0, -1.0, 2.0, 1.0, &
         2.0, 4.0, 5.0, 6.0, 4.0, 5.0, 1.0, 0.0, 8.0, 3.0, 5.0, 1.0, 0.0, 8.0, &
         4.0, 4.0, 1.0, 0.0, 8.0, 4.0, 3.0, 1.0, 0.0, 8.0, 4.0, 2.0, 1.0, 0.0, &
         8.0, 3.0, 2.0, 1.0, 12.0, 10.0, 1.0, -1.0, 3.0, 3.0, 0.0 /)
    lake_parameters => &
        get_lake_parameters_from_array(lake_parameters_as_array, &
                                       6,6,3,3)
    call assert_equals(size(lake_parameters),3)
    call assert_equals(lake_parameters(1)%&
                      &lake_parameters_pointer%center_coords_lat,4)
    call assert_equals(lake_parameters(1)%&
                      &lake_parameters_pointer%center_coords_lon,5)
    call assert_equals(lake_parameters(1)%&
                      &lake_parameters_pointer%center_cell_coarse_coords_lat,2)
    call assert_equals(lake_parameters(1)%&
                      &lake_parameters_pointer%center_cell_coarse_coords_lon,3)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%lake_number,1)
    call assert_false(lake_parameters(1)%lake_parameters_pointer%is_primary)
    call assert_true(lake_parameters(1)%lake_parameters_pointer%is_leaf)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%primary_lake,3)
    call assert_equals(size(lake_parameters(1)%lake_parameters_pointer%secondary_lakes),0)

    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%coords_lat,4)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%coords_lon, 5)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%fill_threshold,0.0_dp)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%height,5.0_dp)

    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%coords_lat,3)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%coords_lon,5)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%fill_threshold,6.0_dp)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%height,8.0_dp)

    call assert_equals(lake_parameters(1)%lake_parameters_pointer%outflow_points%keys,&
                       expected_outflow_keys_lake_one,1)
    call assert_false(lake_parameters(1)%lake_parameters_pointer%outflow_points%values(1)%&
                       &redirect_pointer%use_local_redirect)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%outflow_points%values(1)%&
                       &redirect_pointer%local_redirect_target_lake_number,2)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%outflow_points%values(1)%&
                       &redirect_pointer%non_local_redirect_target_lat,2)
    call assert_equals(lake_parameters(1)%lake_parameters_pointer%outflow_points%values(1)%&
                       &redirect_pointer%non_local_redirect_target_lon,2)

    call assert_equals(lake_parameters(2)%&
                      &lake_parameters_pointer%center_coords_lat,4)
    call assert_equals(lake_parameters(2)%&
                      &lake_parameters_pointer%center_coords_lon,2)
    call assert_equals(lake_parameters(2)%&
                      &lake_parameters_pointer%center_cell_coarse_coords_lat,2)
    call assert_equals(lake_parameters(2)%&
                      &lake_parameters_pointer%center_cell_coarse_coords_lon,1)
    call assert_equals(lake_parameters(2)%lake_parameters_pointer%lake_number,2)
    call assert_false(lake_parameters(2)%lake_parameters_pointer%is_primary)
    call assert_true(lake_parameters(2)%lake_parameters_pointer%is_leaf)
    call assert_equals(lake_parameters(2)%lake_parameters_pointer%primary_lake,3)
    call assert_equals(size(lake_parameters(2)%lake_parameters_pointer%secondary_lakes),0)

    call assert_equals(lake_parameters(2)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%coords_lat,4)
    call assert_equals(lake_parameters(2)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%coords_lon,2)
    call assert_equals(lake_parameters(2)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%fill_threshold,0.0_dp)
    call assert_equals(lake_parameters(2)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(2)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%height,5.0_dp)

    call assert_equals(lake_parameters(2)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%coords_lat,3)
    call assert_equals(lake_parameters(2)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%coords_lon,2)
    call assert_equals(lake_parameters(2)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%fill_threshold,2.0_dp)
    call assert_equals(lake_parameters(2)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(2)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%height,6.0_dp)

    call assert_equals(lake_parameters(2)%lake_parameters_pointer%filling_order(3)%&
                       &cell_pointer%coords_lat,4)
    call assert_equals(lake_parameters(2)%lake_parameters_pointer%filling_order(3)%&
                       &cell_pointer%coords_lon,3)
    call assert_equals(lake_parameters(2)%lake_parameters_pointer%filling_order(3)%&
                       &cell_pointer%fill_threshold,8.0_dp)
    call assert_equals(lake_parameters(2)%lake_parameters_pointer%filling_order(3)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(2)%lake_parameters_pointer%filling_order(3)%&
                       &cell_pointer%height,8.0_dp)

    call assert_equals(lake_parameters(2)%lake_parameters_pointer%outflow_points%keys,&
                       expected_outflow_keys_lake_two,1)
    call assert_true(lake_parameters(2)%lake_parameters_pointer%outflow_points%values(1)%&
                      &redirect_pointer%use_local_redirect)
    call assert_equals(lake_parameters(2)%lake_parameters_pointer%outflow_points%values(1)%&
                       &redirect_pointer%local_redirect_target_lake_number,1)
    call assert_equals(lake_parameters(2)%lake_parameters_pointer%outflow_points%values(1)%&
                       &redirect_pointer%non_local_redirect_target_lat,-1)
    call assert_equals(lake_parameters(2)%lake_parameters_pointer%outflow_points%values(1)%&
                       &redirect_pointer%non_local_redirect_target_lon,-1)

    call assert_equals(lake_parameters(3)%&
                      &lake_parameters_pointer%center_coords_lat,4)
    call assert_equals(lake_parameters(3)%&
                      &lake_parameters_pointer%center_coords_lon,5)
    call assert_equals(lake_parameters(3)%&
                      &lake_parameters_pointer%center_cell_coarse_coords_lat,2)
    call assert_equals(lake_parameters(3)%&
                      &lake_parameters_pointer%center_cell_coarse_coords_lon,3)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%lake_number,3)
    call assert_true(lake_parameters(3)%lake_parameters_pointer%is_primary)
    call assert_false(lake_parameters(3)%lake_parameters_pointer%is_leaf)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%primary_lake,-1)
    call assert_equals(size(lake_parameters(3)%lake_parameters_pointer%secondary_lakes),2)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%secondary_lakes,&
                       expected_secondary_lakes,2)


    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%coords_lat,4)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%coords_lon,5)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%fill_threshold,0.0_dp)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(1)%&
                       &cell_pointer%height,8.0_dp)

    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%coords_lat,3)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%coords_lon,5)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%fill_threshold,0.0_dp)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(2)%&
                       &cell_pointer%height,8.0_dp)

    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(3)%&
                       &cell_pointer%coords_lat,4)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(3)%&
                       &cell_pointer%coords_lon,4)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(3)%&
                       &cell_pointer%fill_threshold,0.0_dp)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(3)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(3)%&
                       &cell_pointer%height,8.0_dp)

    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(4)%&
                       &cell_pointer%coords_lat,4)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(4)%&
                       &cell_pointer%coords_lon,3)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(4)%&
                       &cell_pointer%fill_threshold,0.0_dp)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(4)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(4)%&
                       &cell_pointer%height,8.0_dp)

    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(5)%&
                       &cell_pointer%coords_lat,4)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(5)%&
                       &cell_pointer%coords_lon,2)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(5)%&
                       &cell_pointer%fill_threshold,0.0_dp)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(5)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(5)%&
                       &cell_pointer%height,8.0_dp)

    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(6)%&
                       &cell_pointer%coords_lat,3)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(6)%&
                       &cell_pointer%coords_lon,2)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(6)%&
                       &cell_pointer%fill_threshold,12.0_dp)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(6)%&
                       &cell_pointer%height_type,flood_height)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%filling_order(6)%&
                       &cell_pointer%height,10.0_dp)

    call assert_equals(lake_parameters(3)%lake_parameters_pointer%outflow_points%keys,&
                       expected_outflow_keys_lake_three,1)
    call assert_false(lake_parameters(3)%lake_parameters_pointer%outflow_points%values(1)%&
                      &redirect_pointer%use_local_redirect)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%outflow_points%values(1)%&
                       &redirect_pointer%local_redirect_target_lake_number,-1)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%outflow_points%values(1)%&
                       &redirect_pointer%non_local_redirect_target_lat,3)
    call assert_equals(lake_parameters(3)%lake_parameters_pointer%outflow_points%values(1)%&
                       &redirect_pointer%non_local_redirect_target_lon,3)
    deallocate(expected_outflow_keys_lake_one)
    deallocate(expected_outflow_keys_lake_two)
    deallocate(expected_outflow_keys_lake_three)
    deallocate(expected_secondary_lakes)
    deallocate(lake_parameters_as_array)
    do i = 1,size(lake_parameters)
      call clean_lake_parameters(lake_parameters(i)%lake_parameters_pointer)
      deallocate(lake_parameters(i)%lake_parameters_pointer)
    end do
    deallocate(lake_parameters)
end subroutine testArrayDecoderTwoLakes

end module l2_lake_model_array_decoder_test_mod
