module flow_test_mod
use fruit
implicit none

contains

    subroutine setup
    end subroutine setup

    subroutine teardown
    end subroutine teardown

    subroutine testSmallGrid
    use flow
    use cotat_parameters_mod
    integer, dimension(:,:), allocatable :: input_fine_river_directions
    integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
    integer, dimension(:,:), allocatable :: coarse_river_directions_lat_index
    integer, dimension(:,:), allocatable :: coarse_river_directions_lon_index
    integer, dimension(:,:), allocatable :: coarse_river_direction_lat_index_expected_result
    integer, dimension(:,:), allocatable :: coarse_river_direction_lon_index_expected_result
        allocate(input_fine_river_directions(20,20))
        allocate(input_fine_total_cumulative_flow(20,20))
        allocate(coarse_river_directions_lat_index(5,5))
        allocate(coarse_river_directions_lon_index(5,5))
        allocate(coarse_river_direction_lat_index_expected_result(5,5))
        allocate(coarse_river_direction_lon_index_expected_result(5,5))
        MUFP = 1.5
        run_check_for_sinks = .True.
        yamazaki_max_range = 9
        yamazaki_wrap = .True.
        input_fine_river_directions = transpose(reshape((/ &
!
        3,3,2,2, 1,1,4,2, 8,8,8,4, 2,2,2,2, 2,2,2,2, &
        6,3,3,2, 4,4,4,2, 6,6,2,2, 2,2,2,2, 2,2,2,2, &
        6,6,3,2, 6,6,6,6, 6,3,2,4, 2,1,4,4, 2,1,4,4, &
        6,6,3,3, 9,9,9,4, 6,6,2,4, 1,1,4,4, 1,4,4,4, &
!
        9,6,6,3, 6,6,6,6, 3,8,2,2, 3,2,2,2, 3,2,2,4, &
        9,9,9,8, 6,6,6,6, 2,2,2,2, 6,3,2,2, 6,2,2,1, &
        9,9,9,8, 8,2,8,2, 2,2,2,2, 6,6,6,6, 6,6,6,6, &
        9,6,6,3, 8,2,8,2, 2,2,2,2, 6,6,6,3, 2,4,9,9, &
!
        3,2,1,2, 4,4,4,1, 4,2,2,1, 3,6,6,2, 3,6,6,6, &
        6,3,2,2, 4,4,1,1, 3,2,3,2, 3,3,6,2, 6,3,3,6, &
        6,6,2,1, 4,4,2,4, 3,2,4,2, 6,3,3,2, 6,6,3,3, &
        6,6,2,4, 4,4,2,4, 3,2,4,3, 6,6,3,6, 6,6,6,6, &
!
        2,1,2,1, 2,2,2,2, 2,2,2,6, 6,6,6,6, 3,6,6,2, &
        2,1,3,2, 2,2,3,2, 2,2,2,2, 6,6,6,6, 6,3,6,2, &
        2,4,4,2, 2,2,6,3, 2,3,2,2, 9,9,9,9, 6,6,3,2, &
        2,4,4,2, 6,6,6,6, 1,6,6,6, 6,2,2,2, 6,6,6,6, &
!
        2,3,3,2, 2,2,1,1, 3,2,6,6, 6,6,3,2, 3,3,6,2, &
        4,3,3,2, 1,1,1,2, 6,6,6,8, 2,2,2,3, 3,3,3,2, &
        6,6,6,2, 2,2,2,2, 6,6,6,8, 6,6,6,6, 6,6,6,6, &
        6,6,6,2, 2,2,2,2, 6,6,6,8, 9,8,8,8, 9,9,9,9 /), &
        ((/20,20/))))

        input_fine_total_cumulative_flow = transpose(reshape((/ &
        1, 1, 1, 1,  1, 2, 1, 1,  1, 1, 2, 1,   1, 1, 1, 1,  1, 1, 1, 1, &
        1, 3, 3, 8,  5, 2, 1, 2,  1, 2, 3, 1,   2, 2, 2, 2,  2, 2, 2, 2, &
        1, 2, 6,12,  1, 3, 5,10, 11,12, 6, 2,   3, 9, 6, 3,  3, 9, 6, 3, &
        1, 3, 4,19,  1, 1, 2, 1,  1, 3,23, 1,  13, 3, 2, 1, 16, 3, 2, 1, &
!
        1, 2,50,61, 20,21,22,23, 24, 1,24,14,   4, 1, 1,17,  1, 1, 2, 1, &
        1,47, 3, 3, 64,65,68,69, 70,25,25,15,   1, 7, 2,18,  1, 4, 3, 1, &
       46, 2, 1, 1,  2, 1, 2, 1, 71,26,26,16,   1, 2,12,31, 32,37,42,44, &
        1, 1, 2, 3,  1, 2, 1, 2, 72,27,27,17,   1, 2, 3, 4,  2, 1, 1, 1, &
!
        4, 1, 1, 9,  8, 4, 1,76, 73,28,28,18,   1, 1, 2, 3,  7, 1, 2, 3, &
        2, 9, 1,12,  2, 1,77, 1,  1,29,47, 1,   1, 2, 1, 5,  1, 9, 1, 1, &
        1, 2,13,92, 79,78, 3, 1,  1,32, 1,49,   1, 3, 3, 6,  1, 2,12, 2, &
       29,30,139,3,  2, 1, 5, 1,  1,35, 1,50,   1, 2, 6,10, 11,12,13,26, &
!
        1, 1,140,1,  1, 1, 6, 1,  1,37, 1, 1,  52,53,54,61, 62, 1, 2, 3, &
        3, 1,142,1,  2, 2, 7, 2,  2,38, 2, 1,   1, 3, 5, 7,  9,72, 1, 5, &
        7, 2,1,144,  3, 3, 1,11,  3,39, 3, 2,   1, 1, 1, 1,  1, 2,75, 6, &
       95, 2,1,145,  4, 8, 9,10, 25, 1,44,47,  48,49, 1, 1,  1, 2, 3,85, &
!
       96, 1,1,146,  1, 1, 1,26,  1, 1, 1,16,  17,67,69, 2,  1, 1, 1, 2, &
       97, 1,2,148,  2, 3,27, 1,  1, 4, 5,14,   1, 1, 1,72,  1, 2, 2,100, &
      197,198,200,353, 4,28,1,2,  1, 2, 3, 8,   2, 6, 9,11, 84,87,91,195, &
        1, 2,3,357,  5,29, 2, 3,  1, 2, 3, 4,   1, 1, 1, 1,  1, 1, 1,  1 /), &
        ((/20,20/))))

        coarse_river_direction_lat_index_expected_result = transpose(reshape((/ &
                                                                        4,1,3,3,2, &
                                                                        2,2,3,2,2, &
                                                                        4,3,4,3,3, &
                                                                        5,5,5,4,5, &
                                                                        6,6,5,5,5 /), &
                                                                        (/5,5/)))
        coarse_river_direction_lon_index_expected_result = transpose(reshape((/ &
                                                                        3,3,3,3,4, &
                                                                        2,3,2,5,1, &
                                                                        1,1,4,5,1, &
                                                                        1,2,4,5,5, &
                                                                        1,2,4,5,1 /), &
                                                                        (/5,5/)))
        call flow_latlon(input_fine_river_directions,input_fine_total_cumulative_flow, &
                         coarse_river_directions_lat_index,coarse_river_directions_lon_index)
        call assert_equals(coarse_river_directions_lon_index, &
                           coarse_river_direction_lon_index_expected_result,5,5)
        call assert_equals(coarse_river_directions_lat_index, &
                           coarse_river_direction_lat_index_expected_result,5,5)
    end subroutine testSmallGrid

    subroutine testSmallGridTwo
    use flow
    use cotat_parameters_mod
    integer, dimension(:,:), allocatable :: input_fine_river_directions
    integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
    integer, dimension(:,:), allocatable :: coarse_river_directions_lat_index
    integer, dimension(:,:), allocatable :: coarse_river_directions_lon_index
    integer, dimension(:,:), allocatable :: coarse_river_direction_lat_index_expected_result
    integer, dimension(:,:), allocatable :: coarse_river_direction_lon_index_expected_result
        allocate(input_fine_river_directions(20,20))
        allocate(input_fine_total_cumulative_flow(20,20))
        allocate(coarse_river_directions_lat_index(5,5))
        allocate(coarse_river_directions_lon_index(5,5))
        allocate(coarse_river_direction_lat_index_expected_result(5,5))
        allocate(coarse_river_direction_lon_index_expected_result(5,5))
        MUFP = 1.5
        run_check_for_sinks = .True.
        yamazaki_max_range = 9
        yamazaki_wrap = .True.
        input_fine_river_directions = transpose(reshape((/ &
!
        3,3,2,2, 6,6,6,6, 6,6,6,3, 3,2,2,2, 2,2,2,2, &
        6,3,3,2, 8,8,8,8, 8,8,8,2, 2,2,2,2, 2,2,2,2, &
        6,6,3,2, 6,6,6,6, 6,3,2,4, 2,6,2,4, 2,1,4,4, &
        6,6,3,3, 9,9,9,4, 6,6,2,4, 1,6,6,2, 4,4,4,4, &
!
        9,6,6,3, 6,6,6,6, 3,8,2,2, 3,2,2,2, 3,2,2,4, &
        9,9,9,8, 6,6,6,6, 2,2,2,2, 6,3,2,2, 6,2,2,1, &
        9,9,9,8, 8,2,8,2, 2,2,2,2, 6,6,6,6, 6,6,6,6, &
        9,6,6,3, 8,2,8,2, 2,2,2,2, 6,6,6,3, 2,4,9,9, &
!
        3,2,1,2, 4,4,4,1, 4,2,2,1, 3,6,6,2, 3,6,6,6, &
        6,3,2,2, 4,4,1,1, 3,2,3,2, 3,3,6,2, 6,3,3,6, &
        6,6,2,1, 4,4,2,4, 3,2,4,2, 6,3,3,2, 6,6,3,3, &
        6,6,2,4, 4,4,2,4, 3,2,4,3, 6,6,3,6, 6,6,6,6, &
!
        2,1,2,1, 2,2,2,2, 2,2,2,6, 6,6,6,6, 3,6,6,2, &
        2,1,3,2, 2,2,3,2, 2,2,2,2, 6,6,6,6, 6,3,6,2, &
        2,4,4,2, 2,2,6,3, 2,3,2,2, 9,9,9,9, 6,6,3,2, &
        2,4,4,2, 6,6,6,6, 1,6,6,6, 6,2,2,2, 6,6,6,6, &
!
        2,3,3,2, 2,2,1,1, 3,2,6,6, 6,6,3,2, 3,3,6,2, &
        4,3,3,2, 1,1,1,2, 6,6,6,8, 2,2,2,3, 3,3,3,2, &
        6,6,6,2, 2,2,2,2, 6,6,6,8, 6,6,6,6, 6,6,6,6, &
        6,6,6,2, 2,2,2,2, 6,6,6,8, 9,8,8,8, 9,9,9,9 /), &
        ((/20,20/))))

        input_fine_total_cumulative_flow = transpose(reshape((/ &
      1,  1,  1,  1, 2, 4, 6, 8,10,12,14,15, 1, 1, 1, 1, 1, 1, 1,  1, &
      1,  3,  3,  2, 1, 1, 1, 1, 1, 1, 1, 1,16, 3, 2, 2, 2, 2, 2,  2, &
      1,  2,  6,  6, 1, 3, 5, 8, 9,10, 3, 2,17, 4,10, 3, 3, 9, 6,  3, &
      1,  3,  4, 13, 1, 1, 2, 1, 1, 3,18, 1,18, 1,12,29,16, 3, 2,  1, &
      1,  2, 60, 71,14,15,16,17,18, 1,19,19, 1, 1, 1,30, 1, 1, 2,  1, &
      1, 57,  3,  3,74,75,78,79,80,19,20,20, 1, 4, 2,31, 1, 4, 3,  1, &
     56,  2,  1,  1, 2, 1, 2, 1,81,20,21,21, 1, 2, 9,41,42,47,52, 54, &
      1,  1,  2,  3, 1, 2, 1, 2,82,21,22,22, 1, 2, 3, 4, 2, 1, 1,  1, &
      4,  1,  1,  9, 8, 4, 1,86,83,22,23,23, 1, 1, 2, 3, 7, 1, 2,  3, &
      2,  9,  1, 12, 2, 1,87, 1, 1,23,47, 1, 1, 2, 1, 5, 1, 9, 1,  1, &
      1,  2, 13,102,89,88, 3, 1, 1,26, 1,49, 1, 3, 3, 6, 1, 2,12,  2, &
     29, 30,149,  3, 2, 1, 5, 1, 1,29, 1,50, 1, 2, 6,10,11,12,13, 26, &
      1,  1,150,  1, 1, 1, 6, 1, 1,31, 1, 1,52,53,54,61,62, 1, 2,  3, &
      3,  1,152,  1, 2, 2, 7, 2, 2,32, 2, 1, 1, 3, 5, 7, 9,72, 1,  5, &
      7,  2,  1,154, 3, 3, 1,11, 3,33, 3, 2, 1, 1, 1, 1, 1, 2,75,  6, &
     95,  2,  1,155, 4, 8, 9,10,25, 1,38,41,42,43, 1, 1, 1, 2, 3, 85, &
     96,  1,  1,156, 1, 1, 1,26, 1, 1, 1,16,17,61,63, 2, 1, 1, 1,  2, &
     97,  1,  2,158, 2, 3,27, 1, 1, 4, 5,14, 1, 1, 1,66, 1, 2, 2,100, &
    191,192,194,357, 4,28, 1, 2, 1, 2, 3, 8, 2, 6, 9,11,78,81,85,189, &
      1,  2,  3,361, 5,29, 2, 3, 1, 2, 3, 4, 1, 1, 1, 1, 1, 1, 1,  1 /), &
        ((/20,20/))))

        coarse_river_direction_lat_index_expected_result = transpose(reshape((/ &
                                                                        4,3,3,2,2, &
                                                                        2,2,3,2,2, &
                                                                        4,3,4,3,3, &
                                                                        5,5,5,4,5, &
                                                                        6,6,5,5,5 /), &
                                                                        (/5,5/)))
        coarse_river_direction_lon_index_expected_result = transpose(reshape((/ &
                                                                        3,3,3,4,4, &
                                                                        2,3,2,5,1, &
                                                                        1,1,4,5,1, &
                                                                        1,2,4,5,5, &
                                                                        1,2,4,5,1 /), &
                                                                        (/5,5/)))
        call flow_latlon(input_fine_river_directions,input_fine_total_cumulative_flow, &
                         coarse_river_directions_lat_index,coarse_river_directions_lon_index)
        call assert_equals(coarse_river_directions_lon_index, &
                           coarse_river_direction_lon_index_expected_result,5,5)
        call assert_equals(coarse_river_directions_lat_index, &
                           coarse_river_direction_lat_index_expected_result,5,5)
    end subroutine

    subroutine testSmallGridThree
    use flow
    use cotat_parameters_mod
    integer, dimension(:,:), allocatable :: input_fine_river_directions
    integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
    integer, dimension(:,:), allocatable :: coarse_river_directions_lat_index
    integer, dimension(:,:), allocatable :: coarse_river_directions_lon_index
    integer, dimension(:,:), allocatable :: coarse_river_direction_lat_index_expected_result
    integer, dimension(:,:), allocatable :: coarse_river_direction_lon_index_expected_result
        allocate(input_fine_river_directions(12,12))
        allocate(input_fine_total_cumulative_flow(12,12))
        allocate(coarse_river_directions_lat_index(3,3))
        allocate(coarse_river_directions_lon_index(3,3))
        allocate(coarse_river_direction_lat_index_expected_result(3,3))
        allocate(coarse_river_direction_lon_index_expected_result(3,3))
        MUFP = 1.5
        run_check_for_sinks = .True.
        yamazaki_max_range = 9
        yamazaki_wrap = .True.
        input_fine_river_directions = transpose(reshape((/ &
!
        0,4,4,2, 3,2,1,8, 3,2,1,8, &
        8,7,4,2, 6,0,6,8, 6,0,6,8, &
        8,8,8,2, 6,2,6,8, 6,5,6,8, &
        8,8,8,5, 9,0,6,8, 9,8,6,8, &
!
        5,4,4,0, 2,2,2,2, 2,2,2,2, &
        8,8,8,8, 2,2,2,2, 2,2,2,2, &
        8,8,8,8, 2,2,4,4, 2,2,4,4, &
        8,8,8,8, 2,1,4,2, 2,1,4,2, &
!
        6,6,0,5, 2,3,3,2, 2,3,3,2, &
        2,1,8,2, 3,3,3,2, 3,3,3,2, &
        2,4,8,2, 6,5,3,2, 6,0,3,2, &
        0,6,6,2, 9,6,6,2, 9,6,6,2  /), &
        ((/12,12/))))

        input_fine_total_cumulative_flow = transpose(reshape((/ &
           12,2,1,1, 1, 1,1, 7, 1, 1,1, 7,&
            3,6,3,2, 1, 5,1, 6, 1, 5,1, 6,&
            2,2,2,3, 1, 3,1, 4, 1, 4,1, 4,&
            1,1,1,4, 1, 4,1, 2, 1, 1,1, 2,&
           12,8,4,4, 1, 1,1, 1, 1, 1,1, 1,&
            3,3,3,3, 2, 2,2, 2, 2, 2,2, 2,&
            2,2,2,2, 3, 9,6, 3, 3, 9,6, 3,&
            1,1,1,1, 4,11,1, 1, 4,11,1, 1,&
            1,2,5,1,16, 1,1, 2,16, 1,1, 2,&
            1,1,2,1,17, 1,2, 4,17, 1,2, 4,&
            4,1,1,2, 1,20,2, 7, 1,20,2, 7,&
            5,1,2,5, 1, 1,2,12, 1, 1,2,12/), &
            ((/12,12/))))

        coarse_river_direction_lat_index_expected_result = transpose(reshape((/ &
                                                                       -2,-2,0, &
                                                                       -1,3,3, &
                                                                       -2,-1,-2 /), &
                                                                        (/3,3/)))
        coarse_river_direction_lon_index_expected_result = transpose(reshape((/ &
                                                                       -2,-2,3, &
                                                                       -1,2,3, &
                                                                       -2,-1,-2 /), &
                                                                        (/3,3/)))
        call flow_latlon(input_fine_river_directions,input_fine_total_cumulative_flow, &
                         coarse_river_directions_lat_index,coarse_river_directions_lon_index)
        call assert_equals(coarse_river_directions_lat_index, &
                           coarse_river_direction_lat_index_expected_result,3,3)
        call assert_equals(coarse_river_directions_lon_index, &
                           coarse_river_direction_lon_index_expected_result,3,3)
    end subroutine

end module flow_test_mod
