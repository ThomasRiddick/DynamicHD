module cotat_plus_test_mod
use fruit
implicit none

contains

    subroutine setup
    end subroutine setup

    subroutine teardown
    end subroutine teardown

    subroutine testSmallGrid
    use cotat_plus
    use cotat_parameters_mod
    integer, dimension(:,:), allocatable :: input_fine_river_directions
    integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
    integer, dimension(:,:), allocatable :: output_coarse_river_directions
    integer, dimension(5,5) :: expected_result
        allocate(input_fine_river_directions(15,15))
        allocate(input_fine_total_cumulative_flow(15,15))
        allocate(output_coarse_river_directions(5,5))
        output_coarse_river_directions = -999
        MUFP = 1.5
        area_threshold = 9
        run_check_for_sinks = .True.
        input_fine_river_directions = transpose(reshape((/ &
!
            -1,-1,-1, -1,0,4,   2,2,2, 2,2,2, 3,2,2, &
            -1,-1,-1, -1,-1,-1, 0,4,4, 4,4,1, 4,4,1, &
            -1,-1,-1, -1,-1,9,  8,8,8, 1,7,7, 4,4,4, &
!
            -1,-1,-1, -1,-1,0, 4,4,4,       6,8,5, 8,4,8, &
            -1,0,4, 4,4,5, 7,1,8,       4,6,7, 6,5,4, &
            -1,0,4, 4,5,7, 4,4,1,       9,6,8, 1,6,2, &
!
            -1,-1,0, 4,6,8, 4,4,6,       6,6,7, 4,2,5, &
            -1,-1,-1, 7,6,8, 4,1,3,       9,9,8, 8,7,4, &
            -1,0,0, 4,6,8, 4,4,5,       1,5,5, 9,1,7, &
!
            0,8,7, 7,7,7, 7,4,4,       6,7,4, 4,1,2, &
            8,8,8, 8,1,2, 6,7,5,       9,8,6, 8,7,4, &
            9,2,7, 4,4,2, 9,8,7,       9,8,4, 9,8,8, &
!
            6,6,8, 4,8,1, 2,1,8,       3,8,2, 3,5,8, &
            4,6,8, 7,8,7, 4,4,3,       3,2,6, 6,5,8, &
            9,8,8, 7,8,5, 8,9,8,       6,6,8, 9,8,7  &
!
            /), shape(transpose(input_fine_river_directions))))
        input_fine_total_cumulative_flow = transpose(reshape((/ &
!
        1,  1,  1,    1,  1,  1,    1,  1,  1,    1,  1,  1,    1,  1,  1, &
        1,  1,  1,    1,  1,  1,   52, 48, 45,   42, 11,  6,    4,  3,  2, &
        1,  1,  1,    1,  1,  1,    1,  1,  1,    1, 29,  9,    8,  5,  2, &
!
        1,  1,  1,    1,  1,  8,    6,  5,  4,    1, 22,  1,    2,  1,  1, &
        1, 55, 54,   53, 52,  1,    1,  1,  2,    1,  2, 20,    1,  3,  1, &
        1,  3,  2,    1,  1, 51,    3,  1,  1,    1, 16, 17,    1,  1,  2, &
!
        1,  1,  3,    1,  1, 47,    3,  2,  1,    2,  4, 15,    7,  1,  3, &
        1,  1,  1,    1,  1, 42,    1,  1,  1,    1,  1,  1,    1,  5,  1, &
        1, 35,  5,    2,  2, 39,    3,  1,  1,   24,  1,  1,    1,  1,  1, &
!
        2, 32,  2,    2,  1,  1,   33, 26, 25,    1, 22, 14,   13,  1,  1, &
        1, 31,  1,    1,  1,  1,    1,  6,  1,    1,  5,  1,    3,  8,  5, &
        1,  1, 29,   15, 13,  2,    1,  1,  2,    1,  3,  1,    1,  1,  3, &
!
        1,  3, 13,    1, 12,  3,    1,  1,  1,    1,  1,  1,    1,  1,  2, &
        1,  4,  7,    1,  5,  6,    5,  1,  3,    1,  2, 11,   12, 17,  1, &
        1,  1,  1,    1,  1,  1,    1,  1,  1,    4,  8,  9,    1,  1,  1  &
!
            /), shape(transpose(input_fine_total_cumulative_flow))))

        expected_result = transpose(reshape((/ &
        !
            -1,6,0,4,4, &
             0,4,4,8,5, &
             0,7,4,8,7, &
             8,4,7,4,7, &
             8,7,7,6,5 &
        !
            /), shape(transpose(expected_result))))

        call cotat_plus_latlon(input_fine_river_directions,input_fine_total_cumulative_flow,&
                               output_coarse_river_directions)
        call assert_equals(expected_result,output_coarse_river_directions,5,5)

    end subroutine testSmallGrid

    subroutine testSmallGridTwo
    use cotat_plus
    use cotat_parameters_mod
    integer, dimension(:,:), allocatable :: input_fine_river_directions
    integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
    integer, dimension(:,:), allocatable :: output_coarse_river_directions
    integer, dimension(5,5) :: expected_result
        allocate(input_fine_river_directions(15,15))
        allocate(input_fine_total_cumulative_flow(15,15))
        allocate(output_coarse_river_directions(5,5))
        output_coarse_river_directions = -999
        MUFP = 1.5
        area_threshold = 9
        run_check_for_sinks = .True.
        input_fine_river_directions = transpose(reshape((/ &
!
            -1,-1,-1, -1,0,4,   2,2,2, 2,2,2, 5,5,2, &
            -1,-1,-1, -1,-1,-1, 0,4,4, 4,4,1, 8,8,2, &
            4,-1,-1, -1,-1,9,  8,8,8, 1,7,3, 2,1,4, &
!
            -1,-1,-1, -1,-1,0, 4,4,4,   6,8,2, 1,6,8, &
            -1,0,4, 4,4,5, 7,1,8,       4,6,1, 9,6,8, &
            -1,0,4, 4,5,7, 4,4,1,       9,1,8, 5,9,8, &
!
            -1,-1,0, 4,6,8, 4,4,6,       2,6,7, 4,2,5, &
            -1,-1,-1, 7,6,8, 4,1,3,       5,9,8, 8,7,4, &
            -1,0,0, 4,6,8, 4,4,5,       1,5,5, 9,1,7, &
!
            0,8,7, 7,7,7, 7,4,4,       6,7,4, 4,1,2, &
            8,8,8, 8,1,2, 6,7,5,       9,8,6, 8,7,4, &
            9,2,7, 4,4,2, 9,8,7,       9,8,4, 9,8,8, &
!
            6,6,8, 4,8,1, 2,1,8,       3,8,2, 3,5,8, &
            4,6,8, 7,8,7, 4,4,3,       3,2,6, 6,5,8, &
            9,8,8, 7,8,5, 8,9,8,       6,6,8, 9,8,7  &
!
            /), shape(transpose(input_fine_river_directions))))
        input_fine_total_cumulative_flow = transpose(reshape((/ &
!
        1,  1,  1,    1,  1,  1,    1,  1,  1,    1,  1,  1,    5,  8,  3, &
        1,  1,  1,    1,  1,  1,   52, 48, 45,   42, 11,  6,    5,  6,  9, &
        529,  1,  1,    1,  1,  1,    1,  1,  1,    1, 29,  9,    5,  560,  530, &
!
        1,  1,  1,    1,  1,  8,    6,  5,  4,    1, 22, 108,    570,1,  9, &
        1, 55, 54,   53, 52,  1,    1,  1,  2,    1,  2, 711,    1,  1,  7, &
        1,  3,  2,    1,  1, 51,    3,  1,  1,    1, 821, 17,    1,  1,  4, &
!
        1,  1,  3,    1,  1, 47,    3,  2,  1,    825,  4, 15,    7,  1,  3, &
        1,  1,  1,    1,  1, 42,    1,  1,  1,    826,  1,  1,    1,  5,  1, &
        1, 35,  5,    2,  2, 39,    3,  1,  1,   24,  1,  1,    1,  1,  1, &
!
        2, 32,  2,    2,  1,  1,   33, 26, 25,    1, 22, 14,   13,  1,  1, &
        1, 31,  1,    1,  1,  1,    1,  6,  1,    1,  5,  1,    3,  8,  5, &
        1,  1, 29,   15, 13,  2,    1,  1,  2,    1,  3,  1,    1,  1,  3, &
!
        1,  3, 13,    1, 12,  3,    1,  1,  1,    1,  1,  1,    1,  1,  2, &
        1,  4,  7,    1,  5,  6,    5,  1,  3,    1,  2, 11,   12, 17,  1, &
        1,  1,  1,    1,  1,  1,    1,  1,  1,    4,  8,  9,    1,  1,  1  &
!
            /), shape(transpose(input_fine_total_cumulative_flow))))

        expected_result = transpose(reshape((/ &
        !
            4,6,0,4,2, &
             0,4,4,2,4, &
             0,7,4,5,7, &
             8,4,7,4,7, &
             8,7,7,6,5 &
        !
            /), shape(transpose(expected_result))))

        call cotat_plus_latlon(input_fine_river_directions,input_fine_total_cumulative_flow,&
                               output_coarse_river_directions)
        call assert_equals(expected_result,output_coarse_river_directions,5,5)

    end subroutine testSmallGridTwo

end module cotat_plus_test_mod
