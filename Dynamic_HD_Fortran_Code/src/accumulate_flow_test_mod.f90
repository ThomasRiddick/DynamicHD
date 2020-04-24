module accumulate_flow_test_mod
use fruit
implicit none

contains

    subroutine setup
    end subroutine setup

    subroutine teardown
    end subroutine teardown

    subroutine testCalculateCumlativeFlow
      use accumulate_flow_mod
      integer, dimension(:), pointer :: input_river_directions
      integer, dimension(:,:), pointer :: cell_neighbors
      integer, dimension(:), pointer :: output_cumulative_flow
      integer, dimension(:), pointer :: expected_output_cumulative_flow
        allocate(input_river_directions(80))
        allocate(cell_neighbors(80,3))
        allocate(output_cumulative_flow(80))
        allocate(expected_output_cumulative_flow(80))
        input_river_directions = (/    2,10,  5,  5, -3, &
                        23, 1, 9, 10, 13, 27, 11, 30, 33, 33, 4, 33, 5, 5, 6, &
             6, 21, 24, -3, 26, 45, 10, 47, 28, 47, 33, 51, 32, -3, -3, -3, -3, -3, 40, 20, &
             60, 24, 24, 62, 44, -3, 26, 46, 30, 49, 52, -3, -3, -3, -3, -3, -3, 73, 60, 40, &
                       41, 61, 64, 65, 66, 48, 50, 78, 52, -3, -3, -3, 80, 75, 59,  &
                                      62, 78, 79, -3, 74 /)
        expected_output_cumulative_flow = (/ 2, 3, 1, 2, 6,&
                        38, 1, 1, 2, 9, 2, 1, 10, 1, 1, 1, 1, 1, 1, 35, &
             2, 1, 39, 42, 1, 19, 3, 2, 1, 14, 1, 6, 5, 0, 0, 0, 0, 0, 1, 34, &
            25, 1,  1, 21, 20, 6, 17, 5, 3, 2, 7, 9, 0, 0, 0, 0, 0, 1, 6, 32, &
                        24, 23, 1, 2, 3, 4, 1, 1, 1, 0, 0, 0, 2, 4, 5,  &
                                             1, 1, 3, 4, 3 /)
        cell_neighbors =  transpose(reshape((/ &
        !1
        5,7,2, &
        !2
        1,10,3, &
        !3
        2,13,4, &
        !4
        3,16,5, &
        !5
        4,19,1, &
        !6
        20,21,7, &
        !7
        1,6,8, &
        !8
        7,23,9, &
        !9
        8,25,10, &
        !10
        2,9,11, &
        !11
        10,27,12, &
        !12
        11,29,13, &
        !13
        3,12,14, &
        !14
        13,31,15, &
        !15
        14,33,16, &
        !16
        4,15,17, &
        !17
        16,35,18, &
        !18
        17,37,19, &
        !19
        5,18,20, &
        !20
        19,39,6, &
        !21
        6,40,22, &
        !22
        21,41,23, &
        !23
        8,22,24, &
        !24
        23,43,25, &
        !25
        24,26,9, &
        !26
        25,45,27, &
        !27
        11,26,28, &
        !28
        27,47,29, &
        !29
        12,28,30, &
        !30
        29,49,31, &
        !31
        14,30,32, &
        !32
        31,51,33, &
        !33
        15,32,34, &
        !34
        33,53,35, &
        !35
        17,34,36, &
        !36
        35,55,37, &
        !37
        18,36,38, &
        !38
        37,57,39, &
        !39
        20,38,40, &
        !40
        39,59,21, &
        !41
        22,60,42, &
        !42
        41,61,43, &
        !43
        24,42,44, &
        !44
        43,63,45, &
        !45
        26,44,46, &
        !46
        45,64,47, &
        !47
        28,46,48, &
        !48
        47,66,49, &
        !49
        30,48,50, &
        !50
        49,67,51, &
        !51
        32,50,52, &
        !52
        51,69,53, &
        !53
        34,52,54, &
        !54
        53,70,55, &
        !55
        36,54,56, &
        !56
        55,72,57, &
        !57
        38,56,58, &
        !58
        57,73,59, &
        !59
        40,58,60, &
        !60
        59,75,41, &
        !61
        42,75,62, &
        !62
        61,76,63, &
        !63
        44,62,64, &
        !64
        46,63,65, &
        !65
        64,77,66, &
        !66
        48,65,67, &
        !67
        50,66,68, &
        !68
        67,78,69, &
        !69
        52,68,70, &
        !70
        54,69,71, &
        !71
        70,79,72, &
        !72
        56,71,73, &
        !73
        58,72,74, &
        !74
        73,80,75, &
        !75
        60,74,61, &
        !76
        62,80,77, &
        !77
        65,76,78, &
        !78
        68,77,79, &
        !79
        71,78,80, &
        !80
        74,79,76 /), (/3,80/)))
        call accumulate_flow_icon_single_index(cell_neighbors, &
                                               input_river_directions, &
                                               output_cumulative_flow)
        call assert_true(all(output_cumulative_flow .eq. &
                             expected_output_cumulative_flow))
        deallocate(input_river_directions)
        deallocate(output_cumulative_flow)
        deallocate(cell_neighbors)
        deallocate(expected_output_cumulative_flow)
    end subroutine testCalculateCumlativeFlow

end module accumulate_flow_test_mod
