module accumulate_flow_test_mod
use fruit
implicit none

contains

    subroutine setup
    end subroutine setup

    subroutine teardown
    end subroutine teardown

    subroutine testCalculateCumlativeFlowLatLon
      use accumulate_flow_mod
      integer, dimension(:,:), pointer :: flow_dirs
      integer, dimension(:,:), pointer :: output_cumulative_flow
      integer, dimension(:,:), pointer :: expected_cumulative_flow
        allocate(flow_dirs(6,6))
        allocate(expected_cumulative_flow(6,6))
        allocate(output_cumulative_flow(6,6))
        flow_dirs =  transpose(reshape((/3,1,4,4,4,4, &
                                         2,4,4,4,7,7, &
                                         3,9,8,7,6,2, &
                                         5,3,6,9,2,1, &
                                         6,7,2,3,1,8, &
                                         5,0,7,0,4,4 /),&
                               shape(transpose(flow_dirs))))
        expected_cumulative_flow = transpose(reshape((/ 1, 7, 6, 5, 3, 1, &
                                                       15, 7, 5, 1, 1, 1, &
                                                       16, 1, 1, 1, 3, 4, &
                                                       22,17, 1, 2, 1, 6, &
                                                        1,21,18, 1, 8, 1, &
                                                        0, 0,19,12, 3, 1 /),&
                                             shape(transpose(expected_cumulative_flow))))
        call accumulate_flow_latlon(flow_dirs, &
                                    output_cumulative_flow)
        call assert_true(all(output_cumulative_flow .eq. &
                             expected_cumulative_flow))
        deallocate(flow_dirs)
        deallocate(expected_cumulative_flow)
        deallocate(output_cumulative_flow)
    end subroutine testCalculateCumlativeFlowLatLon

    subroutine testCalculateCumlativeFlowLatLonWrapped()
      use accumulate_flow_mod
      integer, dimension(:,:), pointer :: flow_dirs_with_wrap
      integer, dimension(:,:), pointer :: output_cumulative_flow
      integer, dimension(:,:), pointer :: expected_cumulative_flow_with_wrap
        allocate(flow_dirs_with_wrap(6,6))
        allocate(expected_cumulative_flow_with_wrap(6,6))
        allocate(output_cumulative_flow(6,6))
        flow_dirs_with_wrap =  transpose(reshape((/1,1,4,4,4,3, &
                                                   2,4,4,4,7,7, &
                                                   3,9,8,7,6,2, &
                                                   4,3,6,9,5,4, &
                                                   6,7,2,3,1,8, &
                                                   7,0,7,0,4,9 /),&
                                     shape(transpose(flow_dirs_with_wrap))))
        expected_cumulative_flow_with_wrap = \
        transpose(reshape((/ 1, 7, 6, 5, 3, 1, &
                            15, 6, 5, 1, 1, 2, &
                            16, 1, 1, 1, 3, 4, &
                            23, 17, 1, 2, 31, 30, &
                             2, 22, 18, 1, 1, 2, &
                              1, 0, 19, 4, 2, 1 /),&
                  shape(transpose(expected_cumulative_flow_with_wrap))))
        call accumulate_flow_latlon(flow_dirs_with_wrap, &
                                    output_cumulative_flow)
        call assert_true(all(output_cumulative_flow .eq. &
                             expected_cumulative_flow_with_wrap))
        deallocate(flow_dirs_with_wrap)
        deallocate(expected_cumulative_flow_with_wrap)
        deallocate(output_cumulative_flow)
    end subroutine testCalculateCumlativeFlowLatLonWrapped

    subroutine testCalculateCumlativeFlowLatLonWithMask()
      use accumulate_flow_mod
      integer, dimension(:,:), pointer :: flow_dirs
      integer, dimension(:,:), pointer :: expected_cumulative_flow_when_using_mask
      integer, dimension(:,:), pointer :: output_cumulative_flow
      logical, dimension(:,:), pointer :: ls_mask
        allocate(flow_dirs(6,6))
        allocate(expected_cumulative_flow_when_using_mask(6,6))
        allocate(ls_mask(6,6))
        allocate(output_cumulative_flow(6,6))
        flow_dirs =  transpose(reshape((/3,1,4,4,4,4, &
                                         2,4,4,4,7,7, &
                                         3,9,8,7,6,2, &
                                         5,3,6,9,2,1, &
                                         6,7,2,3,1,8, &
                                         5,2,7,5,4,4 /),&
                               shape(transpose(flow_dirs))))
        expected_cumulative_flow_when_using_mask = \
            transpose(reshape((/0, 0, 0, 2, 0, 0, &
                                0, 4, 3, 1, 1, 0, &
                                1, 0, 0, 1, 3, 4, &
                                2, 2, 1, 2, 1, 5, &
                                0, 1, 3, 1, 7, 0, &
                                0, 0, 4,11, 3, 1 /),&
                      shape(transpose(expected_cumulative_flow_when_using_mask))))
        ls_mask = transpose(reshape((/.true.,.true.,.true.,.true.,.true.,.true., &
                                      .true.,.true.,.false.,.false.,.false.,.true., &
                                      .false.,.true.,.true.,.false.,.false.,.false., &
                                      .false.,.false.,.false.,.false.,.false.,.false., &
                                      .true.,.false.,.false.,.false.,.false.,.true., &
                                      .true.,.true.,.true.,.false.,.false.,.false. /),&
                            shape(ls_mask)))
        where(ls_mask)
          flow_dirs = 0
        end where
        call accumulate_flow_latlon(flow_dirs, &
                                    output_cumulative_flow)
        call assert_true(all(output_cumulative_flow .eq. &
                             expected_cumulative_flow_when_using_mask))
        deallocate(flow_dirs)
        deallocate(ls_mask)
        deallocate(expected_cumulative_flow_when_using_mask)
        deallocate(output_cumulative_flow)
    end subroutine testCalculateCumlativeFlowLatLonWithMask

    subroutine testCalculateCumlativeFlowLatLonWithBasicLoop()
      use accumulate_flow_mod
      integer, dimension(:,:), pointer :: flow_dirs_with_loop
      integer, dimension(:,:), pointer :: output_cumulative_flow
      integer, dimension(:,:), pointer :: expected_cumulative_flow_with_loop
        allocate(flow_dirs_with_loop(3,3))
        allocate(expected_cumulative_flow_with_loop(3,3))
        allocate(output_cumulative_flow(3,3))
        flow_dirs_with_loop =  transpose(reshape((/6,4,2, &
                                                   6,6,5, &
                                                   6,6,5/),&
                                     shape(transpose(flow_dirs_with_loop))))
        expected_cumulative_flow_with_loop = \
        transpose(reshape((/ 0,0,1, &
                             1,2,4, &
                             1,2,3  /),&
                  shape(transpose(expected_cumulative_flow_with_loop))))
        call accumulate_flow_latlon(flow_dirs_with_loop, &
                                    output_cumulative_flow)
        call assert_true(all(output_cumulative_flow .eq. &
                             expected_cumulative_flow_with_loop))
        deallocate(flow_dirs_with_loop)
        deallocate(expected_cumulative_flow_with_loop)
        deallocate(output_cumulative_flow)
    end subroutine testCalculateCumlativeFlowLatLonWithBasicLoop

    subroutine testCalculateCumlativeFlowLatLonWithLoop()
      use accumulate_flow_mod
      integer, dimension(:,:), pointer :: flow_dirs_with_loop
      integer, dimension(:,:), pointer :: output_cumulative_flow
      integer, dimension(:,:), pointer :: expected_cumulative_flow_with_loop
        allocate(flow_dirs_with_loop(8,8))
        allocate(expected_cumulative_flow_with_loop(8,8))
        allocate(output_cumulative_flow(8,8))
        flow_dirs_with_loop =  transpose(reshape((/1,1,4,4,4,6,3,5, &
                                                   2,4,4,4,7,8,4,4, &
                                                   3,9,8,7,6,8,5,8, &
                                                   4,3,6,9,5,8,5,5, &
                                                   6,7,2,3,1,8,7,5, &
                                                   7,2,7,1,4,9,8,4, &
                                                   5,5,5,5,5,5,5,5, &
                                                   5,5,5,5,5,5,5,5 /),&
                                     shape(transpose(flow_dirs_with_loop))))
        expected_cumulative_flow_with_loop = \
        transpose(reshape((/ 1, 5,   4, 3, 1, 0, 0,  0, &
                            12, 6,   5, 1, 1, 0, 0,  0, &
                            13, 1,   1, 1, 3,10, 0,  1, &
                            19, 14,  1, 2, 0, 6, 0, 20, &
                            1,  18, 15, 1, 1, 1, 4,  2, &
                            1,   1, 16, 4, 2, 1, 2,  1, &
                            0,   2,  5, 0, 0, 0, 0,  0, &
                            0,   0,  0, 0, 0, 0, 0,  0 /),&
                  shape(transpose(expected_cumulative_flow_with_loop))))
        call accumulate_flow_latlon(flow_dirs_with_loop, &
                                    output_cumulative_flow)
        call assert_true(all(output_cumulative_flow .eq. &
                             expected_cumulative_flow_with_loop))
        deallocate(flow_dirs_with_loop)
        deallocate(expected_cumulative_flow_with_loop)
        deallocate(output_cumulative_flow)
    end subroutine testCalculateCumlativeFlowLatLonWithLoop

    subroutine testCalculateCumulativeFlow
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
    end subroutine testCalculateCumulativeFlow

    subroutine testCalculateCumulativeFlowWithZeroBifurcations
      use accumulate_flow_mod
      integer, dimension(:), pointer :: input_river_directions
      integer, dimension(:,:), pointer :: input_bifurcated_river_directions
      integer, dimension(:,:), pointer :: cell_neighbors
      integer, dimension(:), pointer :: output_cumulative_flow
      integer, dimension(:), pointer :: expected_output_cumulative_flow
        allocate(input_river_directions(80))
        allocate(cell_neighbors(80,3))
        allocate(output_cumulative_flow(80))
        allocate(expected_output_cumulative_flow(80))
        allocate(input_bifurcated_river_directions(80,11))
        input_river_directions = (/    6, -3, 11,  -3, 18, &
                        21, 6, 6, 2, 2, 10, 13, 3, 13, 16, 4, 35, 37, 18, 39, &
            21,41,22,25,26,45,11,29,30,49,30,50,15,33,36,-3,36,37,38, 38, &
            42,43,24,63,46,47,28,49,50,51,-3,53,34,55,36,55,56,57,-3,-3, &
                      62,63,-3,63,77,65,50,67,68,54,54,55,-3, -3, -3,  &
                                    62, -3, 68, 71, -3 /)
        input_bifurcated_river_directions(:,:) = -9
        expected_output_cumulative_flow = (/ 1, 9, 4, 7, 1,&
                        4, 1, 1, 1, 7, 6, 1, 3, 1, 5, 6, 1, 3, 1, 1, &
             0, 2, 1, 6, 7, 8, 1, 12, 13, 15, 1, 1, 4, 3, 2, 20, 8, 4, 2, 1, &
             3, 4,  5, 1, 9, 10, 11, 1, 17, 23, 24, 1, 2, 4, 9, 3, 2, 1, 0, 0, &
                        1, 3, 6, 1, 2, 1, 4, 3, 1, 1, 2, 1, 0, 0, 0,  &
                                             1, 3, 1, 1, 0 /)
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
                                               output_cumulative_flow, &
                                               input_bifurcated_river_directions)
        call assert_true(all(output_cumulative_flow .eq. &
                             expected_output_cumulative_flow))
        deallocate(input_river_directions)
        deallocate(output_cumulative_flow)
        deallocate(cell_neighbors)
        deallocate(expected_output_cumulative_flow)
    end subroutine testCalculateCumulativeFlowWithZeroBifurcations

    subroutine testCalculateCumulativeFlowWithBifurcations
      use accumulate_flow_mod
      integer, dimension(:), pointer :: input_river_directions
      integer, dimension(:,:), pointer :: input_bifurcated_river_directions
      integer, dimension(:,:), pointer :: cell_neighbors
      integer, dimension(:), pointer :: output_cumulative_flow
      integer, dimension(:), pointer :: expected_output_cumulative_flow
        allocate(input_river_directions(80))
        allocate(cell_neighbors(80,3))
        allocate(output_cumulative_flow(80))
        allocate(expected_output_cumulative_flow(80))
        allocate(input_bifurcated_river_directions(80,11))
        input_river_directions = (/    6, -3, 11,  -3, 18, &
                        21, 6, 6, 2, 2, 10, 13, 3, 13, 16, 4, 35, 37, 18, 39, &
            21,41,22,25,26,45,11,29,30,49,30,50,15,33,36,-3,36,37,38, 38, &
            42,43,24,63,46,47,28,49,50,51,-3,53,34,55,36,55,56,57,-3,-3, &
                      62,63,-3,63,77,65,50,67,68,54,54,55,-3, -3, -3,  &
                                    62, -3, 68, 71, -3 /)
        input_bifurcated_river_directions(:,:) = -9
        input_bifurcated_river_directions(11,1) = 9
        input_bifurcated_river_directions(34,1) = 35
        input_bifurcated_river_directions(43,1) = 44
        input_bifurcated_river_directions(46,1) = 65
        input_bifurcated_river_directions(46,2) = 27
        input_bifurcated_river_directions(46,3) = 64
        expected_output_cumulative_flow = (/ 1, 35, 4, 7, 1,&
                        4, 1, 1, 17, 17, 16, 1, 3, 1, 5, 6, 1, 3, 1, 1, &
             0, 2, 1, 6, 7, 8, 11, 12, 13, 15, 1, 1, 4, 3, 5, 23, 8, 4, 2, 1, &
             3, 4,  5, 6, 9, 10, 11, 1, 17, 23, 24, 1, 2, 4, 9, 3, 2, 1, 0, 0, &
                        1, 3, 21, 11, 12, 1, 4, 3, 1, 1, 2, 1, 0, 0, 0,  &
                                             1, 13, 1, 1, 0 /)
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
                                               output_cumulative_flow, &
                                               input_bifurcated_river_directions)
        call assert_true(all(output_cumulative_flow .eq. &
                             expected_output_cumulative_flow))
        deallocate(input_river_directions)
        deallocate(output_cumulative_flow)
        deallocate(cell_neighbors)
        deallocate(expected_output_cumulative_flow)
    end subroutine testCalculateCumulativeFlowWithBifurcations

end module accumulate_flow_test_mod
