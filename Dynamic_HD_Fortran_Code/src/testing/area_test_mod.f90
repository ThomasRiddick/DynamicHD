module area_test_mod
use fruit
implicit none

contains

    subroutine setup
    end subroutine setup

    subroutine teardown
    end subroutine teardown

    subroutine testOneOutflowCellSetupOne
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            allocate(input_fine_river_directions(9,9))
            allocate(input_fine_total_cumulative_flow(9,9))
            MUFP = 1.5
            area_threshold = 9
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 4,4,4, 4,4,4, 4,6,6,&
                                                               4,4,4, 4,4,4, 4,7,6,&
                                                               4,4,4, 4,4,6, 9,6,6,&
                                                               5,5,5, 3,6,9, 5,5,5,&
                                                               5,5,6, 6,8,8, 5,5,5,&
                                                               5,5,5, 6,9,8, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5 /),&
                                           shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 19,18,17, 16,15,14, 13,1,1,&
                                                                    1,1,1, 1,1,1, 1,12,1,&
                                                                    1,1,1, 1,1,1, 11,1,1,&
                                                                    1,1,1, 1,5,10, 1,1,1,&
                                                                    1,1,1, 2,4,4, 1,1,1,&
                                                                    1,1,1, 1,2,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(7,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
        end subroutine testOneOutflowCellSetupOne

    subroutine testOneOutflowCellSetupTwo
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            allocate(input_fine_river_directions(9,9))
            allocate(input_fine_total_cumulative_flow(9,9))
            MUFP = 1.5
            area_threshold = 4
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 4,4,4, 4,4,4, 4,6,6,&
                                                               4,4,4, 4,4,4, 4,7,6,&
                                                               4,4,4, 4,4,6, 9,6,6,&
                                                               5,5,5, 3,6,9, 5,5,5,&
                                                               5,5,6, 6,8,8, 5,5,5,&
                                                               5,5,5, 6,9,8, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5 /),&
                                           shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 19,18,17, 16,15,14, 13,1,1,&
                                                                    1,1,1, 1,1,1, 1,12,1,&
                                                                    1,1,1, 1,1,1, 11,1,1,&
                                                                    1,1,1, 1,5,10, 1,1,1,&
                                                                    1,1,1, 2,4,4, 1,1,1,&
                                                                    1,1,1, 1,2,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(8,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
        end subroutine testOneOutflowCellSetupTwo

    subroutine testOneOutflowCellSetupThree
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            allocate(input_fine_river_directions(9,9))
            allocate(input_fine_total_cumulative_flow(9,9))
            MUFP = 1.5
            area_threshold = 2
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 4,4,4, 4,4,4, 4,6,6,&
                                                               4,4,4, 4,4,4, 4,7,6,&
                                                               4,4,4, 4,4,6, 9,6,6,&
                                                               5,5,5, 3,6,9, 5,5,5,&
                                                               5,5,6, 6,8,8, 5,5,5,&
                                                               5,5,5, 6,9,8, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5 /),&
                                           shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 19,18,17, 16,15,14, 13,1,1,&
                                                                    1,1,1, 1,1,1, 1,12,1,&
                                                                    1,1,1, 1,1,1, 11,1,1,&
                                                                    1,1,1, 1,5,10, 1,1,1,&
                                                                    1,1,1, 2,4,4, 1,1,1,&
                                                                    1,1,1, 1,2,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(9,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
        end subroutine testOneOutflowCellSetupThree

    subroutine testOneOutflowCellSetupFour
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            allocate(input_fine_river_directions(9,9))
            allocate(input_fine_total_cumulative_flow(9,9))
            MUFP = 1.5
            area_threshold = 5
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,6,&
                                                               5,5,5, 5,5,5, 5,9,5,&
                                                               5,5,5, 5,5,5, 9,5,5,&
                                                               5,5,5, 3,2,9, 4,4,4,&
                                                               5,5,6, 6,3,7, 5,5,5,&
                                                               5,5,5, 1,4,4, 5,5,5,&
                                                               5,5,1, 5,5,5, 5,5,5,&
                                                               5,2,5, 5,5,5, 5,5,9,&
                                                               5,6,6, 6,6,6, 6,9,5 /),&
                                           shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,116,&
                                                                    1,1,1, 1,1,1, 1,115,1,&
                                                                    1,1,1, 1,1,1, 114,1,1,&
                                                                    1,1,1, 1,2,113, 112,111,110,&
                                                                    1,1,1, 2,5,1, 1,1,1,&
                                                                    1,1,1, 8,7,5, 1,1,1,&
                                                                    1,1,9, 1,1,1, 1,1,1,&
                                                                    1,10,1, 1,1,1, 1,1,18,&
                                                                    11,12,13, 14,15,16, 17,1,1 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(2,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
        end subroutine testOneOutflowCellSetupFour

    subroutine testOneOutflowCellSetupFive
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            allocate(input_fine_river_directions(9,9))
            allocate(input_fine_total_cumulative_flow(9,9))
            MUFP = 1.1
            area_threshold = 5
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,6,&
                                                               5,5,5, 5,5,5, 5,9,5,&
                                                               5,5,5, 5,5,5, 9,5,5,&
                                                               5,5,5, 3,2,9, 4,4,4,&
                                                               5,5,6, 6,3,7, 5,5,5,&
                                                               5,5,5, 1,4,4, 5,5,5,&
                                                               5,5,1, 5,5,5, 5,5,5,&
                                                               5,2,5, 5,5,5, 5,5,9,&
                                                               5,6,6, 6,6,6, 6,9,5 /),&
                                           shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,116,&
                                                                    1,1,1, 1,1,1, 1,115,1,&
                                                                    1,1,1, 1,1,1, 114,1,1,&
                                                                    1,1,1, 1,2,113, 112,111,110,&
                                                                    1,1,1, 2,5,1, 1,1,1,&
                                                                    1,1,1, 8,7,5, 1,1,1,&
                                                                    1,1,9, 1,1,1, 1,1,1,&
                                                                    1,10,1, 1,1,1, 1,1,18,&
                                                                    11,12,13, 14,15,16, 17,1,1 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(9,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupFive

    subroutine testOneOutflowCellSetupSix
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            allocate(input_fine_river_directions(9,9))
            allocate(input_fine_total_cumulative_flow(9,9))
            MUFP = 1.1
            area_threshold = 5
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,8,5, 5,5,5,&
                                                               5,5,5, 5,8,5, 5,5,5,&
                                                               5,5,5, 5,8,5, 5,5,5,&
                                                               5,5,5, 1,6,7, 5,5,5,&
                                                               5,5,5, 9,2,2, 5,5,5,&
                                                               5,5,5, 6,2,4, 6,6,6,&
                                                               5,5,5, 5,6,9, 5,5,5,&
                                                               5,2,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5 /),&
                                           shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,6,1, 1,1,1,&
                                                                    1,1,1, 1,5,1, 1,1,1,&
                                                                    1,1,1, 1,4,1, 1,1,1,&
                                                                    1,1,1, 1,2,3, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,5,1, 8,9,10,&
                                                                    1,1,1, 1,6,7, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(6,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupSix

    subroutine testOneOutflowCellSetupSeven
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 4.5
            area_threshold = 1
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,2,5, 8,5,5, 5,5,5, 5,&
                                                               5,1,7, 8,4,4, 5,5,5, 5,&
                                                               3,5,8, 8,1,7, 5,5,5, 5,&
                                                               5,6,1, 7,4,7, 6,6,6, 5,&
                                                               5,1,5, 5,8,5, 5,5,5, 5,&
                                                               7,2,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 8,1,1, 1,1,1, 1,&
                                                                    1,1,1, 7,1,1, 1,1,1, 1,&
                                                                    1,55,1, 6,1,1, 1,1,1, 1,&
                                                                    1,56,54, 5,3,1, 1,1,1, 1,&
                                                                    57,1,53, 1,31,1, 1,1,1, 1,&
                                                                    1,58,59, 52,22,32, 33,34,35, 1,&
                                                                    1,60,1, 1,23,1, 1,1,1, 1,&
                                                                    61,1,1, 1,24,1, 1,1,1, 1,&
                                                                    1,1,1, 1,25,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(8,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
        end subroutine testOneOutflowCellSetupSeven

    subroutine testOneOutflowCellSetupEight
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 4.0
            area_threshold = 1
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,2,5, 8,5,5, 5,5,5, 5,&
                                                               5,1,7, 8,4,4, 5,5,5, 5,&
                                                               3,5,8, 8,1,7, 5,5,5, 5,&
                                                               5,6,1, 7,4,7, 6,6,6, 5,&
                                                               5,1,5, 5,8,5, 5,5,5, 5,&
                                                               7,2,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 8,1,1, 1,1,1, 1,&
                                                                    1,1,1, 7,1,1,1,1,1, 1,&
                                                                    1,55,1, 6,1,1, 1,1,1, 1,&
                                                                    1,56,54, 5,3,1, 1,1,1, 1,&
                                                                    57,1,53, 1,31,1, 1,1,1, 1,&
                                                                    1,58,59, 52,22,32, 33,34,35, 1,&
                                                                    1,60,1, 1,23,1, 1,1,1, 1,&
                                                                    61,1,1, 1,24,1, 1,1,1, 1,&
                                                                    1,1,1, 1,25,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(4,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
        end subroutine testOneOutflowCellSetupEight

    subroutine testOneOutflowCellSetupNine
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 3.0
            area_threshold = 1
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,2,5, 8,5,5, 5,5,5, 5,&
                                                               5,1,7, 8,4,4, 5,5,5, 5,&
                                                               3,5,8, 8,1,7, 5,5,5, 5,&
                                                               5,6,1, 7,4,7, 6,6,6, 5,&
                                                               5,1,5, 5,8,5, 5,5,5, 5,&
                                                               7,2,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 8,1,1, 1,1,1, 1,&
                                                                    1,1,1, 7,1,1,1,1,1, 1,&
                                                                    1,55,1, 6,1,1, 1,1,1, 1,&
                                                                    1,56,54, 5,3,1, 1,1,1, 1,&
                                                                    57,1,53, 1,21,1, 1,1,1, 1,&
                                                                    1,58,59, 52,32,22, 23,24,25, 1,&
                                                                    1,60,1, 1,33,1, 1,1,1, 1,&
                                                                    61,1,1, 1,34,1, 1,1,1, 1,&
                                                                    1,1,1, 1,35,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(8,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
        end subroutine testOneOutflowCellSetupNine

      subroutine testOneOutflowCellSetupTen
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 2.4
            area_threshold = 1
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,2,5, 8,5,5, 5,5,5, 5,&
                                                               5,1,7, 8,4,4, 5,5,5, 5,&
                                                               3,5,8, 8,1,7, 5,5,5, 5,&
                                                               5,6,1, 7,4,7, 6,6,6, 5,&
                                                               5,1,5, 5,8,5, 5,5,5, 5,&
                                                               7,2,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 8,1,1, 1,1,1, 1,&
                                                                    1,1,1, 7,1,1,1,1,1, 1,&
                                                                    1,55,1, 6,1,1, 1,1,1, 1,&
                                                                    1,56,54, 5,3,1, 1,1,1, 1,&
                                                                    57,1,53, 1,21,1, 1,1,1, 1,&
                                                                    1,58,59, 52,32,22, 23,24,25, 1,&
                                                                    1,60,1, 1,33,1, 1,1,1, 1,&
                                                                    61,1,1, 1,34,1, 1,1,1, 1,&
                                                                    1,1,1, 1,35,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(4,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
        end subroutine testOneOutflowCellSetupTen

        subroutine testOneOutflowCellSetupEleven
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 2.4
            area_threshold = 2
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,2,5, 8,5,5, 5,5,5, 5,&
                                                               5,1,7, 8,4,4, 5,5,5, 5,&
                                                               3,5,8, 8,1,7, 5,5,5, 5,&
                                                               5,6,1, 7,4,7, 6,6,6, 5,&
                                                               5,1,5, 5,8,5, 5,5,5, 5,&
                                                               7,2,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 8,1,1, 1,1,1, 1,&
                                                                    1,1,1, 7,1,1,1,1,1, 1,&
                                                                    1,55,1, 6,1,1, 1,1,1, 1,&
                                                                    1,56,54, 5,3,1, 1,1,1, 1,&
                                                                    57,1,53, 1,21,1, 1,1,1, 1,&
                                                                    1,58,59, 52,32,22, 23,24,25, 1,&
                                                                    1,60,1, 1,33,1, 1,1,1, 1,&
                                                                    61,1,1, 1,34,1, 1,1,1, 1,&
                                                                    1,1,1, 1,35,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(7,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
        end subroutine testOneOutflowCellSetupEleven

    subroutine testOneOutflowCellSetupTwelve
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 2.4
            area_threshold = 5
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,2,5, 8,5,5, 5,5,5, 5,&
                                                               5,1,7, 8,4,4, 5,5,5, 5,&
                                                               3,5,8, 8,1,7, 5,5,5, 5,&
                                                               5,6,1, 7,4,7, 6,6,6, 5,&
                                                               5,1,5, 5,8,5, 5,5,5, 5,&
                                                               7,2,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 8,1,1, 1,1,1, 1,&
                                                                    1,1,1, 7,1,1,1,1,1, 1,&
                                                                    1,55,1, 6,1,1, 1,1,1, 1,&
                                                                    1,56,54, 5,3,1, 1,1,1, 1,&
                                                                    57,1,53, 1,21,1, 1,1,1, 1,&
                                                                    1,58,59, 52,32,22, 23,24,25, 1,&
                                                                    1,60,1, 1,33,1, 1,1,1, 1,&
                                                                    61,1,1, 1,34,1, 1,1,1, 1,&
                                                                    1,1,1, 1,35,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(4,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
        end subroutine testOneOutflowCellSetupTwelve

    subroutine testOneOutflowCellSetupThirteen
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 2.4
            area_threshold = 10
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,2,5, 8,5,5, 5,5,5, 5,&
                                                               5,1,7, 8,4,4, 5,5,5, 5,&
                                                               3,5,8, 8,1,7, 5,5,5, 5,&
                                                               5,6,1, 7,4,7, 6,6,6, 5,&
                                                               5,1,5, 5,8,5, 5,5,5, 5,&
                                                               7,2,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 8,1,1, 1,1,1, 1,&
                                                                    1,1,1, 7,1,1,1,1,1, 1,&
                                                                    1,55,1, 6,1,1, 1,1,1, 1,&
                                                                    1,56,54, 5,3,1, 1,1,1, 1,&
                                                                    57,1,53, 1,21,1, 1,1,1, 1,&
                                                                    1,58,59, 52,32,22, 23,24,25, 1,&
                                                                    1,60,1, 1,33,1, 1,1,1, 1,&
                                                                    61,1,1, 1,34,1, 1,1,1, 1,&
                                                                    1,1,1, 1,35,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(1,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupThirteen

    subroutine testOneOutflowCellSetupFourteen
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 1.5
            area_threshold = 10
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 3,2,1, 5,5,5, 5,&
                                                               4,4,4, 6,6,6, 6,6,6, 5,&
                                                               5,5,5, 7,8,3, 5,5,5, 5,&
                                                               5,5,5, 5,7,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1,    1,1,1, 1,1,1, 1,&
                                                                    1,1,1,    1,1,1, 1,1,1, 1,&
                                                                    1,1,1,    1,1,1, 1,1,1, 1,&
                                                                    1,1,1,    1,1,1, 1,1,1, 1,&
                                                                    23,24,25, 1,7,8, 9,10,11, 1,&
                                                                    1,1,1,   26,1,1, 1,1,1, 1,&
                                                                    1,1,1,   1,27,1, 1,1,1, 1,&
                                                                    1,1,1,   1,28,1, 1,1,1, 1,&
                                                                    1,1,1,   1,29,1, 1,1,1, 1, &
                                                                    1,1,1,   1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(6,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupFourteen

    subroutine testOneOutflowCellSetupFifteen
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 1.9
            area_threshold = 10
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,8, 5,5,5, 5,&
                                                               5,5,5, 5,5,8, 5,5,5, 5,&
                                                               5,5,5, 5,5,8, 5,5,5, 5,&
                                                               5,5,5, 2,1,8, 5,5,5, 5,&
                                                               4,4,4, 4,4,4, 7,5,5, 5,&
                                                               5,5,5, 8,7,6, 8,5,5, 5,&
                                                               5,5,5, 5,5,8, 8,5,5, 5,&
                                                               5,5,5, 5,5,8, 5,5,5, 5,&
                                                               5,5,5, 5,5,8, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,17, 1,1,1, 1,&
                                                                    1,1,1, 1,1,18, 1,1,1, 1,&
                                                                    1,1,1, 1,1,21, 1,1,1, 1,&
                                                                    1,1,1, 1,1,22, 1,1,1, 1,&
                                                                    10,9,8, 7,2,1, 24,1,1, 1,&
                                                                    1,1,1, 1,1,26, 25,1,1, 1,&
                                                                    1,1,1, 1,1,27, 1,1,1, 1,&
                                                                    1,1,1, 1,1,28, 1,1,1, 1,&
                                                                    1,1,1, 1,1,29, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(8,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupFifteen

    subroutine testOneOutflowCellSetupSixteen
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 2.1
            area_threshold = 10
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,8, 5,5,5, 5,&
                                                               5,5,5, 5,5,8, 5,5,5, 5,&
                                                               5,5,5, 5,5,8, 5,5,5, 5,&
                                                               5,5,5, 2,1,8, 5,5,5, 5,&
                                                               4,4,4, 4,4,4, 7,5,5, 5,&
                                                               5,5,5, 8,7,6, 8,5,5, 5,&
                                                               5,5,5, 5,5,8, 8,5,5, 5,&
                                                               5,5,5, 5,5,8, 5,5,5, 5,&
                                                               5,5,5, 5,5,8, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,17, 1,1,1, 1,&
                                                                    1,1,1, 1,1,18, 1,1,1, 1,&
                                                                    1,1,1, 1,1,21, 1,1,1, 1,&
                                                                    1,1,1, 1,1,22, 1,1,1, 1,&
                                                                    10,9,8, 7,2,1, 24,1,1, 1,&
                                                                    1,1,1, 1,1,26, 25,1,1, 1,&
                                                                    1,1,1, 1,1,27, 1,1,1, 1,&
                                                                    1,1,1, 1,1,28, 1,1,1, 1,&
                                                                    1,1,1, 1,1,29, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(4,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupSixteen

    subroutine testOneOutflowCellSetupSeventeen
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 0.9
            area_threshold = 10
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,5,5, 8,5,5, 5,5,5, 5,&
                                                               6,6,6, 8,2,1, 5,5,5, 5,&
                                                               5,5,5, 2,2,1, 1,4,4, 5,&
                                                               4,4,4, 4,4,1, 5,5,5, 5,&
                                                               5,5,5, 5,2,5, 5,5,5, 5,&
                                                               5,5,5, 5,2,5, 5,5,5, 5,&
                                                               5,5,5, 5,2,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 83,1,1, 1,1,1, 1,&
                                                                    1,1,1, 84,1,1, 1,1,1, 1,&
                                                                    1,1,1, 85,1,1, 1,1,1, 1,&
                                                                    89,88,87, 86,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,3,1, 28,29,30, 1,&
                                                                    10,9,8, 7,5,27, 1,1,1, 1,&
                                                                    1,1,1, 1,26,1, 1,1,1, 1,&
                                                                    1,1,1, 1,25,1, 1,1,1, 1,&
                                                                    1,1,1, 1,24,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(8,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupSeventeen

    subroutine testOneOutflowCellSetupEighteen
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 1.4
            area_threshold = 10
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,5,5, 8,5,5, 5,5,5, 5,&
                                                               6,6,6, 8,2,1, 5,5,5, 5,&
                                                               5,5,5, 2,2,1, 1,4,4, 5,&
                                                               4,4,4, 4,4,1, 5,5,5, 5,&
                                                               5,5,5, 5,2,5, 5,5,5, 5,&
                                                               5,5,5, 5,2,5, 5,5,5, 5,&
                                                               5,5,5, 5,2,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 83,1,1, 1,1,1, 1,&
                                                                    1,1,1, 84,1,1, 1,1,1, 1,&
                                                                    1,1,1, 85,1,1, 1,1,1, 1,&
                                                                    89,88,87, 86,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,3,1, 28,29,30, 1,&
                                                                    10,9,8, 7,5,27, 1,1,1, 1,&
                                                                    1,1,1, 1,26,1, 1,1,1, 1,&
                                                                    1,1,1, 1,25,1, 1,1,1, 1,&
                                                                    1,1,1, 1,24,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(2,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupEighteen

    subroutine testOneOutflowCellSetupNineteen
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 1.5
            area_threshold = 10
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,5,5, 8,5,5, 5,5,5, 5,&
                                                               6,6,6, 8,2,1, 5,5,5, 5,&
                                                               5,5,5, 2,2,1, 1,4,4, 5,&
                                                               4,4,4, 4,4,1, 5,5,5, 5,&
                                                               5,5,5, 5,2,5, 5,5,5, 5,&
                                                               5,5,5, 5,2,5, 5,5,5, 5,&
                                                               5,5,5, 5,2,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 83,1,1, 1,1,1, 1,&
                                                                    1,1,1, 84,1,1, 1,1,1, 1,&
                                                                    1,1,1, 85,1,1, 1,1,1, 1,&
                                                                    89,88,87, 86,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,3,1, 28,29,30, 1,&
                                                                    10,9,8, 7,5,27, 1,1,1, 1,&
                                                                    1,1,1, 1,26,1, 1,1,1, 1,&
                                                                    1,1,1, 1,25,1, 1,1,1, 1,&
                                                                    1,1,1, 1,24,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(4,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupNineteen

    subroutine testOneOutflowCellSetupTwenty
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 10.0
            area_threshold = 10
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,5,5, 8,5,5, 5,5,5, 5,&
                                                               6,6,6, 8,4,2, 5,5,5, 5,&
                                                               5,5,5, 8,7,2, 5,5,5, 5,&
                                                               5,5,5, 8,6,2, 5,5,5, 5,&
                                                               5,5,5, 5,5,2, 5,5,5, 5,&
                                                               5,5,5, 5,5,2, 5,5,5, 5,&
                                                               5,5,5, 5,5,2, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 56,1,1, 1,1,1, 1,&
                                                                    1,1,1, 55,1,1, 1,1,1, 1,&
                                                                    1,1,1, 54,1,1, 1,1,1, 1,&
                                                                    46,47,48, 53,1,1, 1,1,1, 1,&
                                                                    1,1,1, 2,1,2, 1,1,1, 1,&
                                                                    1,1,1, 1,1,5, 1,1,1, 1,&
                                                                    1,1,1, 1,1,5, 1,1,1, 1,&
                                                                    1,1,1, 1,1,7, 1,1,1, 1,&
                                                                    1,1,1, 1,1,8, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(8,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupTwenty

    subroutine testOneOutflowCellSetupTwentyOne
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 10.0
            area_threshold = 10
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,5,5, 8,5,5, 5,5,5, 5,&
                                                               5,5,5, 8,5,5, 5,5,5, 5,&
                                                               6,6,6, 8,4,2, 5,5,5, 5,&
                                                               5,5,5, 8,3,2, 5,5,5, 5,&
                                                               5,5,5, 8,6,2, 5,5,5, 5,&
                                                               5,5,5, 5,5,2, 5,5,5, 5,&
                                                               5,5,5, 5,5,2, 5,5,5, 5,&
                                                               5,5,5, 5,5,2, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 56,1,1, 1,1,1, 1,&
                                                                    1,1,1, 55,1,1, 1,1,1, 1,&
                                                                    1,1,1, 54,1,1, 1,1,1, 1,&
                                                                    46,47,48, 53,1,1, 1,1,1, 1,&
                                                                    1,1,1, 2,1,2, 1,1,1, 1,&
                                                                    1,1,1, 1,1,5, 1,1,1, 1,&
                                                                    1,1,1, 1,1,5, 1,1,1, 1,&
                                                                    1,1,1, 1,1,7, 1,1,1, 1,&
                                                                    1,1,1, 1,1,8, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(2,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupTwentyOne

    subroutine testOneOutflowCellSetupTwentyTwo
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 10.0
            area_threshold = 10
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 7,8,9, 5,5,5, 5,&
                                                               5,5,5, 4,5,6, 5,5,5, 5,&
                                                               5,5,5, 1,2,3, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,9,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(5,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupTwentyTwo

    subroutine testOneOutflowCellSetupTwentyThree
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 10.0
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,8, 8,5,5, 5,&
                                                               5,5,5, 5,5,8, 8,5,5, 5,&
                                                               5,5,5, 5,5,9, 9,5,5, 5,&
                                                               5,5,6, 6,9,9, 5,5,5, 5,&
                                                               5,5,5, 4,5,6, 5,5,5, 5,&
                                                               5,5,5, 1,2,3, 5,5,5, 5,&
                                                               5,5,5, 5,2,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,6, 4,1,1, 1,&
                                                                    1,1,1, 1,1,5, 3,1,1, 1,&
                                                                    1,1,1, 1,1,4, 2,1,1, 1,&
                                                                    1,1,1, 2,3,1, 1,1,1, 1,&
                                                                    1,1,1, 1,7,1, 1,1,1, 1,&
                                                                    1,1,1, 1,2,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(5,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupTwentyThree

    subroutine testOneOutflowCellSetupTwentyFour
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 10.0
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,8, 8,5,5, 5,&
                                                               5,5,5, 5,5,8, 8,5,5, 5,&
                                                               5,5,5, 5,5,9, 9,5,5, 5,&
                                                               5,5,6, 6,9,9, 5,5,5, 5,&
                                                               5,5,5, 4,6,5, 5,5,5, 5,&
                                                               5,5,5, 1,2,3, 5,5,5, 5,&
                                                               5,5,5, 5,2,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,6, 4,1,1, 1,&
                                                                    1,1,1, 1,1,5, 3,1,1, 1,&
                                                                    1,1,1, 1,1,4, 2,1,1, 1,&
                                                                    1,1,1, 2,3,1, 1,1,1, 1,&
                                                                    1,1,1, 1,6,7, 1,1,1, 1,&
                                                                    1,1,1, 1,2,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(5,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupTwentyFour

    subroutine testOneOutflowCellSetupTwentyFive
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 10.0
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,8, 8,5,5, 5,&
                                                               5,5,5, 5,5,8, 8,5,5, 5,&
                                                               5,5,5, 5,5,8, 9,5,5, 5,&
                                                               6,6,6, 6,9,9, 5,5,5, 5,&
                                                               5,5,5, 4,6,5, 5,5,5, 5,&
                                                               5,5,5, 1,2,3, 5,5,5, 5,&
                                                               5,5,5, 5,2,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,4, 4,1,1, 1,&
                                                                    1,1,1, 1,1,5, 3,1,1, 1,&
                                                                    1,1,1, 1,1,6, 2,1,1, 1,&
                                                                    11,10,9, 8,7,1, 1,1,1, 1,&
                                                                    1,1,1, 1,6,7, 1,1,1, 1,&
                                                                    1,1,1, 1,2,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(8,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupTwentyFive

    subroutine testOneOutflowCellSetupTwentySix
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 10.0
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,8, 8,5,5, 5,&
                                                               5,5,5, 5,5,8, 8,5,5, 5,&
                                                               5,5,5, 5,5,8, 9,5,5, 5,&
                                                               6,6,6, 6,9,9, 5,5,5, 5,&
                                                               5,5,5, 4,6,5, 5,5,5, 5,&
                                                               5,5,5, 1,2,3, 5,5,5, 5,&
                                                               5,5,5, 5,2,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,5, 4,1,1, 1,&
                                                                    1,1,1, 1,1,6, 3,1,1, 1,&
                                                                    1,1,1, 1,1,7, 2,1,1, 1,&
                                                                    12,11,10, 9,8,1, 1,1,1, 1,&
                                                                    1,1,1, 1,6,7, 1,1,1, 1,&
                                                                    1,1,1, 1,8,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(8,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupTwentySix

    subroutine testOneOutflowCellSetupTwentySeven
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 10.0
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,8, 8,5,5, 5,&
                                                               5,5,5, 5,5,8, 8,5,5, 5,&
                                                               5,5,5, 5,5,8, 9,5,5, 5,&
                                                               6,6,6, 6,9,9, 5,5,5, 5,&
                                                               5,5,5, 2,6,5, 4,4,4, 5,&
                                                               5,5,5, 5,4,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,19, 4,1,1, 1,&
                                                                    1,1,1, 1,1,18, 3,1,1, 1,&
                                                                    1,1,1, 1,1,17, 2,1,1, 1,&
                                                                    12,13,14, 15,16,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,21, 19,18,17, 1,&
                                                                    1,1,1, 23,22,1, 1,1,1, 1,&
                                                                    1,1,1, 1,21,1, 1,1,1, 1,&
                                                                    1,1,1, 1,20,1, 1,1,1, 1,&
                                                                    1,1,1, 1,19,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(5,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupTwentySeven

    subroutine testOneOutflowCellSetupTwentyEight
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 10.0
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,8, 8,5,5, 5,&
                                                               5,5,5, 5,5,8, 8,5,5, 5,&
                                                               5,5,5, 5,5,8, 9,5,5, 5,&
                                                               6,6,6, 6,9,9, 5,5,5, 5,&
                                                               5,5,5, 2,6,5, 4,4,4, 5,&
                                                               5,5,5, 5,4,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,19, 4,1,1, 1,&
                                                                    1,1,1, 1,1,18, 3,1,1, 1,&
                                                                    1,1,1, 1,1,17, 2,1,1, 1,&
                                                                    12,13,14, 15,16,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,4, 3,2,1, 1,&
                                                                    1,1,1, 23,22,1, 1,1,1, 1,&
                                                                    1,1,1, 1,21,1, 1,1,1, 1,&
                                                                    1,1,1, 1,20,1, 1,1,1, 1,&
                                                                    1,1,1, 1,19,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(5,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupTwentyEight

    subroutine testOneOutflowCellSetupTwentyNine
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 10.0
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,8, 5,5,1, 5,&
                                                               5,5,5, 5,5,8, 5,1,5, 5,&
                                                               5,5,5, 5,5,8, 1,5,5, 5,&
                                                               6,6,6, 6,9,8, 5,5,5, 5,&
                                                               5,5,5, 2,6,5, 4,4,4, 5,&
                                                               5,5,5, 5,4,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,51, 4,1,33, 1,&
                                                                    1,1,1, 1,1,50, 1,32,1, 1,&
                                                                    1,1,1, 1,1,49, 31,1,1, 1,&
                                                                    12,13,14, 15,32,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,4, 3,2,1, 1,&
                                                                    1,1,1, 6,5,1, 1,1,1, 1,&
                                                                    1,1,1, 1,4,1, 1,1,1, 1,&
                                                                    1,1,1, 1,3,1, 1,1,1, 1,&
                                                                    1,1,1, 1,2,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(8,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupTwentyNine

    subroutine testOneOutflowCellSetupThirty
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 1.1
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,8, 5,5,1, 5,&
                                                               5,5,5, 5,5,8, 5,1,5, 5,&
                                                               5,5,5, 5,5,8, 1,5,5, 5,&
                                                               6,6,6, 6,8,8, 5,5,5, 5,&
                                                               5,5,5, 2,6,5, 4,4,4, 5,&
                                                               5,5,5, 6,2,4, 5,5,5, 5,&
                                                               5,5,5, 5,2,5, 5,5,5, 5,&
                                                               5,5,5, 5,2,5, 5,5,5, 5,&
                                                               5,5,5, 5,2,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,51, 4,1,33, 1,&
                                                                    1,1,1, 1,1,50, 1,32,1, 1,&
                                                                    1,1,1, 1,1,49, 31,1,1, 1,&
                                                                    12,13,14, 15,16,48, 1,1,1, 1,&
                                                                    1,1,1, 1,1,24, 23,22,21, 1,&
                                                                    1,1,1, 1,4,1, 1,1,1, 1,&
                                                                    1,1,1, 1,5,1, 1,1,1, 1,&
                                                                    1,1,1, 1,6,1, 1,1,1, 1,&
                                                                    1,1,1, 1,7,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(5,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupThirty

    subroutine testOneOutflowCellSetupThirtyOne
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 1.1
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,4,5, 1,4,5, 5,&
                                                               5,5,5, 5,5,7, 5,5,7, 5,&
                                                               5,5,5, 6,6,3, 5,5,8, 5,&
                                                               5,5,5, 6,9,8, 6,9,5, 5,&
                                                               5,5,5, 8,9,8, 5,5,5, 5,&
                                                               6,6,9, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 21,20,1, 18,17,1, 1,&
                                                                    1,1,1, 1,1,19, 1,1,16, 1,&
                                                                    1,1,1, 1,2,12, 1,1,15, 1,&
                                                                    1,1,1, 5,6,3, 13,14,1, 1,&
                                                                    1,1,1, 4,1,1, 1,1,1, 1,&
                                                                    1,2,3, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(8,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupThirtyOne

    subroutine testOneOutflowCellSetupThirtyTwo
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 1.1
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, -1,4,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,7, 5,&
                                                               5,5,5, 6,6,3, 5,5,8, 5,&
                                                               5,5,5, 6,9,8, 6,9,5, 5,&
                                                               5,5,5, 8,9,8, 5,5,5, 5,&
                                                               6,6,9, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 18,17,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,16, 1,&
                                                                    1,1,1, 1,2,12, 1,1,15, 1,&
                                                                    1,1,1, 5,6,3, 13,14,1, 1,&
                                                                    1,1,1, 4,1,1, 1,1,1, 1,&
                                                                    1,2,3, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(9,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupThirtyTwo

        subroutine testOneOutflowCellSetupThirtyThree
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 1.1
            area_threshold = 1
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, -1,4,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,7, 5,&
                                                               5,5,5, 6,6,3, 5,5,8, 5,&
                                                               5,5,5, 6,9,8, 6,9,5, 5,&
                                                               5,5,5, 8,9,8, 5,5,5, 5,&
                                                               6,6,9, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 18,17,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,16, 1,&
                                                                    1,1,1, 1,2,12, 1,1,15, 1,&
                                                                    1,1,1, 5,6,3, 13,14,1, 1,&
                                                                    1,1,1, 4,1,1, 1,1,1, 1,&
                                                                    1,2,3, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(6,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupThirtyThree

    subroutine testOneOutflowCellSetupThirtyFour
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 1.1
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 6,6,3, 5,5,5, 5,&
                                                               5,5,5, 6,9,8, 0,5,5, 5,&
                                                               5,5,5, 8,9,8, 5,5,5, 5,&
                                                               6,6,9, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,2,12, 1,1,1, 1,&
                                                                    1,1,1, 5,6,3, 13,1,1, 1,&
                                                                    1,1,1, 4,1,1, 1,1,1, 1,&
                                                                    1,2,3, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(6,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupThirtyFour

    subroutine testOneOutflowCellSetupThirtyFive
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 1.1
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,2, 4,4,4, 5,&
                                                               5,5,5, 8,8,2, 5,5,5, 5,&
                                                               5,5,5, 8,8,5, 5,5,5, 5,&
                                                               6,6,9, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,14,1, 1,1,1, 1,&
                                                                    1,1,1, 1,13,1, 1,1,1, 1,&
                                                                    1,1,1, 1,12,1, 1,1,1, 1,&
                                                                    1,1,1, 6,11,4, 3,2,1, 1,&
                                                                    1,1,1, 5,10,5, 1,1,1, 1,&
                                                                    1,1,1, 4,9,6, 1,1,1, 1,&
                                                                    1,2,3, 1,8,1, 1,1,1, 1,&
                                                                    1,1,1, 1,7,1, 1,1,1, 1,&
                                                                    1,1,1, 1,6,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(5,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupThirtyFive

    subroutine testOneOutflowCellSetupThirtySix
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 1.1
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,2, 4,4,4, 5,&
                                                               5,5,5, 8,8,2, 5,5,5, 5,&
                                                               5,5,5, 8,8,5, 5,5,5, 5,&
                                                               6,6,9, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,14,1, 1,1,1, 1,&
                                                                    1,1,1, 1,13,1, 1,1,1, 1,&
                                                                    1,1,1, 1,12,1, 1,1,1, 1,&
                                                                    1,1,1, 6,11,4, 3,2,1, 1,&
                                                                    1,1,1, 5,10,5, 1,1,1, 1,&
                                                                    1,1,1, 4,9,6, 1,1,1, 1,&
                                                                    1,2,3, 1,8,1, 1,1,1, 1,&
                                                                    1,1,1, 1,7,1, 1,1,1, 1,&
                                                                    1,1,1, 1,6,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(5,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupThirtySix

    subroutine testOneOutflowCellSetupThirtySeven
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 1.1
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,2, 4,4,4, 5,&
                                                               5,5,5, 8,8,2, 5,5,5, 5,&
                                                               5,5,5, 8,8,5, 5,5,5, 5,&
                                                               6,6,9, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,14,1, 1,1,1, 1,&
                                                                    1,1,1, 1,13,1, 1,1,1, 1,&
                                                                    1,1,1, 1,12,1, 1,1,1, 1,&
                                                                    1,1,1, 6,12,4, 3,2,1, 1,&
                                                                    1,1,1, 5,11,5, 1,1,1, 1,&
                                                                    1,1,1, 4,10,6, 1,1,1, 1,&
                                                                    1,2,3, 1,9,1, 1,1,1, 1,&
                                                                    1,1,1, 1,8,1, 1,1,1, 1,&
                                                                    1,1,1, 1,7,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(8,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupThirtySeven

    subroutine testOneOutflowCellSetupThirtyEight
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 1.1
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,2, 4,4,4, 5,&
                                                               5,5,5, 8,8,2, 5,5,5, 5,&
                                                               5,5,5, 8,8,5, 5,5,5, 5,&
                                                               6,6,9, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,16,1, 1,1,1, 1,&
                                                                    1,1,1, 1,15,1, 1,1,1, 1,&
                                                                    1,1,1, 1,14,1, 1,1,1, 1,&
                                                                    1,1,1, 6,13,4, 3,2,1, 1,&
                                                                    1,1,1, 5,12,5, 1,1,1, 1,&
                                                                    1,1,1, 4,11,6, 1,1,1, 1,&
                                                                    1,2,3, 1,10,1, 1,1,1, 1,&
                                                                    1,1,1, 1,9,1, 1,1,1, 1,&
                                                                    1,1,1, 1,8,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(8,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupThirtyEight

    subroutine testOneOutflowCellSetupThirtyNine
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 1.1
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, -1,-1,0, 5,5,5, 5,&
                                                               5,5,5, 0,0,8, 5,5,5, 5,&
                                                               5,5,5, 8,8,8, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,2, 1,1,1, 1,&
                                                                    1,1,1, 2,5,1, 1,1,1, 1,&
                                                                    1,1,1, 1,4,1, 1,1,1, 1,&
                                                                    1,1,1, 1,3,1, 1,1,1, 1,&
                                                                    1,1,1, 1,2,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            call dir_based_rdirs_cell%set_contains_river_mouths(.TRUE.)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(0,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupThirtyNine

    subroutine testOneOutflowCellSetupForty
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 1.1
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, -1,6,0, 5,5,5, 5,&
                                                               5,5,5, 0,8,8, 5,5,5, 5,&
                                                               5,5,5, 8,8,8, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,6,8, 1,1,1, 1,&
                                                                    1,1,1, 2,5,2, 1,1,1, 1,&
                                                                    1,1,1, 1,4,1, 1,1,1, 1,&
                                                                    1,1,1, 1,3,1, 1,1,1, 1,&
                                                                    1,1,1, 1,2,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            call dir_based_rdirs_cell%set_contains_river_mouths(.TRUE.)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(0,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupForty

    subroutine testOneOutflowCellSetupFortyOne
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 0.9
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, -1,6,0, 5,5,5, 5,&
                                                               5,5,5, 0,8,8, 5,5,5, 5,&
                                                               5,5,5, 8,8,6, 5,5,5, 5,&
                                                               5,5,5, 5,9,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,3,5, 1,1,1, 1,&
                                                                    1,1,1, 2,2,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,6, 1,1,1, 1,&
                                                                    1,1,1, 1,5,1, 1,1,1, 1,&
                                                                    1,1,1, 1,4,1, 1,1,1, 1,&
                                                                    1,1,1, 1,3,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            call dir_based_rdirs_cell%set_contains_river_mouths(.TRUE.)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(0,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupFortyOne

    subroutine testOneOutflowCellSetupFortyTwo
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 0.9
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, -1,6,0, 5,5,5, 5,&
                                                               5,5,5, 5,8,8, 5,5,5, 5,&
                                                               5,5,5, 8,8,6, 5,5,5, 5,&
                                                               5,5,5, 5,9,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,3,5, 1,1,1, 1,&
                                                                    1,1,1, 2,2,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,5, 1,1,1, 1,&
                                                                    1,1,1, 1,4,1, 1,1,1, 1,&
                                                                    1,1,1, 1,3,1, 1,1,1, 1,&
                                                                    1,1,1, 1,2,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            call dir_based_rdirs_cell%set_contains_river_mouths(.TRUE.)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(6,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupFortyTwo

    subroutine testOneOutflowCellSetupFortyThree
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 0.9
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, -1,6,0, 5,5,5, 5,&
                                                               5,5,5, 5,8,8, 5,5,5, 5,&
                                                               5,5,5, 8,8,6, 5,5,5, 5,&
                                                               5,5,5, 5,9,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,3,5, 1,1,1, 1,&
                                                                    1,1,1, 2,2,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,6, 1,1,1, 1,&
                                                                    1,1,1, 1,5,1, 1,1,1, 1,&
                                                                    1,1,1, 1,4,1, 1,1,1, 1,&
                                                                    1,1,1, 1,3,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            call dir_based_rdirs_cell%set_contains_river_mouths(.TRUE.)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(6,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupFortyThree

    subroutine testOneOutflowCellSetupFortyFour
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 0.9
            area_threshold = 10
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5,&
                                                               5,5,5, -1,6,0, 5,5,5, 5,&
                                                               5,5,5, 5,8,8, 5,5,5, 5,&
                                                               6,6,6, 8,8,6, 5,5,5, 5,&
                                                               5,5,5, 5,9,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,8,5, 5,5,5, 5,&
                                                               5,5,5, 5,5,5, 5,5,5, 5 /),&
                                           shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,1,1, 1,1,1, 1,&
                                                                    1,1,1, 1,3,5, 1,1,1, 1,&
                                                                    1,1,1, 25,2,1, 1,1,1, 1,&
                                                                    21,22,23, 24,1,6, 1,1,1, 1,&
                                                                    1,1,1, 1,5,1, 1,1,1, 1,&
                                                                    1,1,1, 1,4,1, 1,1,1, 1,&
                                                                    1,1,1, 1,3,1, 1,1,1, 1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            call dir_based_rdirs_cell%set_contains_river_mouths(.TRUE.)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(5,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupFortyFour

    subroutine testOneOutflowCellSetupFortyFive
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 1.5
            area_threshold = 9
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 3, 2, 1, 1, 1, 1, 9, 9, 9,5,&
                                                               3, 2, 1, 1, 2, 1, 1, 9, 9,5,&
                                                               3, 2, 1, 2, 1, 4, 1, 1, 9,5,&
                                                               6, 3, 2, 1, 1, 3, 3, 2, 1,5,&
                                                               2 ,6, 3, 2, 1, 6, 3, 3, 2,5,&
                                                               3, 9, 6, 3, 2, 1, 1, 2, 1,5,&
                                                               6 ,9, 9, 3, 2, 1, 3, 3, 2,5,&
                                                               9, 8, 7, 6, 3, 3, 3, 3, 3,5,&
                                                               8, 7, 7, 9, 6, 3, 3, 6, 6,5,&
                                                               5, 5, 5, 5, 5, 5, 5, 5, 5,5 /),&
                                                    shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,15, 4, 1, 2, 1, 1, 1, 2,5,&
                                                                    2,21, 2, 3, 2, 1, 1, 1, 1,5,&
                                                                    2,26, 4, 1, 6, 2, 1, 1, 1,5,&
                                                                    4,37, 1, 8, 1, 2, 2, 1, 1,5,&
                                                                    1, 1,49, 2, 1, 1, 4, 5, 1,5,&
                                                                    9, 1,  210,  264, 1, 1, 1, 5,10,5,&
                                                                    1,  209, 1, 1,  267, 2, 1,16, 1,5,&
                                                                    195, 2, 1, 1,  273, 1, 1, 2,20,5,&
                                                                    1, 1, 1, 1, 2, 278, 2, 2, 1053,5,&
                                                                    5, 5, 5, 5, 5, 5, 5, 5, 5,5 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            call dir_based_rdirs_cell%set_contains_river_mouths(.TRUE.)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(2,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupFortyFive

    subroutine testOneOutflowCellSetupFortySix
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer :: result0
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
            !Extra lon column acts as buffer. Extra row makes specification of values simpler
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            MUFP = 1.5
            area_threshold = 9
            run_check_for_sinks = .True.
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 5, 5, 2, 5, 5, 5, 5, 5, 5, 5,&
                                                               5, 5, 2, 5, 5, 5, 5, 5, 5, 5,&
                                                               5, 5, 3, 2, 5, 5, 5, 5, 5, 5,&
                                                               5, 5, 1, 4, 7, 4, 5, 5, 5, 5,&
                                                               5 ,1, 5, 6, 8, 8, 5, 5, 5, 5,&
                                                               1, 5, 5, 6, 9, 8, 5, 5, 5, 5,&
                                                               5 ,5, 5, 5, 5, 8, 5, 5, 5, 5,&
                                                               5, 5, 5, 5, 5, 5, 5, 5, 5, 5,&
                                                               5, 5, 5, 5, 5, 5, 5, 5, 5, 5,&
                                                               5, 5, 5, 5, 5, 5, 5, 5, 5, 5 /),&
                                                    shape(transpose(input_fine_river_directions))))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1, 1,31, 1, 1, 1, 1, 1, 1, 1,&
                                                                    1, 1,32, 1, 1, 1, 1, 1, 1, 1,&
                                                                    1, 1,33,10, 1, 1, 1, 1, 1, 1,&
                                                                    1, 1,45,44, 9, 4, 1, 1, 1, 1,&
                                                                    1,46, 1, 3, 4, 3, 1, 1, 1, 1,&
                                                                   47, 1, 1, 1, 2, 2, 1, 1, 1, 1,&
                                                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,&
                                                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,&
                                                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,&
                                                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1 /),&
                                                         shape(transpose(input_fine_river_directions))))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            call dir_based_rdirs_cell%set_contains_river_mouths(.TRUE.)
            output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
            select type (output_coarse_river_direction)
            type is (dir_based_direction_indicator)
                result0 = output_coarse_river_direction%get_direction()
            end select
            call assert_equals(4,result0)
            call dir_based_rdirs_cell%destructor()
            deallocate(output_coarse_river_direction)
    end subroutine testOneOutflowCellSetupFortySix

    subroutine testCellCumulativeFlowGeneration
     use area_mod
        use coords_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(3,3) :: expected_cell_cumulative_flow_result
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
        class(*), dimension(:,:), pointer :: cmltv_flow_array
            expected_cell_cumulative_flow_result = transpose(reshape((/ 1,4,9, &
                                                                        1,3,4, &
                                                                        1,2,1 /),&
                                                   shape(expected_cell_cumulative_flow_result)))
            allocate(input_fine_river_directions(9,9))
            allocate(input_fine_total_cumulative_flow(9,9))
            allocate(dir_based_direction_indicator::output_coarse_river_direction)
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 4,4,4, 4,4,4, 4,6,6,&
                                                               4,4,4, 4,4,4, 4,7,6,&
                                                               4,4,4, 4,4,6, 9,6,6,&
                                                               5,5,5, 3,6,9, 5,5,5,&
                                                               5,5,6, 6,8,8, 5,5,5,&
                                                               5,5,5, 6,9,8, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5 /),&
                                           shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 17,16,15, 14,13,12, 11,1,1,&
                                                                    1,1,1, 1,1,1, 1,10,1,&
                                                                    1,1,1, 1,1,1, 9,1,1,&
                                                                    1,1,1, 1,3,8, 1,1,1,&
                                                                    1,1,1, 1,1,4, 1,1,1,&
                                                                    1,1,1, 1,2,3, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            cmltv_flow_array => dir_based_rdirs_cell%test_generate_cell_cumulative_flow()
            select type (cmltv_flow_array)
            type is (integer)
                call assert_true(all(expected_cell_cumulative_flow_result .eq. cmltv_flow_array))
            class default
                call assert_true(.False.)
            end select
            deallocate(cmltv_flow_array)
            deallocate(input_fine_river_directions)
            deallocate(input_fine_total_cumulative_flow)
            deallocate(output_coarse_river_direction)
            call dir_based_rdirs_cell%destructor()
    end subroutine testCellCumulativeFlowGeneration

    subroutine testCellCumulativeFlowGenerationTwo
        use area_mod
        use coords_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(3,3) :: expected_cell_cumulative_flow_result
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
        class(*), dimension(:,:), pointer :: cmltv_flow_array
            expected_cell_cumulative_flow_result = transpose(reshape((/ 1,1,1, &
                                                                        2,1,1, &
                                                                        4,1,1 /),&
                                                   shape(expected_cell_cumulative_flow_result)))
            allocate(input_fine_river_directions(10,10))
            allocate(input_fine_total_cumulative_flow(10,10))
            allocate(dir_based_direction_indicator::output_coarse_river_direction)
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_fine_river_directions = transpose(reshape((/ 3, 2, 1, 1, 1, 1, 9, 9, 9,5,&
                                                               3, 2, 1, 1, 2, 1, 1, 9, 9,5,&
                                                               3, 2, 1, 2, 1, 4, 1, 1, 9,5,&
                                                               6, 3, 2, 1, 1, 3, 3, 2, 1,5,&
                                                               2 ,6, 3, 2, 1, 6, 3, 3, 2,5,&
                                                               3, 9, 6, 3, 2, 1, 1, 2, 1,5,&
                                                               6 ,9, 9, 3, 2, 1, 3, 3, 2,5,&
                                                               9, 8, 7, 6, 3, 3, 3, 3, 3,5,&
                                                               8, 7, 7, 9, 6, 3, 3, 6, 6,5,&
                                                               5, 5, 5, 5, 5, 5, 5, 5, 5,5 /),&
                                                    shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,15, 4, 1, 2, 1, 1, 1, 2,5,&
                                                                    2,21, 2, 3, 2, 1, 1, 1, 1,5,&
                                                                    2,26, 4, 1, 6, 2, 1, 1, 1,5,&
                                                                    4,37, 1, 8, 1, 2, 2, 1, 1,5,&
                                                                    1, 1,49, 2, 1, 1, 4, 5, 1,5,&
                                                                    9, 1,  210,  264, 1, 1, 1, 5,10,5,&
                                                                    1,  209, 1, 1,  267, 2, 1,16, 1,5,&
                                                                    195, 2, 1, 1,  273, 1, 1, 2,20,5,&
                                                                    1, 1, 1, 1, 2, 278, 2, 2, 1053,5,&
                                                                    5, 5, 5, 5, 5, 5, 5, 5, 5,5 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow)
            cmltv_flow_array => dir_based_rdirs_cell%test_generate_cell_cumulative_flow()
            select type (cmltv_flow_array)
            type is (integer)
                call assert_true(all(expected_cell_cumulative_flow_result .eq. cmltv_flow_array))
            class default
                call assert_true(.False.)
            end select
            deallocate(cmltv_flow_array)
            deallocate(input_fine_river_directions)
            deallocate(input_fine_total_cumulative_flow)
            deallocate(output_coarse_river_direction)
            call dir_based_rdirs_cell%destructor()
    end subroutine testCellCumulativeFlowGenerationTwo

    subroutine testCheckFieldForLocalizedLoops
        use area_mod
        use coords_mod
        integer, dimension(:,:), allocatable :: river_directions
        logical, dimension(:,:), allocatable :: expected_results
        logical, dimension(:,:), pointer     :: cells_to_reprocess
        type(latlon_dir_based_rdirs_field) :: dir_based_rdirs_field
        type(latlon_section_coords) :: field_section_coords
            allocate(river_directions(5,5))
            allocate(expected_results(5,5))
            river_directions = transpose(reshape((/ 7,6,3,5,2,&
                                                    7,6,4,7,8,&
                                                    4,5,1,0,6,&
                                                    3,9,8,1,3,&
                                                    4,7,-1,1,3 /),&
                                         shape(river_directions)))
            expected_results = transpose(reshape((/ .False.,.False.,.True.,.False.,.True.,&
                                                    .False.,.True.,.True.,.True.,.True.,&
                                                    .True.,.False.,.True.,.False.,.True.,&
                                                    .True.,.True.,.False.,.False.,.False.,&
                                                    .False.,.True.,.False.,.False.,.False. /),&
                                         shape(river_directions)))
            field_section_coords = latlon_section_coords(1,1,5,5)
            dir_based_rdirs_field = latlon_dir_based_rdirs_field(field_section_coords,river_directions)
            cells_to_reprocess => dir_based_rdirs_field%check_field_for_localized_loops()
            call assert_equals(expected_results,cells_to_reprocess,5,5)
            deallocate(cells_to_reprocess)
            call dir_based_rdirs_field%destructor()
    end subroutine testCheckFieldForLocalizedLoops

    subroutine testYamazakiFindDownstreamCellOne
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            allocate(input_fine_river_directions(9,9))
            allocate(input_fine_total_cumulative_flow(9,9))
            allocate(input_yamazaki_outlet_pixels(9,9))
            allocate(initial_outlet_pixel,source=latlon_coords(4,6))
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 4,4,4, 4,4,4, 4,6,6,&
                                                               4,4,4, 4,4,4, 4,7,6,&
                                                               4,4,4, 4,4,6, 9,6,6,&
                                                               5,5,5, 3,6,9, 5,5,5,&
                                                               5,5,6, 6,8,8, 5,5,5,&
                                                               5,5,5, 6,9,8, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5 /),&
                                          shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 19,18,17, 16,15,14, 13,1,1,&
                                                                    1,1,1, 1,1,1, 1,12,1,&
                                                                    1,1,1, 1,1,1, 11,1,1,&
                                                                    1,1,1, 1,5,10, 1,1,1,&
                                                                    1,1,1, 2,4,4, 1,1,1,&
                                                                    1,1,1, 1,2,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1 /),&
                                                         shape(input_fine_river_directions)))
            input_yamazaki_outlet_pixels = transpose(reshape((/ 1,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,1, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,1)
                call assert_equals(result%lon,1)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellOne

    subroutine testYamazakiFindDownstreamCellTwo
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            allocate(input_fine_river_directions(9,9))
            allocate(input_fine_total_cumulative_flow(9,9))
            allocate(input_yamazaki_outlet_pixels(9,9))
            allocate(initial_outlet_pixel,source=latlon_coords(4,6))
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 4,4,4, 4,4,4, 4,6,6,&
                                                               4,4,4, 4,4,4, 4,7,6,&
                                                               4,4,4, 4,4,6, 9,6,6,&
                                                               5,5,5, 3,6,9, 5,5,5,&
                                                               5,5,6, 6,8,8, 5,5,5,&
                                                               5,5,5, 6,9,8, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5 /),&
                                          shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 19,18,17, 16,15,14, 13,1,1,&
                                                                    1,1,1, 1,1,1, 1,12,1,&
                                                                    1,1,1, 1,1,1, 11,1,1,&
                                                                    1,1,1, 1,5,10, 1,1,1,&
                                                                    1,1,1, 2,4,4, 1,1,1,&
                                                                    1,1,1, 1,2,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1 /),&
                                                         shape(input_fine_river_directions)))
            input_yamazaki_outlet_pixels = transpose(reshape((/ 1,0,0, 1,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,1, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,1)
                call assert_equals(result%lon,2)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellTwo

    subroutine testYamazakiFindDownstreamCellThree
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            allocate(input_fine_river_directions(9,9))
            allocate(input_fine_total_cumulative_flow(9,9))
            allocate(input_yamazaki_outlet_pixels(9,9))
            allocate(initial_outlet_pixel,source=latlon_coords(4,6))
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 4,4,4, 4,4,4, 4,6,6,&
                                                               4,4,4, 4,4,4, 4,7,6,&
                                                               4,4,4, 4,4,6, 9,6,6,&
                                                               5,5,5, 3,6,9, 5,5,5,&
                                                               5,5,6, 6,8,8, 5,5,5,&
                                                               5,5,5, 6,9,8, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5 /),&
                                          shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 19,18,17, 16,15,14, 13,1,1,&
                                                                    1,1,1, 1,1,1, 1,12,1,&
                                                                    1,1,1, 1,1,1, 11,1,1,&
                                                                    1,1,1, 1,5,10, 1,1,1,&
                                                                    1,1,1, 2,4,4, 1,1,1,&
                                                                    1,1,1, 1,2,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1 /),&
                                                         shape(input_fine_river_directions)))
            input_yamazaki_outlet_pixels = transpose(reshape((/ 1,0,0, 0,0,0, 1,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,1, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,1)
                call assert_equals(result%lon,3)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellThree

    subroutine testYamazakiFindDownstreamCellFour
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            allocate(input_fine_river_directions(9,9))
            allocate(input_fine_total_cumulative_flow(9,9))
            allocate(input_yamazaki_outlet_pixels(9,9))
            allocate(initial_outlet_pixel,source=latlon_coords(4,6))
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 4,4,4, 4,4,4, 4,4,6,&
                                                               4,4,4, 4,4,4, 4,4,6,&
                                                               4,4,4, 4,4,4, 4,4,6,&
                                                               5,5,5, 3,6,6, 3,5,5,&
                                                               5,5,6, 6,8,8, 5,3,5,&
                                                               5,5,5, 6,9,8, 5,5,1,&
                                                               5,5,5, 5,5,5, 5,1,5,&
                                                               4,4,4, 4,4,4, 4,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5 /),&
                                          shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,5,10, 11,1,1,&
                                                                    1,1,1, 2,4,4, 1,12,1,&
                                                                    1,1,1, 1,2,1, 1,1,13,&
                                                                    1,1,1, 1,1,1, 1,14,1,&
                                                                    21,20,19, 18,17,16, 15,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1 /),&
                                                         shape(input_fine_river_directions)))
            input_yamazaki_outlet_pixels = transpose(reshape((/ 1,0,0, 0,0,0, 1,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,1, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 1,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,3)
                call assert_equals(result%lon,2)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellFour

    subroutine testYamazakiFindDownstreamCellFive
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            allocate(input_fine_river_directions(9,9))
            allocate(input_fine_total_cumulative_flow(9,9))
            allocate(input_yamazaki_outlet_pixels(9,9))
            allocate(initial_outlet_pixel,source=latlon_coords(4,6))
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 4,4,4, 4,4,4, 4,4,6,&
                                                               4,4,4, 4,4,4, 4,4,6,&
                                                               4,4,4, 4,4,4, 4,4,6,&
                                                               5,5,5, 3,6,6, 3,5,5,&
                                                               5,5,6, 6,8,8, 5,3,5,&
                                                               5,5,5, 6,9,8, 5,5,1,&
                                                               5,5,5, 5,5,5, 5,1,5,&
                                                               4,4,4, 4,4,4, 4,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5 /),&
                                          shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,5,10, 11,1,1,&
                                                                    1,1,1, 2,4,4, 1,12,1,&
                                                                    1,1,1, 1,2,1, 1,1,13,&
                                                                    1,1,1, 1,1,1, 1,14,1,&
                                                                    21,20,19, 18,17,16, 15,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1 /),&
                                                         shape(input_fine_river_directions)))
            input_yamazaki_outlet_pixels = transpose(reshape((/ 1,0,0, 0,0,0, 1,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,1, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                1,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,3)
                call assert_equals(result%lon,1)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellFive

    subroutine testYamazakiFindDownstreamCellSix
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            allocate(input_fine_river_directions(9,9))
            allocate(input_fine_total_cumulative_flow(9,9))
            allocate(input_yamazaki_outlet_pixels(9,9))
            allocate(initial_outlet_pixel,source=latlon_coords(4,6))
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 4,4,4, 4,4,4, 4,4,6,&
                                                               4,4,4, 4,4,4, 4,4,6,&
                                                               4,4,4, 4,4,4, 4,4,6,&
                                                               5,5,5, 3,6,6, 3,5,5,&
                                                               5,5,6, 6,8,8, 5,3,5,&
                                                               5,5,5, 6,9,8, 5,5,1,&
                                                               5,5,5, 5,5,5, 5,1,5,&
                                                               4,4,4, 4,4,4, 4,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5 /),&
                                          shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,5,10, 11,1,1,&
                                                                    1,1,1, 2,4,4, 1,12,1,&
                                                                    1,1,1, 1,2,1, 1,1,13,&
                                                                    1,1,1, 1,1,1, 1,14,1,&
                                                                    21,20,19, 18,17,16, 15,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1 /),&
                                                         shape(input_fine_river_directions)))
            input_yamazaki_outlet_pixels = transpose(reshape((/ 1,0,0, 0,0,0, 1,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,1, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 1,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,3)
                call assert_equals(result%lon,3)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellSix

    subroutine testYamazakiFindDownstreamCellSeven
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            allocate(input_fine_river_directions(9,9))
            allocate(input_fine_total_cumulative_flow(9,9))
            allocate(input_yamazaki_outlet_pixels(9,9))
            allocate(initial_outlet_pixel,source=latlon_coords(4,6))
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 4,4,4, 4,4,4, 4,4,6,&
                                                               4,4,4, 4,4,4, 4,4,6,&
                                                               4,4,4, 4,4,4, 4,4,6,&
                                                               5,5,5, 3,6,6, 3,5,5,&
                                                               5,5,6, 6,8,8, 5,3,5,&
                                                               5,5,5, 6,9,8, 5,5,1,&
                                                               5,5,5, 5,5,5, 5,1,5,&
                                                               4,4,4, 4,4,4, 4,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5 /),&
                                          shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,5,10, 11,1,1,&
                                                                    1,1,1, 2,4,4, 1,12,1,&
                                                                    1,1,1, 1,2,1, 1,1,13,&
                                                                    1,1,1, 1,1,1, 1,14,1,&
                                                                    21,20,19, 18,17,16, 15,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1 /),&
                                                         shape(input_fine_river_directions)))
            input_yamazaki_outlet_pixels = transpose(reshape((/ 1,0,0, 0,0,0, 1,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,1, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,1,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 1,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,2)
                call assert_equals(result%lon,3)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellSeven

    subroutine testYamazakiFindDownstreamCellEight
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            MUFP = 1.5
            allocate(input_fine_river_directions(9,9))
            allocate(input_fine_total_cumulative_flow(9,9))
            allocate(input_yamazaki_outlet_pixels(9,9))
            allocate(initial_outlet_pixel,source=latlon_coords(4,6))
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 4,4,4, 4,4,4, 4,4,8,&
                                                               4,4,4, 4,4,4, 4,4,8,&
                                                               4,4,4, 4,4,4, 4,6,8,&
                                                               5,5,5, 3,6,6, 9,5,5,&
                                                               5,5,5, 6,8,8, 5,5,5,&
                                                               5,5,5, 6,9,8, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5 /),&
                                          shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,15,&
                                                                    1,1,1, 1,1,1, 1,1,14,&
                                                                    1,1,1, 1,1,1, 1,12,13,&
                                                                    1,1,1, 1,5,10, 11,1,1,&
                                                                    1,1,1, 2,4,4, 1,1,1,&
                                                                    1,1,1, 1,2,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1 /),&
                                                         shape(input_fine_river_directions)))
            input_yamazaki_outlet_pixels = transpose(reshape((/ 1,0,0, 0,0,0, 0,0,1,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,1, 1,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 1,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,1)
                call assert_equals(result%lon,3)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellEight

    subroutine testYamazakiFindDownstreamCellNine
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            MUFP = 1.0
            allocate(input_fine_river_directions(9,9))
            allocate(input_fine_total_cumulative_flow(9,9))
            allocate(input_yamazaki_outlet_pixels(9,9))
            allocate(initial_outlet_pixel,source=latlon_coords(4,6))
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 4,4,4, 4,4,4, 4,4,8,&
                                                               4,4,4, 4,4,4, 4,4,8,&
                                                               4,4,4, 4,4,4, 4,6,8,&
                                                               5,5,5, 3,6,6, 9,5,5,&
                                                               5,5,5, 6,8,8, 5,5,5,&
                                                               5,5,5, 6,9,8, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5 /),&
                                          shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,15,&
                                                                    1,1,1, 1,1,1, 1,1,14,&
                                                                    1,1,1, 1,1,1, 1,12,13,&
                                                                    1,1,1, 1,5,10, 11,1,1,&
                                                                    1,1,1, 2,4,4, 1,1,1,&
                                                                    1,1,1, 1,2,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1 /),&
                                                         shape(input_fine_river_directions)))
            input_yamazaki_outlet_pixels = transpose(reshape((/ 1,0,0, 0,0,0, 0,0,1,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,1, 1,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 1,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,2)
                call assert_equals(result%lon,3)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellNine

    subroutine testYamazakiFindDownstreamCellTen
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            MUFP = 1.0
            allocate(input_fine_river_directions(9,9))
            allocate(input_fine_total_cumulative_flow(9,9))
            allocate(input_yamazaki_outlet_pixels(9,9))
            allocate(initial_outlet_pixel,source=latlon_coords(4,6))
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 4,4,4, 4,4,4, 4,4,8,&
                                                               4,4,4, 4,4,4, 4,4,8,&
                                                               4,4,4, 4,4,4, 6,6,8,&
                                                               5,5,5, 3,6,6, 8,5,5,&
                                                               5,5,5, 6,8,8, 5,5,5,&
                                                               5,5,5, 6,9,8, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5 /),&
                                          shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,16,&
                                                                    1,1,1, 1,1,1, 1,1,15,&
                                                                    1,1,1, 1,1,1, 12,13,14,&
                                                                    1,1,1, 1,5,10, 11,1,1,&
                                                                    1,1,1, 2,4,4, 1,1,1,&
                                                                    1,1,1, 1,2,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1 /),&
                                                         shape(input_fine_river_directions)))
            input_yamazaki_outlet_pixels = transpose(reshape((/ 1,0,0, 0,0,0, 0,0,1,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,1, 1,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 1,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,1)
                call assert_equals(result%lon,3)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
   end subroutine testYamazakiFindDownstreamCellTen

   subroutine testYamazakiFindDownstreamCellEleven
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            MUFP = 0.5
            allocate(input_fine_river_directions(9,9))
            allocate(input_fine_total_cumulative_flow(9,9))
            allocate(input_yamazaki_outlet_pixels(9,9))
            allocate(initial_outlet_pixel,source=latlon_coords(4,6))
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 4,4,4, 4,4,4, 4,4,8,&
                                                               4,4,4, 4,4,4, 4,4,8,&
                                                               4,4,4, 4,4,4, 6,6,8,&
                                                               5,5,5, 3,6,6, 8,5,5,&
                                                               5,5,5, 6,8,8, 5,5,5,&
                                                               5,5,5, 6,9,8, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5 /),&
                                          shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,16,&
                                                                    1,1,1, 1,1,1, 1,1,15,&
                                                                    1,1,1, 1,1,1, 12,13,14,&
                                                                    1,1,1, 1,5,10, 11,1,1,&
                                                                    1,1,1, 2,4,4, 1,1,1,&
                                                                    1,1,1, 1,2,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1,&
                                                                    1,1,1, 1,1,1, 1,1,1 /),&
                                                         shape(input_fine_river_directions)))
            input_yamazaki_outlet_pixels = transpose(reshape((/ 1,0,0, 0,0,0, 0,0,1,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,1, 1,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 1,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,2)
                call assert_equals(result%lon,3)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellEleven

    subroutine testYamazakiFindDownstreamCellTwelve
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            MUFP = 0.5
            allocate(input_fine_river_directions(9,9))
            allocate(input_fine_total_cumulative_flow(9,9))
            allocate(input_yamazaki_outlet_pixels(9,9))
            allocate(initial_outlet_pixel,source=latlon_coords(5,4))
            cell_section_coords = latlon_section_coords(4,4,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 4,4,4, 4,4,4, 4,4,5,&
                                                               4,4,4, 4,4,4, 4,4,5,&
                                                               4,4,4, 4,4,4, 4,4,5,&
                                                               5,5,5, 2,2,4, 5,5,5,&
                                                               5,1,4, 4,4,4, 5,5,5,&
                                                               7,5,5, 8,7,4, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5,&
                                                               5,5,5, 5,5,5, 5,5,5 /),&
                                          shape(input_fine_river_directions)))
            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1,  1,1,1, 1,1,1, &
                                                                    1,1,1,  1,1,1, 1,1,1, &
                                                                    1,1,1,  1,1,1, 1,1,1, &
                                                                    1,1,1,  2,2,1, 1,1,1, &
                                                                    1,9,8,  4,4,1, 1,1,1, &
                                                                    10,1,1, 1,2,1, 1,1,1, &
                                                                    1,1,1,  1,1,1, 1,1,1, &
                                                                    1,1,1,  1,1,1, 1,1,1, &
                                                                    1,1,1,  1,1,1, 1,1,1/),&
                                                         shape(input_fine_river_directions)))
            input_yamazaki_outlet_pixels = transpose(reshape((/ 1,0,0, 0,0,0, 0,0,1,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 1,0,0,&
                                                                0,0,0, 1,0,0, 0,0,0,&
                                                                1,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0,&
                                                                0,0,0, 0,0,0, 1,0,0,&
                                                                0,0,0, 0,0,0, 0,0,0 /),&
                                                         shape(input_fine_river_directions)))
            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,2)
                call assert_equals(result%lon,1)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellTwelve

    subroutine testYamazakiFindDownstreamCellThirteen
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            MUFP = 1.5
            allocate(input_fine_river_directions(18,18))
            allocate(input_fine_total_cumulative_flow(18,18))
            allocate(input_yamazaki_outlet_pixels(18,18))
            allocate(initial_outlet_pixel,source=latlon_coords(9,9))
            cell_section_coords = latlon_section_coords(7,7,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 6,2,1, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 6,3,2, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 9,6,3, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 2,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 2,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 2,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 2,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 6,2,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,3,5, 5,6,6, 6,6,6, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,6, 6,8,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5/),&
                                          shape(input_fine_river_directions)))

            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,2,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,6,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,9, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 10,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 11,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 12,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 13,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 14,15,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,16,1, 1,20,21, 22,23,24, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,17, 18,19,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1/), &
                                                         shape(input_fine_river_directions)))

            input_yamazaki_outlet_pixels = transpose(reshape((/ 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,1, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,1,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0/), &
                                                         shape(input_fine_river_directions)))

            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,6)
                call assert_equals(result%lon,5)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellThirteen

    subroutine testYamazakiFindDownstreamCellFourteen
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            MUFP = 1.5
            allocate(input_fine_river_directions(18,18))
            allocate(input_fine_total_cumulative_flow(18,18))
            allocate(input_yamazaki_outlet_pixels(18,18))
            allocate(initial_outlet_pixel,source=latlon_coords(8,7))
            cell_section_coords = latlon_section_coords(7,7,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 3,2,1, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,2,4, 4,4,4, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,2,5, 8,8,7, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,2,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,3,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,3, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 6,2,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,2,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 1,4,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,2, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               4,4,4, 4,4,4, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5/),&
                                          shape(input_fine_river_directions)))

            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,10,9, 8,6,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,11,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,12,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,13,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,14, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 15,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,16,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 18,17,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,19, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    25,24,23, 22,21,20, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1/), &
                                                         shape(input_fine_river_directions)))

            input_yamazaki_outlet_pixels = transpose(reshape((/ 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 1,0,0, 0,0,0, 0,0,1, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,1,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 1,0,0, 0,0,0, 0,1,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,1,0, 0,0,0, &
                                                                1,0,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0/), &
                                                         shape(input_fine_river_directions)))

            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,5)
                call assert_equals(result%lon,3)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellFourteen

    subroutine testYamazakiFindDownstreamCellFifteen
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            MUFP = 1.5
            allocate(input_fine_river_directions(18,18))
            allocate(input_fine_total_cumulative_flow(18,18))
            allocate(input_yamazaki_outlet_pixels(18,18))
            allocate(initial_outlet_pixel,source=latlon_coords(8,7))
            cell_section_coords = latlon_section_coords(7,7,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 3,2,1, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,2,4, 4,4,4, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,2,5, 8,8,7, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,2,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,3,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,3, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 6,2,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,2,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 1,4,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,2, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               4,4,4, 4,4,4, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5/),&
                                          shape(input_fine_river_directions)))

            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,10,9, 8,6,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,11,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,12,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,13,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,14, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 15,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,16,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 18,17,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,19, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    25,24,23, 22,21,20, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1/), &
                                                         shape(input_fine_river_directions)))

            input_yamazaki_outlet_pixels = transpose(reshape((/ 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 1,0,0, 0,0,0, 0,0,1, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,1,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 1,0,0, 0,0,0, 0,1,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,1,0, 0,0,0, &
                                                                1,0,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0/), &
                                                         shape(input_fine_river_directions)))

            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,5)
                call assert_equals(result%lon,3)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellFifteen

    subroutine testYamazakiFindDownstreamCellSixteen
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            MUFP = 1.5
            allocate(input_fine_river_directions(18,18))
            allocate(input_fine_total_cumulative_flow(18,18))
            allocate(input_yamazaki_outlet_pixels(18,18))
            allocate(initial_outlet_pixel,source=latlon_coords(5,9))
            cell_section_coords = latlon_section_coords(4,7,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                               1,1,1, 1,1,1, 6,6,6, 6,3,1, 1,1,1, 1,1,1, &
                                                               1,1,1, 1,1,9, 1,1,1, 1,1,3, 1,1,1, 1,1,1, &

                                                               1,1,1, 1,9,1, 1,1,1, 1,1,1, 2,1,1, 1,1,1, &
                                                               1,1,1, 1,8,1, 1,1,1, 1,1,1, 2,1,1, 1,1,1, &
                                                               1,1,1, 1,8,1, 1,1,1, 1,1,1, 2,1,1, 1,1,1, &

                                                               1,1,1, 1,8,1, 3,2,1, 1,1,1, 2,1,1, 1,1,1, &
                                                               1,1,1, 1,1,7, 4,4,4, 1,1,1, 2,1,1, 1,1,1, &
                                                               1,1,1, 1,1,1, 8,8,7, 1,1,1, 2,1,1, 1,1,1, &

                                                               1,1,1, 1,1,1, 1,1,1, 1,1,1, 2,1,1, 1,1,1, &
                                                               4,4,4, 4,4,4, 4,4,4, 4,4,4, 4,1,1, 1,1,1, &
                                                               1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5/),&
                                          shape(input_fine_river_directions)))

            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 15,16,17, 18,19,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,14, 1,1,1, 1,1,20, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,13,1, 1,1,1, 1,1,1, 21,1,1, 1,1,1, &
                                                                    1,1,1, 1,12,1, 1,1,1, 1,1,1, 22,1,1, 1,1,1, &
                                                                    1,1,1, 1,11,1, 1,1,1, 1,1,1, 23,1,1, 1,1,1, &

                                                                    1,1,1, 1,10,1, 1,1,1, 1,1,1, 24,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,9, 8,6,1, 1,1,1, 25,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 26,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 27,1,1, 1,1,1, &
                                                                    40,39,38, 37,36,35, 34,33,32, 31,30,29, 28,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1/), &
                                                         shape(input_fine_river_directions)))

            input_yamazaki_outlet_pixels = transpose(reshape((/ 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,1, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0/), &
                                                         shape(input_fine_river_directions)))

            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,5)
                call assert_equals(result%lon,2)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellSixteen

    subroutine testYamazakiFindDownstreamCellSeventeen
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            MUFP = 1.5
            yamazaki_max_range = 5
            allocate(input_fine_river_directions(18,18))
            allocate(input_fine_total_cumulative_flow(18,18))
            allocate(input_yamazaki_outlet_pixels(18,18))
            allocate(initial_outlet_pixel,source=latlon_coords(5,9))
            cell_section_coords = latlon_section_coords(4,7,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                               1,1,1, 1,1,1, 6,6,6, 6,3,1, 1,1,1, 1,1,1, &
                                                               1,1,1, 1,1,9, 1,1,1, 1,1,3, 1,1,1, 1,1,1, &

                                                               1,1,1, 1,9,1, 1,1,1, 1,1,1, 2,1,1, 1,1,1, &
                                                               1,1,1, 1,8,1, 1,1,1, 1,1,1, 2,1,1, 1,1,1, &
                                                               1,1,1, 1,8,1, 1,1,1, 1,1,1, 2,1,1, 1,1,1, &

                                                               1,1,1, 1,8,1, 3,2,1, 1,1,1, 2,1,1, 1,1,1, &
                                                               1,1,1, 1,1,7, 4,4,4, 1,1,1, 2,1,1, 1,1,1, &
                                                               1,1,1, 1,1,1, 8,8,7, 1,1,1, 2,1,1, 1,1,1, &

                                                               1,1,1, 1,1,1, 1,1,1, 1,1,1, 2,1,1, 1,1,1, &
                                                               4,4,4, 4,4,4, 4,4,4, 4,4,4, 4,1,1, 1,1,1, &
                                                               1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5/),&
                                          shape(input_fine_river_directions)))

            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 15,16,17, 18,19,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,14, 1,1,1, 1,1,20, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,13,1, 1,1,1, 1,1,1, 21,1,1, 1,1,1, &
                                                                    1,1,1, 1,12,1, 1,1,1, 1,1,1, 22,1,1, 1,1,1, &
                                                                    1,1,1, 1,11,1, 1,1,1, 1,1,1, 23,1,1, 1,1,1, &

                                                                    1,1,1, 1,10,1, 1,1,1, 1,1,1, 24,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,9, 8,6,1, 1,1,1, 25,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 26,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 27,1,1, 1,1,1, &
                                                                    40,39,38, 37,36,35, 34,33,32, 31,30,29, 28,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1/), &
                                                         shape(input_fine_river_directions)))

            input_yamazaki_outlet_pixels = transpose(reshape((/ 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,1, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0/), &
                                                         shape(input_fine_river_directions)))

            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,2)
                call assert_equals(result%lon,4)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellSeventeen

    subroutine testYamazakiFindDownstreamCellEighteen
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            MUFP = 1.5
            yamazaki_max_range = 20
            allocate(input_fine_river_directions(18,18))
            allocate(input_fine_total_cumulative_flow(18,18))
            allocate(input_yamazaki_outlet_pixels(18,18))
            allocate(initial_outlet_pixel,source=latlon_coords(11,7))
            cell_section_coords = latlon_section_coords(10,7,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                               1,1,1, 1,1,1, 6,6,6, 6,3,1, 1,1,1, 1,1,1, &
                                                               1,1,1, 1,1,9, 1,1,1, 1,1,3, 1,1,1, 1,1,1, &

                                                               1,1,1, 1,9,1, 1,1,1, 1,1,1, 2,1,1, 1,1,1, &
                                                               1,1,1, 1,8,1, 1,1,1, 1,1,1, 2,1,1, 1,1,1, &
                                                               1,1,1, 1,8,1, 1,1,1, 1,1,1, 2,1,1, 1,1,1, &

                                                               1,1,1, 1,8,1, 3,2,1, 1,1,1, 2,1,1, 1,1,1, &
                                                               1,1,1, 1,1,7, 4,4,4, 1,1,1, 2,1,1, 1,1,1, &
                                                               1,1,1, 1,1,1, 8,8,7, 1,1,1, 2,1,1, 1,1,1, &

                                                               1,1,1, 1,1,1, 1,1,1, 1,1,1, 2,1,1, 1,1,1, &
                                                               4,4,4, 4,4,4, 4,4,4, 4,4,4, 4,1,1, 1,1,1, &
                                                               1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5/),&
                                          shape(input_fine_river_directions)))

            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 15,16,17, 18,19,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,14, 1,1,1, 1,1,20, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,13,1, 1,1,1, 1,1,1, 21,1,1, 1,1,1, &
                                                                    1,1,1, 1,12,1, 1,1,1, 1,1,1, 22,1,1, 1,1,1, &
                                                                    1,1,1, 1,11,1, 1,1,1, 1,1,1, 23,1,1, 1,1,1, &

                                                                    1,1,1, 1,10,1, 1,1,1, 1,1,1, 24,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,9, 8,6,1, 1,1,1, 25,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 26,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 27,1,1, 1,1,1, &
                                                                    40,39,38, 37,36,35, 34,33,32, 31,30,29, 28,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1/), &
                                                         shape(input_fine_river_directions)))

            input_yamazaki_outlet_pixels = transpose(reshape((/ 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0/), &
                                                         shape(input_fine_river_directions)))

            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,5)
                call assert_equals(result%lon,2)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellEighteen

    subroutine testYamazakiFindDownstreamCellNineteen
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            MUFP = 1.5
            allocate(input_fine_river_directions(18,18))
            allocate(input_fine_total_cumulative_flow(18,18))
            allocate(input_yamazaki_outlet_pixels(18,18))
            allocate(initial_outlet_pixel,source=latlon_coords(16,16))
            cell_section_coords = latlon_section_coords(16,16,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 6,6,6, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,9, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 7,4,4, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 8,7,4, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 8,6,8/),&
                                          shape(input_fine_river_directions)))

            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 7,8,9, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,6, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 5,2,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 2,4,2, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1/), &
                                                         shape(input_fine_river_directions)))

            input_yamazaki_outlet_pixels = transpose(reshape((/ 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,1, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,1, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 1,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0/), &
                                                         shape(input_fine_river_directions)))

            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,5)
                call assert_equals(result%lon,6)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellNineteen

    subroutine testYamazakiFindDownstreamCellTwenty
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            MUFP = 1.5
            allocate(input_fine_river_directions(18,18))
            allocate(input_fine_total_cumulative_flow(18,18))
            allocate(input_yamazaki_outlet_pixels(18,18))
            allocate(initial_outlet_pixel,source=latlon_coords(16,16))
            cell_section_coords = latlon_section_coords(16,16,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,6, 6,6,6, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,8, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 7,4,4, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 8,7,4, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 8,6,8/),&
                                          shape(input_fine_river_directions)))

            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,7, 8,9,10, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,6, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 5,2,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 2,4,2, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1/), &
                                                         shape(input_fine_river_directions)))

            input_yamazaki_outlet_pixels = transpose(reshape((/ 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,1, 0,0,1, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 1,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0/), &
                                                         shape(input_fine_river_directions)))

            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,5)
                call assert_equals(result%lon,5)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellTwenty

    subroutine testYamazakiFindDownstreamCellTwentyOne
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            MUFP = 1.5
            yamazaki_max_range = 100
            allocate(input_fine_river_directions(18,18))
            allocate(input_fine_total_cumulative_flow(18,18))
            allocate(input_yamazaki_outlet_pixels(18,18))
            allocate(initial_outlet_pixel,source=latlon_coords(8,9))
            cell_section_coords = latlon_section_coords(7,7,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,2,5, 5,8,5, &
                                                               5,2,4, 4,4,4, 4,4,4, 4,4,4, 4,4,5, 5,8,5, &
                                                               5,2,5, 5,5,5, 5,5,5, 5,5,5, 5,8,5, 5,8,5, &

                                                               5,2,5, 5,5,5, 5,5,5, 5,5,5, 5,8,5, 5,8,5, &
                                                               5,2,5, 5,2,4, 4,4,4, 4,4,5, 5,8,5, 5,8,5, &
                                                               5,2,5, 5,2,5, 5,5,5, 5,8,5, 5,8,5, 5,8,5, &

                                                               5,2,5, 5,2,5, 2,2,2, 5,8,5, 5,8,5, 5,8,5, &
                                                               5,2,5, 5,2,5, 6,6,6, 6,8,5, 5,8,5, 5,8,5, &
                                                               5,2,5, 5,2,5, 8,8,8, 5,5,5, 5,8,5, 5,8,5, &

                                                               5,2,5, 5,2,5, 5,5,5, 5,5,5, 5,8,5, 5,8,5, &
                                                               5,2,5, 5,6,6, 6,6,6, 6,6,6, 6,8,5, 5,8,5, &
                                                               5,2,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,8,5, &

                                                               5,2,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,8,5, &
                                                               5,6,6, 6,6,6, 6,6,6, 6,6,6, 6,6,6, 6,8,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5/),&
                                          shape(input_fine_river_directions)))

            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1,   1,1,1,    1,1,1,    1,1,1,    1,4,1,    1,100,1, &
                                                                    1,59,58, 57,56,55, 54,53,52, 51,50,49, 48,47,46, 1,99,1, &
                                                                    1,60,1,  1,1,1,    1,1,1,    1,1,1,    1,43,1,    1,98,1, &

                                                                    1,61,1,  1,1,1,   1,1,1,    1,1,1,   1,42,1, 1,97,1, &
                                                                    1,62,1,  1,20,19, 18,17,16, 15,14,1, 1,41,1, 1,96,1, &
                                                                    1,63,1,  1,21,1,  1,1,1,    1,13,1,  1,40,1, 1,95,1, &

                                                                    1,64,1, 1,22,1, 1,1,1, 1,12,1,  1,39,1, 1,94,1, &
                                                                    1,65,1, 1,23,1, 3,6,9, 10,11,1, 1,38,1, 1,93,1, &
                                                                    1,66,1, 1,24,1, 1,1,1, 1,1,1,   1,37,1, 1,92,1, &

                                                                    1,67,1, 1,25,1,  1,1,1,    1,1,1,    1,36,1,  1,91,1, &
                                                                    1,68,1, 1,26,27, 28,29,30, 31,32,33, 34,35,1, 1,90,1, &
                                                                    1,69,1, 1,1,1,   1,1,1,    1,1,1,    1,1,1,   1,89,1, &

                                                                    1,70,1,  1,1,1,    1,1,1,    1,1,1,    1,1,1,    1,88,1, &
                                                                    1,71,72, 73,74,75, 76,78,79, 80,81,82, 83,84,85, 86,87,1, &
                                                                    1,1,1,   1,1,1,    1,1,1,    1,1,1,    1,1,1,    1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1/), &
                                                         shape(input_fine_river_directions)))

            input_yamazaki_outlet_pixels = transpose(reshape((/ 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,1,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,1, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0/), &
                                                         shape(input_fine_river_directions)))

            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,1)
                call assert_equals(result%lon,6)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellTwentyOne

    subroutine testYamazakiFindDownstreamCellTwentyTwo
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            MUFP = 1.5
            allocate(input_fine_river_directions(18,18))
            allocate(input_fine_total_cumulative_flow(18,18))
            allocate(input_yamazaki_outlet_pixels(18,18))
            allocate(initial_outlet_pixel,source=latlon_coords(9,18))
            cell_section_coords = latlon_section_coords(7,16,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 3,2,2, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 6,6,2, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 9,6,3, &

                                                               3,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,6,6, 6,6,2, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,2, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,2, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,2, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,2, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,2, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,2, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,2, 5,5,5, 5,5,5, 5,5,5, 5,5,5/),&
                                          shape(input_fine_river_directions)))

            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,5,7, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,9, &

                                                                    10,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,11,12, 13,14,15, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,16, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,17, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,18, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,19, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,20, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,20, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,20, 1,1,1, 1,1,1, 1,1,1, 1,1,1/), &
                                                         shape(input_fine_river_directions)))

            input_yamazaki_outlet_pixels = transpose(reshape((/ 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,1, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,1, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0/), &
                                                         shape(input_fine_river_directions)))

            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,5)
                call assert_equals(result%lon,2)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellTwentyTwo

    subroutine testYamazakiFindDownstreamCellTwentyThree
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: result
            MUFP = 1.5
            allocate(input_fine_river_directions(18,18))
            allocate(input_fine_total_cumulative_flow(18,18))
            allocate(input_yamazaki_outlet_pixels(18,18))
            allocate(initial_outlet_pixel,source=latlon_coords(17,4))
            cell_section_coords = latlon_section_coords(16,4,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 2,1,2, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               4,4,4, 4,4,6, 6,2,5, 5,5,5, 5,5,5, 5,1,4, &
                                                               5,5,5, 8,7,8, 5,2,5, 5,5,5, 5,5,5, 2,5,5/),&
                                          shape(input_fine_river_directions)))

            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    9,8,7, 6,1,3, 4,5,1, 1,1,1, 1,1,1, 1,11,10, &
                                                                    1,1,1, 1,1,1, 1,6,1, 1,1,1, 1,1,1, 12,1,1/), &
                                                         shape(input_fine_river_directions)))

            input_yamazaki_outlet_pixels = transpose(reshape((/ 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,1, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0/), &
                                                         shape(input_fine_river_directions)))

            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            result=>dir_based_rdirs_cell%yamazaki_test_find_downstream_cell(initial_outlet_pixel)
            call dir_based_rdirs_cell%destructor()
            select type (result)
            type is (latlon_coords)
                call assert_equals(result%lat,6)
                call assert_equals(result%lon,6)
            end select
            deallocate(result)
            deallocate(initial_outlet_pixel)
    end subroutine testYamazakiFindDownstreamCellTwentyThree

    subroutine testYamazakiLatLonCalculateRiverDirectionsAsIndicesOne
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        integer, dimension(:,:,:), allocatable :: coarse_river_direction_indices
        integer, dimension(:,:,:), allocatable :: coarse_river_direction_indices_expected_result
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
            MUFP = 1.5
            allocate(input_fine_river_directions(18,18))
            allocate(input_fine_total_cumulative_flow(18,18))
            allocate(input_yamazaki_outlet_pixels(18,18))
            allocate(coarse_river_direction_indices(6,6,2))
            allocate(coarse_river_direction_indices_expected_result(6,6,2))
            cell_section_coords = latlon_section_coords(16,4,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 2,1,2, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               4,4,4, 4,4,6, 6,2,5, 5,5,5, 5,5,5, 5,1,4, &
                                                               5,5,5, 8,7,8, 5,2,5, 5,5,5, 5,5,5, 2,5,5/),&
                                          shape(input_fine_river_directions)))

            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    9,8,7, 6,1,3, 4,5,1, 1,1,1, 1,1,1, 1,11,10, &
                                                                    1,1,1, 1,1,1, 1,6,1, 1,1,1, 1,1,1, 12,1,1/), &
                                                         shape(input_fine_river_directions)))

            input_yamazaki_outlet_pixels = transpose(reshape((/ 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, -1,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,1, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0/), &
                                                         shape(input_fine_river_directions)))
            coarse_river_direction_indices =  0

            coarse_river_direction_indices_expected_result(:,:,1) = &
                transpose(reshape((/ 0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &

                                     0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &
                                     0,-1,0, 0,0,0 /), &
                          (/6,6/)))

            coarse_river_direction_indices_expected_result(:,:,2) = &
                transpose(reshape((/ 0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &

                                     0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &
                                     0,-1,0, 0,0,0 /), &
                          (/6,6/)))

            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            call dir_based_rdirs_cell%yamazaki_calculate_river_directions_as_indices(coarse_river_direction_indices)
            call dir_based_rdirs_cell%destructor()
            call assert_true(all(coarse_river_direction_indices .eq. coarse_river_direction_indices_expected_result))
    end subroutine testYamazakiLatLonCalculateRiverDirectionsAsIndicesOne

    subroutine testYamazakiLatLonCalculateRiverDirectionsAsIndicesTwo
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        integer, dimension(:,:,:), allocatable :: coarse_river_direction_indices
        integer, dimension(:,:,:), allocatable :: coarse_river_direction_indices_expected_result
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
            MUFP = 1.5
            allocate(input_fine_river_directions(18,18))
            allocate(input_fine_total_cumulative_flow(18,18))
            allocate(input_yamazaki_outlet_pixels(18,18))
            allocate(coarse_river_direction_indices(6,6,2))
            allocate(coarse_river_direction_indices_expected_result(6,6,2))
            cell_section_coords = latlon_section_coords(16,7,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 2,1,2, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               4,4,4, 4,4,6, 6,2,5, 5,5,5, 5,5,5, 5,1,4, &
                                                               5,5,5, 8,7,8, 5,2,5, 5,5,5, 5,5,5, 2,5,5/),&
                                          shape(input_fine_river_directions)))

            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    9,8,7, 6,1,3, 4,5,1, 1,1,1, 1,1,1, 1,11,10, &
                                                                    1,1,1, 1,1,1, 1,6,1, 1,1,1, 1,1,1, 12,1,1/), &
                                                         shape(input_fine_river_directions)))

            input_yamazaki_outlet_pixels = transpose(reshape((/ 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,1, 0,0,2, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,-2,0, 0,0,0, 0,0,0, 0,0,0/), &
                                                         shape(input_fine_river_directions)))
            coarse_river_direction_indices =  0

            coarse_river_direction_indices_expected_result(:,:,1) = &
                transpose(reshape((/ 0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &

                                     0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &
                                     0,0,-2, 0,0,0 /), &
                          (/6,6/)))

            coarse_river_direction_indices_expected_result(:,:,2) = &
                transpose(reshape((/ 0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &

                                     0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &
                                     0,0,-2, 0,0,0 /), &
                          (/6,6/)))

            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            call dir_based_rdirs_cell%yamazaki_calculate_river_directions_as_indices(coarse_river_direction_indices)
            call dir_based_rdirs_cell%destructor()
            call assert_true(all(coarse_river_direction_indices .eq. coarse_river_direction_indices_expected_result))
    end subroutine testYamazakiLatLonCalculateRiverDirectionsAsIndicesTwo

    subroutine testYamazakiLatLonCalculateRiverDirectionsAsIndicesThree
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        integer, dimension(:,:,:), allocatable :: coarse_river_direction_indices
        integer, dimension(:,:,:), allocatable :: coarse_river_direction_indices_expected_result
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
            MUFP = 1.5
            allocate(input_fine_river_directions(18,18))
            allocate(input_fine_total_cumulative_flow(18,18))
            allocate(input_yamazaki_outlet_pixels(18,18))
            allocate(coarse_river_direction_indices(6,6,2))
            allocate(coarse_river_direction_indices_expected_result(6,6,2))
            cell_section_coords = latlon_section_coords(16,4,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 2,1,2, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               4,4,4, 4,4,6, 6,2,5, 5,5,5, 5,5,5, 5,1,4, &
                                                               5,5,5, 8,7,8, 5,2,5, 5,5,5, 5,5,5, 2,5,5/),&
                                          shape(input_fine_river_directions)))

            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    9,8,7, 6,1,3, 4,5,1, 1,1,1, 1,1,1, 1,11,10, &
                                                                    1,1,1, 1,1,1, 1,6,1, 1,1,1, 1,1,1, 12,1,1/), &
                                                         shape(input_fine_river_directions)))

            input_yamazaki_outlet_pixels = transpose(reshape((/ 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                2,0,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 1,0,0/), &
                                                         shape(input_fine_river_directions)))
            coarse_river_direction_indices =  0

            coarse_river_direction_indices_expected_result(:,:,1) = &
                transpose(reshape((/ 0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &

                                     0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &
                                     0,6,0, 0,0,0 /), &
                          (/6,6/)))

            coarse_river_direction_indices_expected_result(:,:,2) = &
                transpose(reshape((/ 0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &

                                     0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &
                                     0,6,0, 0,0,0 /), &
                          (/6,6/)))

            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            call dir_based_rdirs_cell%yamazaki_calculate_river_directions_as_indices(coarse_river_direction_indices)
            call dir_based_rdirs_cell%destructor()
            call assert_true(all(coarse_river_direction_indices .eq. coarse_river_direction_indices_expected_result))
    end subroutine testYamazakiLatLonCalculateRiverDirectionsAsIndicesThree

    subroutine testYamazakiLatLonCalculateRiverDirectionsAsIndicesFour
        use area_mod
        use coords_mod
        use cotat_parameters_mod
        integer, dimension(:,:), allocatable :: input_fine_river_directions
        integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: input_yamazaki_outlet_pixels
        integer, dimension(:,:,:), allocatable :: coarse_river_direction_indices
        integer, dimension(:,:,:), allocatable :: coarse_river_direction_indices_expected_result
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: input_yamazaki_section_coords
            MUFP = 1.5
            allocate(input_fine_river_directions(18,18))
            allocate(input_fine_total_cumulative_flow(18,18))
            allocate(input_yamazaki_outlet_pixels(18,18))
            allocate(coarse_river_direction_indices(6,6,2))
            allocate(coarse_river_direction_indices_expected_result(6,6,2))
            cell_section_coords = latlon_section_coords(16,4,3,3)
            input_yamazaki_section_coords = latlon_section_coords(1,1,18,18)
            input_fine_river_directions = transpose(reshape((/ 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &

                                                               5,5,5, 2,1,2, 5,5,5, 5,5,5, 5,5,5, 5,5,5, &
                                                               4,4,4, 4,4,6, 6,2,5, 5,5,5, 5,5,5, 5,1,4, &
                                                               5,5,5, 8,7,8, 5,2,5, 5,5,5, 5,5,5, 2,5,5/),&
                                          shape(input_fine_river_directions)))

            input_fine_total_cumulative_flow = transpose(reshape((/ 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &

                                                                    1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                                    9,8,7, 6,1,3, 4,5,1, 1,1,1, 1,1,1, 1,11,10, &
                                                                    1,1,1, 1,1,1, 1,6,1, 1,1,1, 1,1,1, 12,1,1/), &
                                                         shape(input_fine_river_directions)))

            input_yamazaki_outlet_pixels = transpose(reshape((/ 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &

                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                2,0,0, 2,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, &
                                                                0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 1,0,0/), &
                                                         shape(input_fine_river_directions)))
            coarse_river_direction_indices =  0

            coarse_river_direction_indices_expected_result(:,:,1) = &
                transpose(reshape((/ 0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &

                                     0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &
                                     0,6,0, 0,0,0 /), &
                          (/6,6/)))

            coarse_river_direction_indices_expected_result(:,:,2) = &
                transpose(reshape((/ 0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &

                                     0,0,0, 0,0,0, &
                                     0,0,0, 0,0,0, &
                                     0,6,0, 0,0,0 /), &
                          (/6,6/)))

            dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                input_fine_river_directions, input_fine_total_cumulative_flow, &
                input_yamazaki_outlet_pixels,input_yamazaki_section_coords)
            call dir_based_rdirs_cell%yamazaki_calculate_river_directions_as_indices(coarse_river_direction_indices)
            call dir_based_rdirs_cell%destructor()
            call assert_true(all(coarse_river_direction_indices .eq. coarse_river_direction_indices_expected_result))
    end subroutine testYamazakiLatLonCalculateRiverDirectionsAsIndicesFour

end module area_test_mod
