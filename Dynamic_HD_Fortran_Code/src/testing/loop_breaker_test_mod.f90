module loop_breaker_test_mod
use fruit
implicit none
contains

    subroutine setup
    end subroutine setup

    subroutine teardown
    end subroutine teardown

    subroutine testLoopBreaker
        use loop_breaker_mod
        integer, dimension(:,:), pointer :: coarse_rdirs
        integer, dimension(:,:), pointer :: coarse_cumulative_flow
        integer, dimension(:,:), pointer :: coarse_catchments
        integer, dimension(:,:), pointer :: fine_rdirs
        integer, dimension(:,:), pointer :: fine_cumulative_flow
        integer, dimension(:,:), pointer :: expected_results
        integer, dimension(5) :: loop_nums_list
        integer, dimension(:,:), pointer :: result0
        class(*), dimension(:,:), pointer :: result_ptr
        type(latlon_dir_based_rdirs_loop_breaker) :: dir_based_rdirs_loop_breaker
            allocate(coarse_rdirs(5,5))
            allocate(coarse_cumulative_flow(5,5))
            allocate(coarse_catchments(5,5))
            allocate(expected_results(5,5))
            allocate(fine_rdirs(15,15))
            allocate(fine_cumulative_flow(15,15))
            loop_nums_list = (/ 25,61,9,58,13 /)
            coarse_rdirs = transpose(reshape((/ 6,1,6,6,1,&
                                                8,4,4,8,7, &
                                                1,-1,5,0,6, &
                                                2,4,9,8,8, &
                                                9,6,8,7,6 /), &
                                                shape(transpose(coarse_rdirs))))
            coarse_cumulative_flow = transpose(reshape((/ 0,0,1,0,0, &
                                                          0,2,1,0,1, &
                                                          0,0,1,0,0, &
                                                          0,0,0,1,0, &
                                                          0,0,0,1,1 /), &
                                                          shape(transpose(coarse_cumulative_flow))))
            coarse_catchments = transpose(reshape((/ 25,25,61,61,61, &
                                                     25,25,25,61,61, &
                                                     13,0,14,58,13,  &
                                                     9,9,58,19,13, &
                                                     9,58,58,58,9 /), &
                                                     shape(transpose(coarse_catchments))))
            fine_rdirs = transpose(reshape((/ 6,6,6, 6,6,2, 6,6,3, 6,6,6, 6,6,2, &
                                              6,6,6, 6,6,6, 9,8,4, 4,6,6, 6,6,2, &
                                              6,6,6, 6,6,8, 8,8,8,  6,6,6, 6,6,2, &
!
                                              6,6,6, 6,6,8, 8,8,8, 6,6,6, 6,6,2, &
                                              6,6,6, 6,6,8, 8,8,8, 6,6,6, 6,6,2, &
                                              6,6,6, 6,6,3, 8,8,8, 6,6,6, 6,6,6, &

                                              4,4,4, 0,1,2, -1,-1,-1, 5,5,5, 7,6,8, &
                                              4,4,4, 0,1,2, 5,6,2,    1,1,1, 8,6,6, &
                                              4,4,4, 0,1,2, 1,5,2,    2,3,8, 8,6,6, &
!
                                              4,6,6, 6,9,6, 3,2,1, 1,2,3,  8,8,8, &
                                              6,6,6, 6,8,4, 4,5,6, 4,5,6,  8,8,8, &
                                              6,6,6, 6,8,4, 9,8,7, 7,8,9,  8,8,8, &
!
                                              8,8,8, 6,6,6, 8,8,8, 3,3,3,  2,2,2, &
                                              8,8,8, 6,6,6, 8,8,8, 3,3,3,  2,2,2, &
                                              8,8,8, 6,6,6, 8,8,6, 3,3,3,  2,2,2 /),&
                                              shape(transpose(fine_rdirs))))
            fine_cumulative_flow = transpose(reshape((/ 1,1,205, 1,1,1,   1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,206, 207,1,1, 61,1,1, 1,1,1, &
                                                        1,1,1,  1,1,1,    1,1,1, 1,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,1,  1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1,  1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,221,  1,1,1, 1,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 207,1,56, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1,   1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1,   1,1,1, &
!
                                                        32,1,1, 35,36,37, 1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1, 1,90,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,11, 1,1,1, 1,1,1 /),&
                                                        shape(transpose(fine_cumulative_flow))))
            expected_results = transpose(reshape((/ 6,6,6,4,1,&
                                                    8,4,4,8,7, &
                                                    1,-1,5,0,7, &
                                                    2,6,5,8,8, &
                                                    9,6,8,7,6 /), &
                                                    shape(transpose(coarse_rdirs))))
            dir_based_rdirs_loop_breaker = &
                latlon_dir_based_rdirs_loop_breaker(coarse_catchments,coarse_cumulative_flow,&
                coarse_rdirs,fine_rdirs,fine_cumulative_flow)
            call dir_based_rdirs_loop_breaker%break_loops(loop_nums_list)
            result_ptr => dir_based_rdirs_loop_breaker%latlon_get_loop_free_rdirs()
            call dir_based_rdirs_loop_breaker%destructor()
            select type(result_ptr)
            type is (integer)
                result0 => result_ptr
            end select
            call assert_equals(expected_results,result0,5,5)
            deallocate(coarse_rdirs)
            deallocate(coarse_cumulative_flow)
            deallocate(coarse_catchments)
            deallocate(fine_rdirs)
            deallocate(fine_cumulative_flow)
            deallocate(expected_results)
            deallocate(result0)
    end subroutine testLoopBreaker

    subroutine testLoopBreakerTwo
        use loop_breaker_mod
        integer, dimension(:,:), pointer :: coarse_rdirs
        integer, dimension(:,:), pointer :: coarse_cumulative_flow
        integer, dimension(:,:), pointer :: coarse_catchments
        integer, dimension(:,:), pointer :: fine_rdirs
        integer, dimension(:,:), pointer :: fine_cumulative_flow
        integer, dimension(:,:), pointer :: expected_results
        integer, dimension(4) :: loop_nums_list
        integer, dimension(:,:), pointer :: result0
        class(*), dimension(:,:), pointer :: result_ptr
        type(latlon_dir_based_rdirs_loop_breaker) :: dir_based_rdirs_loop_breaker
            allocate(coarse_rdirs(5,5))
            allocate(coarse_cumulative_flow(5,5))
            allocate(coarse_catchments(5,5))
            allocate(expected_results(5,5))
            allocate(fine_rdirs(15,15))
            allocate(fine_cumulative_flow(15,15))
            loop_nums_list = (/ 10,20,30,40 /)
            coarse_rdirs = transpose(reshape((/ 6,1,5,6,2,&
                                                8,5,5,5,7, &
                                                5,5,5,5,5, &
                                                3,5,5,5,2, &
                                                8,4,5,9,4 /), &
                                                shape(transpose(coarse_rdirs))))
            coarse_cumulative_flow = transpose(reshape((/ 0,0,1,0,0, &
                                                          0,1,1,1,0, &
                                                          1,1,1,1,1, &
                                                          0,1,1,1,0, &
                                                          0,0,1,0,0 /), &
                                                          shape(transpose(coarse_cumulative_flow))))
            coarse_catchments = transpose(reshape((/ 10,10,1,30,30, &
                                                     10, 1,1, 1,30, &
                                                      1, 1,1, 1, 1,  &
                                                     20, 1,1, 1,40, &
                                                     20,20,1,40,40 /), &
                                                     shape(transpose(coarse_catchments))))
            fine_rdirs = transpose(reshape((/ 6,6,6, 6,6,2, 6,6,3, 6,6,6, 6,6,2, &
                                              6,6,6, 6,6,6, 9,8,4, 4,6,6, 6,6,2, &
                                              6,6,6, 6,6,3, 8,8,8,  1,6,6, 6,6,2, &
!
                                              6,6,6, 6,6,8, 8,8,8, 6,6,6, 6,6,2, &
                                              6,6,6, 6,6,8, 8,8,8, 6,6,6, 6,6,2, &
                                              6,6,6, 6,6,3, 8,8,8, 6,6,6, 6,6,6, &

                                              4,4,4, 0,1,2, -1,-1,-1, 5,5,5, 7,6,8, &
                                              4,4,4, 0,1,2, 5,6,2,    1,1,1, 8,6,6, &
                                              4,4,4, 0,1,2, 1,5,2,    2,3,8, 8,6,6, &
!
                                              4,6,6, 6,9,6, 3,2,1, 1,2,3,  8,8,8, &
                                              6,6,6, 6,8,4, 4,5,6, 4,5,6,  8,8,8, &
                                              6,6,6, 6,8,4, 9,8,7, 7,8,9,  8,8,8, &
!
                                              8,8,8, 6,6,9, 8,8,8, 3,3,3,  7,2,2, &
                                              8,8,8, 6,6,6, 8,8,8, 3,3,3,  2,2,2, &
                                              8,8,8, 6,6,6, 8,8,6, 3,3,3,  2,2,2 /),&
                                              shape(transpose(fine_rdirs))))
            fine_cumulative_flow = transpose(reshape((/ 1,1,205, 1,1,1,   1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,206, 207,1,1, 61,1,1, 1,1,1, &
                                                        1,1,1,  1,1,209,    1,1,1, 210,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,1,  1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1,  1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,221,  1,1,1, 1,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 207,1,56, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1,   1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1,   1,1,1, &
!
                                                        32,1,1, 35,36,37, 1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1, 1,90,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,77, 1,1,1, 1,1,1, 101,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,11, 1,1,1, 1,1,1 /),&
                                                        shape(transpose(fine_cumulative_flow))))
            expected_results = transpose(reshape((/ 6,3,5,1,2,&
                                                    8,5,5,5,7, &
                                                    5,5,5,5,5, &
                                                    3,5,5,5,2, &
                                                    8,9,5,9,7 /), &
                                                    shape(transpose(coarse_rdirs))))
            dir_based_rdirs_loop_breaker = &
                latlon_dir_based_rdirs_loop_breaker(coarse_catchments,coarse_cumulative_flow,&
                coarse_rdirs,fine_rdirs,fine_cumulative_flow)
            call dir_based_rdirs_loop_breaker%break_loops(loop_nums_list)
            result_ptr => dir_based_rdirs_loop_breaker%latlon_get_loop_free_rdirs()
            call dir_based_rdirs_loop_breaker%destructor()
            select type(result_ptr)
            type is (integer)
                result0 => result_ptr
            end select
            call assert_equals(expected_results,result0,5,5)
            deallocate(coarse_rdirs)
            deallocate(coarse_cumulative_flow)
            deallocate(coarse_catchments)
            deallocate(fine_rdirs)
            deallocate(fine_cumulative_flow)
            deallocate(expected_results)
            deallocate(result0)
    end subroutine testLoopBreakerTwo

subroutine testLoopBreakerThree
        use loop_breaker_mod
        integer, dimension(:,:), pointer :: coarse_rdirs
        integer, dimension(:,:), pointer :: coarse_cumulative_flow
        integer, dimension(:,:), pointer :: coarse_catchments
        integer, dimension(:,:), pointer :: fine_rdirs
        integer, dimension(:,:), pointer :: fine_cumulative_flow
        integer, dimension(:,:), pointer :: expected_results
        integer, dimension(4) :: loop_nums_list
        integer, dimension(:,:), pointer :: result0
        class(*), dimension(:,:), pointer :: result_ptr
        type(latlon_dir_based_rdirs_loop_breaker) :: dir_based_rdirs_loop_breaker
            allocate(coarse_rdirs(5,5))
            allocate(coarse_cumulative_flow(5,5))
            allocate(coarse_catchments(5,5))
            allocate(expected_results(5,5))
            allocate(fine_rdirs(15,15))
            allocate(fine_cumulative_flow(15,15))
            loop_nums_list = (/ 10,20,30,40 /)
            coarse_rdirs = transpose(reshape((/ 6,1,5,6,2, &
                                                8,5,5,5,7, &
                                                5,5,5,5,5, &
                                                3,5,5,5,2, &
                                                8,4,5,9,4 /), &
                                                shape(transpose(coarse_rdirs))))
            coarse_cumulative_flow = transpose(reshape((/ 0,0,1,0,0, &
                                                          0,1,1,1,0, &
                                                          1,1,1,1,1, &
                                                          0,1,1,1,0, &
                                                          0,0,1,0,0 /), &
                                                          shape(transpose(coarse_cumulative_flow))))
            coarse_catchments = transpose(reshape((/ 10,10,1,30,30, &
                                                     10, 1,1, 1,30, &
                                                      1, 1,1, 1, 1,  &
                                                     20, 1,1, 1,40, &
                                                     20,20,1,40,40 /), &
                                                     shape(transpose(coarse_catchments))))
            fine_rdirs = transpose(reshape((/ 6,6,6, 6,6,2, 6,6,3, 6,6,6, 6,6,2, &
                                              6,6,6, 6,6,6, 9,8,4, 4,6,6, 6,6,2, &
                                              6,6,6, 6,6,1, 8,8,8,  7,6,6, 6,6,2, &
!
                                              6,6,6, 6,6,8, 8,8,8, 6,6,6, 6,6,2, &
                                              6,6,6, 6,6,8, 8,8,8, 6,6,6, 6,6,2, &
                                              6,6,6, 6,6,3, 8,8,8, 6,6,6, 6,6,6, &

                                              4,4,4, 0,1,2, -1,-1,-1, 5,5,5, 7,6,8, &
                                              4,4,4, 0,1,2, 5,6,2,    1,1,1, 8,6,6, &
                                              4,4,4, 0,1,2, 1,5,2,    2,3,8, 8,6,6, &
!
                                              4,6,6, 6,9,6, 3,2,1, 1,2,3,  8,8,8, &
                                              6,6,6, 6,8,4, 4,5,6, 4,5,6,  8,8,8, &
                                              6,6,6, 6,8,4, 9,8,7, 7,8,9,  8,8,8, &
!
                                              8,8,8, 6,6,3, 8,8,8, 9,3,3,  4,2,2, &
                                              8,8,8, 6,6,6, 8,8,8, 3,3,3,  2,2,2, &
                                              8,8,8, 6,6,6, 8,8,6, 3,3,3,  2,2,2 /),&
                                              shape(transpose(fine_rdirs))))
            fine_cumulative_flow = transpose(reshape((/ 1,1,205, 1,1,1,   1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,206, 207,1,1, 61,1,1, 1,1,1, &
                                                        1,1,1,  1,1,209,    1,1,1, 210,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,1,  1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1,  1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,221,  1,1,1, 1,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 207,1,56, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1,   1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1,   1,1,1, &
!
                                                        32,1,1, 35,36,37, 1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1, 1,90,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,77, 1,1,1, 102,1,1, 101,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,11, 1,1,1, 1,1,1 /),&
                                                        shape(transpose(fine_cumulative_flow))))
            expected_results = transpose(reshape((/ 6,2,5,4,2, &
                                                    8,5,5,5,7, &
                                                    5,5,5,5,5, &
                                                    3,5,5,5,2, &
                                                    8,6,5,8,4 /), &
                                                    shape(transpose(coarse_rdirs))))
            dir_based_rdirs_loop_breaker = &
                latlon_dir_based_rdirs_loop_breaker(coarse_catchments,coarse_cumulative_flow,&
                coarse_rdirs,fine_rdirs,fine_cumulative_flow)
            call dir_based_rdirs_loop_breaker%break_loops(loop_nums_list)
            result_ptr => dir_based_rdirs_loop_breaker%latlon_get_loop_free_rdirs()
            call dir_based_rdirs_loop_breaker%destructor()
            select type(result_ptr)
            type is (integer)
                result0 => result_ptr
            end select
            call assert_equals(expected_results,result0,5,5)
            deallocate(coarse_rdirs)
            deallocate(coarse_cumulative_flow)
            deallocate(coarse_catchments)
            deallocate(fine_rdirs)
            deallocate(fine_cumulative_flow)
            deallocate(expected_results)
            deallocate(result0)
    end subroutine testLoopBreakerThree

subroutine testLoopBreakerFour
        use loop_breaker_mod
        integer, dimension(:,:), pointer :: coarse_rdirs
        integer, dimension(:,:), pointer :: coarse_cumulative_flow
        integer, dimension(:,:), pointer :: coarse_catchments
        integer, dimension(:,:), pointer :: fine_rdirs
        integer, dimension(:,:), pointer :: fine_cumulative_flow
        integer, dimension(:,:), pointer :: expected_results
        integer, dimension(4) :: loop_nums_list
        integer, dimension(:,:), pointer :: result0
        class(*), dimension(:,:), pointer :: result_ptr
        type(latlon_dir_based_rdirs_loop_breaker) :: dir_based_rdirs_loop_breaker
            allocate(coarse_rdirs(5,5))
            allocate(coarse_cumulative_flow(5,5))
            allocate(coarse_catchments(5,5))
            allocate(expected_results(5,5))
            allocate(fine_rdirs(15,15))
            allocate(fine_cumulative_flow(15,15))
            loop_nums_list = (/ 10,20,30,40 /)
            coarse_rdirs = transpose(reshape((/ 6,1,5,6,2, &
                                                8,5,5,5,7, &
                                                5,5,5,5,5, &
                                                3,5,5,5,2, &
                                                8,4,5,9,4 /), &
                                                shape(transpose(coarse_rdirs))))
            coarse_cumulative_flow = transpose(reshape((/ 0,0,1,0,0, &
                                                          0,1,1,1,0, &
                                                          1,1,1,1,1, &
                                                          0,1,1,1,0, &
                                                          0,0,1,0,0 /), &
                                                          shape(transpose(coarse_cumulative_flow))))
            coarse_catchments = transpose(reshape((/ 10,10,1,30,30, &
                                                     10, 1,1, 1,30, &
                                                      1, 1,1, 1, 1,  &
                                                     20, 1,1, 1,40, &
                                                     20,20,1,40,40 /), &
                                                     shape(transpose(coarse_catchments))))
            fine_rdirs = transpose(reshape((/ 6,6,6, 6,6,2, 6,6,3, 6,6,6, 6,6,2, &
                                              6,6,6, 6,6,6, 9,8,4, 4,6,6, 6,6,2, &
                                              6,6,6, 6,6,9, 8,8,8,  3,6,6, 6,6,2, &
!
                                              6,6,6, 6,6,8, 8,8,8, 6,6,6, 6,6,2, &
                                              6,6,6, 6,6,8, 8,8,8, 6,6,6, 6,6,2, &
                                              6,6,6, 6,6,3, 8,8,8, 6,6,6, 6,6,6, &

                                              4,4,4, 0,1,2, -1,-1,-1, 5,5,5, 7,6,8, &
                                              4,4,4, 0,1,2, 5,6,2,    1,1,1, 8,6,6, &
                                              4,4,4, 0,1,2, 1,5,2,    2,3,8, 8,6,6, &
!
                                              4,6,6, 6,9,6, 3,2,1, 1,2,3,  8,8,8, &
                                              6,6,6, 6,8,4, 4,5,6, 4,5,6,  8,8,8, &
                                              6,6,6, 6,8,4, 9,8,7, 7,8,9,  8,8,8, &
!
                                              8,8,8, 6,6,7, 8,8,8, 1,3,3,  1,2,2, &
                                              8,8,8, 6,6,6, 8,8,8, 3,3,3,  2,2,2, &
                                              8,8,8, 6,6,6, 8,8,6, 3,3,3,  2,2,2 /),&
                                              shape(transpose(fine_rdirs))))
            fine_cumulative_flow = transpose(reshape((/ 1,1,205, 1,1,1,   1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,206, 207,1,1, 61,1,1, 1,1,1, &
                                                        1,1,1,  1,1,209,    1,1,1, 210,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,1,  1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1,  1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,221,  1,1,1, 1,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 207,1,56, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1,   1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1,   1,1,1, &
!
                                                        32,1,1, 35,36,37, 1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1, 1,90,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,77, 1,1,1, 102,1,1, 101,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,11, 1,1,1, 1,1,1 /),&
                                                        shape(transpose(fine_cumulative_flow))))
            expected_results = transpose(reshape((/ 6,6,5,2,2, &
                                                    8,5,5,5,7, &
                                                    5,5,5,5,5, &
                                                    3,5,5,5,2, &
                                                    8,8,5,4,4 /), &
                                                    shape(transpose(coarse_rdirs))))
            dir_based_rdirs_loop_breaker = &
                latlon_dir_based_rdirs_loop_breaker(coarse_catchments,coarse_cumulative_flow,&
                coarse_rdirs,fine_rdirs,fine_cumulative_flow)
            call dir_based_rdirs_loop_breaker%break_loops(loop_nums_list)
            result_ptr => dir_based_rdirs_loop_breaker%latlon_get_loop_free_rdirs()
            call dir_based_rdirs_loop_breaker%destructor()
            select type(result_ptr)
            type is (integer)
                result0 => result_ptr
            end select
            call assert_equals(expected_results,result0,5,5)
            deallocate(coarse_rdirs)
            deallocate(coarse_cumulative_flow)
            deallocate(coarse_catchments)
            deallocate(fine_rdirs)
            deallocate(fine_cumulative_flow)
            deallocate(expected_results)
            deallocate(result0)
end subroutine testLoopBreakerFour

subroutine testLoopBreakerFive
        use loop_breaker_mod
        integer, dimension(:,:), pointer :: coarse_rdirs
        integer, dimension(:,:), pointer :: coarse_cumulative_flow
        integer, dimension(:,:), pointer :: coarse_catchments
        integer, dimension(:,:), pointer :: fine_rdirs
        integer, dimension(:,:), pointer :: fine_cumulative_flow
        integer, dimension(:,:), pointer :: expected_results
        integer, dimension(4) :: loop_nums_list
        integer, dimension(:,:), pointer :: result0
        class(*), dimension(:,:), pointer :: result_ptr
        type(latlon_dir_based_rdirs_loop_breaker) :: dir_based_rdirs_loop_breaker
            allocate(coarse_rdirs(5,5))
            allocate(coarse_cumulative_flow(5,5))
            allocate(coarse_catchments(5,5))
            allocate(expected_results(5,5))
            allocate(fine_rdirs(15,15))
            allocate(fine_cumulative_flow(15,15))
            loop_nums_list = (/ 10,20,30,40 /)
            coarse_rdirs = transpose(reshape((/ 6,1,5,6,2, &
                                                8,5,5,5,7, &
                                                5,5,5,5,5, &
                                                3,5,5,5,2, &
                                                8,4,5,9,4 /), &
                                                shape(transpose(coarse_rdirs))))
            coarse_cumulative_flow = transpose(reshape((/ 0,0,1,0,0, &
                                                          0,1,1,1,0, &
                                                          1,1,1,1,1, &
                                                          0,1,1,1,0, &
                                                          0,0,1,0,0 /), &
                                                          shape(transpose(coarse_cumulative_flow))))
            coarse_catchments = transpose(reshape((/ 10,10,1,30,30, &
                                                     10, 1,1, 1,30, &
                                                      1, 1,1, 1, 1,  &
                                                     20, 1,1, 1,40, &
                                                     20,20,1,40,40 /), &
                                                     shape(transpose(coarse_catchments))))
            fine_rdirs = transpose(reshape((/ 6,6,6, 6,6,2, 6,6,3, 6,6,6, 6,6,2, &
                                              6,6,6, 6,6,6, 9,8,4, 4,6,6, 6,6,2, &
                                              6,6,6, 6,1,6, 8,8,8,  6,6,6, 6,6,2, &
!
                                              6,6,6, 6,6,8, 8,8,8, 6,6,6, 6,6,2, &
                                              6,6,6, 6,6,8, 8,8,8, 6,6,6, 7,6,2, &
                                              6,6,6, 6,6,3, 8,8,8, 6,6,6, 6,6,6, &

                                              4,4,4, 0,1,2, -1,-1,-1, 5,5,5, 7,6,8, &
                                              4,4,4, 0,1,2, 5,6,2,    1,1,1, 8,6,6, &
                                              4,4,4, 0,1,2, 1,5,2,    2,3,8, 8,6,6, &
!
                                              4,6,6, 6,9,6, 3,2,1, 1,2,3,  8,8,8, &
                                              6,6,3, 6,8,4, 4,5,6, 4,5,6,  8,8,8, &
                                              6,6,6, 6,8,4, 9,8,7, 7,8,9,  8,8,8, &
!
                                              8,8,8, 6,6,7, 8,8,8, 3,9,3,  1,2,2, &
                                              8,8,8, 6,6,6, 8,8,8, 3,3,3,  2,2,2, &
                                              8,8,8, 6,6,6, 8,8,6, 3,3,3,  2,2,2 /),&
                                              shape(transpose(fine_rdirs))))
            fine_cumulative_flow = transpose(reshape((/ 1,1,205, 1,1,1,   1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,206, 207,1,1, 61,1,1, 1,1,1, &
                                                        1,1,1,  1,210,209,    1,1,1, 210,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,1,  1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1,  1,1,1, 1,1,1, 211,1,1, &
                                                        1,1,1, 1,1,221,  1,1,1, 1,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 207,1,56, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1,   1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1,   1,1,1, &
!
                                                        32,1,1, 35,36,37, 1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,104, 1,1,1, 1,90,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,77, 1,1,1, 102,103,1, 101,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,11, 1,1,1, 1,1,1 /),&
                                                        shape(transpose(fine_cumulative_flow))))
            expected_results = transpose(reshape((/ 6,2,5,6,2, &
                                                    8,5,5,5,4, &
                                                    5,5,5,5,5, &
                                                    6,5,5,5,2, &
                                                    8,4,5,8,4 /), &
                                                    shape(transpose(coarse_rdirs))))
            dir_based_rdirs_loop_breaker = &
                latlon_dir_based_rdirs_loop_breaker(coarse_catchments,coarse_cumulative_flow,&
                coarse_rdirs,fine_rdirs,fine_cumulative_flow)
            call dir_based_rdirs_loop_breaker%break_loops(loop_nums_list)
            result_ptr => dir_based_rdirs_loop_breaker%latlon_get_loop_free_rdirs()
            call dir_based_rdirs_loop_breaker%destructor()
            select type(result_ptr)
            type is (integer)
                result0 => result_ptr
            end select
            call assert_equals(expected_results,result0,5,5)
            deallocate(coarse_rdirs)
            deallocate(coarse_cumulative_flow)
            deallocate(coarse_catchments)
            deallocate(fine_rdirs)
            deallocate(fine_cumulative_flow)
            deallocate(expected_results)
            deallocate(result0)
end subroutine testLoopBreakerFive

subroutine testLoopBreakerSix
        use loop_breaker_mod
        integer, dimension(:,:), pointer :: coarse_rdirs
        integer, dimension(:,:), pointer :: coarse_cumulative_flow
        integer, dimension(:,:), pointer :: coarse_catchments
        integer, dimension(:,:), pointer :: fine_rdirs
        integer, dimension(:,:), pointer :: fine_cumulative_flow
        integer, dimension(:,:), pointer :: expected_results
        integer, dimension(4) :: loop_nums_list
        integer, dimension(:,:), pointer :: result0
        class(*), dimension(:,:), pointer :: result_ptr
        type(latlon_dir_based_rdirs_loop_breaker) :: dir_based_rdirs_loop_breaker
            allocate(coarse_rdirs(5,5))
            allocate(coarse_cumulative_flow(5,5))
            allocate(coarse_catchments(5,5))
            allocate(expected_results(5,5))
            allocate(fine_rdirs(15,15))
            allocate(fine_cumulative_flow(15,15))
            loop_nums_list = (/ 10,20,30,40 /)
            coarse_rdirs = transpose(reshape((/ 6,1,5,6,2, &
                                                8,5,5,5,7, &
                                                5,5,5,5,5, &
                                                3,5,5,5,2, &
                                                8,4,5,9,4 /), &
                                                shape(transpose(coarse_rdirs))))
            coarse_cumulative_flow = transpose(reshape((/ 0,0,1,0,0, &
                                                          0,1,1,1,0, &
                                                          1,1,1,1,1, &
                                                          0,1,1,1,0, &
                                                          0,0,1,0,0 /), &
                                                          shape(transpose(coarse_cumulative_flow))))
            coarse_catchments = transpose(reshape((/ 10,10,1,30,30, &
                                                     10, 1,1, 1,30, &
                                                      1, 1,1, 1, 1,  &
                                                     20, 1,1, 1,40, &
                                                     20,20,1,40,40 /), &
                                                     shape(transpose(coarse_catchments))))
            fine_rdirs = transpose(reshape((/ 6,6,6, 6,6,2, 6,6,3, 6,6,6, 6,6,2, &
                                              6,6,6, 6,6,6, 9,8,4, 4,6,6, 6,6,2, &
                                              6,6,6, 6,2,6, 8,8,8,  6,6,6, 6,6,2, &
!
                                              6,6,6, 6,6,8, 8,8,8, 6,6,6, 6,6,2, &
                                              6,6,6, 6,6,8, 8,8,8, 6,6,6, 4,6,2, &
                                              6,6,6, 6,6,3, 8,8,8, 6,6,6, 6,6,6, &

                                              4,4,4, 0,1,2, -1,-1,-1, 5,5,5, 7,6,8, &
                                              4,4,4, 0,1,2, 5,6,2,    1,1,1, 8,6,6, &
                                              4,4,4, 0,1,2, 1,5,2,    2,3,8, 8,6,6, &
!
                                              4,6,6, 6,9,6, 3,2,1, 1,2,3,  8,8,8, &
                                              6,6,6, 6,8,4, 4,5,6, 4,5,6,  8,8,8, &
                                              6,6,6, 6,8,4, 9,8,7, 7,8,9,  8,8,8, &
!
                                              8,8,8, 6,6,7, 8,8,8, 3,8,3,  1,2,2, &
                                              8,8,8, 6,6,6, 8,8,8, 3,3,3,  2,2,2, &
                                              8,8,8, 6,6,6, 8,8,6, 3,3,3,  2,2,2 /),&
                                              shape(transpose(fine_rdirs))))
            fine_cumulative_flow = transpose(reshape((/ 1,1,205, 1,1,1,   1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,206, 207,1,1, 61,1,1, 1,1,1, &
                                                        1,1,1,  1,210,209,    1,1,1, 210,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,1,  1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1,  1,1,1, 1,1,1, 211,1,1, &
                                                        1,1,1, 1,1,221,  1,1,1, 1,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 207,1,56, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1,   1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1,   1,1,1, &
!
                                                        32,1,1, 35,36,37, 1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,104, 1,1,1, 1,90,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,77, 1,1,1, 102,103,1, 101,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,11, 1,1,1, 1,1,1 /),&
                                                        shape(transpose(fine_cumulative_flow))))
            expected_results = transpose(reshape((/ 6,2,5,6,2, &
                                                    8,5,5,5,4, &
                                                    5,5,5,5,5, &
                                                    6,5,5,5,2, &
                                                    8,4,5,8,4 /), &
                                                    shape(transpose(coarse_rdirs))))
            dir_based_rdirs_loop_breaker = &
                latlon_dir_based_rdirs_loop_breaker(coarse_catchments,coarse_cumulative_flow,&
                coarse_rdirs,fine_rdirs,fine_cumulative_flow)
            call dir_based_rdirs_loop_breaker%break_loops(loop_nums_list)
            result_ptr => dir_based_rdirs_loop_breaker%latlon_get_loop_free_rdirs()
            call dir_based_rdirs_loop_breaker%destructor()
            select type(result_ptr)
            type is (integer)
                result0 => result_ptr
            end select
            call assert_equals(expected_results,result0,5,5)
                        deallocate(coarse_rdirs)
            deallocate(coarse_cumulative_flow)
            deallocate(coarse_catchments)
            deallocate(fine_rdirs)
            deallocate(fine_cumulative_flow)
            deallocate(expected_results)
            deallocate(result0)
end subroutine testLoopBreakerSix

subroutine testLoopBreakerSeven
        use loop_breaker_mod
        integer, dimension(:,:), pointer :: coarse_rdirs
        integer, dimension(:,:), pointer :: coarse_cumulative_flow
        integer, dimension(:,:), pointer :: coarse_catchments
        integer, dimension(:,:), pointer :: fine_rdirs
        integer, dimension(:,:), pointer :: fine_cumulative_flow
        integer, dimension(:,:), pointer :: expected_results
        integer, dimension(4) :: loop_nums_list
        integer, dimension(:,:), pointer :: result0
        class(*), dimension(:,:), pointer :: result_ptr
        type(latlon_dir_based_rdirs_loop_breaker) :: dir_based_rdirs_loop_breaker
            allocate(coarse_rdirs(5,5))
            allocate(coarse_cumulative_flow(5,5))
            allocate(coarse_catchments(5,5))
            allocate(expected_results(5,5))
            allocate(fine_rdirs(15,15))
            allocate(fine_cumulative_flow(15,15))
            loop_nums_list = (/ 10,20,30,40 /)
            coarse_rdirs = transpose(reshape((/ 6,1,5,6,2, &
                                                8,5,5,5,7, &
                                                5,5,5,5,5, &
                                                3,5,5,5,2, &
                                                8,4,5,9,4 /), &
                                                shape(transpose(coarse_rdirs))))
            coarse_cumulative_flow = transpose(reshape((/ 0,0,1,0,0, &
                                                          0,1,1,1,0, &
                                                          1,1,1,1,1, &
                                                          0,1,1,1,0, &
                                                          0,0,1,0,0 /), &
                                                          shape(transpose(coarse_cumulative_flow))))
            coarse_catchments = transpose(reshape((/ 10,10,1,30,30, &
                                                     10, 1,1, 1,30, &
                                                      1, 1,1, 1, 1,  &
                                                     20, 1,1, 1,40, &
                                                     20,20,1,40,40 /), &
                                                     shape(transpose(coarse_catchments))))
            fine_rdirs = transpose(reshape((/ 6,6,6, 6,6,2, 6,6,3, 6,6,6, 6,6,2, &
                                              6,6,6, 6,6,6, 9,8,4, 4,6,6, 6,6,2, &
                                              6,6,6, 6,3,6, 8,8,8,  6,6,6, 6,6,2, &
!
                                              6,6,6, 6,6,8, 8,8,8, 6,6,6, 6,6,2, &
                                              6,6,6, 6,6,8, 8,8,8, 6,6,6, 1,6,2, &
                                              6,6,6, 6,6,3, 8,8,8, 6,6,6, 6,6,6, &

                                              4,4,4, 0,1,2, -1,-1,-1, 5,5,5, 7,6,8, &
                                              4,4,4, 0,1,2, 5,6,2,    1,1,1, 8,6,6, &
                                              4,4,4, 0,1,2, 1,5,2,    2,3,8, 8,6,6, &
!
                                              4,6,6, 6,9,6, 3,2,1, 1,2,3,  8,8,8, &
                                              6,6,9, 6,8,4, 4,5,6, 4,5,6,  8,8,8, &
                                              6,6,6, 6,8,4, 9,8,7, 7,8,9,  8,8,8, &
!
                                              8,8,8, 6,6,7, 8,8,8, 3,7,3,  1,2,2, &
                                              8,8,8, 6,6,6, 8,8,8, 3,3,3,  2,2,2, &
                                              8,8,8, 6,6,6, 8,8,6, 3,3,3,  2,2,2 /),&
                                              shape(transpose(fine_rdirs))))
            fine_cumulative_flow = transpose(reshape((/ 1,1,205, 1,1,1,   1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,206, 207,1,1, 61,1,1, 1,1,1, &
                                                        1,1,1,  1,210,209,    1,1,1, 210,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,1,  1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1,  1,1,1, 1,1,1, 211,1,1, &
                                                        1,1,1, 1,1,221,  1,1,1, 1,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 207,1,56, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1,   1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1,   1,1,1, &
!
                                                        32,1,1, 35,36,37, 1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,104, 1,1,1, 1,90,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
!
                                                        1,1,1, 1,1,77, 1,1,1, 102,103,1, 101,1,1, &
                                                        1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, &
                                                        1,1,1, 1,1,1, 1,1,11, 1,1,1, 1,1,1 /),&
                                                        shape(transpose(fine_cumulative_flow))))
            expected_results = transpose(reshape((/ 6,2,5,6,2, &
                                                    8,5,5,5,4, &
                                                    5,5,5,5,5, &
                                                    6,5,5,5,2, &
                                                    8,4,5,8,4 /), &
                                                    shape(transpose(coarse_rdirs))))
            dir_based_rdirs_loop_breaker = &
                latlon_dir_based_rdirs_loop_breaker(coarse_catchments,coarse_cumulative_flow,&
                coarse_rdirs,fine_rdirs,fine_cumulative_flow)
            call dir_based_rdirs_loop_breaker%break_loops(loop_nums_list)
            result_ptr => dir_based_rdirs_loop_breaker%latlon_get_loop_free_rdirs()
            call dir_based_rdirs_loop_breaker%destructor()
            select type(result_ptr)
            type is (integer)
                result0 => result_ptr
            end select
            call assert_equals(expected_results,result0,5,5)
            deallocate(coarse_rdirs)
            deallocate(coarse_cumulative_flow)
            deallocate(coarse_catchments)
            deallocate(fine_rdirs)
            deallocate(fine_cumulative_flow)
            deallocate(expected_results)
            deallocate(result0)
end subroutine testLoopBreakerSeven

end module loop_breaker_test_mod
