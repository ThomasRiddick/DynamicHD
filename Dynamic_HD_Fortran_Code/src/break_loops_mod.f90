module break_loops_mod
use loop_breaker_mod
implicit none

    contains

    !> Main routine for latitude longitude loop breaking. Inputs are pointers to the course and fine
    !! river directions and cumulative flows and to the course catchments. Outputs the new loop
    !! free course river directions via the same course river direction argument as used for input.
    subroutine break_loops_latlon(course_rdirs,course_cumulative_flow,course_catchments, &
                                  fine_rdirs,fine_cumulative_flow,loop_nums_list)
        integer, dimension(:,:), pointer, intent(inout) :: course_rdirs
        integer, dimension(:,:), pointer :: course_cumulative_flow
        integer, dimension(:,:), pointer :: course_catchments
        integer, dimension(:,:), pointer :: fine_rdirs
        integer, dimension(:,:), pointer :: fine_cumulative_flow
        class(*), dimension(:,:), pointer :: output_course_rdirs
        integer, dimension(:) :: loop_nums_list
        type(latlon_dir_based_rdirs_loop_breaker) :: dir_based_rdirs_loop_breaker
            dir_based_rdirs_loop_breaker = latlon_dir_based_rdirs_loop_breaker(course_catchments,course_cumulative_flow,&
                                                                               course_rdirs,fine_rdirs,fine_cumulative_flow)
            call dir_based_rdirs_loop_breaker%break_loops(loop_nums_list)
            output_course_rdirs => dir_based_rdirs_loop_breaker%latlon_get_loop_free_rdirs()
            call dir_based_rdirs_loop_breaker%destructor()
            select type(output_course_rdirs)
            type is (integer)
                course_rdirs => output_course_rdirs
            end select
    end subroutine break_loops_latlon

end module break_loops_mod
