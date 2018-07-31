module break_loops_mod
use loop_breaker_mod
implicit none

    contains

    !> Main routine for latitude longitude loop breaking. Inputs are pointers to the coarse and fine
    !! river directions and cumulative flows and to the coarse catchments. Outputs the new loop
    !! free coarse river directions via the same coarse river direction argument as used for input.
    subroutine break_loops_latlon(coarse_rdirs,coarse_cumulative_flow,coarse_catchments, &
                                  fine_rdirs,fine_cumulative_flow,loop_nums_list)
        integer, dimension(:,:), pointer, intent(inout) :: coarse_rdirs
        integer, dimension(:,:), pointer :: coarse_cumulative_flow
        integer, dimension(:,:), pointer :: coarse_catchments
        integer, dimension(:,:), pointer :: fine_rdirs
        integer, dimension(:,:), pointer :: fine_cumulative_flow
        class(*), dimension(:,:), pointer :: output_coarse_rdirs
        integer, dimension(:) :: loop_nums_list
        type(latlon_dir_based_rdirs_loop_breaker) :: dir_based_rdirs_loop_breaker
            dir_based_rdirs_loop_breaker = latlon_dir_based_rdirs_loop_breaker(coarse_catchments,coarse_cumulative_flow,&
                                                                               coarse_rdirs,fine_rdirs,fine_cumulative_flow)
            call dir_based_rdirs_loop_breaker%break_loops(loop_nums_list)
            output_coarse_rdirs => dir_based_rdirs_loop_breaker%latlon_get_loop_free_rdirs()
            call dir_based_rdirs_loop_breaker%destructor()
            select type(output_coarse_rdirs)
            type is (integer)
                coarse_rdirs => output_coarse_rdirs
            end select
    end subroutine break_loops_latlon

end module break_loops_mod
