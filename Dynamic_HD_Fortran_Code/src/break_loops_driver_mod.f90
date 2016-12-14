module break_loops_driver_mod
use break_loops_mod
implicit none

    contains

    subroutine break_loops_latlon_f2py_wrapper(course_rdirs,course_cumulative_flow, &
        course_catchments,fine_rdirs,fine_cumulative_flow,loop_nums_list, &
        nlat_course,nlon_course,nlat_fine,nlon_fine,nloop_nums_list)
        integer, intent(in) :: nlat_course,nlon_course
        integer, intent(in) :: nlat_fine,nlon_fine
        integer, intent(in) :: nloop_nums_list
        integer, dimension(nlat_course,nlon_course), intent(inout) :: course_rdirs
        integer, dimension(nlat_course,nlon_course), intent(in) :: &
            course_cumulative_flow
        integer, dimension(nlat_course,nlon_course), intent(in) :: &
            course_catchments
        integer, dimension(nlat_fine,nlon_fine), intent(in) :: fine_rdirs
        integer, dimension(nlat_fine,nlon_fine), intent(in) :: &
            fine_cumulative_flow
        integer, dimension(nloop_nums_list), intent(in) :: loop_nums_list
        integer, dimension(:,:), pointer :: course_rdirs_ptr
        integer, dimension(:,:), pointer :: course_cumulative_flow_ptr
        integer, dimension(:,:), pointer :: course_catchments_ptr
        integer, dimension(:,:), pointer :: fine_rdirs_ptr
        integer, dimension(:,:), pointer :: fine_cumulative_flow_ptr
            allocate(course_rdirs_ptr,source=course_rdirs)
            allocate(course_cumulative_flow_ptr,source=course_cumulative_flow)
            allocate(course_catchments_ptr,source=course_catchments)
            allocate(fine_rdirs_ptr,source=fine_rdirs)
            allocate(fine_cumulative_flow_ptr,source=fine_cumulative_flow)
            call break_loops_latlon(course_rdirs_ptr,course_cumulative_flow_ptr, &
                                    course_catchments_ptr,fine_rdirs_ptr, &
                                    fine_cumulative_flow_ptr,loop_nums_list)
            course_rdirs = course_rdirs_ptr
    end subroutine break_loops_latlon_f2py_wrapper

end module break_loops_driver_mod
