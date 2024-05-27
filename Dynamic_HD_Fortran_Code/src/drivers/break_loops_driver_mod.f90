module break_loops_driver_mod
use break_loops_mod
implicit none

contains

    !> Fortran2Python (f2Py) wrapper for the latitude longitude loop break algorithms main routine,
    !! takes the necessary array bounds as arguments along with the input and output fields. The
    !! input fields are the coarse river directions, coarse cumulative flow, coarse catchments,
    !! fine river direction, fine cumulative flow and a list of loop numbers to remove (as a
    !! 1D array).
    subroutine break_loops_latlon_f2py_wrapper(coarse_rdirs,coarse_cumulative_flow, &
        coarse_catchments,fine_rdirs,fine_cumulative_flow,loop_nums_list, &
        nlat_coarse,nlon_coarse,nlat_fine,nlon_fine,nloop_nums_list)
        integer, intent(in) :: nlat_coarse,nlon_coarse
        integer, intent(in) :: nlat_fine,nlon_fine
        integer, intent(in) :: nloop_nums_list
        integer, dimension(nlat_coarse,nlon_coarse), intent(inout) :: coarse_rdirs
        integer, dimension(nlat_coarse,nlon_coarse), intent(in) :: &
            coarse_cumulative_flow
        integer, dimension(nlat_coarse,nlon_coarse), intent(in) :: &
            coarse_catchments
        integer, dimension(nlat_fine,nlon_fine), intent(in) :: fine_rdirs
        integer, dimension(nlat_fine,nlon_fine), intent(in) :: &
            fine_cumulative_flow
        integer, dimension(nloop_nums_list), intent(in) :: loop_nums_list
        integer, dimension(:,:), pointer :: coarse_rdirs_ptr
        integer, dimension(:,:), pointer :: coarse_cumulative_flow_ptr
        integer, dimension(:,:), pointer :: coarse_catchments_ptr
        integer, dimension(:,:), pointer :: fine_rdirs_ptr
        integer, dimension(:,:), pointer :: fine_cumulative_flow_ptr
            !To compile on Linux with gfortran seem to need to specify array sizes
            allocate(coarse_rdirs_ptr(nlat_coarse,nlon_coarse),source=coarse_rdirs)
            allocate(coarse_cumulative_flow_ptr(nlat_coarse,nlon_coarse), &
                source=coarse_cumulative_flow)
            allocate(coarse_catchments_ptr(nlat_coarse,nlon_coarse),source=coarse_catchments)
            allocate(fine_rdirs_ptr(nlat_fine,nlon_fine),source=fine_rdirs)
            allocate(fine_cumulative_flow_ptr(nlat_fine,nlon_fine),source=fine_cumulative_flow)
            call break_loops_latlon(coarse_rdirs_ptr,coarse_cumulative_flow_ptr, &
                                    coarse_catchments_ptr,fine_rdirs_ptr, &
                                    fine_cumulative_flow_ptr,loop_nums_list)
            coarse_rdirs = coarse_rdirs_ptr
    end subroutine break_loops_latlon_f2py_wrapper

    subroutine break_loops_iic_llp_f2py_wrap( &
            input_fine_rdirs,input_fine_total_cumulative_flow,cell_numbers, &
            coarse_cumulative_flow,coarse_catchments,coarse_rdirs, loop_nums_list, &
            cell_neighbors,pixel_center_lats,pixel_center_lons,cell_vertices_lats, &
            cell_vertices_lons,nlat_fine,nlon_fine,ncells_coarse)
        integer, intent(in) :: nlat_fine,nlon_fine
        integer, intent(in) :: ncells_coarse
        integer, parameter :: MAX_NAME_LENGTH = 1000
        real(kind=double_precision), parameter :: ABS_TOL_FOR_DEG = 0.00001
        real(kind=double_precision), parameter :: PI = 4.0*atan(1.0)
        real, intent(in), dimension(nlat_fine,nlon_fine), target :: input_fine_rdirs
        integer, intent(in), dimension(nlat_fine,nlon_fine), target :: &
            input_fine_total_cumulative_flow
        integer, intent(in), dimension(nlat_fine,nlon_fine), target :: cell_numbers
        integer, intent(in), dimension(ncells_coarse), target :: coarse_cumulative_flow
        integer, intent(in), dimension(ncells_coarse), target :: coarse_catchments
        integer, intent(inout), dimension(ncells_coarse), target :: coarse_rdirs
        integer, intent(in), dimension(:) :: loop_nums_list
        real, intent(in), dimension(nlat_fine), target :: pixel_center_lats
        real, intent(in), dimension(nlon_fine), target :: pixel_center_lons
        real(kind=8), intent(in), dimension(ncells_coarse,3) :: cell_vertices_lats
        real(kind=8), intent(in), dimension(ncells_coarse,3) :: cell_vertices_lons
        integer, intent(in), dimension(ncells_coarse,3), target :: cell_neighbors
        real(kind=double_precision), dimension(:,:), pointer :: cell_vertices_lats_ptr
        real(kind=double_precision), dimension(:,:), pointer :: cell_vertices_lons_ptr
        real(kind=double_precision), dimension(:), pointer :: pixel_center_lats_ptr
        real(kind=double_precision), dimension(:), pointer :: pixel_center_lons_ptr
        integer, dimension(:), pointer :: coarse_rdirs_ptr
        integer, dimension(:), pointer :: coarse_cumulative_flow_ptr
        integer, dimension(:), pointer :: coarse_catchments_ptr
        integer, dimension(:,:), pointer :: cell_numbers_ptr
        integer, dimension(:,:), pointer :: cell_neighbors_ptr
        integer, dimension(:,:), pointer :: input_fine_total_cumulative_flow_ptr
        integer, dimension(:,:), pointer :: input_fine_rdirs_ptr
            allocate(cell_vertices_lats_ptr(ncells_coarse,3))
            allocate(cell_vertices_lons_ptr(ncells_coarse,3))
            allocate(pixel_center_lats_ptr(nlat_fine))
            allocate(pixel_center_lons_ptr(nlon_fine))
            allocate(input_fine_rdirs_ptr(nlat_fine,nlon_fine))
            cell_vertices_lats_ptr = real(cell_vertices_lats,double_precision)
            cell_vertices_lons_ptr = real(cell_vertices_lons,double_precision)
            pixel_center_lats_ptr = real(pixel_center_lats,double_precision)
            pixel_center_lons_ptr = real(pixel_center_lons,double_precision)
            coarse_rdirs_ptr => coarse_rdirs
            coarse_cumulative_flow_ptr => coarse_cumulative_flow
            coarse_catchments_ptr => coarse_catchments
            cell_numbers_ptr => cell_numbers
            cell_neighbors_ptr => cell_neighbors
            input_fine_total_cumulative_flow_ptr => input_fine_total_cumulative_flow
            input_fine_rdirs_ptr = nint(input_fine_rdirs)
            call break_loops_icon_icosohedral_cell_latlon_pixel(coarse_rdirs_ptr, &
                                                                coarse_cumulative_flow_ptr, &
                                                                coarse_catchments_ptr, &
                                                                input_fine_rdirs_ptr, &
                                                                input_fine_total_cumulative_flow_ptr, &
                                                                loop_nums_list, &
                                                                cell_neighbors_ptr, &
                                                                cell_vertices_lats_ptr, &
                                                                cell_vertices_lons_ptr, &
                                                                pixel_center_lats_ptr, &
                                                                pixel_center_lons_ptr, &
                                                                cell_numbers_ptr)
            deallocate(cell_vertices_lats_ptr)
            deallocate(cell_vertices_lons_ptr)
            deallocate(pixel_center_lats_ptr)
            deallocate(pixel_center_lons_ptr)
            deallocate(input_fine_rdirs_ptr)
    end subroutine break_loops_iic_llp_f2py_wrap

end module break_loops_driver_mod
