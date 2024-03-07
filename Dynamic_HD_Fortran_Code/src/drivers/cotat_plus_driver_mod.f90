module cotat_plus_driver_mod
use cotat_plus
implicit none

contains

    !> Fortran2Python (F2Py) wrapper for the latitude longitude implementation of the COTAT+ algorithm. Input
    !! is the fine cumulative flow and fine river directions along with the file path to the COTAT+ parameters
    !! namelist file. Output is the coarse river directions. Also takes the bounds of those arrays as arguments.
    subroutine cotat_plus_latlon_f2py_wrapper(input_fine_river_directions,input_fine_total_cumulative_flow,&
                                              output_coarse_river_directions,cotat_parameters_filepath,&
                                              nlat_fine,nlon_fine,nlat_coarse,nlon_coarse)
        integer, intent(in) :: nlat_fine,nlon_fine,nlat_coarse,nlon_coarse
        integer, intent(in), dimension(nlat_fine,nlon_fine) :: input_fine_river_directions
        integer, intent(in), dimension(nlat_fine,nlon_fine) :: input_fine_total_cumulative_flow
        integer, intent(out), dimension(nlat_coarse,nlon_coarse) :: output_coarse_river_directions
        character(len=*),intent(in) :: cotat_parameters_filepath
            write (*,*) "Running COTAT+ up-scaling algorithm"
            call cotat_plus_latlon(input_fine_river_directions,input_fine_total_cumulative_flow,&
                                   output_coarse_river_directions,cotat_parameters_filepath)
    end subroutine

    subroutine cotat_plus_latlon_f2py_worker_wrapper()
            call cotat_plus_latlon()
    end subroutine cotat_plus_latlon_f2py_worker_wrapper

    subroutine cotat_plus_icon_icosohedral_cell_latlon_pixel_f2py_wrapper( &
            input_fine_rdirs,input_fine_total_cumulative_flow, &
            cell_neighbors,pixel_center_lats,pixel_center_lons, &
            cell_vertices_lats,cell_vertices_lons, &
            write_cell_numbers, &
            output_coarse_next_cell_index, &
            nlat_fine,nlon_fine,ncells_coarse, &
            cotat_parameters_filename, &
            output_cell_numbers)
        integer, intent(in) :: ncells_coarse
        integer, intent(in) :: nlat_fine,nlon_fine
        integer, parameter :: MAX_NAME_LENGTH = 1000
        real(kind=double_precision), parameter :: ABS_TOL_FOR_DEG = 0.00001
        real(kind=double_precision), parameter :: PI = 4.0*atan(1.0)
        real, intent(in), dimension(nlat_fine,nlon_fine) :: input_fine_rdirs
        integer, intent(in), dimension(nlat_fine,nlon_fine) :: input_fine_total_cumulative_flow
        real, intent(in), dimension(nlat_fine) :: pixel_center_lats
        real, intent(in), dimension(nlon_fine) :: pixel_center_lons
        real, intent(in), target, &
            dimension(ncells_coarse,3) :: cell_vertices_lats
        real, intent(in), target, &
            dimension(ncells_coarse,3) :: cell_vertices_lons
        integer, intent(in), dimension(ncells_coarse,3), target :: cell_neighbors
        integer, dimension(:,:), pointer :: cell_neighbors_ptr
        integer, intent(out), dimension(ncells_coarse) :: output_coarse_next_cell_index
        logical, intent(in) :: write_cell_numbers
        character(len=MAX_NAME_LENGTH), intent(in), optional :: &
           cotat_parameters_filename
        integer, intent(out), dimension(nlat_fine,nlon_fine), &
            optional, target :: output_cell_numbers
        real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lats_alloc
        real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lons_alloc
        integer, dimension(:,:), pointer :: output_cell_numbers_ptr
            cell_neighbors_ptr => cell_neighbors
            output_cell_numbers_ptr => output_cell_numbers
            allocate(cell_vertices_lats_alloc(ncells_coarse,3))
            allocate(cell_vertices_lons_alloc(ncells_coarse,3))
            cell_vertices_lats_alloc = real(cell_vertices_lats,double_precision)
            cell_vertices_lons_alloc = real(cell_vertices_lons,double_precision)
            if (write_cell_numbers) then
                if (.not. present(cotat_parameters_filename)) then
                   write(*,*) "can't write cell numbers in cotat parameters file not specified"
                   stop
                end if
                call cotat_plus_icon_icosohedral_cell_latlon_pixel(nint(input_fine_rdirs),&
                                                                 input_fine_total_cumulative_flow,&
                                                                 output_coarse_next_cell_index,&
                                                                 real(pixel_center_lats,double_precision),&
                                                                 real(pixel_center_lons,double_precision),&
                                                                 cell_vertices_lats_alloc,&
                                                                 cell_vertices_lons_alloc,&
                                                                 cell_neighbors_ptr, &
                                                                 write_cell_numbers, &
                                                                 cotat_parameters_filename,&
                                                                 output_cell_numbers_ptr)
            else
                if (present(cotat_parameters_filename) .and. &
                    cotat_parameters_filename /= '') then
                    call cotat_plus_icon_icosohedral_cell_latlon_pixel(nint(input_fine_rdirs),&
                                                                       input_fine_total_cumulative_flow,&
                                                                       output_coarse_next_cell_index,&
                                                                       real(pixel_center_lats,double_precision),&
                                                                       real(pixel_center_lons,double_precision),&
                                                                       cell_vertices_lats_alloc,&
                                                                       cell_vertices_lons_alloc,&
                                                                       cell_neighbors_ptr, &
                                                                       write_cell_numbers, &
                                                                       cotat_parameters_filename)
                else
                    call cotat_plus_icon_icosohedral_cell_latlon_pixel(nint(input_fine_rdirs),&
                                                                       input_fine_total_cumulative_flow,&
                                                                       output_coarse_next_cell_index,&
                                                                       real(pixel_center_lats,double_precision),&
                                                                       real(pixel_center_lons,double_precision),&
                                                                       cell_vertices_lats_alloc,&
                                                                       cell_vertices_lons_alloc,&
                                                                       cell_neighbors_ptr, &
                                                                       write_cell_numbers)
                end if
            end if
            deallocate(cell_vertices_lats_alloc)
            deallocate(cell_vertices_lons_alloc)
    end subroutine cotat_plus_icon_icosohedral_cell_latlon_pixel_f2py_wrapper

end module cotat_plus_driver_mod
