module flow
use area_mod
use coords_mod
use cotat_parameters_mod
implicit none

contains

    !> Main routine for the latitude longitude version of the (re-worked) FLOW algorithm.
    !! takes the fine river direction and total cumulative flow as input alongside the
    !! filepath to the COTAT+ parameter namelist file (this serves as the parameters
    !! file for the FLOW algorithm as well as the COTAT+ algorithm). Returns the index
    !! based course river direction as output.
    subroutine flow_latlon(input_fine_river_directions,input_fine_total_cumulative_flow,&
                           output_course_river_directions_lat_index,&
                           output_course_river_directions_lon_index,&
                           cotat_parameters_filepath)
        integer, dimension(:,:) :: input_fine_river_directions
        integer, dimension(:,:) :: input_fine_total_cumulative_flow
        integer, dimension(:,:) :: output_course_river_directions_lat_index
        integer, dimension(:,:) :: output_course_river_directions_lon_index
        integer, dimension(:,:,:), allocatable :: output_course_river_directions_indices
        integer, dimension(:,:), allocatable :: yamazaki_outlet_pixels
        integer, dimension(:,:), allocatable :: expanded_input_fine_river_directions
        integer, dimension(:,:), allocatable :: expanded_input_fine_total_cumulative_flow
        character(len=*), optional :: cotat_parameters_filepath
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: yamazaki_section_coords
        integer :: nlat_course, nlon_course
        integer :: nlat_fine, nlon_fine
        integer :: scale_factor
        integer :: i,j
        logical :: use_lcda_criterion = .TRUE.
            if (present(cotat_parameters_filepath)) then
                call read_cotat_parameters_namelist(cotat_parameters_filepath)
            end if
            nlat_course = size(output_course_river_directions_lat_index,1)
            nlon_course = size(output_course_river_directions_lat_index,2)
            nlat_fine = size(input_fine_river_directions,1)
            nlon_fine = size(input_fine_river_directions,2)
            scale_factor = nlat_fine/nlat_course
            allocate(expanded_input_fine_river_directions(nlat_fine+2*scale_factor,nlon_fine))
            allocate(expanded_input_fine_total_cumulative_flow(nlat_fine+2*scale_factor,nlon_fine))
            allocate(yamazaki_outlet_pixels(nlat_fine+2*scale_factor,nlon_fine))
            allocate(output_course_river_directions_indices(nlat_course,nlon_course,2))
            yamazaki_section_coords = latlon_section_coords(1,1,nlat_fine+2*scale_factor,nlon_fine)
            yamazaki_outlet_pixels = 0
            expanded_input_fine_river_directions(scale_factor+1:nlat_fine+scale_factor,1:nlon_fine) = &
                input_fine_river_directions
            expanded_input_fine_river_directions(1:scale_factor,1:nlon_fine) = -1
            expanded_input_fine_river_directions(nlat_fine+scale_factor+1:nlat_fine+2*scale_factor,1:nlon_fine) = 5
            expanded_input_fine_total_cumulative_flow(scale_factor+1:nlat_fine+scale_factor,1:nlon_fine) = &
                input_fine_total_cumulative_flow
            expanded_input_fine_total_cumulative_flow(1:scale_factor,1:nlon_fine) = 1
            expanded_input_fine_total_cumulative_flow(nlat_fine+scale_factor+1:nlat_fine+2*scale_factor,1:nlon_fine) = 1
            do j = 1,nlon_course
                do i = 1,nlat_course
                    cell_section_coords = latlon_section_coords(i*scale_factor+1,(j-1)*scale_factor+1,scale_factor,scale_factor)
                    dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                        expanded_input_fine_river_directions, expanded_input_fine_total_cumulative_flow, &
                        yamazaki_outlet_pixels,yamazaki_section_coords)
                    call dir_based_rdirs_cell%mark_ocean_and_river_mouth_points()
                    call dir_based_rdirs_cell%yamazaki_mark_cell_outlet_pixel(use_lcda_criterion)
                    call dir_based_rdirs_cell%destructor()
                end do
            end do
            do j = 1,nlon_course
                do i = 1,nlat_course
                    cell_section_coords = latlon_section_coords(i*scale_factor+1,(j-1)*scale_factor+1,scale_factor,scale_factor)
                    dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                        expanded_input_fine_river_directions, expanded_input_fine_total_cumulative_flow, &
                        yamazaki_outlet_pixels,yamazaki_section_coords)
                    call dir_based_rdirs_cell%yamazaki_calculate_river_directions_as_indices&
                        (output_course_river_directions_indices)
                    call dir_based_rdirs_cell%destructor()
                end do
            end do
            deallocate(expanded_input_fine_river_directions)
            deallocate(expanded_input_fine_total_cumulative_flow)
            output_course_river_directions_lat_index = output_course_river_directions_indices(:,:,1)
            output_course_river_directions_lon_index = output_course_river_directions_indices(:,:,2)
            deallocate(output_course_river_directions_indices)
    end subroutine flow_latlon

end module flow
