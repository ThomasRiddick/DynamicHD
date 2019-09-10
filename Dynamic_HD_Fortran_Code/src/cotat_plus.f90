module cotat_plus
use area_mod
use coords_mod
use cotat_parameters_mod
implicit none

contains

    !> Main routine for the latitude-longitude version of the COTAT plus routine, takes as input the
    !! fine river direction and total cumulative flow and the filepath to the COTAT+ parameters
    !! namelist file. Returns as output via an argument with intent OUT the generated coarse river
    !! directions.
    subroutine cotat_plus_latlon(input_fine_river_directions,input_fine_total_cumulative_flow,&
                                 output_coarse_river_directions,cotat_parameters_filepath)
        integer, dimension(:,:) :: input_fine_river_directions
        integer, dimension(:,:) :: input_fine_total_cumulative_flow
        integer, dimension(:,:) :: output_coarse_river_directions
        integer, dimension(:,:), allocatable :: expanded_input_fine_river_directions
        integer, dimension(:,:), allocatable :: expanded_input_fine_total_cumulative_flow
        logical, dimension(:,:), pointer :: cells_to_reprocess
        character(len=*), optional :: cotat_parameters_filepath
        type(latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        type(latlon_dir_based_rdirs_field) :: dir_based_rdirs_field
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: field_section_coords
        integer :: nlat_coarse, nlon_coarse
        integer :: nlat_fine, nlon_fine
        integer :: scale_factor
        integer :: i,j
            if (present(cotat_parameters_filepath)) then
                call read_cotat_parameters_namelist(cotat_parameters_filepath)
            end if
            nlat_coarse = size(output_coarse_river_directions,1)
            nlon_coarse = size(output_coarse_river_directions,2)
            nlat_fine = size(input_fine_river_directions,1)
            nlon_fine = size(input_fine_river_directions,2)
            scale_factor = nlat_fine/nlat_coarse
            allocate(expanded_input_fine_river_directions(nlat_fine+2*scale_factor,nlon_fine))
            allocate(expanded_input_fine_total_cumulative_flow(nlat_fine+2*scale_factor,nlon_fine))
            expanded_input_fine_river_directions(scale_factor+1:nlat_fine+scale_factor,1:nlon_fine) = &
                input_fine_river_directions
            expanded_input_fine_river_directions(1:scale_factor,1:nlon_fine) = -1
            expanded_input_fine_river_directions(nlat_fine+scale_factor+1:nlat_fine+2*scale_factor,1:nlon_fine) = 5
            expanded_input_fine_total_cumulative_flow(scale_factor+1:nlat_fine+scale_factor,1:nlon_fine) = &
                input_fine_total_cumulative_flow
            expanded_input_fine_total_cumulative_flow(1:scale_factor,1:nlon_fine) = 1
            expanded_input_fine_total_cumulative_flow(nlat_fine+scale_factor+1:nlat_fine+2*scale_factor,1:nlon_fine) = 1
            do j = 1,nlon_coarse
                do i = 1,nlat_coarse
                    cell_section_coords = latlon_section_coords(i*scale_factor+1,(j-1)*scale_factor+1,scale_factor,scale_factor)
                    dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                        expanded_input_fine_river_directions, expanded_input_fine_total_cumulative_flow)
                    call dir_based_rdirs_cell%mark_ocean_and_river_mouth_points()
                    output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
                    call dir_based_rdirs_cell%destructor()
                    select type (output_coarse_river_direction)
                    type is (dir_based_direction_indicator)
                        output_coarse_river_directions(i,j) = output_coarse_river_direction%get_direction()
                    end select
                    deallocate(output_coarse_river_direction)
                end do
            end do
            field_section_coords = latlon_section_coords(1,1,nlat_coarse,nlon_coarse)
            dir_based_rdirs_field = latlon_dir_based_rdirs_field(field_section_coords,output_coarse_river_directions)
            cells_to_reprocess => dir_based_rdirs_field%check_field_for_localized_loops()
            call dir_based_rdirs_field%destructor()
            MUFP = 0.5
            do j = 1,nlon_coarse
                do i = 1,nlat_coarse
                    if ( .not. cells_to_reprocess(i,j) ) cycle
                    cell_section_coords = latlon_section_coords(i*scale_factor+1,(j-1)*scale_factor+1,scale_factor,scale_factor)
                    dir_based_rdirs_cell = latlon_dir_based_rdirs_cell(cell_section_coords, &
                        expanded_input_fine_river_directions, expanded_input_fine_total_cumulative_flow)
                    call dir_based_rdirs_cell%mark_ocean_and_river_mouth_points()
                    output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
                    call dir_based_rdirs_cell%destructor()
                    select type (output_coarse_river_direction)
                    type is (dir_based_direction_indicator)
                        output_coarse_river_directions(i,j) = output_coarse_river_direction%get_direction()
                    end select
                    deallocate(output_coarse_river_direction)
                end do
            end do
            deallocate(cells_to_reprocess)
            deallocate(expanded_input_fine_river_directions)
            deallocate(expanded_input_fine_total_cumulative_flow)
    end subroutine cotat_plus_latlon

    subroutine cotat_plus_icon_icosohedral_cell_latlon_pixel

    end subroutine cotat_plus_icon_icosohedral_cell_latlon_pixel

end module cotat_plus
