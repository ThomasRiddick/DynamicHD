module cotat_plus
use area_mod
use coords_mod
use map_non_coincident_grids_mod
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

    subroutine cotat_plus_icon_icosohedral_cell_latlon_pixel(input_fine_river_directions,&
                                                             input_fine_total_cumulative_flow,&
                                                             output_coarse_next_cell_index,&
                                                             pixel_center_lats,&
                                                             pixel_center_lons,&
                                                             cell_vertices_lats, &
                                                             cell_vertices_lons, &
                                                             cell_neighbors, &
                                                             cotat_parameters_filepath)
        type(icon_icosohedral_grid), pointer :: coarse_grid
        type(icon_icosohedral_cell_latlon_pixel_non_coincident_grid_mapper) :: ncg_mapper
        type(irregular_latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        integer, dimension(:,:), intent(in) :: input_fine_river_directions
        integer, dimension(:,:) :: input_fine_total_cumulative_flow
        integer, dimension(:) :: output_coarse_next_cell_index
        class(field_section), pointer :: cell_numbers
        integer, dimension(:,:), pointer :: cell_numbers_data
        class(*), dimension(:,:), pointer :: cell_numbers_data_ptr
        integer, dimension(:,:), pointer :: cell_neighbors
        real(kind=double_precision), dimension(:), target :: pixel_center_lats
        real(kind=double_precision), dimension(:), target :: pixel_center_lons
        class(*), dimension(:,:), pointer :: pixel_center_lats_ptr_2d
        class(*), dimension(:,:), pointer :: pixel_center_lons_ptr_2d
        class(latlon_field_section), pointer :: pixel_center_lats_field
        class(latlon_field_section), pointer :: pixel_center_lons_field
        real(kind=double_precision), dimension(:,:) :: cell_vertices_lats
        real(kind=double_precision), dimension(:,:) :: cell_vertices_lons
        class(*), dimension(:),pointer :: cell_vertex_coords_data
        class(icon_single_index_field_section), pointer :: cell_vertex_coords
        integer, dimension(:,:), pointer :: secondary_neighbors
        integer, dimension(:,:), pointer :: cell_primary_and_secondary_neighbors
        class(latlon_section_coords),pointer :: fine_grid_shape
        class(generic_1d_section_coords),pointer :: coarse_grid_shape
        class(irregular_latlon_section_coords),pointer :: cell_section_coords
        character(len=*), optional :: cotat_parameters_filepath
        integer :: i
            if (present(cotat_parameters_filepath)) then
                call read_cotat_parameters_namelist(cotat_parameters_filepath)
            end if
            do i = 1,size(pixel_center_lons)
                select type (pixel_center_lats_ptr_2d)
                type is (real(kind=double_precision))
                    pixel_center_lats_ptr_2d(:,i) = pixel_center_lats(:)
                end select
            end do
            do i = 1,size(pixel_center_lats)
                select type (pixel_center_lons_ptr_2d)
                type is (real(kind=double_precision))
                    pixel_center_lons_ptr_2d(i,:) = pixel_center_lons(:)
                end select
            end do
            pixel_center_lats_field => latlon_field_section(pixel_center_lats_ptr_2d,fine_grid_shape)
            pixel_center_lons_field => latlon_field_section(pixel_center_lons_ptr_2d,fine_grid_shape)
            coarse_grid => icon_icosohedral_grid(cell_neighbors)
            call coarse_grid%calculate_secondary_neighbors()
            secondary_neighbors => coarse_grid%get_cell_secondary_neighbors()
            cell_primary_and_secondary_neighbors(:,1:3) = cell_neighbors(:,:)
            cell_primary_and_secondary_neighbors(:,4:12) = secondary_neighbors(:,:)
            allocate(unstructured_grid_vertex_coords::cell_vertex_coords_data(size(cell_vertices_lats,1)))
            do i = 1,size(cell_vertices_lats,1)
                select type (data_element => cell_vertex_coords_data(i) )
                type is (unstructured_grid_vertex_coords)
                    data_element = &
                        unstructured_grid_vertex_coords(cell_vertices_lats(1,:),cell_vertices_lons(1,:))
                end select
            end do
            allocate(coarse_grid_shape,source=generic_1d_section_coords(cell_neighbors,&
                                                                        secondary_neighbors))
            cell_vertex_coords => icon_single_index_field_section(cell_vertex_coords_data,coarse_grid_shape)
            ncg_mapper = &
                icon_icosohedral_cell_latlon_pixel_non_coincident_grid_mapper(pixel_center_lats_field,&
                                                                              pixel_center_lons_field,&
                                                                              cell_vertex_coords,&
                                                                              fine_grid_shape)
            cell_numbers => ncg_mapper%generate_cell_numbers()
            select type(cell_numbers)
            class is (latlon_field_section)
                cell_numbers_data_ptr => cell_numbers%data
                select type (cell_numbers_data_ptr)
                type is (integer)
                    cell_numbers_data => cell_numbers_data_ptr
                end select
            end select
            call ncg_mapper%generate_limits()
            do i = 1,coarse_grid%num_points
                allocate(cell_section_coords,source=irregular_latlon_section_coords(i,cell_numbers_data, &
                                                                                    ncg_mapper%section_min_lats, &
                                                                                    ncg_mapper%section_min_lons, &
                                                                                    ncg_mapper%section_max_lats, &
                                                                                    ncg_mapper%section_max_lons))
                dir_based_rdirs_cell = &
                    irregular_latlon_dir_based_rdirs_cell(cell_section_coords, &
                                                          input_fine_river_directions, &
                                                          input_fine_total_cumulative_flow, &
                                                          cell_primary_and_secondary_neighbors)
                call dir_based_rdirs_cell%mark_ocean_and_river_mouth_points()
                output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
                call dir_based_rdirs_cell%destructor()
                select type (output_coarse_river_direction)
                type is (index_based_direction_indicator)
                    output_coarse_next_cell_index(i) = output_coarse_river_direction%get_direction()
                end select
                deallocate(output_coarse_river_direction)
            end do
    end subroutine cotat_plus_icon_icosohedral_cell_latlon_pixel

end module cotat_plus
