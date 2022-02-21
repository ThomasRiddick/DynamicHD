module break_loops_mod
use loop_breaker_mod
use coords_mod
use map_non_coincident_grids_mod
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


    subroutine break_loops_icon_icosohedral_cell_latlon_pixel(coarse_rdirs,coarse_cumulative_flow,coarse_catchments, &
                                                              fine_rdirs,fine_cumulative_flow, &
                                                              loop_nums_list,cell_neighbors, &
                                                              cell_vertices_lats,cell_vertices_lons, &
                                                              pixel_center_lats,pixel_center_lons,&
                                                              cell_numbers_data, &
                                                              longitudal_range_centered_on_zero_in)
        integer, dimension(:), pointer, intent(inout) :: coarse_rdirs
        integer, dimension(:), pointer :: coarse_cumulative_flow
        integer, dimension(:), pointer :: coarse_catchments
        integer, dimension(:,:), pointer :: fine_rdirs
        integer, dimension(:,:), pointer :: fine_cumulative_flow
        integer, dimension(:,:), pointer :: expanded_fine_rdirs
        integer, dimension(:,:), pointer :: expanded_fine_cumulative_flow
        class(*), dimension(:), pointer :: output_coarse_rdirs
        integer, dimension(:) :: loop_nums_list
        type(icon_icosohedral_cell_latlon_pixel_dir_based_rdirs_loop_breaker) :: dir_based_rdirs_loop_breaker
        type(generic_1d_section_coords) :: coarse_section_coords
        class(latlon_section_coords),pointer :: fine_grid_shape
        integer, dimension(:,:), pointer :: cell_neighbors
        integer, dimension(:,:), pointer :: cell_secondary_neighbors
        class(field_section), pointer :: cell_numbers
        integer, dimension(:,:), pointer :: cell_numbers_data
        class(*), dimension(:,:), pointer :: cell_numbers_data_ptr
        integer, dimension(:,:), pointer :: expanded_cell_numbers_data
        real(kind=double_precision), dimension(:), pointer :: pixel_center_lats
        real(kind=double_precision), dimension(:), pointer :: pixel_center_lons
        class(*), dimension(:,:), pointer :: pixel_center_lats_ptr_2d
        class(*), dimension(:,:), pointer :: pixel_center_lons_ptr_2d
        class(latlon_field_section), pointer :: pixel_center_lats_field
        class(latlon_field_section), pointer :: pixel_center_lons_field
        type(icon_icosohedral_grid), pointer :: coarse_grid
        class(*), dimension(:),pointer :: cell_vertex_coords_data
        class(icon_single_index_field_section), pointer :: cell_vertex_coords
        real(kind=double_precision) :: fine_grid_zero_line
        real(kind=double_precision), dimension(:,:), pointer :: cell_vertices_lats
        real(kind=double_precision), dimension(:,:), pointer :: cell_vertices_lons
        class(generic_1d_section_coords),pointer :: coarse_grid_shape
        logical :: longitudal_range_centered_on_zero
        logical, optional :: longitudal_range_centered_on_zero_in
        type(icon_icosohedral_cell_latlon_pixel_ncg_mapper) :: ncg_mapper
        integer :: i
        integer :: nlat_fine, nlon_fine
            if (present(longitudal_range_centered_on_zero_in)) then
              longitudal_range_centered_on_zero = longitudal_range_centered_on_zero_in
            else
              longitudal_range_centered_on_zero = .false.
            end if
            write(*,*) "Entering Fortran loop breaking code"
            if (pixel_center_lons(1) < -1.0) then
                fine_grid_zero_line = -180.0
            else
                fine_grid_zero_line = 0.0
            end if
            allocate(fine_grid_shape,source=latlon_section_coords(1,1,size(pixel_center_lats),&
                                                                      size(pixel_center_lons),&
                                                                      fine_grid_zero_line))
            allocate(real(kind=double_precision)::&
                     pixel_center_lats_ptr_2d(size(pixel_center_lats),&
                                              size(pixel_center_lons)))
            allocate(real(kind=double_precision)::&
                     pixel_center_lons_ptr_2d(size(pixel_center_lats),&
                                              size(pixel_center_lons)))
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
            cell_secondary_neighbors => coarse_grid%get_cell_secondary_neighbors()
            allocate(unstructured_grid_vertex_coords::cell_vertex_coords_data(size(cell_vertices_lats,1)))
            select type (cell_vertex_coords_data)
            type is (unstructured_grid_vertex_coords)
                do i = 1,size(cell_vertices_lats,1)
                    cell_vertex_coords_data(i) = &
                        unstructured_grid_vertex_coords(cell_vertices_lats(i,:),cell_vertices_lons(i,:))
                end do
            end select
            allocate(coarse_grid_shape,source=generic_1d_section_coords(cell_neighbors,&
                                                                        cell_secondary_neighbors))
            cell_vertex_coords => icon_single_index_field_section(cell_vertex_coords_data,coarse_grid_shape)
            ncg_mapper = &
                icon_icosohedral_cell_latlon_pixel_ncg_mapper(pixel_center_lats_field,&
                                                              pixel_center_lons_field,&
                                                              cell_vertex_coords,&
                                                              fine_grid_shape, &
                                                              longitudal_range_centered_on_zero)
            allocate(cell_numbers_data_ptr,source=cell_numbers_data)
            cell_numbers => latlon_field_section(cell_numbers_data_ptr,fine_grid_shape)
            call ncg_mapper%set_cell_numbers(cell_numbers)
            call ncg_mapper%generate_limits()
            coarse_section_coords = generic_1d_section_coords(cell_neighbors,cell_secondary_neighbors)
            nlat_fine = size(fine_rdirs,1)
            nlon_fine = size(fine_rdirs,2)
            allocate(expanded_fine_rdirs(nlat_fine+2,nlon_fine))
            allocate(expanded_fine_cumulative_flow(nlat_fine+2,nlon_fine))
            allocate(expanded_cell_numbers_data(nlat_fine+2,nlon_fine))
            expanded_fine_rdirs(2:nlat_fine+1,1:nlon_fine) = fine_rdirs
            expanded_fine_rdirs(1,1:nlon_fine) = -1
            expanded_fine_rdirs(nlat_fine+2,1:nlon_fine) = 5
            expanded_fine_cumulative_flow(2:nlat_fine+1,1:nlon_fine) = &
                fine_cumulative_flow
            expanded_fine_cumulative_flow(1,1:nlon_fine) = 1
            expanded_fine_cumulative_flow(nlat_fine+2,1:nlon_fine) = 1
            expanded_cell_numbers_data(2:nlat_fine+1,1:nlon_fine) = &
                cell_numbers_data
            expanded_cell_numbers_data(1,1:nlon_fine) = 0
            expanded_cell_numbers_data(nlat_fine+2,1:nlon_fine) = 0
            dir_based_rdirs_loop_breaker = &
                icon_icosohedral_cell_latlon_pixel_dir_based_rdirs_loop_breaker(coarse_catchments,&
                                                                                coarse_cumulative_flow,&
                                                                                coarse_rdirs,&
                                                                                coarse_section_coords, &
                                                                                expanded_fine_rdirs,&
                                                                                expanded_fine_cumulative_flow,&
                                                                                expanded_cell_numbers_data, &
                                                                                ncg_mapper%section_min_lats, &
                                                                                ncg_mapper%section_min_lons, &
                                                                                ncg_mapper%section_max_lats, &
                                                                                ncg_mapper%section_max_lons)
            call dir_based_rdirs_loop_breaker%break_loops(loop_nums_list)
            output_coarse_rdirs => dir_based_rdirs_loop_breaker%icon_icosohedral_cell_get_loop_free_rdirs()
            call dir_based_rdirs_loop_breaker%destructor()
            select type(output_coarse_rdirs)
            type is (integer)
                coarse_rdirs => output_coarse_rdirs
            end select
    end subroutine break_loops_icon_icosohedral_cell_latlon_pixel

end module break_loops_mod
