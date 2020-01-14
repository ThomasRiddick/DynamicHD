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
                                                             longitudal_range_centered_on_zero_in, &
                                                             cotat_parameters_filepath, &
                                                             cell_numbers_out)
        type(icon_icosohedral_grid), pointer :: coarse_grid
        type(icon_icosohedral_cell_latlon_pixel_non_coincident_grid_mapper) :: ncg_mapper
        type(irregular_latlon_dir_based_rdirs_cell) :: dir_based_rdirs_cell
        class(direction_indicator), pointer :: output_coarse_river_direction
        type(icon_single_index_index_based_rdirs_field) :: index_based_rdirs_field
        integer, dimension(:,:), intent(in) :: input_fine_river_directions
        integer, dimension(:,:), intent(in) :: input_fine_total_cumulative_flow
        integer, dimension(:,:), allocatable :: expanded_input_fine_river_directions
        integer, dimension(:,:), allocatable :: expanded_input_fine_total_cumulative_flow
        real(kind=double_precision), dimension(:), target, intent(in) :: pixel_center_lats
        real(kind=double_precision), dimension(:), target, intent(in):: pixel_center_lons
        real(kind=double_precision), dimension(:,:), intent(in) :: cell_vertices_lats
        real(kind=double_precision), dimension(:,:), intent(in) :: cell_vertices_lons
        logical, dimension(:), allocatable :: cells_to_reprocess
        logical, dimension(:,:), pointer :: cells_to_reprocess_local
        integer, dimension(:), intent(out) :: output_coarse_next_cell_index
        class(field_section), pointer :: cell_numbers
        integer, dimension(:,:), pointer :: cell_numbers_data
        integer, dimension(:,:), pointer :: expanded_cell_numbers_data
        class(*), dimension(:,:), pointer :: cell_numbers_data_ptr
        integer, dimension(:,:), pointer :: cell_neighbors
        class(*), dimension(:,:), pointer :: pixel_center_lats_ptr_2d
        class(*), dimension(:,:), pointer :: pixel_center_lons_ptr_2d
        class(latlon_field_section), pointer :: pixel_center_lats_field
        class(latlon_field_section), pointer :: pixel_center_lons_field
        class(*), dimension(:),pointer :: cell_vertex_coords_data
        class(icon_single_index_field_section), pointer :: cell_vertex_coords
        integer, dimension(:,:), pointer :: secondary_neighbors
        integer, dimension(:,:), pointer :: cell_primary_and_secondary_neighbors
        class(latlon_section_coords),pointer :: fine_grid_shape
        class(generic_1d_section_coords),pointer :: coarse_grid_shape
        type(irregular_latlon_section_coords) :: cell_section_coords
        character(len=*), optional :: cotat_parameters_filepath
        logical, optional :: longitudal_range_centered_on_zero_in
        logical :: longitudal_range_centered_on_zero
        real(kind=double_precision) :: fine_grid_zero_line
        integer, dimension(:,:), pointer, optional :: cell_numbers_out
        integer :: i
        integer :: nlat_fine, nlon_fine
        integer :: j
  character(len=250) :: debug, debug2
  character(len=30)  :: output
            if (present(longitudal_range_centered_on_zero_in)) then
              longitudal_range_centered_on_zero = longitudal_range_centered_on_zero_in
            else
              longitudal_range_centered_on_zero = .false.
            end if
            write(*,*) "Entering Fortran COTAT+ upscaling code"
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
            allocate(cell_primary_and_secondary_neighbors(size(cell_vertices_lats,1),12))
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
            select type (cell_vertex_coords_data)
            type is (unstructured_grid_vertex_coords)
                do i = 1,size(cell_vertices_lats,1)
                    cell_vertex_coords_data(i) = &
                        unstructured_grid_vertex_coords(cell_vertices_lats(i,:),cell_vertices_lons(i,:))
                end do
            end select
            allocate(coarse_grid_shape,source=generic_1d_section_coords(cell_neighbors,&
                                                                        secondary_neighbors))
            cell_vertex_coords => icon_single_index_field_section(cell_vertex_coords_data,coarse_grid_shape)
            ncg_mapper = &
                icon_icosohedral_cell_latlon_pixel_non_coincident_grid_mapper(pixel_center_lats_field,&
                                                                              pixel_center_lons_field,&
                                                                              cell_vertex_coords,&
                                                                              fine_grid_shape, &
                                                                              longitudal_range_centered_on_zero)
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
            nlat_fine = size(input_fine_river_directions,1)
            nlon_fine = size(input_fine_river_directions,2)
            allocate(expanded_input_fine_river_directions(nlat_fine+2,nlon_fine))
            allocate(expanded_input_fine_total_cumulative_flow(nlat_fine+2,nlon_fine))
            allocate(expanded_cell_numbers_data(nlat_fine+2,nlon_fine))
            expanded_input_fine_river_directions(2:nlat_fine+1,1:nlon_fine) = &
                input_fine_river_directions
            expanded_input_fine_river_directions(1,1:nlon_fine) = -1
            expanded_input_fine_river_directions(nlat_fine+2,1:nlon_fine) = 5
            expanded_input_fine_total_cumulative_flow(2:nlat_fine+1,1:nlon_fine) = &
                input_fine_total_cumulative_flow
            expanded_input_fine_total_cumulative_flow(1,1:nlon_fine) = 1
            expanded_input_fine_total_cumulative_flow(nlat_fine+2,1:nlon_fine) = 1
            expanded_cell_numbers_data(2:nlat_fine+1,1:nlon_fine) = &
                cell_numbers_data
            expanded_cell_numbers_data(1,1:nlon_fine) = 0
            expanded_cell_numbers_data(nlat_fine+2,1:nlon_fine) = 0
            do i = 1,coarse_grid%num_points
                if(mod(i,(coarse_grid%num_points/10)) == 0) then
                    write(output,*) 100*i/coarse_grid%num_points
                    write(*,*) trim(output) // " % complete"
                end if
               cell_section_coords = irregular_latlon_section_coords(i,expanded_cell_numbers_data, &
                                                                     ncg_mapper%section_min_lats, &
                                                                     ncg_mapper%section_min_lons, &
                                                                     ncg_mapper%section_max_lats, &
                                                                     ncg_mapper%section_max_lons,&
                                                                     1)
                dir_based_rdirs_cell = &
                    irregular_latlon_dir_based_rdirs_cell(cell_section_coords, &
                                                          expanded_input_fine_river_directions, &
                                                          expanded_input_fine_total_cumulative_flow, &
                                                          cell_primary_and_secondary_neighbors)
                call dir_based_rdirs_cell%mark_ocean_and_river_mouth_points()
                output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
                call dir_based_rdirs_cell%destructor()
                select type (output_coarse_river_direction)
                type is (index_based_direction_indicator)
                    output_coarse_next_cell_index(i) = output_coarse_river_direction%get_direction()
                end select
                deallocate(output_coarse_river_direction)
                call cell_section_coords%irregular_latlon_section_coords_destructor()
                call dir_based_rdirs_cell%destructor()
            end do
            index_based_rdirs_field = &
                icon_single_index_index_based_rdirs_field(coarse_grid_shape,&
                                                          output_coarse_next_cell_index)
            cells_to_reprocess_local => index_based_rdirs_field%check_field_for_localized_loops()
            call index_based_rdirs_field%destructor()
            allocate(cells_to_reprocess(size(cells_to_reprocess_local,1)))
            cells_to_reprocess(:) = cells_to_reprocess_local(:,1)
            MUFP = 0.5
            do i = 1,coarse_grid%num_points
                if ( .not. cells_to_reprocess(i) ) cycle
                write(*,*) "Reprocessing Loop in Cell: "
                write(*,*) i
                cell_section_coords = irregular_latlon_section_coords(i,expanded_cell_numbers_data, &
                                                                     ncg_mapper%section_min_lats, &
                                                                     ncg_mapper%section_min_lons, &
                                                                     ncg_mapper%section_max_lats, &
                                                                     ncg_mapper%section_max_lons,&
                                                                     1)
                dir_based_rdirs_cell = &
                    irregular_latlon_dir_based_rdirs_cell(cell_section_coords, &
                                                          expanded_input_fine_river_directions, &
                                                          expanded_input_fine_total_cumulative_flow, &
                                                          cell_primary_and_secondary_neighbors)
                call dir_based_rdirs_cell%mark_ocean_and_river_mouth_points()
                output_coarse_river_direction => dir_based_rdirs_cell%process_cell()
                call dir_based_rdirs_cell%destructor()
                select type (output_coarse_river_direction)
                type is (index_based_direction_indicator)
                    output_coarse_next_cell_index(i) = output_coarse_river_direction%get_direction()
                end select
                deallocate(output_coarse_river_direction)
                call cell_section_coords%irregular_latlon_section_coords_destructor()
                call dir_based_rdirs_cell%destructor()
            end do
            write(*,*) "COTAT+ upscaling complete"
            write(*,"(5(5X(I2)5X))") (output_coarse_next_cell_index(i), i=1,5)
            write(*,"(15(XI2X))") (output_coarse_next_cell_index(i), i=6,20)
            write(*,"(20(XI2))") (output_coarse_next_cell_index(i), i=21,40)
            write(*,"(20(XI2))") (output_coarse_next_cell_index(i), i=41,60)
            write(*,"(15(XI2X))") (output_coarse_next_cell_index(i), i=61,75)
            write(*,"(5(5X(I2)5X))") (output_coarse_next_cell_index(i), i=76,80)
            write(*,*) " "
                do i = 2,13
      write(debug,'(20(I3))') (expanded_cell_numbers_data(i,j),j=1,20)
      write(*,*) trim(debug)
    end do
    do i = 2,13
      write(debug2,'(XXXXXXXXXXXXX20(I3))') (expanded_cell_numbers_data(i,j),j=21,40)
      write(*,*) trim(debug2)
    end do
    call ncg_mapper%icon_icosohedral_cell_latlon_pixel_ncg_mapper_destructor()
    call coarse_grid_shape%generic_1d_section_coords_destructor()
    if (present(cell_numbers_out)) then
        allocate(cell_numbers_out(nlat_fine,nlon_fine))
        cell_numbers_out = expanded_cell_numbers_data(2:nlat_fine+1,:)
    end if
    deallocate(expanded_cell_numbers_data)
    deallocate(expanded_input_fine_river_directions)
    deallocate(expanded_input_fine_total_cumulative_flow)
    deallocate(fine_grid_shape)
    deallocate(coarse_grid_shape)
    deallocate(pixel_center_lats_ptr_2d)
    deallocate(pixel_center_lons_ptr_2d)
    deallocate(cell_primary_and_secondary_neighbors)
    deallocate(cell_vertex_coords_data)
    deallocate(cell_vertex_coords)
    deallocate(pixel_center_lats_field)
    deallocate(pixel_center_lons_field)
    deallocate(coarse_grid)
    deallocate(secondary_neighbors)
    end subroutine cotat_plus_icon_icosohedral_cell_latlon_pixel

end module cotat_plus
