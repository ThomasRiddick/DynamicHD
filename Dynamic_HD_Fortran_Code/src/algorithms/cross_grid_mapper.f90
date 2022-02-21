module cross_grid_mapper
use precision_mod
use coords_mod
use map_non_coincident_grids_mod
implicit none

contains

  subroutine cross_grid_mapper_latlon_to_icon(pixel_center_lats, &
                                              pixel_center_lons, &
                                              cell_vertices_lats, &
                                              cell_vertices_lons, &
                                              cell_neighbors, &
                                              output_cell_numbers, &
                                              longitudal_range_centered_on_zero_in)
  real(kind=double_precision), dimension(:), allocatable :: pixel_center_lats
  real(kind=double_precision), dimension(:), allocatable :: pixel_center_lons
  real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lats
  real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lons
  integer, dimension(:,:), pointer :: cell_neighbors
  integer, dimension(:,:), pointer, intent(out) :: output_cell_numbers
  logical, optional :: longitudal_range_centered_on_zero_in
  class(latlon_section_coords),pointer :: fine_grid_shape
  class(generic_1d_section_coords),pointer :: coarse_grid_shape
  type(icon_icosohedral_grid), pointer :: coarse_grid
  type(icon_icosohedral_cell_latlon_pixel_ncg_mapper) :: ncg_mapper
  class(*), dimension(:,:), pointer :: cell_numbers_data_ptr
        integer, dimension(:,:), pointer :: cell_numbers_data
  class(*), dimension(:),pointer :: cell_vertex_coords_data
  class(*), dimension(:,:), pointer :: pixel_center_lats_ptr_2d
  class(*), dimension(:,:), pointer :: pixel_center_lons_ptr_2d
  class(latlon_field_section), pointer :: pixel_center_lats_field
  class(latlon_field_section), pointer :: pixel_center_lons_field
  integer, dimension(:,:), pointer :: cell_primary_and_secondary_neighbors
  integer, dimension(:,:), pointer :: secondary_neighbors
  class(field_section), pointer :: cell_numbers
  class(icon_single_index_field_section), pointer :: cell_vertex_coords
  real(kind=double_precision) :: fine_grid_zero_line
  logical :: longitudal_range_centered_on_zero
  integer :: i
    if (present(longitudal_range_centered_on_zero_in)) then
      longitudal_range_centered_on_zero = longitudal_range_centered_on_zero_in
    else
      longitudal_range_centered_on_zero = .false.
    end if
    write(*,*) "Entering Cross Grid Mapping Generation Code"
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
        icon_icosohedral_cell_latlon_pixel_ncg_mapper(pixel_center_lats_field,&
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
    allocate(output_cell_numbers(size(cell_numbers_data,1),size(cell_numbers_data,2)))
    output_cell_numbers(:,:) = cell_numbers_data(:,:)
    call ncg_mapper%icon_icosohedral_cell_latlon_pixel_ncg_mapper_destructor()
    call coarse_grid_shape%generic_1d_section_coords_destructor()
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
  end subroutine cross_grid_mapper_latlon_to_icon

end module cross_grid_mapper
