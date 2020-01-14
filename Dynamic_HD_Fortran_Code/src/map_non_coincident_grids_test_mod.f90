module map_non_coincident_grids_test_mod
use fruit
use map_non_coincident_grids_mod
implicit none

contains

subroutine setup
end subroutine setup

subroutine teardown
end subroutine teardown

subroutine testLatLonToIconGrids
  type(icon_icosohedral_grid), pointer :: coarse_grid
  type(icon_icosohedral_cell_latlon_pixel_non_coincident_grid_mapper) :: ncg_mapper
  integer, dimension(:,:), pointer :: cell_neighbors
  integer, dimension(:,:), pointer :: secondary_neighbors
  real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lats
  real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lons
  real(kind=double_precision), dimension(:), allocatable :: pixel_center_lats
  real(kind=double_precision), dimension(:), allocatable :: pixel_center_lons
  class(*), dimension(:,:), pointer :: pixel_center_lats_ptr_2d
  class(*), dimension(:,:), pointer :: pixel_center_lons_ptr_2d
  type(latlon_field_section), pointer :: pixel_center_lats_field
  type(latlon_field_section), pointer :: pixel_center_lons_field
  class(*), dimension(:),pointer :: cell_vertex_coords_data
  class(icon_single_index_field_section), pointer :: cell_vertex_coords
  type(generic_1d_section_coords),pointer :: coarse_grid_shape
  type(latlon_section_coords),pointer :: fine_grid_shape
  class(field_section), pointer :: cell_numbers
  integer, dimension(:,:), pointer :: cell_numbers_data
  class(*), dimension(:,:), pointer :: cell_numbers_data_ptr
  integer :: i,j
  character(len=250) :: debug, debug2
    allocate(fine_grid_shape,source=latlon_section_coords(1,1,12,40))
    allocate(cell_vertices_lats(80,3))
    allocate(cell_vertices_lons(80,3))
    allocate(pixel_center_lats(12))
    allocate(pixel_center_lons(40))
    allocate(real(kind=double_precision)::pixel_center_lats_ptr_2d(12,40))
    allocate(real(kind=double_precision)::pixel_center_lons_ptr_2d(12,40))
    allocate(cell_neighbors(80,3))
    cell_neighbors = transpose( reshape((/ 5,7,2, &
                                           1,10,3, &
                                           2,13,4, &
                                           3,16,5, &
                                           4,19,1, &
                                           20,21,7, &
                                           1,6,8, &
                                           7,23,9, &
                                           8,25,10, &
                                           2,9,11, &
                                           10,27,12, &
                                           11,29,13, &
                                           3,12,14, &
                                           13,31,15, &
                                           14,33,16, &
                                           4,15,17, &
                                           16,35,18, &
                                           17,37,19, &
                                           5,18,20, &
                                           19,39,6, &
                                           6,40,22, &
                                           21,41,23, &
                                           8,22,24, &
                                           23,43,25, &
                                           24,26,9, &
                                           25,45,27, &
                                           11,26,28, &
                                           27,47,29, &
                                           12,28,30, &
                                           29,49,31, &
                                           14,30,32, &
                                           31,51,33, &
                                           15,32,34, &
                                           33,53,35, &
                                           17,34,36, &
                                           35,55,37, &
                                           18,36,38, &
                                           37,57,39, &
                                           20,38,40, &
                                           39,59,21, &
                                           22,60,42, &
                                           41,61,43, &
                                           24,42,44, &
                                           43,63,45, &
                                           26,44,46, &
                                           45,64,47, &
                                           28,46,48, &
                                           47,66,49, &
                                           30,48,50, &
                                           49,67,51, &
                                           32,50,52, &
                                           51,69,53, &
                                           34,52,54, &
                                           53,70,55, &
                                           36,54,56, &
                                           55,72,57, &
                                           38,56,58, &
                                           57,73,59, &
                                           40,58,60, &
                                           59,75,41, &
                                           42,75,62, &
                                           61,76,63, &
                                           44,62,64, &
                                           46,63,65, &
                                           64,77,66, &
                                           48,65,67, &
                                           50,66,68, &
                                           67,78,69, &
                                           52,68,70, &
                                           54,69,71, &
                                           70,79,72, &
                                           56,71,73, &
                                           58,72,74, &
                                           73,80,75, &
                                           60,74,61, &
                                           62,80,77, &
                                           65,76,78, &
                                           68,77,79, &
                                           71,78,80, &
                                           74,79,76 /), (/3,80/)))
    cell_vertices_lats = transpose(reshape((/  90.0,  60.0,  60.0, &
                                               90.0,  60.0,  60.0, &
                                               90.0,  60.0,  60.0, &
                                               90.0,  60.0,  60.0, &
                                               90.0,  60.0,  60.0, &
                                               60.0,  30.0,  30.0, &
                                               60.0,  30.0,  60.0, &
                                               60.0,  30.0,  30.0, &
                                               60.0,  30.0,  30.0, &
                                               60.0,  30.0,  60.0, &
                                               60.0,  30.0,  30.0, &
                                               60.0,  30.0,  30.0, &
                                               60.0,  30.0,  60.0, &
                                               60.0,  30.0,  30.0, &
                                               60.0,  30.0,  30.0, &
                                               60.0,  30.0,  60.0, &
                                               60.0,  30.0,  30.0, &
                                               60.0,  30.0,  30.0, &
                                               60.0,  30.0,  60.0, &
                                               60.0,  30.0,  30.0, &
                                               30.0,   0.0,  30.0, &
                                               30.0,   0.0,   0.0, &
                                               30.0,   0.0,  30.0, &
                                               30.0,   0.0,   0.0, &
                                               30.0,   0.0,  30.0, &
                                               30.0,   0.0,   0.0, &
                                               30.0,   0.0,  30.0, &
                                               30.0,   0.0,   0.0, &
                                               30.0,   0.0,  30.0, &
                                               30.0,   0.0,   0.0, &
                                               30.0,   0.0,  30.0, &
                                               30.0,   0.0,   0.0, &
                                               30.0,   0.0,  30.0, &
                                               30.0,   0.0,   0.0, &
                                               30.0,   0.0,  30.0, &
                                               30.0,   0.0,   0.0, &
                                               30.0,   0.0,  30.0, &
                                               30.0,   0.0,   0.0, &
                                               30.0,   0.0,  30.0, &
                                               30.0,   0.0,   0.0, &
                                                0.0, -30.0,   0.0, &
                                                0.0, -30.0, -30.0, &
                                                0.0, -30.0,   0.0, &
                                                0.0, -30.0, -30.0, &
                                                0.0, -30.0,   0.0, &
                                                0.0, -30.0, -30.0, &
                                                0.0, -30.0,   0.0, &
                                                0.0, -30.0, -30.0, &
                                                0.0, -30.0,   0.0, &
                                                0.0, -30.0, -30.0, &
                                                0.0, -30.0,   0.0, &
                                                0.0, -30.0, -30.0, &
                                                0.0, -30.0,   0.0, &
                                                0.0, -30.0, -30.0, &
                                                0.0, -30.0,   0.0, &
                                                0.0, -30.0, -30.0, &
                                                0.0, -30.0,   0.0, &
                                                0.0, -30.0, -30.0, &
                                                0.0, -30.0,   0.0, &
                                                0.0, -30.0, -30.0, &
                                              -30.0, -60.0, -30.0, &
                                              -30.0, -60.0, -60.0, &
                                              -30.0, -60.0, -30.0, &
                                              -30.0, -60.0, -30.0, &
                                              -30.0, -60.0, -60.0, &
                                              -30.0, -60.0, -30.0, &
                                              -30.0, -60.0, -30.0, &
                                              -30.0, -60.0, -60.0, &
                                              -30.0, -60.0, -30.0, &
                                              -30.0, -60.0, -30.0, &
                                              -30.0, -60.0, -60.0, &
                                              -30.0, -60.0, -30.0, &
                                              -30.0, -60.0, -30.0, &
                                              -30.0, -60.0, -60.0, &
                                              -30.0, -60.0, -30.0, &
                                              -60.0, -90.0, -60.0, &
                                              -60.0, -90.0, -60.0, &
                                              -60.0, -90.0, -60.0, &
                                              -60.0, -90.0, -60.0, &
                                              -60.0, -90.0, -60.0 /), (/3,80/)))

    cell_vertices_lons = transpose(reshape((/ 0.0, 324.0,  36.0, &
                                              0.0,  36.0, 108.0, &
                                              0.0, 108.0, 180.0, &
                                              0.0, 180.0, 252.0, &
                                              0.0, 252.0, 324.0, &
                                            324.0, 324.0,   0.0, &
                                            324.0,   0.0,  36.0, &
                                             36.0,   0.0,  36.0, &
                                             36.0,  36.0,  72.0, &
                                             36.0,  72.0, 108.0, &
                                            108.0,  72.0, 108.0, &
                                            108.0, 108.0, 144.0, &
                                            108.0, 144.0, 180.0, &
                                            180.0, 144.0, 180.0, &
                                            180.0, 180.0, 216.0, &
                                            180.0, 216.0, 252.0, &
                                            252.0, 216.0, 252.0, &
                                            252.0, 252.0, 288.0, &
                                            252.0, 288.0, 324.0, &
                                            324.0, 288.0, 324.0, &
                                            324.0, 342.0,   0.0, &
                                              0.0, 342.0,  18.0, &
                                              0.0,  18.0,  36.0, &
                                             36.0,  18.0,  54.0, &
                                             36.0,  54.0,  72.0, &
                                             72.0,  54.0,  90.0, &
                                             72.0,  90.0, 108.0, &
                                            108.0,  90.0, 126.0, &
                                            108.0, 126.0, 144.0, &
                                            144.0, 126.0, 162.0, &
                                            144.0, 162.0, 180.0, &
                                            180.0, 162.0, 198.0, &
                                            180.0, 198.0, 216.0, &
                                            216.0, 198.0, 234.0, &
                                            216.0, 234.0, 252.0, &
                                            252.0, 234.0, 270.0, &
                                            252.0, 270.0, 288.0, &
                                            288.0, 270.0, 306.0, &
                                            288.0, 306.0, 324.0, &
                                            324.0, 306.0, 342.0, &
                                            342.0,   0.0,  18.0, &
                                             18.0,   0.0,  36.0, &
                                             18.0,  36.0,  54.0, &
                                             54.0,  36.0,  72.0, &
                                             54.0,  72.0,  90.0, &
                                             90.0,  72.0, 108.0, &
                                             90.0, 108.0, 126.0, &
                                            126.0, 108.0, 144.0, &
                                            126.0, 144.0, 162.0, &
                                            162.0, 144.0, 180.0, &
                                            162.0, 180.0, 198.0, &
                                            198.0, 180.0, 216.0, &
                                            198.0, 216.0, 234.0, &
                                            234.0, 216.0, 252.0, &
                                            234.0, 252.0, 270.0, &
                                            270.0, 252.0, 288.0, &
                                            270.0, 288.0, 306.0, &
                                            306.0, 288.0, 324.0, &
                                            306.0, 324.0, 342.0, &
                                            342.0, 324.0,   0.0, &
                                              0.0,   0.0,  36.0, &
                                             36.0,   0.0,  72.0, &
                                             36.0,  72.0,  72.0, &
                                             72.0,  72.0, 108.0, &
                                            108.0,  72.0, 144.0, &
                                            108.0, 144.0, 144.0, &
                                            144.0, 144.0, 180.0, &
                                            180.0, 144.0, 216.0, &
                                            180.0, 216.0, 216.0, &
                                            216.0, 216.0, 252.0, &
                                            252.0, 216.0, 288.0, &
                                            252.0, 288.0, 288.0, &
                                            288.0, 288.0, 324.0, &
                                            324.0, 288.0,   0.0, &
                                            324.0,   0.0,   0.0, &
                                              0.0,   0.0,  72.0, &
                                             72.0,   0.0, 144.0, &
                                            144.0,   0.0, 216.0, &
                                            216.0,   0.0, 288.0, &
                                            288.0,   0.0,   0.0 /), (/3,80/)))
    pixel_center_lats = (/ 82.5, 67.5, 52.5, 37.5, 22.5, 7.5, &
                           -7.5,-22.5,-37.5,-52.5,-67.5,-82.5 /)
    pixel_center_lons = (/ 4.5, 13.5, 22.5, 31.5, 40.5, 49.5, 58.5, 67.5, 76.5, 85.5, &
                          94.5,103.5,112.5,121.5,130.5,139.5,148.5,157.5,166.5,175.5, &
                         184.5,193.5,202.5,211.5,220.5,229.5,238.5,247.5,256.5,265.5, &
                         274.5,283.5,292.5,301.5,310.5,319.5,328.5,337.5,346.5,355.5 /)
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
    do i = 1,12
      write(debug,'(20(I3))') (cell_numbers_data(i,j),j=1,20)
      write(*,*) trim(debug)
    end do
    do i = 1,12
      write(debug2,'(XXXXXXXXXXXXX20(I3))') (cell_numbers_data(i,j),j=21,40)
      write(*,*) trim(debug2)
    end do
    call ncg_mapper%generate_limits()
    call ncg_mapper%icon_icosohedral_cell_latlon_pixel_ncg_mapper_destructor()
    call coarse_grid_shape%generic_1d_section_coords_destructor()
    deallocate(coarse_grid)
    deallocate(fine_grid_shape)
    deallocate(coarse_grid_shape)
    deallocate(cell_vertices_lats)
    deallocate(cell_vertices_lons)
    deallocate(pixel_center_lats)
    deallocate(pixel_center_lons)
    deallocate(pixel_center_lats_field)
    deallocate(pixel_center_lons_field)
    deallocate(pixel_center_lats_ptr_2d)
    deallocate(pixel_center_lons_ptr_2d)
    deallocate(cell_neighbors)
    deallocate(secondary_neighbors)
    deallocate(cell_vertex_coords_data)
    deallocate(cell_vertex_coords)
end subroutine testLatLonToIconGrids

end module map_non_coincident_grids_test_mod
