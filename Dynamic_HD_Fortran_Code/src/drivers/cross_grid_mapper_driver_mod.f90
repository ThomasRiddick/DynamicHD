module cross_grid_mapper_driver_mod

use cross_grid_mapper
use precision_mod
implicit none

contains

  subroutine cross_grid_mapper_latlon_to_icon_f2py_wrapper(&
      cell_neighbors,pixel_center_lats,pixel_center_lons, &
      cell_vertices_lats,cell_vertices_lons, &
      output_cell_numbers, &
      nlat_fine,nlon_fine,ncells_coarse)
    integer, intent(in) :: ncells_coarse
    integer, intent(in) :: nlat_fine,nlon_fine
    real, intent(in), dimension(nlat_fine) :: pixel_center_lats
    real, intent(in), dimension(nlon_fine) :: pixel_center_lons
    real, intent(in), dimension(ncells_coarse,3) :: cell_vertices_lats
    real, intent(in), dimension(ncells_coarse,3) :: cell_vertices_lons
    integer, intent(in), dimension(ncells_coarse,3), target :: cell_neighbors
    integer, intent(out), dimension(:,:),pointer :: output_cell_numbers
    integer, dimension(:,:), pointer :: cell_neighbors_ptr
    real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lats_alloc
    real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lons_alloc
    real(kind=double_precision), dimension(:), allocatable :: pixel_center_lats_alloc
    real(kind=double_precision), dimension(:), allocatable :: pixel_center_lons_alloc
      allocate(cell_vertices_lats_alloc(ncells_coarse,3))
      allocate(cell_vertices_lons_alloc(ncells_coarse,3))
      allocate(pixel_center_lats_alloc(nlat_fine))
      allocate(pixel_center_lons_alloc(nlon_fine))
      cell_neighbors_ptr => cell_neighbors
      pixel_center_lats_alloc = real(pixel_center_lats,double_precision)
      pixel_center_lons_alloc = real(pixel_center_lons,double_precision)
      cell_vertices_lats_alloc = real(cell_vertices_lats,double_precision)
      cell_vertices_lons_alloc = real(cell_vertices_lons,double_precision)
      call cross_grid_mapper_latlon_to_icon(pixel_center_lats_alloc, &
                                            pixel_center_lons_alloc, &
                                            cell_vertices_lats_alloc, &
                                            cell_vertices_lons_alloc, &
                                            cell_neighbors_ptr, &
                                            output_cell_numbers, &
                                            .true.)
      deallocate(cell_vertices_lats_alloc)
      deallocate(cell_vertices_lons_alloc)
      deallocate(pixel_center_lats_alloc)
      deallocate(pixel_center_lons_alloc)
  end subroutine cross_grid_mapper_latlon_to_icon_f2py_wrapper

end module cross_grid_mapper_driver_mod
