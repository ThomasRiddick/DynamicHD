module cross_grid_mapper_driver_mod

use cross_grid_mapper
use precision_mod
implicit none

contains

  subroutine cross_grid_mapper_latlon_to_icon_f2py_wrapper( &
      cell_neighbors,pixel_center_lats,pixel_center_lons, &
      cell_vertices_lats,cell_vertices_lons, &
      output_cell_numbers, &
      nlat_fine,nlon_fine,ncells_coarse)
    integer, intent(in) :: ncells_coarse
    integer, intent(in) :: nlat_fine,nlon_fine
    real, intent(in), dimension(nlat_fine) :: pixel_center_lats
    real, intent(in), dimension(nlon_fine) :: pixel_center_lons
    !Need to set precision using a constant for f2py to work correctly
    real(kind=8), intent(in), dimension(ncells_coarse,3) :: cell_vertices_lats
    real(kind=8), intent(in), dimension(ncells_coarse,3) :: cell_vertices_lons
    integer, intent(in), dimension(ncells_coarse,3), target :: cell_neighbors
    integer, intent(out), dimension(nlat_fine,nlon_fine) :: output_cell_numbers
    integer, dimension(:,:), pointer :: cell_neighbors_ptr
    real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lats_deg
    real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lons_deg
    real(kind=double_precision), dimension(:), allocatable :: pixel_center_lats_alloc
    real(kind=double_precision), dimension(:), allocatable :: pixel_center_lons_alloc
    real(kind=double_precision), parameter :: ABS_TOL_FOR_DEG = 0.00001
    real(kind=double_precision), parameter :: PI = 4.0*atan(1.0)
    integer, dimension(:,:), pointer :: output_cell_numbers_ptr
    integer :: i,j
      allocate(pixel_center_lats_alloc(nlat_fine))
      allocate(pixel_center_lons_alloc(nlon_fine))
      cell_neighbors_ptr => cell_neighbors
      pixel_center_lats_alloc = real(pixel_center_lats,double_precision)
      pixel_center_lons_alloc = real(pixel_center_lons,double_precision)
      allocate(cell_vertices_lats_deg(ncells_coarse,3))
      allocate(cell_vertices_lons_deg(ncells_coarse,3))
      do j = 1,3
        do i=1,ncells_coarse
          cell_vertices_lats_deg(i,j) = cell_vertices_lats(i,j)*(180.0/PI)
          if(cell_vertices_lats_deg(i,j) > 90.0 - ABS_TOL_FOR_DEG) then
            cell_vertices_lats_deg(i,j) = 90.0
          end if
          if(cell_vertices_lats_deg(i,j) < -90.0 + ABS_TOL_FOR_DEG) then
            cell_vertices_lats_deg(i,j) = -90.0
          end if
          cell_vertices_lons_deg(i,j) = cell_vertices_lons(i,j)*(180.0/PI)
        end do
      end do
      call cross_grid_mapper_latlon_to_icon(pixel_center_lats_alloc, &
                                            pixel_center_lons_alloc, &
                                            cell_vertices_lats_deg, &
                                            cell_vertices_lons_deg, &
                                            cell_neighbors_ptr, &
                                            output_cell_numbers_ptr, &
                                            .true.)
      output_cell_numbers(:,:) = output_cell_numbers_ptr(:,:)
      deallocate(cell_vertices_lats_deg)
      deallocate(cell_vertices_lons_deg)
      deallocate(pixel_center_lats_alloc)
      deallocate(pixel_center_lons_alloc)
  end subroutine cross_grid_mapper_latlon_to_icon_f2py_wrapper

end module cross_grid_mapper_driver_mod
