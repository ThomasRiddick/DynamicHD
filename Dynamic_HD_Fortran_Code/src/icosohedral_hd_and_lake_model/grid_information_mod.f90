module grid_information_mod

  !An object containing information about an icon
  !icosohedral grid
  type, public :: gridinformation
  integer :: ncells !The number of cells in the grid
  real,pointer,dimension(:) :: clat !the latitudes of the cell centers
  real,pointer,dimension(:) :: clon !the longitudes of the cell centers
  real,pointer,dimension(:,:) :: clat_bounds !the latitudes of the boundaries of the cells
  real,pointer,dimension(:,:) :: clon_bounds !the longitudes of the boundaries of the cells
  end type gridinformation

end module grid_information_mod
