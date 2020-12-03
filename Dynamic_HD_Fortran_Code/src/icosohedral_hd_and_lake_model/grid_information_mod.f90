module grid_information_mod

  !Get the double precision parameter from the lake model
  use icosohedral_lake_model_mod, only: dp
  !An object containing information about an icon
  !icosohedral grid
  type, public :: gridinformation
  integer :: ncells !The number of cells in the grid
  real(dp),pointer,dimension(:) :: clat !the latitudes of the cell centers
  real(dp),pointer,dimension(:) :: clon !the longitudes of the cell centers
  real(dp),pointer,dimension(:,:) :: clat_bounds !the latitudes of the boundaries of the cells
  real(dp),pointer,dimension(:,:) :: clon_bounds !the longitudes of the boundaries of the cells
  end type gridinformation

end module grid_information_mod
