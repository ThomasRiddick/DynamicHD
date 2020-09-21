module grid_information_mod
  type, public :: gridinformation
  integer :: ncells
  real,pointer,dimension(:) :: clat
  real,pointer,dimension(:) :: clon
  real,pointer,dimension(:,:) :: clat_bounds
  real,pointer,dimension(:,:) :: clon_bounds
  end type gridinformation
end module grid_information_mod
