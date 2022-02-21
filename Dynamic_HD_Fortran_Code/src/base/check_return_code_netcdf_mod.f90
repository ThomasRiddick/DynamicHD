module check_return_code_netcdf_mod

use netcdf
implicit none

contains

subroutine check_return_code(return_code)
  integer, intent(in) :: return_code
  if(return_code /= nf90_noerr) then
    print *,trim(nf90_strerror(return_code))
    stop
  end if
end subroutine check_return_code

end module check_return_code_netcdf_mod
