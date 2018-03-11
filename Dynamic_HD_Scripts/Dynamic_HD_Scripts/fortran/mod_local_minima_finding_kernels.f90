MODULE mod_local_minima_finding_kernels

IMPLICIT NONE

CONTAINS

FUNCTION latlong_grid_is_local_minimum_kernel(orog_section) RESULT(is_mininum)


REAL*8, INTENT(IN), DIMENSION(1:9) :: orog_section

!Local Variables
INTEGER :: i
REAL*8 :: centre_value
REAL*8 :: is_minimum

    is_minimum = 1.0
    DO i = 1,8
        IF (i == 5) CYCLE
        centre_value = orog_section(5)
        IF (orog_section(i) < centre_value) is_minimum = 0.0
    END DO

END FUNCTION latlong_grid_is_local_minimum_kernel
END MODULE mod_local_minima_finding_kernels
