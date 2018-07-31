MODULE mod_local_minima_finding_kernels

IMPLICIT NONE

CONTAINS

FUNCTION latlon_local_min(orog_section) RESULT(is_minimum)


REAL*8, INTENT(IN), DIMENSION(1:9) :: orog_section

!Local Variables
INTEGER :: i
REAL*8 :: centre_value
REAL*8 :: is_minimum

    is_minimum = 1.0
    DO i = 1,9
        IF (i == 5) CYCLE
        centre_value = orog_section(5)
        IF (orog_section(i) < centre_value) is_minimum = 0.0
    END DO

END FUNCTION latlon_local_min
END MODULE mod_local_minima_finding_kernels
