MODULE mod_grid_m_nbrs_kernels

IMPLICIT NONE

CONTAINS

FUNCTION HDgrid_masked_nbrs_kernel(input_mask) RESULT(number_nbrs)

IMPLICIT NONE

REAL, INTENT(IN), DIMENSION(1:9) :: input_mask

!Local Variables
INTEGER :: i  !A counter
REAL  :: number_nbrs  !The calculated flow direction to return
 number_nbrs = 0.0
 IF (input_mask(5) == 1.0) THEN
    DO i = 1,9
      IF(i==5) CYCLE
      IF(input_mask(i) == 1.0) number_nbrs = number_nbrs + 1.0
    ENDDO
 END IF

END FUNCTION HDgrid_masked_nbrs_kernel

END MODULE mod_grid_m_nbrs_kernels
