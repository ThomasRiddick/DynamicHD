MODULE mod_res_size_init_kernel

!Module containing kernel function to help with the initialisation of initial
!reservoirs using the numpy idimage generic filter processing method

IMPLICIT NONE

CONTAINS

FUNCTION latlongrid_res_size_init_krnl(reservoir_size_section) &
    RESULT(reservoir_size)

!Initialise initial reservoirs that are zero to the maximum value in there
!neighborhood, which is defined as the cell itself and its eight direction
!neighbors, works for a latitude longitude grid

!Kind parameters for double precision from JSBACH, could potentially be moved
!to a seperate module to produce a more unified kind scheme with HD scripting
INTEGER, PARAMETER :: pd =  12
INTEGER, PARAMETER :: rd = 307
INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(pd,rd)

!Variables with intent IN
!A 3 by 3 input reservoir size section flattened into a single dimensional
!array
REAL(dp), INTENT(IN), DIMENSION(1:9) :: reservoir_size_section

!Local Variables
!The calculated reservoir size to return; this will be the highest value in the
!input section if the center value is zero. Otherwise the center value is returned
!unchanged
REAL(dp) :: reservoir_size

    !All zeros should of been explicitly set to zero therefore don't require any
    !floating point error tolerance here
    IF (reservoir_size_section(5) == 0.0) THEN
        reservoir_size = MAXVAL(reservoir_size_section)
    ELSE
        reservoir_size = reservoir_size_section(5)
    END IF

END FUNCTION latlongrid_res_size_init_krnl

END MODULE mod_res_size_init_kernel
