MODULE mod_grid_flow_direction_kernels

!Module containing flow direction finding kernels

IMPLICIT NONE

CONTAINS

FUNCTION HDgrid_fdir_kernel(orog_section) RESULT(flow_direction)

!Kernel for find the flow direction (given as a number from 1-9 according to
!the directions given by a numeric keypad on a keyboard; this number must be
!have type float64/REAL*8 even though it only ever takes integer values to
!satify type checking. If muliply mimimum occur then take the first unless
!there is another one in the centre of the cell in which case set flow to 5
!(a sink point)

!Note: should ideally replace REAL*8 with a more modern SELECT_REAL_KIND based
!kind specification

IMPLICIT NONE

!Variables with intent IN
!A 3 by 3 input orography section flattened into a single dimensional array
REAL*8, INTENT(IN), DIMENSION(1:9) :: orog_section

!Local Variables
INTEGER :: i               !A counter
INTEGER :: min_coord       !The coordinate of the minimum value
REAL*8  :: flow_direction  !The calculated flow direction to return
!The flow directions; simply an array initialised with counting numbers
REAL*8, DIMENSION(1:9) :: flow_directions = [(REAL(i,8),i=1,9)]

 !Find the location of the first minimum in orog_section
 min_coord = MINLOC(orog_section,1)
 !And use that as an index to find the flow direction
 flow_direction = flow_directions(min_coord)
 !Check if this grid cell is also a minimum
 IF ( orog_section(5) == MINVAL(orog_section) ) THEN
    flow_direction = 5
 END IF

END FUNCTION HDgrid_fdir_kernel

END MODULE mod_grid_flow_direction_kernels
