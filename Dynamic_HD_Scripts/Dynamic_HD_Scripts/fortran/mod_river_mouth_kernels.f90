MODULE mod_river_mouth_kernels

!Module containing kernels for marking rivers mouths using the ndimage generic
!filter method of numpy

IMPLICIT NONE

CONTAINS

FUNCTION latlongrid_river_mouth_kernel(flow_dirs_section) &
    RESULT(flow_direction)

!Kernel to mark a sea point as river mouth if any of its neighbors flow into it

IMPLICIT NONE

!Kind parameters for double precision from JSBACH, could potentially be moved
!to a seperate module to produce a more unified kind scheme with HD scripting
INTEGER, PARAMETER :: pd =  12
INTEGER, PARAMETER :: rd = 307
INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(pd,rd)

!Set up the direction values for special points (sea points and river mouths)
REAL(dp), PARAMETER :: sea_point_flow_dir_value         = -1.0
REAL(dp), PARAMETER :: river_mouth_point_flow_dir_value =  0.0

!Variables with intent IN
!A 3 by 3 input river directions section flattened into a single dimensional
!array
REAL(dp),INTENT(IN),DIMENSION(1:9) :: flow_dirs_section

!Local Variables
!The calculated flow direction value to return; this will either be the input
!value or -1 if this is a sea point which is a river mouth (has a river
!flowing into it)
REAL(dp) :: flow_direction
!The inversion (i.e. opposite) of the flow directions in a numeric keypad style
!grid
REAL(dp),DIMENSION(1:9) :: inverted_flow_directions = (/ 3,2,1,6,5,4,9,8,7 /)

    !Only process this point if it is a sea point, otherwise return its
    !value unchanged
    IF( flow_dirs_section(5) == sea_point_flow_dir_value ) THEN
        !Check if any rivers flow to it, if so make it a river mouth point
        !if not then return its value unchanged (i.e. leave it is a sea-point)
        IF ( ANY(flow_dirs_section == inverted_flow_directions) ) THEN
            flow_direction = river_mouth_point_flow_dir_value
        ELSE
            flow_direction = sea_point_flow_dir_value
        END IF
    ELSE
        flow_direction = flow_dirs_section(5)
    END IF

END FUNCTION latlongrid_river_mouth_kernel

END MODULE mod_river_mouth_kernels
