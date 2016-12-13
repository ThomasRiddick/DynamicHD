MODULE mod_iterate_paths_map

!Module that has function used to generate maps of how many cells flow to each
!point. iterate_paths_map and sparse_iterator (for when less than a certain
!fraction of the points still need to be calculated) are called by the external
!python wrapper while count_accumulated_inflow is a internal function

IMPLICIT NONE

CONTAINS

FUNCTION iterate_paths_map(river_directions,paths_map,nlat,nlong) RESULT(unfinished)
                 
!Performs 1 iteration of the process of calculating a map of how many cells flow
!to each point

!Returns: Flag if the iterative process is complete (true) or not (false)

IMPLICIT NONE

!Variables with intent IN

!number of rows in original input array (excluding extra top/bottom
!edge rows added by a higher level routine; e.g. 720/360 for a half
!degree grid not 722/362 which will be actual size of input arrays)
INTEGER, INTENT(IN) :: nlat,nlong
!River directions input using keypad notation (1-9)
INTEGER, INTENT(IN), DIMENSION(:,:) :: river_directions

!Accumulated flow so far to each cell 
INTEGER, INTENT(INOUT), DIMENSION(:,:) :: paths_map
!Result
LOGICAL :: unfinished

!Local Variables 
INTEGER :: i,j                !loop counters (for lat and long)
!Temporary arrays to store wrapped sections in
INTEGER, DIMENSION(3,3) :: river_directions_section_temp
INTEGER, DIMENSION(3,3) :: paths_map_section_temp

 IF (ANY(paths_map == 0)) THEN
    !If there are still points to be calculated then perform another iteration
    unfinished = .TRUE.
    !Latitude and longitude contain the extra two rows and columns add by a higher
    !level routine
    DO i=1,nlat+2
        DO j=1,nlong
            !Note difference in indices from python (e.g. first row has index 1 
            !not zero)
            IF(i == 1 .OR. i == nlat+2 ) THEN
            !Deal with first and last columns and rows individually
                paths_map(i,j) = 1 
            ELSE IF (j == 1) THEN
                river_directions_section_temp(1:3,2:3) = river_directions(i-1:i+1,j:j+1)
                river_directions_section_temp(1:3,1:1) = river_directions(i-1:i+1,nlong:nlong)
                paths_map_section_temp(1:3,2:3) = paths_map(i-1:i+1,j:j+1)
                paths_map_section_temp(1:3,1:1) = paths_map(i-1:i+1,nlong:nlong)
                paths_map(i,j) = count_accumulated_inflow(river_directions_section_temp, &
                                                          paths_map_section_temp)
            ELSE IF (j == nlong) THEN
                river_directions_section_temp(1:3,1:2) = river_directions(i-1:i+1,j-1:j)
                river_directions_section_temp(1:3,3:3) = river_directions(i-1:i+1,1:1)
                paths_map_section_temp(1:3,1:2) = paths_map(i-1:i+1,j-1:j)
                paths_map_section_temp(1:3,3:3) = paths_map(i-1:i+1,1:1)
                paths_map(i,j) = count_accumulated_inflow(river_directions_section_temp, &
                                                          paths_map_section_temp)
            ELSE
            !Pass a 3 by 3 section around this cell to counting function
            !Note the difference in the indexing from that used in python 
            !(+1 instead of +2)
                paths_map(i,j) = count_accumulated_inflow(                                &
                                    river_directions(i-1:i+1,j-1:j+1),                    &
                                    paths_map(i-1:i+1,j-1:j+1))
            END IF
        END DO 
    END DO
 ELSE
    !If there are no points to be calculated then return appropriate flag  
    unfinished = .FALSE.
 END IF

END FUNCTION iterate_paths_map

SUBROUTINE sparse_iterator(river_directions,paths_map,nlat,nlong)

!Perform iteration of the process of calculating the last remaining ('sparse')
!points in a map of how many cells flow to each point

IMPLICIT NONE

!Variables with intent IN

!number of rows in original input array (excluding extra top/bottom
!edge rows added by a higher level routine); e.g. 720/360 for a
!half degree grid not 720/362 which will be actual size of input arrays)
INTEGER, INTENT(IN) :: nlat,nlong
!River directions input using keypad notation (1-9)
INTEGER, INTENT(IN), DIMENSION(1:nlat+2,1:nlong) :: river_directions

!Accumulated flow so far to each cell
INTEGER, INTENT(INOUT), DIMENSION(1:nlat+2,1:nlong) :: paths_map

!Local Variables
INTEGER :: i,j                !loop counters (for lat and long)
!the array of locations of the remaining points; would be better to replace
!this with a list
INTEGER, DIMENSION(2,SIZE(paths_map)) :: remaining_points
!the current number of remaining points
INTEGER :: npoints_current
!the number of remaining points for the next iteration
INTEGER :: npoints_new
INTEGER :: counter            !a loop counter
!Temporary arrays to store wrapped sections in
INTEGER, DIMENSION(3,3) :: river_directions_section_temp
INTEGER, DIMENSION(3,3) :: paths_map_section_temp

!variables
npoints_current = 0
npoints_new     = 0

!iterate over the paths_map array and find which points still need to be
!calculated
DO i=1,nlat+2
 DO j=1,nlong
  IF (paths_map(i,j) == 0) THEN
   npoints_current = npoints_current + 1
   remaining_points(:,npoints_current) = (/i,j/)
  END IF
 END DO
END DO

DO WHILE ( npoints_current > 0 )
 npoints_new = 0
 !iterate over the current remaining points calculating the accumulated inflow
 !where possible. Add points that can't be calculated to a new reduced list of
 !uncalculated points
 DO counter=1,npoints_current
  i = remaining_points(1,counter)
  j = remaining_points(2,counter)
  !Note difference in indices from python (e.g. first row has index 1
  !not zero)
  IF(i == 1 .OR. i == nlat+2) THEN
   !Shouldn't need to treat edge cases within the sparse iterator normally
   !but include code just in case
   !Deal with first and last columns and rows individually
   paths_map(i,j) = 1
  ELSE IF (j == 1) THEN
   river_directions_section_temp(1:3,2:3) = river_directions(i-1:i+1,j:j+1)
   river_directions_section_temp(1:3,1:1) = river_directions(i-1:i+1,nlong:nlong)
   paths_map_section_temp(1:3,2:3) = paths_map(i-1:i+1,j:j+1)
   paths_map_section_temp(1:3,1:1) = paths_map(i-1:i+1,nlong:nlong)
   paths_map(i,j) = count_accumulated_inflow(river_directions_section_temp, &
                                              paths_map_section_temp)
  ELSE IF (j == nlong) THEN
   river_directions_section_temp(1:3,1:2) = river_directions(i-1:i+1,j-1:j)
   river_directions_section_temp(1:3,3:3) = river_directions(i-1:i+1,1:1)
   paths_map_section_temp(1:3,1:2) = paths_map(i-1:i+1,j-1:j)
   paths_map_section_temp(1:3,3:3) = paths_map(i-1:i+1,1:1)
   paths_map(i,j) = count_accumulated_inflow(river_directions_section_temp, &
                                              paths_map_section_temp)
  ELSE
   !Pass a 3 by 3 section around this cell to counting function
   !Note the difference in the indexing from that used in python
   !(+1 instead of +2)
   paths_map(i,j) = count_accumulated_inflow(river_directions(i-1:i+1,j-1:j+1), &
                                              paths_map(i-1:i+1,j-1:j+1))
  END IF
   !if this point is still zero add it to a new list of points for the next
   !iteration
  IF ( paths_map(i,j) == 0 ) THEN
    npoints_new = npoints_new + 1
    remaining_points(:,npoints_new) = (/i,j/)
  END IF
 END DO !for DO counter=1,npoints_current
 IF (npoints_current == npoints_new) THEN
  WRITE (*,*) "WARNING: An iteration of the sparse iterator has produced " // &
              "no reduction in the number of points to be processed. "     // &
              "Will stop iterating and return current results."
  !Break out of current loop
  EXIT
 END IF
 npoints_current = npoints_new
 WRITE (*,'(A,I10)') "Remaining points to process: ", npoints_current
END DO  !for DO WHILE ( npoints_current > 0 )

END SUBROUTINE sparse_iterator

FUNCTION count_accumulated_inflow(river_directions_section,paths_map_section) &
    RESULT(flow_to_cell)

!count the accumulated inflow to a given cell if alls it neighbour that flow
!into it have been calculated already otherwise return zero

IMPLICIT NONE

!Variables with intent IN

!River directions input using keypad notation (1-9)
INTEGER, INTENT(IN), DIMENSION(:,:) :: river_directions_section 

!Variables with intent INOUT

!Accumulated flow so far to each cell 
INTEGER, INTENT(IN), DIMENSION(:,:) :: paths_map_section

!RESULT
INTEGER :: flow_to_cell       !number of points count as flowing into cell from
                              !neighbours

!Local variables
LOGICAL :: uncalculated_inflow !flag that a neighbouring cell that flow to this
                               !one has not had its inflow calculated yet
!The value of flow direction that surrounding cell need to have to flow to this 
!cell. This is the value directly across the numeric keypad from the flow 
!direction
INTEGER, PARAMETER, DIMENSION(1:9) :: values_for_inflow_flat = (/3,6,9, 2,5,8,&
                                                                 1,4,7/)
!Values as 3 by 3 array 
INTEGER, DIMENSION(1:3,1:3) :: values_for_inflow
INTEGER :: i,j                !loop counters

 !setup values for inflow
 values_for_inflow = RESHAPE(values_for_inflow_flat,(/3,3/))
 !further setup
 flow_to_cell = 0
 uncalculated_inflow = .FALSE.
 !Note the change in indices compared to the fortran code
 DO i=1,3
    DO j=1,3
        IF ( i == 2 .AND. j == 2) THEN
            !flow to self 
            flow_to_cell = flow_to_cell + 1
            !with this we have counted flow to self so don't run the rest of
            !the code in the loop by using ELSE IF
        ELSE IF ( values_for_inflow(i,j) == river_directions_section(i,j) ) THEN
            !inflow from a neighbour. First check if neighbour has been 
            !calculated yet; if so add it to inflow 
            IF ( paths_map_section(i,j) /= 0) THEN
                flow_to_cell = flow_to_cell + paths_map_section(i,j)
            ELSE
                uncalculated_inflow = .TRUE.
            END IF
        END IF
    END DO
 END DO

 IF (uncalculated_inflow) THEN
    !If this cell is not yet calculatable return zero
    flow_to_cell = 0
 ELSE IF (flow_to_cell < 1) THEN
    WRITE(*,*) 'Error: inflow less than 1'
    CALL EXIT(1)
 END IF

END FUNCTION count_accumulated_inflow

END MODULE mod_iterate_paths_map
