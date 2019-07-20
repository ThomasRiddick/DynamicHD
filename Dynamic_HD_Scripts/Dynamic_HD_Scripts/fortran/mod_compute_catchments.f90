MODULE mod_compute_catchments

!This module contain various subroutines related to the computation of
!catchments. Two of these can be called as externally. The main catchment
!number computing function is 'compute_catchments'; all the internal subroutines
!are directly or indirectly called by this. The other function that can be called
!externally, which is not called by 'compute_catchments' but intended to called
!seperately by this modules python wrapper, is relabel_catchments. This simple
!gives new labels to an existing field/set of catchments

IMPLICIT NONE

!Define parameters

!Define codes within river flow direction used to identify various sinks
INTEGER, PARAMETER :: param_ocean_point =  -1
INTEGER, PARAMETER :: param_coast_point =   0
INTEGER, PARAMETER :: param_sink_point  =  5

!Define internal labels for different sink/pseudo-sink types
INTEGER, PARAMETER :: param_no_sink_found                 = 0
INTEGER, PARAMETER :: param_coast_type_sink               = 1
INTEGER, PARAMETER :: param_ocean_type_sink               = 2
INTEGER, PARAMETER :: param_local_type_sink               = 3
INTEGER, PARAMETER :: param_unknown_river_direction_value = 4
INTEGER, PARAMETER :: param_flow_over_pole                = 5
INTEGER, PARAMETER :: param_circular_flow                 = 6

!file unit for loops found log file
INTEGER, PARAMETER :: fileunit                           = 1

CONTAINS

SUBROUTINE compute_catchments(rivdir_field, circ_flow_check_period_factor, &
                              log_file_for_loops,nlong,nlat, &
                              sink_types_found, catchments_field)

!Given a river direction field compute the different catchments within it and
!label them in the order of discovery by looping over all points in developing
!catchments_field and the flow from points that are not yet in any catchment
!until a catchment is found for them.

IMPLICIT NONE

!Variables with intent IN
!File to log the location of any loops found in
CHARACTER(LEN=*), INTENT(INOUT)           :: log_file_for_loops
!Field of river flow direction (labelled according to directions on a numeric
!keypad)
INTEGER, INTENT(IN)                         :: nlong !number of longitude points
INTEGER, INTENT(IN)                         :: nlat  !number of latitude points
INTEGER, INTENT(IN), DIMENSION(nlong,nlat)  :: rivdir_field
!The total number of grid points is divide by this factor to
!give a frequency with which to check for circular flows
INTEGER, INTENT(IN)                         :: circ_flow_check_period_factor

!Variables with intent OUT
!Field of values labelling catchments according to a number
INTEGER, INTENT(OUT), DIMENSION(nlong,nlat) :: catchments_field
!The number of each sink type (index by label number) found so far
INTEGER, INTENT(OUT), DIMENSION(6) :: &
 sink_types_found

!Local variables
!co-ordinates of possible starting point under consideration
INTEGER :: i,j
INTEGER :: catchment_number
INTEGER :: sink_type_found
INTEGER, PARAMETER :: print_out_frequency = 25

!initialize variables
catchments_field = 0
catchment_number = 1
sink_type_found  = 0

!Open a file for logging any loops found
OPEN(UNIT=fileunit, FILE=log_file_for_loops, STATUS='replace', &
     FORM='formatted',ACTION="WRITE")
WRITE(fileunit,*) 'Loops found in catchments:'
CLOSE(fileunit)

!Consider possible start points (unfilled in catchments in catchments_field)
!and follow the flow direction from any start point found until the river merges
!into another catchment or finds a sink point.
DO i=1,nlong
 DO j=1,nlat
  IF ( catchments_field(i,j) == 0 ) THEN
   CALL follow_river(rivdir_field,catchments_field,catchment_number, i, j, &
                     circ_flow_check_period_factor,log_file_for_loops,nlong, &
                     nlat,sink_type_found)
    IF ( sink_type_found /= 0 ) THEN
    !If we have found a sink increment the count of the number of this type
    !of sink by one
     sink_types_found(sink_type_found) = sink_types_found(sink_type_found) + 1
    END IF
  END IF
 END DO
 !Print out only as frequently as specified by print out frequency
 IF ( MOD(i,print_out_frequency) == 0) THEN
    WRITE (*,*) 'Finished processing column: ', i
 END IF
END DO

END SUBROUTINE compute_catchments

SUBROUTINE follow_river(rivdir_field,catchments_field,catchment_number,     &
                        initial_i,initial_j, circ_flow_check_period_factor, &
                        log_file_for_loops,nlong,nlat,sink_type_found)

!Follow a given river till it reaches a sink or merges into a catchment that
!has been defined already. First try to find next grid cell and deal with
!special case of flow over pole and flow wrapping over vertical edges of grid.
!Then label the catchment if it has either merged with another or found a sink.
!Then check for loops (this check is only called occasionally).

IMPLICIT NONE

!Variables with intent IN
!File to log the location of any loops found in
CHARACTER(LEN=*), INTENT(IN)                :: log_file_for_loops
!Field of river flow direction (labelled according to directions on a numeric
!keypad)
INTEGER, INTENT(IN), DIMENSION(nlong,nlat) :: rivdir_field
!Co-ordinates of grid point to start at
INTEGER, INTENT(IN)                        :: initial_i, initial_j
!The total number of grid points is divide by this factor to
!give a frequency with which to check for circular flows
INTEGER, INTENT(IN)                        :: circ_flow_check_period_factor
INTEGER, INTENT(IN)                        :: nlong !number of longitude points
INTEGER, INTENT(IN)                        :: nlat !number of latitude points

!Fields with intent INTOUT
!Field of values labelling catchments according to a number
INTEGER, INTENT(INOUT), DIMENSION(nlong,nlat) :: catchments_field
!The number with which to label this catchment if it is a new catchment
INTEGER, INTENT(INOUT)                        :: catchment_number

!Variables with intent OUT
INTEGER, INTENT(OUT) :: sink_type_found

!Local Variables
!List of grid points in this catchment found so far
!It would be better to use an actual list than a single very long array
!however this would take longer to implement
INTEGER, DIMENSION(2,SIZE(catchments_field)) :: grid_points_in_catchment

!Number of points in grid_points_in_catchment
INTEGER :: grid_point_count
!Did the last iteration bring us to the end of the river; either a sink or
!the edge of another catchment
LOGICAL :: end_of_river_found
INTEGER :: i,j   !Coordinate of current gridpoint or after update next gridpoint
INTEGER :: counter ! a loop counter
!Number of iteration to wait before checking for a circular flow
INTEGER :: circ_flow_check_period
INTEGER :: num_points !Total number of grid points

!initialise variables
num_points = nlong*nlat
circ_flow_check_period = FLOOR(1.0*num_points/circ_flow_check_period_factor)
grid_points_in_catchment = 0
grid_point_count = 0
sink_type_found = 0
end_of_river_found = .FALSE.
i = initial_i
j = initial_j

DO WHILE (.NOT. end_of_river_found)
 grid_point_count = grid_point_count + 1
 grid_points_in_catchment(1:2,grid_point_count) = (/ i, j /)
 !compute which cell to go to next, this will update i and j
 CALL compute_next_grid_cell(rivdir_field(i,j),i,j,sink_type_found)
 !Fix cases where river flows over pole by setting cell to sink
 IF ( j < 1 ) THEN
    j = 1
    sink_type_found = param_flow_over_pole
 ELSE IF ( j > nlat ) THEN
    j = nlat
    sink_type_found = param_flow_over_pole
 END IF
 !Fix cases where river flows over side of grid (wrap)
 IF ( i < 1 ) THEN
    i = nlong - i
 ELSE IF ( i > nlong ) THEN
    i = i - nlong
 END IF
 !check if this cell is a sink
 IF ( sink_type_found /= 0 ) THEN
    !if so end this river search and label all points in the catchment found
    !with the next catchment number
    end_of_river_found = .TRUE.
    CALL label_catchment(catchment_number,catchments_field, &
                         grid_points_in_catchment,grid_point_count,nlong,nlat)
    catchment_number = catchment_number + 1
    !check if the next grid cell is another catchment
 ELSE IF ( catchments_field(i,j) /= 0 ) THEN
    !if so end this river search and label all points in the catchment found
    !with the catchment number of the other catchment
    end_of_river_found = .TRUE.
    !note catchments_field is the number to label the catchment with
    !the second argument is the entire catchment_field part of which
    !will be labelled
    CALL label_catchment(catchments_field(i,j),catchments_field, &
                         grid_points_in_catchment,grid_point_count,nlong,nlat)
    !occasionally check incase flow is circular
 ELSE IF ( MOD(grid_point_count,circ_flow_check_period) == 0 .OR. &
          grid_point_count >= num_points ) THEN
    !check for circular flow by seeing if current coordinates are already
    !in catchment
    DO counter=1,grid_point_count
        IF ( ALL(grid_points_in_catchment(1:2,counter) == (/ i, j /)) ) THEN
            !if river is circular stop calculating this catchment and record
            !problem. Note if all is just a requirement that both the i and j
            !values for this particular counter value match
            WRITE (*,'(A)') 'Warning: Circular flow found; continuing to next catchment'
            !writing to a file; appear to the easiest way to get a unknown amount of
            !information out of Fortran cleanly
            OPEN(UNIT=fileunit, FILE=log_file_for_loops, STATUS='old', &
                 FORM='formatted',POSITION="append",ACTION="write")
            WRITE(fileunit,'(I10)') catchment_number
            CLOSE(fileunit)
            sink_type_found = param_circular_flow
            end_of_river_found = .TRUE.
            CALL label_catchment(catchment_number,catchments_field,grid_points_in_catchment, &
                         grid_point_count,nlong,nlat)
            catchment_number = catchment_number + 1
            EXIT
        END IF
    END DO
    !if the grid point count has reached its maximum and no loop has been
    !found then something has gone very wrong
    IF (grid_point_count >= num_points .AND. .NOT. end_of_river_found) THEN
        WRITE (*,*) 'ERROR: Trying to store more co-ordinates than' // &
                        ' there are grid points without any apparent loops'
        CALL EXIT(1)
    END IF
 END IF
END DO
END SUBROUTINE follow_river

SUBROUTINE compute_next_grid_cell(value,i,j,sink_type_found)

!This subroutine returns the value of i or j of the next grid point given the
!input river flow direction file. If a sink has been found it returns the
!input values of i and j unchanged and also a sink_type_found value (that is
!not param_no_sink_found) to indicate what type of sink has been found

IMPLICIT NONE

!Variables with intent IN
!Value of river direction field for current grid point
INTEGER, INTENT(IN)    :: value

!Variables with intent INOUT
!Co-ordinates in grid. Initially those of current cell; will be updated to
!those of next cell. If a sink is found return those of current cell instead
INTEGER, INTENT(INOUT) :: i,j

!Variable with intent OUT
!The type of sink found; see parameter value at the start of module
INTEGER, INTENT(OUT)   :: sink_type_found

 !Setup variables
 !Initially set no sink found
 sink_type_found = param_no_sink_found

 !Decide on next action; declare this a sink or
 !calculate i and j of next cell
 SELECT CASE (value)
    CASE (param_coast_point)  ! A coast point
     sink_type_found = param_coast_type_sink
    CASE (param_ocean_point)  ! A ocean point
     sink_type_found = param_ocean_type_sink
    CASE (param_sink_point)   ! A local (land) sink
     sink_type_found = param_local_type_sink
    CASE (1:3)                ! Value points to a cell in row j - 1
     j = j + 1
     i = i + (value - 2)
    CASE (4,6)                ! Value points to a cell in row j
     i = i + (value - 5)
    CASE (7:9)                ! Value points to a cell in row j + 1
     j = j - 1
     i = i + (value - 8)
    CASE DEFAULT
     sink_type_found = param_unknown_river_direction_value
 END SELECT

END SUBROUTINE

SUBROUTINE label_catchment(catchment_number,catchments_field, &
                           grid_points_in_catchment,grid_point_count,nlong,nlat)

!Takes a list of points and a catchment number and numbers those points in the
!returned catchments_field array with that number

IMPLICIT NONE

!Labels the list of points given with the given catchment number
!Variables with intent IN
!The number with which to label this catchment
INTEGER, INTENT(IN) :: catchment_number
!List of grid points in this catchment found so far
!It would be better to use an actual list than a single very long array
!however this would take longer to implement
INTEGER, INTENT(IN), DIMENSION(:,:)   :: grid_points_in_catchment
!number of longitude points
INTEGER, INTENT(IN)                          :: nlong
!number of latitude points
INTEGER, INTENT(IN)                          :: nlat
!Number of points in grid_points_in_catchment
INTEGER, INTENT(IN) :: grid_point_count
!Variable with intent INOUT
!Field of values labelling catchments according to a number
INTEGER, INTENT(INOUT), DIMENSION(nlong,nlat) :: catchments_field
!local variables
INTEGER               :: i      !loop counter
INTEGER, DIMENSION(2) :: coords !coordinates of gridpoint

!iterate through as many elements in the list of points in the catchment
!as are filled
DO i=1,grid_point_count
 !label grid points in list with the given catchment number
 coords = grid_points_in_catchment(1:2,i)
 !check the coordinates are valid
 IF ( coords(1) < 0 .OR. coords(1) > nlong .OR. &
      coords(2) < 0 .OR. coords(2) > nlat) THEN
  WRITE (*,*) 'ERROR: Trying to add invalid set of coordinates to a catchment'
  CALL EXIT(0)
 END IF
 !label
 catchments_field(coords(1),coords(2)) = catchment_number
END DO

END SUBROUTINE label_catchment

SUBROUTINE relabel_catchments(catchments,old_to_new_label_map,nlong,nlat)

!Relabel catchments according to size. Take a 1D array filled of new catchment
!label numbers index by old label number and loop over a field of catchment
!replacing the old label number with the new ones

IMPLICIT NONE

!Variables with intent in
!An array with the new labels sorted so the index is the old
!label
INTEGER, INTENT(IN), DIMENSION(:) :: old_to_new_label_map
!number of longitude points
INTEGER, INTENT(IN) :: nlong
!number of latitude points
INTEGER, INTENT(IN) :: nlat
!Variables with intent inout
!Map of the world catchments to be relabelled
INTEGER, INTENT(INOUT), DIMENSION(nlong,nlat) :: catchments
!local variables
INTEGER :: i,j !counters

 !Iterate overall all cells and look up value in old_to_new_label_map
 !(which index by old label number) and replace the label with the
 !value found; which is the new label number
 DO i=1,nlong
  DO j=1,nlat
    IF (catchments(i,j) /= 0) THEN
        catchments(i,j) = old_to_new_label_map(catchments(i,j))
    END IF
  END DO
 END DO

END SUBROUTINE relabel_catchments

END MODULE mod_compute_catchments
