MODULE mod_topo_io

!Module containing subroutines to read and write FORTRAN binary data arrays using
!a big endian 32 bit float format (REAL(4))

IMPLICIT NONE
  
CONTAINS
 
SUBROUTINE read_topo(filename,nlat,nlong,topdata)

!Read arrays of fortran binary data
 
IMPLICIT NONE

!Variable with intent IN
INTEGER, INTENT(IN)             :: nlat                 !Dimensions of grid
INTEGER, INTENT(IN)             :: nlong                !Dimensions of grid
CHARACTER(len=*), INTENT(IN)    :: filename             !filename
!Variables with intent OUT
!topography field; note order of the indices
REAL(4), INTENT(OUT)            :: topdata(nlong,nlat)

!Local Variables
INTEGER                         :: fileunit=1           !file unit
INTEGER                         :: ios                  !file reading error code

   !Open the file, read into topodata and close
   OPEN(UNIT=fileunit, FILE=filename, STATUS='old', FORM='unformatted',        &
        IOSTAT=ios, CONVERT='big_endian')
   READ(fileunit) topdata
   CLOSE(fileunit)

END SUBROUTINE read_topo

SUBROUTINE write_topo(filename,topodata)

!Write arrays of fortran binary data

IMPLICIT NONE

!Variables with intent IN
CHARACTER(len=*), INTENT(IN)           :: filename             !filename
!topography field
REAL(4), INTENT(IN), DIMENSION(:,:)    :: topodata

!Local Variables
INTEGER                         :: fileunit=1           !file unit
INTEGER                         :: ios                  !file reading error code

   !Open the file, write topodata and close
   OPEN(UNIT=fileunit, FILE=filename, STATUS='replace', FORM='unformatted',        &
        IOSTAT=ios, CONVERT='big_endian')
   WRITE(fileunit) topodata
   CLOSE(fileunit)

END SUBROUTINE write_topo

END MODULE mod_topo_io
