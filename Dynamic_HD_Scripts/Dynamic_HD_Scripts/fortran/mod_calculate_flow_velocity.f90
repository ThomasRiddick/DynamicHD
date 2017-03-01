module mod_calculate_flow_velocity

!Moudle contains a function to compute the flow velocity of a river using
!the retention parameter and the cell area and river directio  for diagnostic
!purposes. This is reserving a previous calculation to find the retention parameter

implicit none

contains

function calculate_flow_velocity(retention_constant,FDIR,DLON,nlat,nlon) &
result(flow_velocities)

!As stated above, reserves the retention constant calculation to get the flow parameter

implicit none

!Kind parameters for double precision from JSBACH, could potentially be moved
!to a seperate module to produce a more unified kind scheme with HD scripting
integer, parameter :: pd =  12
integer, parameter :: rd = 307
integer, parameter :: dp = SELECTED_REAL_KIND(pd,rd)

!Variables with intent IN
!Dimension of the input array
integer, intent(in) :: nlat, nlon
!Input retention constant array (in days?)
real(dp), intent(in), dimension(nlon,nlat) :: retention_constants
!River directions array (given as double for historical reasons although it always
!takes integer values).
real(dp), intent(in), dimension(nlon,nlat) :: FDIR
!Longitudal grid spacings (in metres)
real(dp), intent(in), dimension(nlat) :: DLON
!Local variables
!Flow velocities calculated (in m/s)
real(dp), dimension(nlon,nlat) :: flow_velocities
!Loop counters
integer :: JB,JL
!Distance between grid cells centres
real(dp) :: DX
!River flow retention constant normalisation constant
real(dp), parameter :: ARF_K0 = 0.4112
!Grid spacing normalisation parameter (metres)
real(dp), parameter :: ARF_DX = 228000.
!River flow velocity normalisation constant
real(dp), parameter :: ARF_V0 = 1.0039
!Latitudal grid spacing (metres)
real(dp), paramter :: DLAT
!From globuse.f value of DLAT
DLAT = DLON(2)

do JB = 1,nlat
    do JL = 1,nlon
         !Calculate relative coordinates between grid cell and neighbor it flows into
         IB = -( INT( (FDIR(JL,JB)-1.)/3. + 0.1) - 1)
         IL = INT( ( (FDIR(JL,JB)+2.)/3. -
     &        INT((FDIR(JL,JB)+2.)/3. + 0.1) ) * 3. + 0.1) -1
        !Thus calculate the distance between the centres of the grid cells
        DX = SQRT(IB*IB*DLAT*DLAT + IL*IL*DLON(JB)*DLON(JB))
        !Reserve the calculation of the retention constant to give a flow direction
        flow_velocities(JL,JB) = ARF_K0 * DX/ARF_DX * ARF_V0/retention_constants(JL,JB)
    end do
end do

end function calculate_flow_velocity

end module mod_calculate_flow_velocity
