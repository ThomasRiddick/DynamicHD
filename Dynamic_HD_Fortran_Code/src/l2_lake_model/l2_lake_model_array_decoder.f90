module l2_lake_model_array_decoder_mod

preproc latlon vs single index

use l2_lake_model_mod

implicit none

type arraydecoder
  integer, dimension(:) :: array
  integer :: current_index
  integer :: object_count
  integer :: object_start_index
  integer :: expected_total_objects
  integer :: expected_object_length

end type arraydecoder

contains

function arraydecoderconstructor(array) result(constructor)
  integer, dimension(:), allocatable, intent(in) :: array
  type(arraydecoder) :: constructor
    constructor%array => array
    constructor%current_index = 2
    constructor%object_count = 0
    constructor%object_start_index = 0
    constructor%expected_total_objects = nint(array(1))
    constructor%expected_object_length = 0
end function arraydecoderconstructor

subroutine start_next_object(decoder)
  type(arraydecoder), intent(inout) :: decoder
    decoder%expected_object_length = nint(decoder%array(decoder.current_index))
    decoder%current_index = decoder.current_index + 1
    decoder%object_count = decoder.object_count + 1
    !Expected object length excludes the first entry (i.e. the length itself)
    decoder%object_start_index = decoder%current_index
end subroutine start_next_object

subroutine finish_object(decoder)
  type(arraydecoder), intent(inout) :: decoder
    if (decoder%expected_object_length /= &
        decoder%current_index - decoder%object_start_index) then
      write(*,*) "Object read incorrectly - length doesn't match expectation"
      stop
    end if
end subroutine finish_object

subroutine finish_array(decoder)
  type(arraydecoder), intent(inout) :: decoder
    if (decoder.object_count /= decoder.expected_total_objects) then
      write(*,*) "Array read incorrectly - number of object doesn't match expectation"
      stop
    end if
    if (length(decoder.array) /= decoder.current_index - 1 ) then
      write(*,*) "Array read incorrectly - length doesn't match expectation"
      stop
    end if
end subroutine finish_array

function read_float(decoder) result(value)
  type(arraydecoder), intent(inout) :: decoder
  real(dp) :: value
    value = decoder%array(decoder%current_index)
    decoder.current_index = decoder%current_index + 1
end function read_float

function read_integer(decoder) result(value)
  type(arraydecoder), intent(inout) :: decoder
  integer :: value
    value = nint(read_float(decoder))
end function read_integer

function read_bool(decoder) result(value)
  type(arraydecoder), intent(inout) :: decoder
  logical :: value
    value = (read_float(decoder) == 1.0_dp)
end function read_bool

function read_coords_latlon(decoder) result(coords_arr)
  type(arraydecoder), intent(inout) :: decoder
  integer, dimension(2) :: coords_arr
  !Reverse indices
    coords_arr(2) = nint(decoder%array(decoder.current_index))
    coords_arr(1) = nint(decoder%array(decoder.current_index+1))
    decoder%current_index = decoder%current_index + 2
end function read_coords

function read_field(decoder) result(field)
  type(arraydecoder), intent(inout) :: decoder
  integer :: field_length
  real(dp), dimension(:), allocatable :: field
    field_length = nint(decoder%array[decoder%current_index])
    decoder%current_index = decoder%current_index + 1
    allocate(field(field_length))
    field(:) = decoder%array(decoder%current_index:decoder%current_index+field_length-1)
    decoder%current_index = decoder%current_index + field_length
end function read_field

function read_integer_field(decoder) result(field)
  type(arraydecoder), intent(inout) :: decoder
  integer :: field_length
  real(dp), dimension(:), allocatable :: field_as_real
  integer, dimension(:), allocatable :: field
    field_as_real = read_field(decoder)
    field_length = size(field_as_real)
    allocate(field(field_length))
    field = nint(field_as_real)
end function read_integer_field

! function read_outflow_points_dict(decoder;single_index=false)
!   type(arraydecoder), intent(inout) :: decoder
!   length::Int64 = Int64(decoder.array[decoder.current_index])
!   decoder.current_index += 1
!   entry_length::Int64 = single_index ? 3 : 4
!   offset::Int64 = single_index ? 0 : 1
!   outflow_points = Dict{Int64,Redirect}()
!   for ___ in 1:length
!     entry::Array{Float64} =
!       decoder.array[decoder.current_index:decoder.current_index+entry_length-1]
!     decoder.current_index += entry_length
!     lake_number::Int64 = Int64(entry[1])
!     is_local::Bool = Bool(entry[3+offset])
!     local coords::CartesianIndex
!     if ! is_local
!       coords_array::Array{Int64} = single_index ? Int64[Int64(entry[2])] :
!                                    [Int64(x) for x in entry[2:3]]
!       coords = CartesianIndex(coords_array...)
!     else
!       coords = CartesianIndex(-1)
!     end
!     redirect::Redirect = Redirect(is_local,lake_number,coords)
!     outflow_points[lake_number] = redirect
!   end
!   return outflow_points
! end

function read_filling_order(decoder,single_index) result(filling_order)
  type(arraydecoder), intent(inout) :: decoder
  type(cell), dimension(:) :: filling_order
  real(dp), dimension(:) :: entry
  logical :: single_index
  real(dp):: threshold
  real(dp):: height
  integer :: length
  integer :: entry_length
  integer :: offset
  integer :: i
    length = nint(decoder%array(decoder%current_index))
    decoder%current_index = decoder%current_index + 1
    if (single_index) then
      entry_length = 4
      offset = 0
    else
      entry_length = 5
      offset = 1
    end if
    allocate(filling_order(length))
    allocate(entry(entry_length))
    do i = 1,length
      entry(:) = &
        decoder%array(decoder%current_index:decoder%current_index+entry_length-1)
      decoder%current_index = decoder%current_index + entry_length
      coords::Array{Int64} = single_index ?
                             Int64[Int64(entry[1])] :
                             [Int64(x) for x in entry[1:2]]
      height_type_int::Int64 = Int64(entry[2+offset])
      height_type::HeightType = height_type_int == 1 ? flood_height : connect_height
      threshold = entry(3+offset)
      height = entry(4+offset)
      filling_order(i) = Cell(CartesianIndex(coords...),height_type,threshold,height))
    end do
end function read_filling_order

function get_lake_parameters_from_array(array::Array{Float64},fine_grid::Grid,
                                        coarse_grid::Grid;single_index=false) result(lake_parameters)
  integer, dimension(:), allocatable, intent(in) :: array
  type(arraydecoder) :: decoder
  integer :: lake_number
  integer :: primary_lake
  integer :: i
    decoder = arraydecoder(array)
    allocate(lake_parameters(decoder%expected_total_objects))
    do i = 1,decoder%expected_total_objects
      call start_next_object(decoder)
      lake_number = read_integer(decoder)
      primary_lake = read_integer(decoder)
      secondary_lakes::Array{Int64} = read_field(decoder;integer_field=true)
      center_coords::CartesianIndex = read_coords(decoder;
                                                  single_index=single_index)
      filling_order::Vector{Cell} =
        read_filling_order(decoder;
                           single_index=single_index)
      outflow_points::Dict{Int64,Redirect} =
        read_outflow_points_dict(decoder;
                                 single_index=single_index)
      finish_object(decoder)
      push!(lake_parameters,LakeParameters(lake_number,
                                           primary_lake,
                                           secondary_lakes,
                                           center_coords,
                                           filling_order,
                                           outflow_points,
                                           fine_grid,
                                           coarse_grid))
    end do
    call finish_array(decoder)
end

end module l2_lake_model_array_decoder_mod

