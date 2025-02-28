module l2_lake_model_array_decoder_mod

use l2_lake_model_mod

implicit none

type arraydecoder
  real(dp), dimension(:), pointer :: array
  integer :: current_index
  integer :: object_count
  integer :: object_start_index
  integer :: expected_total_objects
  integer :: expected_object_length
end type arraydecoder

interface arraydecoder
  procedure :: arraydecoderconstructor
end interface arraydecoder

contains

function arraydecoderconstructor(array) result(constructor)
  real(dp), dimension(:), pointer, intent(in) :: array
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
    decoder%expected_object_length = nint(decoder%array(decoder%current_index))
    decoder%current_index = decoder%current_index + 1
    decoder%object_count = decoder%object_count + 1
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
    if (decoder%object_count /= decoder%expected_total_objects) then
      write(*,*) "Array read incorrectly - number of object doesn't match expectation"
      stop
    end if
    if (size(decoder%array) /= decoder%current_index - 1 ) then
      write(*,*) "Array read incorrectly - length doesn't match expectation"
      stop
    end if
end subroutine finish_array

function read_float(decoder) result(value)
  type(arraydecoder), intent(inout) :: decoder
  real(dp) :: value
    value = decoder%array(decoder%current_index)
    decoder%current_index = decoder%current_index + 1
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

subroutine read_coords(decoder,_COORDS_ARG_coords_out_)
  type(arraydecoder), intent(inout) :: decoder
  _DEF_COORDS_coords_out_ _INTENT_out_
  real(dp), dimension(:), pointer :: coords_as_array
  integer :: entry_length
    _IF_USE_SINGLE_INDEX_
      entry_length = 1
      coords_as_arrays  => decoder%array(decoder%current_index)
    _ELSE_
      coords_as_array  => decoder%array(decoder%current_index:decoder%current_index+1)
      entry_length = 2
    _END_IF_USE_SINGLE_INDEX_
    _GET_COORDS_ _COORDS_coords_out_ _FROM_ _ARRAY_coords_as_array_
    decoder%current_index = decoder%current_index + entry_length
end subroutine read_coords

function read_field(decoder) result(field)
  type(arraydecoder), intent(inout) :: decoder
  integer :: field_length
  real(dp), dimension(:), pointer :: field
    field_length = nint(decoder%array(decoder%current_index))
    decoder%current_index = decoder%current_index + 1
    allocate(field(field_length))
    field(:) = decoder%array(decoder%current_index:decoder%current_index+field_length-1)
    decoder%current_index = decoder%current_index + field_length
end function read_field

function read_integer_field(decoder) result(field)
  type(arraydecoder), intent(inout) :: decoder
  integer :: field_length
  real(dp), dimension(:), pointer :: field_as_real
  integer, dimension(:), pointer :: field
    field_as_real => read_field(decoder)
    field_length = size(field_as_real)
    allocate(field(field_length))
    field = nint(field_as_real)
end function read_integer_field

function read_outflow_points_dict(decoder) result(outflow_points)
  type(arraydecoder), intent(inout) :: decoder
  type(redirectdictionary), pointer :: outflow_points
  real(dp), dimension(:), pointer :: entry
  type(redirect), pointer :: working_redirect
  _DEF_COORDS_coords_
  integer :: length
  integer :: entry_length
  integer :: offset
  integer :: lake_number
  logical :: is_local
  integer :: i
    length = nint(decoder%array(decoder%current_index))
    outflow_points => redirectdictionary(length)
    decoder%current_index = decoder%current_index +  1
    _IF_USE_SINGLE_INDEX_
      entry_length = 3
      offset = 0
    _ELSE_
      entry_length = 4
      offset = 1
    _END_IF_USE_SINGLE_INDEX_
    allocate(entry(entry_length))
    do i = 1,length
      entry => &
        decoder%array(decoder%current_index:decoder%current_index+entry_length-1)
      decoder%current_index = decoder%current_index + entry_length
      lake_number = nint(entry(1))
      is_local = (entry(3+offset) == 1.0_dp)
      if (.not. is_local) then
        _GET_COORDS_ _COORDS_coords_ _FROM_ _ARRAY_entry_ _OFFSET_1_
      else
        _ASSIGN_COORDS_coords_ = _VALUE_-1_
      end if
      working_redirect => redirect(is_local,lake_number,_COORDS_ARG_coords_)
      call add_entry_to_dictionary(outflow_points,lake_number,working_redirect)
    end do
    call finish_dictionary(outflow_points)
end

function read_filling_order(decoder) result(filling_order)
  type(arraydecoder), intent(inout) :: decoder
  type(cellpointer), dimension(:), pointer :: filling_order
  real(dp), dimension(:), pointer :: entry
  logical :: single_index
  real(dp):: threshold
  real(dp):: height
  _DEF_COORDS_coords_
  integer :: height_type
  integer :: height_type_original
  integer :: length
  integer :: entry_length
  integer :: offset
  integer :: i
    length = nint(decoder%array(decoder%current_index))
    decoder%current_index = decoder%current_index + 1
    _IF_USE_SINGLE_INDEX_
      entry_length = 4
      offset = 0
    _ELSE_
      entry_length = 5
      offset = 1
    _END_IF_USE_SINGLE_INDEX_
    allocate(filling_order(length))
    allocate(entry(entry_length))
    do i = 1,length
      entry(:) = &
        decoder%array(decoder%current_index:decoder%current_index+entry_length-1)
      decoder%current_index = decoder%current_index + entry_length
      _GET_COORDS_ _COORDS_coords_ _FROM_ _ARRAY_entry_
      height_type_original = nint(entry(2+offset))
      if (height_type_original == 1) then
        height_type = flood_height
      else
        height_type = connect_height
      end if
      threshold = entry(3+offset)
      height = entry(4+offset)
      filling_order(i) = cellpointer(cell(_COORDS_ARG_coords_, &
                                          height_type, &
                                          threshold, &
                                          height))
    end do
end function read_filling_order

function get_lake_parameters_from_array(array, &
                                        _NPOINTS_LAKE_, &
                                        _NPOINTS_HD_) result(lake_parameters)
  real(dp), dimension(:), pointer, intent(in) :: array
  _DEF_NPOINTS_LAKE_ _INTENT_in_
  _DEF_NPOINTS_HD_ _INTENT_in_
  type(lakeparameterspointer), dimension(:), pointer :: lake_parameters
  integer, dimension(:), pointer :: secondary_lakes
  type(cellpointer), dimension(:), pointer :: filling_order
  type(arraydecoder) :: decoder
  type(redirectdictionary), pointer :: outflow_points
  _DEF_COORDS_center_coords_
  integer :: lake_number
  integer :: primary_lake
  integer :: i
    decoder = arraydecoder(array)
    allocate(lake_parameters(decoder%expected_total_objects))
    do i = 1,decoder%expected_total_objects
      call start_next_object(decoder)
      lake_number = read_integer(decoder)
      primary_lake = read_integer(decoder)
      secondary_lakes => read_integer_field(decoder)
      call read_coords(decoder,_COORDS_ARG_center_coords_)
      filling_order => read_filling_order(decoder)
      outflow_points => read_outflow_points_dict(decoder)
      call finish_object(decoder)
      lake_parameters(i) = lakeparameterspointer(&
                              lakeparameters(lake_number, &
                              primary_lake, &
                              secondary_lakes, &
                              _COORDS_ARG_center_coords_, &
                              filling_order, &
                              outflow_points, &
                              _NPOINTS_LAKE_, &
                              _NPOINTS_HD_))

    end do
    call finish_array(decoder)
end

end module l2_lake_model_array_decoder_mod

