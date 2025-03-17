module  l2_calculate_lake_fractions_mod

implicit none

integer, parameter :: dp = selected_real_kind(12)
! intents

type :: pixel
  integer :: id
  integer :: lake_number
  logical :: filled
  _DEF_COORDS_fine_grid_coords_
  _DEF_COORDS_original_coarse_grid_coords_
  _DEF_COORDS_assigned_coarse_grid_coords_
  logical :: transferred
end type pixel

interface pixel
  procedure :: pixelconstructor
end interface pixel

type :: pixelpointer
  type(pixel), pointer :: pixel_pointer
end type pixelpointer

type :: lakecell
  _DEF_COORDS_coarse_grid_coords_
  integer :: all_lake_potential_pixel_count
  integer :: lake_potential_pixel_count
  integer, dimension(:), pointer :: lake_pixels
  integer, dimension(:), pointer :: lake_pixels_added
  integer :: pixel_count
  integer :: lake_pixel_count
  integer :: lake_pixels_added_count
  integer :: max_pixels_from_lake
  logical :: in_binary_mask
end type lakecell

interface lakecell
  procedure :: lakecellconstructor
end interface lakecell

type :: lakecellpointer
  type(lakecell), pointer :: lake_cell_pointer
end type lakecellpointer

type :: lakeinput
  integer :: lake_number
  _DEF_INDICES_LIST_lake_pixel_coords_list_INDEX_NAME_
  _DEF_INDICES_LIST_potential_lake_pixel_coords_list_INDEX_NAME_
  _DEF_INDICES_LIST_cell_coords_list_INDEX_NAME_
end type lakeinput

interface lakeinput
  procedure :: lakeinputconstructor
end interface lakeinput

type lakeinputpointer
  type(lakeinput), pointer :: lake_input_pointer
end type lakeinputpointer

type :: lakeproperties
  integer :: lake_number
  type(lakecellpointer), dimension(:), pointer :: cell_list
  integer :: lake_pixel_count
end type lakeproperties

interface lakeproperties
  procedure :: lakepropertiesconstructor
end interface lakeproperties

type :: lakepropertiespointer
  type(lakeproperties), pointer :: lake_properties_pointer
end type lakepropertiespointer

contains

!Constructors

function lakecellconstructor(_COORDS_ARG_coarse_grid_coords_, &
                             pixel_count, &
                             potential_lake_pixel_count, &
                             lake_pixels_original) &
    result(constructor)
  _DEF_COORDS_coarse_grid_coords_
  integer :: pixel_count
  integer :: potential_lake_pixel_count
  integer, dimension(:), pointer :: lake_pixels_original
  type(lakecell), pointer :: constructor
  integer :: i
    allocate(constructor)
    _ASSIGN_constructor%_COORDS_coarse_grid_coords_ = _COORDS_coarse_grid_coords_
    constructor%all_lake_potential_pixel_count = -1
    constructor%lake_potential_pixel_count = potential_lake_pixel_count
    allocate(constructor%lake_pixels(pixel_count))
    constructor%lake_pixels(:) = -1
    do i=1,size(lake_pixels_original)
      constructor%lake_pixels(i) = lake_pixels_original(i)
    end do
    allocate(constructor%lake_pixels_added(pixel_count))
    constructor%lake_pixels_added(:) = -1
    constructor%pixel_count = pixel_count
    constructor%lake_pixel_count = size(lake_pixels_original)
    constructor%lake_pixels_added_count = 0
    constructor%max_pixels_from_lake = -1
    constructor%in_binary_mask = .false.
end function lakecellconstructor

function lakeinputconstructor(lake_number, &
                              _INDICES_LIST_lake_pixel_coords_list_INDEX_NAME_, &
                              _INDICES_LIST_potential_lake_pixel_coords_list_INDEX_NAME_, &
                              _INDICES_LIST_cell_coords_list_INDEX_NAME_) result(constructor)
  integer :: lake_number
  _DEF_INDICES_LIST_lake_pixel_coords_list_INDEX_NAME_
  _DEF_INDICES_LIST_potential_lake_pixel_coords_list_INDEX_NAME_
  _DEF_INDICES_LIST_cell_coords_list_INDEX_NAME_
  type(lakeinput), pointer :: constructor
    allocate(constructor)
    constructor%lake_number = lake_number
    _ASSIGN_constructor%_INDICES_LIST_lake_pixel_coords_list_INDEX_NAME_ => &
      _INDICES_LIST_lake_pixel_coords_list_INDEX_NAME_
    _ASSIGN_constructor%_INDICES_LIST_potential_lake_pixel_coords_list_INDEX_NAME_ => &
      _INDICES_LIST_potential_lake_pixel_coords_list_INDEX_NAME_
    _ASSIGN_constructor%_INDICES_LIST_cell_coords_list_INDEX_NAME_ => &
       _INDICES_LIST_cell_coords_list_INDEX_NAME_
end

function lakepropertiesconstructor(lake_number, &
                                   cell_list, &
                                   lake_pixel_count) &
    result(constructor)
  integer :: lake_number
  type(lakecellpointer), dimension(:), pointer :: cell_list
  integer :: lake_pixel_count
  type(lakeproperties), pointer :: constructor
    allocate(constructor)
    constructor%lake_number = lake_number
    constructor%cell_list => cell_list
    constructor%lake_pixel_count = lake_pixel_count
end function lakepropertiesconstructor

function pixelconstructor(id,lake_number,filled, &
                          _COORDS_ARG_fine_grid_coords_, &
                          _COORDS_ARG_original_coarse_grid_coords_, &
                          _COORDS_ARG_assigned_coarse_grid_coords_, &
                          transferred) result(constructor)
  integer :: id
  integer :: lake_number
  logical :: filled
  _DEF_COORDS_fine_grid_coords_
  _DEF_COORDS_original_coarse_grid_coords_
  _DEF_COORDS_assigned_coarse_grid_coords_
  logical :: transferred
  type(pixel), pointer :: constructor
    allocate(constructor)
    constructor%id = id
    constructor%lake_number = lake_number
    constructor%filled = filled
    _ASSIGN_constructor%_COORDS_fine_grid_coords_ = _COORDS_fine_grid_coords_
    _ASSIGN_constructor%_COORDS_original_coarse_grid_coords_ = _COORDS_original_coarse_grid_coords_
    _ASSIGN_constructor%_COORDS_assigned_coarse_grid_coords_ = _COORDS_assigned_coarse_grid_coords_
    constructor%transferred = transferred
end function pixelconstructor

! Utility routines

subroutine insert_pixel(cell_in,pixel_in)
  type(lakecell), pointer :: cell_in
  type(pixel), pointer :: pixel_in
    _ASSIGN_pixel_in%_COORDS_assigned_coarse_grid_coords_ = cell_in%_COORDS_coarse_grid_coords_
    if (_NEQUALS_pixel_in%_COORDS_original_coarse_grid_coords_ /= cell_in%_COORDS_coarse_grid_coords_) then
      cell_in%lake_pixels_added_count = cell_in%lake_pixels_added_count + 1
      cell_in%lake_pixels_added(cell_in%lake_pixels_added_count) = pixel_in%id
      pixel_in%transferred = .true.
    end if
    cell_in%lake_pixel_count = cell_in%lake_pixel_count + 1
    cell_in%lake_pixels(cell_in%lake_pixel_count) = pixel_in%id
    !write(*,*) "inserting pixel"
end subroutine insert_pixel

subroutine extract_pixel(cell_in,pixel_in)
  type(lakecell), pointer :: cell_in
  type(pixel), pointer :: pixel_in
  integer :: i,j
    if (_NEQUALS_pixel_in%_COORDS_original_coarse_grid_coords_ /= cell_in%_COORDS_coarse_grid_coords_) then
      do i = 1,cell_in%lake_pixels_added_count
        if (cell_in%lake_pixels_added(i) == pixel_in%id) then
          cell_in%lake_pixels_added(i) = -1
          if (i < cell_in%lake_pixels_added_count) then
            do j = (i + 1),cell_in%lake_pixels_added_count
              cell_in%lake_pixels_added(j - 1) = cell_in%lake_pixels_added(j)
            end do
            cell_in%lake_pixels_added(cell_in%lake_pixels_added_count) = -1
          end if
          exit
        end if
      end do
      cell_in%lake_pixels_added_count = cell_in%lake_pixels_added_count - 1
      pixel_in%transferred = .false.
    end if
    do i = 1,cell_in%lake_pixel_count
      if (cell_in%lake_pixels(i) == pixel_in%id) then
        cell_in%lake_pixels(i) = -1
        if (i < cell_in%lake_pixel_count) then
          do j = (i + 1),cell_in%lake_pixel_count
            cell_in%lake_pixels(j - 1) = cell_in%lake_pixels(j)
          end do
          cell_in%lake_pixels(cell_in%lake_pixel_count) = -1
        end if
        exit
      end if
    end do
    cell_in%lake_pixel_count = cell_in%lake_pixel_count - 1
    !write(*,*) "extracting pixel"
end subroutine extract_pixel

function extract_any_pixel(cell_in,pixels) result(working_pixel)
  type(lakecell), pointer :: cell_in
  type(pixelpointer), dimension(:), pointer :: pixels
  type(pixel), pointer :: working_pixel
    if (cell_in%lake_pixel_count <= 0) then
      write(*,*) "No pixels to extract"
      stop
    end if
    working_pixel => pixels(cell_in%lake_pixels(1))%pixel_pointer
    call extract_pixel(cell_in,working_pixel)
end function extract_any_pixel

subroutine move_pixel(pixel_in,source_cell,target_cell)
  type(pixel), pointer :: pixel_in
  type(lakecell), pointer :: source_cell
  type(lakecell), pointer :: target_cell
    call extract_pixel(source_cell,pixel_in)
    call insert_pixel(target_cell,pixel_in)
end subroutine move_pixel

function get_lake_cell_from_coords(lake,_COORDS_ARG_coords_) result(cell_out)
  type(lakeproperties), pointer :: lake
  type(lakecell), pointer :: cell_out
  _DEF_COORDS_coords_
  integer :: i
    do i = 1,size(lake%cell_list)
      cell_out => lake%cell_list(i)%lake_cell_pointer
      if (_EQUALS_cell_out%_COORDS_coarse_grid_coords_ == _COORDS_coords_) then
        return
      end if
    end do
end function get_lake_cell_from_coords

subroutine setup_cells_lakes_and_pixels(lakes, &
                                        cell_pixel_counts, &
                                        all_lake_potential_pixel_mask, &
                                        lake_properties, &
                                        pixel_numbers, &
                                        pixels, &
                                        all_lake_potential_pixel_counts, &
                                        _INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_, &
                                        all_lake_total_pixels, &
                                        _NPOINTS_LAKE_)
  type(lakeinputpointer), dimension(:), pointer :: lakes
  integer, dimension(_DIMS_), pointer :: cell_pixel_counts
  type(lakepropertiespointer), dimension(:), pointer :: lake_properties
  logical, dimension(_DIMS_), pointer :: all_lake_potential_pixel_mask
  integer, dimension(_DIMS_), pointer :: pixel_numbers
  type(pixelpointer), dimension(:), pointer :: pixels
  integer, dimension(_DIMS_), pointer :: all_lake_potential_pixel_counts
  _DEF_INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_ _INTENT_in_
  integer, intent(out) :: all_lake_total_pixels
  _DEF_NPOINTS_LAKE_ _INTENT_in_
  type(lakeinput), pointer :: lake_input
  type(lakeproperties), pointer :: lake
  type(lakecellpointer), dimension(:), pointer :: lake_cells
  type(pixelpointer), dimension(:), pointer :: pixels_temp
  type(pixel), pointer :: working_pixel
  type(lakecell), pointer :: working_cell
  integer, dimension(:), pointer :: pixels_in_cell
  integer, dimension(:), pointer :: pixels_in_cell_temp
  _DEF_INDICES_LIST_cell_coords_list_INDEX_NAME_
  _DEF_COORDS_pixel_coords_
  _DEF_COORDS_cell_coords_
  _DEF_COORDS_containing_cell_coords_
  _DEF_COORDS_working_coarse_coords_
  integer :: pixel_number
  integer :: lake_cell_pixel_count
  integer :: potential_lake_pixel_count
  integer :: lake_pixel_count
  integer :: i,j
  integer :: lake_number
  _DEF_LOOP_INDEX_LAKE_
    all_lake_total_pixels = 0
    allocate(pixels_temp(_NPOINTS_TOTAL_LAKE_))
    do lake_number = 1,size(lakes)
      lake_input => lakes(lake_number)%lake_input_pointer
      _ASSIGN_INDICES_LIST_cell_coords_list_INDEX_NAME_ => &
        lake_input%_INDICES_LIST_cell_coords_list_INDEX_NAME_
      allocate(lake_cells(size(_INDICES_LIST_cell_coords_list_INDEX_NAME_FIRST_DIM_)))
      do i = 1,size(lake_input%_INDICES_LIST_potential_lake_pixel_coords_list_INDEX_NAME_FIRST_DIM_)
        _GET_COORDS_ _COORDS_pixel_coords_ _FROM_ lake_input%_INDICES_LIST_potential_lake_pixel_coords_list_INDEX_NAME_ i
        all_lake_total_pixels = all_lake_total_pixels + 1
        pixel_numbers(_COORDS_ARG_pixel_coords_) = all_lake_total_pixels
        _GET_COORDS_ _COORDS_containing_cell_coords_ _FROM_ _INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_ _COORDS_pixel_coords_
        working_pixel => pixel(all_lake_total_pixels,lake_input%lake_number, &
                              .false.,_COORDS_ARG_pixel_coords_, &
                              _COORDS_ARG_containing_cell_coords_,&
                              _COORDS_ARG_containing_cell_coords_,.false.)
        pixels_temp(all_lake_total_pixels) = pixelpointer(working_pixel)
        all_lake_potential_pixel_mask(working_pixel%_COORDS_ARG_fine_grid_coords_) = .true.
      end do
      do i = 1,size(lake_input%_INDICES_LIST_lake_pixel_coords_list_INDEX_NAME_FIRST_DIM_)
        _GET_COORDS_ _COORDS_pixel_coords_ _FROM_ lake_input%_INDICES_LIST_lake_pixel_coords_list_INDEX_NAME_ i
        pixels_temp(pixel_numbers(_COORDS_ARG_pixel_coords_))%pixel_pointer%filled = .true.
      end do
      do i = 1,size(_INDICES_LIST_cell_coords_list_INDEX_NAME_FIRST_DIM_)
        _GET_COORDS_ _COORDS_cell_coords_ _FROM_ _INDICES_LIST_cell_coords_list_INDEX_NAME_ i
        allocate(pixels_in_cell_temp(cell_pixel_counts(_COORDS_ARG_cell_coords_)))
        potential_lake_pixel_count = 0
        lake_pixel_count = 0
        _LOOP_OVER_LAKE_GRID_ _COORDS_LAKE_
          _GET_COORDS_ _COORDS_working_coarse_coords_ _FROM_ _INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_ _COORDS_LAKE_
          if (_EQUALS_COORDS_cell_coords_ == _COORDS_working_coarse_coords_) then
            pixel_number = pixel_numbers(_COORDS_LAKE_)
            if (pixel_number > 0) then
              working_pixel => pixels_temp(pixel_number)%pixel_pointer
              if (_EQUALS_working_pixel%_COORDS_fine_grid_coords_ == _COORDS_LAKE_ .and. &
                  pixels_temp(pixel_number)%pixel_pointer%lake_number == lake_input%lake_number) then
                  potential_lake_pixel_count = potential_lake_pixel_count + 1
                if (pixels_temp(pixel_number)%pixel_pointer%filled) then
                  lake_pixel_count = lake_pixel_count + 1
                  pixels_in_cell_temp(lake_pixel_count) = pixels_temp(pixel_number)%pixel_pointer%id
                end if
              end if
            end if
          end if
        _LOOP_OVER_LAKE_GRID_END_
        allocate(pixels_in_cell(lake_pixel_count))
        do j = 1,lake_pixel_count
           pixels_in_cell(j) =  pixels_in_cell_temp(j)
        end do
        deallocate(pixels_in_cell_temp)
        lake_cells(i)%lake_cell_pointer => &
          lakecell(_COORDS_ARG_cell_coords_, &
                   cell_pixel_counts(_COORDS_ARG_cell_coords_), &
                   potential_lake_pixel_count,pixels_in_cell)
      end do
      lake_cell_pixel_count = size(lake_input%_INDICES_LIST_lake_pixel_coords_list_INDEX_NAME_FIRST_DIM_)
      lake_properties(lake_number) = &
        lakepropertiespointer(lakeproperties(lake_input%lake_number, &
                                             lake_cells, &
                                             lake_cell_pixel_count))
    end do
    allocate(pixels(all_lake_total_pixels))
    do i = 1,all_lake_total_pixels
      pixels(i) = pixels_temp(i)
    end do
    deallocate(pixels_temp)
    _LOOP_OVER_LAKE_GRID_ _COORDS_LAKE_
        if (all_lake_potential_pixel_mask(_COORDS_LAKE_)) then
          _GET_COORDS_ _COORDS_working_coarse_coords_ _FROM_ _INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_ _COORDS_LAKE_
          all_lake_potential_pixel_counts(_COORDS_ARG_working_coarse_coords_) = &
            all_lake_potential_pixel_counts(_COORDS_ARG_working_coarse_coords_) + 1
        end if
    _LOOP_OVER_LAKE_GRID_END_
    do lake_number = 1,size(lake_properties)
      lake => lake_properties(lake_number)%lake_properties_pointer
      do i = 1,size(lake%cell_list)
        working_cell => lake%cell_list(i)%lake_cell_pointer
        working_cell%all_lake_potential_pixel_count = &
          all_lake_potential_pixel_counts(working_cell%_COORDS_ARG_coarse_grid_coords_)
      end do
    end do
end subroutine setup_cells_lakes_and_pixels

subroutine set_max_pixels_from_lake_for_cell(cell_in,lake_pixel_counts_field)
  type(lakecell), pointer :: cell_in
  integer, dimension(_DIMS_), pointer :: lake_pixel_counts_field
  integer :: other_lake_potential_pixels
  integer :: all_lake_pixel_count
  integer :: other_lake_filled_non_lake_pixels
    other_lake_potential_pixels = cell_in%all_lake_potential_pixel_count - &
                                  cell_in%lake_potential_pixel_count
    all_lake_pixel_count = lake_pixel_counts_field(cell_in%_COORDS_ARG_coarse_grid_coords_)
    other_lake_filled_non_lake_pixels = all_lake_pixel_count + &
                                        cell_in%lake_potential_pixel_count - &
                                        cell_in%lake_pixel_count - &
                                        cell_in%all_lake_potential_pixel_count
    if (other_lake_filled_non_lake_pixels < 0) then
      other_lake_filled_non_lake_pixels = 0
    end if
    cell_in%max_pixels_from_lake =  cell_in%pixel_count - other_lake_potential_pixels -  &
                                 other_lake_filled_non_lake_pixels
end subroutine set_max_pixels_from_lake_for_cell

! Main routines

subroutine calculate_lake_fractions(lakes, &
                                    cell_pixel_counts, &
                                    lake_pixel_counts_field, &
                                    lake_fractions_field, &
                                    binary_lake_mask, &
                                    _INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_, &
                                    _NPOINTS_LAKE_, &
                                    _NPOINTS_SURFACE_)
  type(lakeinputpointer), dimension(:), pointer :: lakes
  integer, dimension(_DIMS_), pointer :: cell_pixel_counts
  integer, dimension(_DIMS_), pointer :: lake_pixel_counts_field
  real(dp), dimension(_DIMS_), pointer :: lake_fractions_field
  logical, dimension(_DIMS_), pointer :: binary_lake_mask
  _DEF_INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_ _INTENT_in_
  _DEF_NPOINTS_LAKE_ _INTENT_in_
  _DEF_NPOINTS_SURFACE_ _INTENT_in_
  integer, dimension(_DIMS_), pointer :: pixel_numbers
  integer, dimension(_DIMS_), pointer :: all_lake_potential_pixel_counts
  logical, dimension(_DIMS_), pointer :: all_lake_potential_pixel_mask
  type(lakepropertiespointer), dimension(:), pointer :: lake_properties
  type(lakepropertiespointer), dimension(:), pointer :: lake_properties_temp
  type(lakecellpointer), dimension(:), pointer :: cell_list_temp
  type(pixelpointer), dimension(:), pointer :: pixels
  type(pixel), pointer :: working_pixel
  type(lakeproperties), pointer :: lake
  type(lakecell), pointer :: working_cell
  type(lakecell), pointer :: least_filled_cell
  real(dp) :: lake_fraction
  real(dp) :: max_lake_fraction_found
  integer :: all_lake_total_pixels
  integer :: unprocessed_cells_total_pixel_count
  integer :: min_pixel_count
  integer :: max_lake_pixel_count_found
  integer :: max_lake_fraction_index
  integer :: max_lake_pixel_count_index
  integer :: pixels_to_transfer
  integer :: i,j,k,l,m
  integer :: lake_number
    allocate(all_lake_potential_pixel_mask(_NPOINTS_LAKE_))
    all_lake_potential_pixel_mask(_DIMS_) =  .false.
    allocate(lake_properties(size(lakes)))
    allocate(pixel_numbers(_NPOINTS_LAKE_))
    pixel_numbers(_DIMS_) = 0
    allocate(all_lake_potential_pixel_counts(_NPOINTS_SURFACE_))
    all_lake_potential_pixel_counts(_DIMS_) = 0
    allocate(lake_pixel_counts_field(_NPOINTS_SURFACE_))
    lake_pixel_counts_field(_DIMS_) = 0
    call setup_cells_lakes_and_pixels(lakes,cell_pixel_counts, &
                                      all_lake_potential_pixel_mask, &
                                      lake_properties, &
                                      pixel_numbers,pixels, &
                                      all_lake_potential_pixel_counts, &
                                      _INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_, &
                                      all_lake_total_pixels, &
                                      _NPOINTS_LAKE_)
    allocate(lake_properties_temp(size(lake_properties)))
    do i = 1,size(lake_properties)
      max_lake_pixel_count_found = -1
      do j = 1,size(lake_properties)
        if (associated(lake_properties(j)%lake_properties_pointer)) then
          if (lake_properties(j)%lake_properties_pointer%lake_pixel_count > &
              max_lake_pixel_count_found) then
            max_lake_pixel_count_found = lake_properties(j)%lake_properties_pointer%lake_pixel_count
            max_lake_pixel_count_index = j
          end if
        end if
      end do
      lake_properties_temp(i)%lake_properties_pointer => &
        lake_properties(max_lake_pixel_count_index)%lake_properties_pointer
      lake_properties(max_lake_pixel_count_index)%lake_properties_pointer => null()
    end do
    deallocate(lake_properties)
    lake_properties => lake_properties_temp
    do lake_number = 1,size(lake_properties)
      lake => lake_properties(lake_number)%lake_properties_pointer
      do k = 1,size(lake%cell_list)
        working_cell => lake%cell_list(k)%lake_cell_pointer
        call set_max_pixels_from_lake_for_cell(working_cell,lake_pixel_counts_field)
      end do
      allocate(cell_list_temp(size(lake%cell_list)))
      do l = 1,size(lake%cell_list)
        max_lake_fraction_found = -1.0_dp
        do m = 1,size(lake%cell_list)
          if (associated(lake%cell_list(m)%lake_cell_pointer)) then
            lake_fraction = real(lake%cell_list(m)%lake_cell_pointer%lake_pixel_count,dp)/ &
                            real(lake%cell_list(m)%lake_cell_pointer%pixel_count,dp)
            if (lake_fraction > &
                max_lake_fraction_found) then
              max_lake_fraction_found = lake_fraction
              max_lake_fraction_index = m
            end if
          end if
        end do
        cell_list_temp(l)%lake_cell_pointer => &
          lake%cell_list(max_lake_fraction_index)%lake_cell_pointer
        lake%cell_list(max_lake_fraction_index)%lake_cell_pointer => null()
      end do
      deallocate(lake%cell_list)
      lake%cell_list => cell_list_temp
      j = size(lake%cell_list)
      unprocessed_cells_total_pixel_count = 0
      min_pixel_count = _NPOINTS_TOTAL_LAKE_
      do k = 1,size(lake%cell_list)
        working_cell => lake%cell_list(k)%lake_cell_pointer
        unprocessed_cells_total_pixel_count = unprocessed_cells_total_pixel_count + working_cell%lake_pixel_count
        if (working_cell%pixel_count < min_pixel_count) then
          min_pixel_count = working_cell%pixel_count
        end if
      end do
      if (unprocessed_cells_total_pixel_count < 0.5*min_pixel_count) then
        do i = 1,size(lake%cell_list)
          working_cell => lake%cell_list(i)%lake_cell_pointer
          lake_pixel_counts_field(working_cell%_COORDS_ARG_coarse_grid_coords_) = &
            lake_pixel_counts_field(working_cell%_COORDS_ARG_coarse_grid_coords_)+working_cell%lake_pixel_count
        end do
        cycle
      end if
      do i = 1,size(lake%cell_list)
        if (i == j) then
          exit
        end if
        working_cell => lake%cell_list(i)%lake_cell_pointer
        unprocessed_cells_total_pixel_count = unprocessed_cells_total_pixel_count - &
                                              working_cell%lake_pixel_count
        if (unprocessed_cells_total_pixel_count + &
            working_cell%lake_pixel_count < 0.5*working_cell%pixel_count) then
          exit
        end if
        if (working_cell%max_pixels_from_lake == working_cell%lake_pixel_count) then
          cycle
        end if
        if (working_cell%max_pixels_from_lake < 0.5*working_cell%pixel_count) then
          cycle
        end if
        do while (.true.)
          least_filled_cell => lake%cell_list(j)%lake_cell_pointer
          if (working_cell%lake_pixel_count + least_filled_cell%lake_pixel_count  &
              <= working_cell%max_pixels_from_lake) then
            do k = 1,least_filled_cell%lake_pixel_count
              working_pixel => pixels(least_filled_cell%lake_pixels(least_filled_cell%lake_pixel_count))%pixel_pointer
              call move_pixel(working_pixel,least_filled_cell,working_cell)
              unprocessed_cells_total_pixel_count = unprocessed_cells_total_pixel_count - 1
            end do
            j = j - 1
            if (working_cell%lake_pixel_count == working_cell%max_pixels_from_lake) then
              exit
            end if
            if (i == j) then
              exit
            end if
          else
            pixels_to_transfer = working_cell%max_pixels_from_lake - working_cell%lake_pixel_count
            do k = 1,pixels_to_transfer
              working_pixel => pixels(least_filled_cell%lake_pixels(least_filled_cell%lake_pixel_count))%pixel_pointer
              call move_pixel(working_pixel,least_filled_cell,working_cell)
              unprocessed_cells_total_pixel_count = unprocessed_cells_total_pixel_count - 1
            end do
            exit
          end if
        end do
      end do
      do k = 1,size(lake%cell_list)
        working_cell => lake%cell_list(k)%lake_cell_pointer
        lake_pixel_counts_field(working_cell%_COORDS_ARG_coarse_grid_coords_) = &
          lake_pixel_counts_field(working_cell%_COORDS_ARG_coarse_grid_coords_)+working_cell%lake_pixel_count
      end do
    end do
    allocate(lake_fractions_field(_NPOINTS_SURFACE_))
    lake_fractions_field = &
      real(lake_pixel_counts_field(:,:),dp)/real(cell_pixel_counts(:,:),dp)
    allocate(binary_lake_mask(_NPOINTS_SURFACE_))
    where (lake_fractions_field >= 0.5_dp)
      binary_lake_mask(:,:) = .true.
    elsewhere
      binary_lake_mask(:,:) = .false.
    end where
end subroutine calculate_lake_fractions

subroutine setup_lake_for_fraction_calculation(lakes, &
                                               cell_pixel_counts, &
                                               binary_lake_mask, &
                                               lake_properties, &
                                               pixel_numbers, &
                                               pixels, &
                                               lake_pixel_counts_field, &
                                                _INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_, &
                                               _NPOINTS_LAKE_, &
                                               _NPOINTS_SURFACE_)
  type(lakeinputpointer), dimension(:), pointer :: lakes
  integer, dimension(_DIMS_), pointer :: cell_pixel_counts
  logical, dimension(_DIMS_), pointer :: binary_lake_mask
  type(lakepropertiespointer), dimension(:), pointer :: lake_properties
  integer, dimension(_DIMS_), pointer :: pixel_numbers
  type(pixelpointer), dimension(:), pointer :: pixels
  integer, dimension(_DIMS_), pointer :: lake_pixel_counts_field
  _DEF_INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_ _INTENT_in_
  _DEF_NPOINTS_LAKE_ _INTENT_in_
  _DEF_NPOINTS_SURFACE_ _INTENT_in_
  integer, dimension(_DIMS_), pointer :: all_lake_potential_pixel_counts
  logical, dimension(_DIMS_), pointer :: all_lake_potential_pixel_mask
  type(lakeproperties), pointer :: lake
  type(lakecell), pointer :: working_cell
  integer :: all_lake_total_pixels
  integer :: lake_number
  integer :: i
    allocate(all_lake_potential_pixel_mask(_NPOINTS_LAKE_))
    all_lake_potential_pixel_mask(_DIMS_) =  .false.
    allocate(lake_properties(size(lakes)))
    allocate(pixel_numbers(_NPOINTS_LAKE_))
    pixel_numbers(_DIMS_) = 0
    allocate(all_lake_potential_pixel_counts(_NPOINTS_SURFACE_))
    all_lake_potential_pixel_counts(_DIMS_) = 0
    allocate(lake_pixel_counts_field(_NPOINTS_SURFACE_))
    lake_pixel_counts_field(_DIMS_) =  0
    call setup_cells_lakes_and_pixels(lakes,cell_pixel_counts, &
                                      all_lake_potential_pixel_mask, &
                                      lake_properties, &
                                      pixel_numbers,pixels, &
                                      all_lake_potential_pixel_counts, &
                                      _INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_, &
                                      all_lake_total_pixels, &
                                      _NPOINTS_LAKE_)
    do lake_number = 1,size(lake_properties)
      lake => lake_properties(lake_number)%lake_properties_pointer
      do i = 1,size(lake%cell_list)
        working_cell => lake%cell_list(i)%lake_cell_pointer
        working_cell%in_binary_mask = binary_lake_mask(working_cell%_COORDS_ARG_coarse_grid_coords_)
      end do
    end do
end subroutine setup_lake_for_fraction_calculation

recursive subroutine add_pixel(lake,pixel_in,pixels,lake_pixel_counts_field)
  type(lakeproperties), pointer :: lake
  type(pixel), pointer :: pixel_in
  type(pixelpointer), dimension(:), pointer :: pixels
  integer, dimension(_DIMS_), pointer :: lake_pixel_counts_field
  type(lakecell), pointer :: working_cell
  type(lakecell), pointer :: other_cell
  type(lakecell), pointer :: most_filled_cell
  type(lakecell), pointer :: other_pixel_origin_cell
  type(lakecell), pointer :: working_pixel_origin_cell
  type(lakecell), pointer :: entry
  type(pixel), pointer :: other_pixel
  type(pixel), pointer :: working_pixel
  real(dp) :: working_pixel_origin_cell_lake_fraction
  real(dp) :: other_cell_lake_fraction
  real(dp) :: max_other_pixel_origin_cell_lake_fraction
  real(dp) :: most_filled_cell_lake_fraction
  integer :: i
    do i = 1,size(lake%cell_list)
      entry => lake%cell_list(i)%lake_cell_pointer
      if (_EQUALS_entry%_COORDS_coarse_grid_coords_ == pixel_in%_COORDS_original_coarse_grid_coords_) then
        working_cell => entry
      end if
    end do
    call set_max_pixels_from_lake_for_cell(working_cell,lake_pixel_counts_field)
    if (working_cell%max_pixels_from_lake == working_cell%lake_pixel_count) then
      max_other_pixel_origin_cell_lake_fraction = -1.0_dp
      do i = 1,working_cell%lake_pixels_added_count
        working_pixel => pixels(working_cell%lake_pixels_added(i))%pixel_pointer
        working_pixel_origin_cell => &
          get_lake_cell_from_coords(lake,working_pixel%_COORDS_ARG_original_coarse_grid_coords_)
        working_pixel_origin_cell_lake_fraction = &
          real(working_pixel_origin_cell%lake_pixel_count,dp)/ &
          real(working_pixel_origin_cell%pixel_count,dp)
        if (working_pixel_origin_cell_lake_fraction > max_other_pixel_origin_cell_lake_fraction) then
          other_pixel => working_pixel
          other_pixel_origin_cell =>  working_pixel_origin_cell
          max_other_pixel_origin_cell_lake_fraction = working_pixel_origin_cell%lake_pixel_count
        end if
      end do
      if (max_other_pixel_origin_cell_lake_fraction == -1) then
        write(*,*) "No added pixel to return - logic error"
        stop
      end if
      call extract_pixel(working_cell,other_pixel)
      !Order of next two statements is critical to prevent loops
      call insert_pixel(working_cell,pixel_in)
      call add_pixel(lake,other_pixel,pixels,lake_pixel_counts_field)
    elseif (working_cell%max_pixels_from_lake < working_cell%lake_pixel_count) then
      write(*,*) "Cell has more pixel than possible - logic error"
      stop
    elseif (working_cell%in_binary_mask) then
      call insert_pixel(working_cell,pixel_in)
      lake_pixel_counts_field(working_cell%_COORDS_ARG_coarse_grid_coords_) = &
        lake_pixel_counts_field(working_cell%_COORDS_ARG_coarse_grid_coords_) + 1
    else
      most_filled_cell_lake_fraction = -1.0_dp
      do i = 1,size(lake%cell_list)
        other_cell => lake%cell_list(i)%lake_cell_pointer
        call set_max_pixels_from_lake_for_cell(other_cell,lake_pixel_counts_field)
        other_cell_lake_fraction = real(other_cell%lake_pixel_count,dp)/ &
                                   real(other_cell%pixel_count,dp)
        if (other_cell_lake_fraction > most_filled_cell_lake_fraction .and. &
            other_cell%lake_pixel_count < other_cell%max_pixels_from_lake .and. &
            other_cell%in_binary_mask) then
          most_filled_cell => other_cell
          most_filled_cell_lake_fraction = other_cell_lake_fraction
        end if
      end do
      if (most_filled_cell_lake_fraction >= 0.0_dp) then
        call insert_pixel(most_filled_cell,pixel_in)
        lake_pixel_counts_field(most_filled_cell%_COORDS_ARG_coarse_grid_coords_) = &
          lake_pixel_counts_field(most_filled_cell%_COORDS_ARG_coarse_grid_coords_) + 1
      else
        call insert_pixel(working_cell,pixel_in)
        lake_pixel_counts_field(working_cell%_COORDS_ARG_coarse_grid_coords_) = &
          lake_pixel_counts_field(working_cell%_COORDS_ARG_coarse_grid_coords_) + 1
      end if
    end if
end subroutine add_pixel

subroutine remove_pixel(lake,pixel_in,pixels,lake_pixel_counts_field)
  type(lakeproperties), pointer :: lake
  type(pixel), pointer :: pixel_in
  type(pixelpointer), dimension(:), pointer :: pixels
  integer, dimension(_DIMS_), pointer :: lake_pixel_counts_field
  type(lakecell), pointer :: working_cell
  type(lakecell), pointer :: other_cell
  type(lakecell), pointer :: least_filled_cell
  type(lakecell), pointer :: entry
  type(pixel), pointer :: other_pixel
  real(dp) :: least_filled_cell_lake_fraction
  real(dp) :: other_cell_lake_fraction
  integer :: i
    do i = 1,size(lake%cell_list)
      entry => lake%cell_list(i)%lake_cell_pointer
      if (_EQUALS_entry%_COORDS_coarse_grid_coords_ == pixel_in%_COORDS_assigned_coarse_grid_coords_) then
        working_cell => entry
      end if
    end do
    call extract_pixel(working_cell,pixel_in)
    lake_pixel_counts_field(working_cell%_COORDS_ARG_coarse_grid_coords_) = &
      lake_pixel_counts_field(working_cell%_COORDS_ARG_coarse_grid_coords_) - 1
    if (working_cell%in_binary_mask) then
      least_filled_cell_lake_fraction = 999.0_dp
      do i = 1,size(lake%cell_list)
        other_cell => lake%cell_list(i)%lake_cell_pointer
        other_cell_lake_fraction = real(other_cell%lake_pixel_count,dp)/ &
                                   real(other_cell%pixel_count,dp)
        if (.not. other_cell%in_binary_mask .and.  other_cell%lake_pixel_count > 0 .and. &
             other_cell_lake_fraction < least_filled_cell_lake_fraction) then
           least_filled_cell_lake_fraction = other_cell_lake_fraction
           least_filled_cell => other_cell
        end if
      end do
      if (least_filled_cell_lake_fraction < 999.0_dp) then
        other_pixel => extract_any_pixel(least_filled_cell,pixels)
        lake_pixel_counts_field(least_filled_cell%_COORDS_ARG_coarse_grid_coords_) = &
          lake_pixel_counts_field(least_filled_cell%_COORDS_ARG_coarse_grid_coords_) - 1
        call insert_pixel(working_cell,other_pixel)
        lake_pixel_counts_field(working_cell%_COORDS_ARG_coarse_grid_coords_) = &
          lake_pixel_counts_field(working_cell%_COORDS_ARG_coarse_grid_coords_) + 1
      end if
    end if
end subroutine remove_pixel

end module  l2_calculate_lake_fractions_mod
