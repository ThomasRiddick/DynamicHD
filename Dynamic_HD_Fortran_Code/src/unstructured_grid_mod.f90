module unstructured_grid_mod
implicit none

type, abstract :: unstructured_grid
  private
  integer :: num_points
  integer, dimension(:,:), pointer :: cell_neighbors
  integer, dimension(:,:), pointer :: cell_secondary_neighbors
  integer :: num_primary_neighbors
  integer :: num_secondary_neighbors
  contains
    private
      procedure, public :: generate_cell_neighbors
      procedure, public :: generate_cell_secondary_neighbors
      procedure, public :: generate_edge_cells
      procedure, public :: generate_subfield_indices
      procedure, public :: generate_full_field_indices
      procedure(calculate_secondary_neighbors), deferred :: &
        calculate_secondary_neighbors
end type unstructured_grid

abstract interface
  subroutine calculate_secondary_neighbors(this)
    import unstructured_grid
    class(unstructured_grid), intent(inout) :: this
  end subroutine calculate_secondary_neighbors
end interface

type, extends(unstructured_grid) :: icon_icosohedral_grid
  private
  contains
    private
      procedure :: calculate_secondary_neighbors => icon_icosohedral_calculate_secondary_neighbors
end type icon_icosohedral_grid

contains

  function generate_edge_cells(this,cell_neighbors,cell_secondary_neighbors) &
      result(edge_cells)
    class(unstructured_grid), intent(in) :: this
    integer, dimension(:,:), pointer, intent(in) :: cell_neighbors
    integer, dimension(:,:), pointer, intent(in) :: cell_secondary_neighbors
    integer, dimension(:), pointer :: edge_cells
    integer, dimension(:), pointer :: expanded_edge_cells
    integer :: num_points_subarray
    integer :: num_edge_cells
    integer :: i,j
    logical :: is_edge_cell
      num_points_subarray = size(cell_neighbors,1)
      allocate(expanded_edge_cells(num_points_subarray))
      num_edge_cells = 0
      do i = 1,num_points_subarray
        is_edge_cell = .false.
        do j = 1,this%num_primary_neighbors
          if(cell_neighbors(i,j) == -1) then
            num_edge_cells = num_edge_cells + 1
            expanded_edge_cells(num_edge_cells) = i
            is_edge_cell = .true.
            exit
          end if
        end do
        if (is_edge_cell) exit
        do j = 1,this%num_secondary_neighbors
          if(cell_secondary_neighbors(i,j) == -1) then
            num_edge_cells = num_edge_cells + 1
            expanded_edge_cells(num_edge_cells) = i
            is_edge_cell = .true.
            exit
          end if
        end do
        if (is_edge_cell) exit
      end do
      allocate(edge_cells(num_edge_cells))
      do i = 1,num_edge_cells
        edge_cells(i) = expanded_edge_cells(i)
      end do
  end function generate_edge_cells

  function generate_full_field_indices(this,mask,subfield_indices) result(full_field_indices)
      class(unstructured_grid), intent(in) :: this
      integer, dimension(:), pointer, intent(in) :: subfield_indices
      logical, dimension(:), pointer, intent(in) :: mask
      integer, dimension(:), pointer :: full_field_indices
      integer :: num_points_subarray
      integer :: i
        num_points_subarray = count(mask)
        allocate(full_field_indices(num_points_subarray))
        do i = 1,this%num_points
          if(mask(i)) then
            full_field_indices(subfield_indices(i)) = i
          end if
        end do
  end function generate_full_field_indices

  function generate_subfield_indices(this,mask) result(subfield_indices)
      class(unstructured_grid), intent(in) :: this
      logical, dimension(:), pointer, intent(in) :: mask
      integer, dimension(:), pointer :: subfield_indices
      integer :: i, i_new
        allocate(subfield_indices(this%num_points))
        i_new = 1
        do i = 1,this%num_points
          if(mask(i)) then
            subfield_indices(i) = i_new
            i_new = i + i_new
          else
            subfield_indices(i) = -1
          end if
        end do
  end function generate_subfield_indices

  function generate_cell_neighbors(this,mask,subfield_indices,full_field_indices) &
      result(cell_neighbors)
    class(unstructured_grid), intent(in) :: this
    logical, dimension(:), pointer, intent(in) :: mask
    integer, dimension(:), pointer, intent(in) :: subfield_indices
    integer, dimension(:), pointer, intent(in) :: full_field_indices
    integer, dimension(:,:), pointer :: cell_neighbors
    integer :: num_points_subarray
    integer :: i,nbr_num
      if(all(mask)) then
        cell_neighbors => this%cell_neighbors
      else
        num_points_subarray = count(mask)
        allocate(cell_neighbors(num_points_subarray,this%num_primary_neighbors))
        do i = 1,num_points_subarray
          do nbr_num = 1,this%num_primary_neighbors
            cell_neighbors(i,nbr_num) = subfield_indices(this%cell_neighbors(full_field_indices(i),nbr_num))
          end do
        end do
      end if
  end function generate_cell_neighbors

  function generate_cell_secondary_neighbors(this,subfield_indices,full_field_indices,cell_neighbors) &
      result(cell_secondary_neighbors)
    class(unstructured_grid), intent(inout) :: this
    integer, dimension(:,:), pointer, intent(in) :: cell_neighbors
    integer, dimension(:,:), pointer :: cell_secondary_neighbors
    integer, dimension(:), pointer, intent(in) :: subfield_indices
    integer, dimension(:), pointer, intent(in) :: full_field_indices
    integer :: num_points_subarray
    integer :: i,j
      num_points_subarray = size(cell_neighbors)
      if( .not. associated(this%cell_secondary_neighbors)) then
        call this%calculate_secondary_neighbors()
      end if
      allocate(cell_secondary_neighbors(num_points_subarray,this%num_secondary_neighbors))
      do i = 1,num_points_subarray
        do j = 1,this%num_secondary_neighbors
          cell_secondary_neighbors(i,j) = subfield_indices(this%cell_secondary_neighbors(full_field_indices(i),j))
        end do
      end do
  end function generate_cell_secondary_neighbors

  subroutine icon_icosohedral_calculate_secondary_neighbors(this)
    class(icon_icosohedral_grid), intent(inout) :: this
    integer :: index_over_grid
    integer :: index_over_primary_nbrs,index_over_secondary_nbrs
    integer :: index_over_tertiary_nbrs
    integer :: primary_neighbor_index,secondary_neighbor_index
    integer :: tertiary_neighbor_index
    integer :: second_tertiary_neighbor_index
    integer :: valid_secondary_nbr_count
    integer :: gap_index = 2
    integer :: first_secondary_neighbor_index,second_secondary_neighbor_index
    integer :: second_index_over_secondary_nbrs
    integer :: second_index_over_tertiary_nbrs
    integer :: no_neighbor
      no_neighbor = 0
      allocate(this%cell_secondary_neighbors(this%num_points,9))
      do index_over_grid=1,this%num_points
        !Six secondary neighbors are neighbors of primary neighbors
        do index_over_primary_nbrs=1,3
          primary_neighbor_index = this%cell_neighbors(index_over_grid,index_over_primary_nbrs)
          valid_secondary_nbr_count = 0;
          do index_over_secondary_nbrs=1,3
            !2 rather than 3 times primary neighbor index as we miss out 1 secondary neighbor for each
            !primary neighbor
            secondary_neighbor_index = this%cell_neighbors(primary_neighbor_index,index_over_secondary_nbrs)
            if (secondary_neighbor_index /= index_over_grid) then
              !Note this leaves gaps for the remaining three secondary neighbors
              this%cell_secondary_neighbors(index_over_grid,3*index_over_primary_nbrs + &
                                            valid_secondary_nbr_count) = secondary_neighbor_index
              valid_secondary_nbr_count = valid_secondary_nbr_count + 1
            end if
          end do
        end do
        !Three secondary neighbors are common neighbors of the existing secondary neighbors
        gap_index = 2;
        !Last secondary neighbor is as yet unfilled so loop only up to an index of 8
        do index_over_secondary_nbrs=1,8
          !skip as yet unfilled entries in the secondary neighbors array
          if (mod(index_over_secondary_nbrs,3) == 0) cycle
          first_secondary_neighbor_index = &
              this%cell_secondary_neighbors(index_over_grid,index_over_secondary_nbrs)
          !Last secondary neighbor is as yet unfilled so loop only up to an index of 7
          do second_index_over_secondary_nbrs=index_over_secondary_nbrs+2,8
            if (mod(second_index_over_secondary_nbrs,3) == 0) cycle
              second_secondary_neighbor_index = &
                this%cell_secondary_neighbors(index_over_grid,second_index_over_secondary_nbrs)
            !Some tertiary neighbors are also secondary neighbors
            do index_over_tertiary_nbrs=1,3
              tertiary_neighbor_index = &
                this%cell_neighbors(first_secondary_neighbor_index,index_over_tertiary_nbrs)
              !Test to see if this one of the twelve 5-point vertices in the grid
              if (second_secondary_neighbor_index == tertiary_neighbor_index) then
                this%cell_secondary_neighbors(index_over_grid,gap_index) = no_neighbor
                gap_index = gap_index + 3;
                cycle;
              end if
              do second_index_over_tertiary_nbrs=1,3
                second_tertiary_neighbor_index = &
                  this%cell_neighbors(second_secondary_neighbor_index,second_index_over_tertiary_nbrs)
                if(second_tertiary_neighbor_index == tertiary_neighbor_index) then
                  this%cell_secondary_neighbors(index_over_grid,gap_index) = tertiary_neighbor_index
                  gap_index = gap_index + 3;
                end if
              end do
            end do
          end do
        end do
      end do
  end subroutine icon_icosohedral_calculate_secondary_neighbors

end module unstructured_grid_mod
