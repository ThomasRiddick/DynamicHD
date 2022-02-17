module convert_rdirs_to_indices

contains

  subroutine convert_rdirs_to_latlon_indices(rdirs, &
                                             next_cell_index_lat, &
                                             next_cell_index_lon)
    integer, dimension(:,:), pointer, intent(in) :: rdirs
    integer, dimension(:,:), pointer, intent(inout) :: next_cell_index_lat
    integer, dimension(:,:), pointer, intent(inout) :: next_cell_index_lon
    integer :: rdir
    integer :: i,j
      do i = 1,size(rdirs,1)
        do j = 1,size(rdirs,2)
          rdir = rdirs(i,j)
          if (rdir == 5) then
            next_cell_index_lat(i,j) = -5
            next_cell_index_lon(i,j) = -5
          else if (rdir == 0 .or. rdir == -1 .or. rdir == -2) then
            next_cell_index_lat(i,j) = rdir
            next_cell_index_lon(i,j) = rdir
          else if (rdir <= 9 .or. rdir >= 1) then
            if (rdir == 7 .or. rdir == 8 .or. rdir == 9) then
              next_cell_index_lat(i,j) = i - 1
            else if (rdir == 4 .or. rdir == 6) then
              next_cell_index_lat(i,j) = i
            else if (rdir == 1 .or. rdir == 2 .or. rdir == 3) then
              next_cell_index_lat(i,j) = i + 1
            end if
            if (rdir == 7 .or. rdir == 4 .or. rdir == 1) then
              next_cell_index_lon(i,j) = j - 1
            else if (rdir == 8 .or. rdir == 2) then
              next_cell_index_lon(i,j) = j
            else if (rdir == 9 .or. rdir == 6 .or. rdir == 3) then
              next_cell_index_lon(i,j) = j + 1
            end if
          else
            stop 'Invalid river direction'
          end if
        end do
      end do
  end subroutine convert_rdirs_to_latlon_indices

end module convert_rdirs_to_indices
