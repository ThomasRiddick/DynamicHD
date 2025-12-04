module calculate_lake_fractions_test_mod

use fruit
implicit none

contains

subroutine testLakeFractionCalculationTest1
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: expected_binary_lake_mask
   logical, dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Single lake
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      lake_number = 1
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(lake_pixel_mask)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, &
         0, 0, 0, 0, 16, &
         0, 0, 0, 0, 14 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.0, 0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 1.0, &
         0.0, 0.0, 0.0, 0.0, 0.875 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_binary_lake_mask(nlat_surface,nlon_surface))
      expected_binary_lake_mask = transpose(reshape((/   &
          .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .True., &
         .False., .False., .False., .False., .True. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      mask = (lake_pixel_mask == 1)
      npixels = count(mask)
      allocate(lake_pixel_coords_list_lat(npixels))
      allocate(lake_pixel_coords_list_lon(npixels))
      allocate(potential_lake_pixel_coords_list_lat(npixels))
      allocate(potential_lake_pixel_coords_list_lon(npixels))
      counter = 1
      do i = 1,nlat_lake
        do j = 1,nlon_lake
          if (mask(i,j)) then
            lake_pixel_coords_list_lat(counter) = i
            lake_pixel_coords_list_lon(counter) = j
            potential_lake_pixel_coords_list_lat(counter) = i
            potential_lake_pixel_coords_list_lon(counter) = j
            counter = counter + 1
          end if
        end do
      end do
      allocate(cell_mask(nlat_surface,nlon_surface))
      cell_mask(:,:) = .false.
      do i=1,size(potential_lake_pixel_coords_list_lat)
          lat = corresponding_surface_cell_lat_index( &
            potential_lake_pixel_coords_list_lat(i), &
            potential_lake_pixel_coords_list_lon(i))
          lon = corresponding_surface_cell_lon_index( &
            potential_lake_pixel_coords_list_lat(i), &
            potential_lake_pixel_coords_list_lon(i))
          cell_mask(lat,lon) = .true.
      end do
      allocate(cell_coords_list_lat(count(cell_mask)))
      allocate(cell_coords_list_lon(count(cell_mask)))
      counter = 1
      do i = 1,nlat_surface
        do j = 1,nlon_surface
          if (cell_mask(i,j)) then
            cell_coords_list_lat(counter) = i
            cell_coords_list_lon(counter) = j
            counter = counter + 1
          end if
        end do
      end do
      lakes(1)%lake_input_pointer => lakeinput(lake_number, &
                                               lake_pixel_coords_list_lat, &
                                               lake_pixel_coords_list_lon, &
                                               potential_lake_pixel_coords_list_lat, &
                                               potential_lake_pixel_coords_list_lon, &
                                               cell_coords_list_lat, &
                                               cell_coords_list_lon)
      call calculate_lake_fractions(lakes, &
                                    cell_pixel_counts, &
                                    non_lake_mask, &
                                    lake_pixel_counts_field, &
                                    lake_fractions_field, &
                                    binary_lake_mask, &
                                    corresponding_surface_cell_lat_index, &
                                    corresponding_surface_cell_lon_index, &
                                    nlat_lake,nlon_lake, &
                                    nlat_surface,nlon_surface)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,nlat_surface,nlon_surface)
      call assert_equals(lake_fractions_field, expected_lake_fractions_field, &
                         nlat_surface,nlon_surface)
      call assert_equals(binary_lake_mask, expected_binary_lake_mask, &
                         nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(lake_fractions_field)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest1

subroutine testLakeFractionCalculationTest2
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: expected_binary_lake_mask
   logical, dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Multiple lakes
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 2, 2, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 7, 0, &
         0, 2, 2, 2,  2, 0, 0, 0,  0, 0, 3, 3,  0, 0, 0, 0,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 6,  6, 6, 6, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 4, 4, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 4, 4, 4,  4, 4, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 4, 4, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 5, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 5, 5, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 5, 0, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 1, 0, &
         0, 0, 0, 5,  5, 5, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  0, 5, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(lake_pixel_mask)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          5, 1, 2, 0, 1, &
         0, 0, 0, 0, 11, &
         0, 9, 0, 0, 0, &
         0, 0, 0, 0, 16, &
         0, 8, 0, 0, 14 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.3125, 0.0625, 0.125, 0.0, 0.0625, &
         0.0, 0.0, 0.0, 0.0, 0.6875, &
         0.0, 0.5625, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 1.0, &
         0.0, 0.5, 0.0, 0.0, 0.875 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_binary_lake_mask(nlat_surface,nlon_surface))
      expected_binary_lake_mask = transpose(reshape((/   &
          .False., .False., .False., .False., .False., &
         .False., .False., .False., .False.,  .True., &
         .False.,  .True., .False., .False., .False., &
         .False., .False., .False., .False.,  .True., &
         .False.,  .True., .False., .False.,  .True. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      allocate(cell_mask(nlat_surface,nlon_surface))
      do lake_number = 1,7
        mask = (lake_pixel_mask == lake_number)
        npixels = count(mask)
        allocate(lake_pixel_coords_list_lat(npixels))
        allocate(lake_pixel_coords_list_lon(npixels))
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              lake_pixel_coords_list_lat(counter) = i
              lake_pixel_coords_list_lon(counter) = j
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        cell_mask(:,:) = .false.
        do i=1,size(potential_lake_pixel_coords_list_lat)
            lat = corresponding_surface_cell_lat_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            lon = corresponding_surface_cell_lon_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            cell_mask(lat,lon) = .true.
        end do
        allocate(cell_coords_list_lat(count(cell_mask)))
        allocate(cell_coords_list_lon(count(cell_mask)))
        counter = 1
        do i = 1,nlat_surface
          do j = 1,nlon_surface
            if (cell_mask(i,j)) then
              cell_coords_list_lat(counter) = i
              cell_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        lakes(lake_number)%lake_input_pointer => &
          lakeinput(lake_number, &
          lake_pixel_coords_list_lat, &
          lake_pixel_coords_list_lon, &
          potential_lake_pixel_coords_list_lat, &
          potential_lake_pixel_coords_list_lon, &
          cell_coords_list_lat, &
          cell_coords_list_lon)
      end do
      call calculate_lake_fractions(lakes, &
                                    cell_pixel_counts, &
                                    non_lake_mask, &
                                    lake_pixel_counts_field, &
                                    lake_fractions_field, &
                                    binary_lake_mask, &
                                    corresponding_surface_cell_lat_index, &
                                    corresponding_surface_cell_lon_index, &
                                    nlat_lake,nlon_lake, &
                                    nlat_surface,nlon_surface)
      call assert_equals(lake_pixel_counts_field, &
                           expected_lake_pixel_counts_field,nlat_surface,nlon_surface)
      call assert_equals(lake_fractions_field, expected_lake_fractions_field, &
                                            nlat_surface,nlon_surface)
      call assert_equals(binary_lake_mask, expected_binary_lake_mask, &
                                            nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(lake_fractions_field)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest2

subroutine testLakeFractionCalculationTest3
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: expected_binary_lake_mask
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Multiple lakes
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 1, 0,  1, 1, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 7, 0, &
         0, 0, 1, 0,  1, 1, 0, 1,  1, 1, 1, 0,  0, 0, 8, 8,  0, 7, 7, 0, &
         0, 0, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0,  0, 0, 8, 8,  0, 7, 7, 0, &
         0, 0, 1, 1,  1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 8,  0, 7, 7, 0, &
         0, 0, 1, 1,  1, 0, 1, 1,  1, 1, 1, 0,  0, 0, 0, 0,  0, 7, 7, 0, &
         0, 0, 1, 1,  1, 2, 1, 1,  1, 1, 0, 0,  0, 0, 8, 8,  8, 8, 0, 9, &
         0, 0, 1, 0,  1, 2, 1, 1,  0, 0, 0, 0,  0, 0, 8, 8,  8, 8, 9, 9, &
         0, 0, 1, 0,  2, 2, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 9, 9, &
         0, 0, 1, 1,  2, 2, 1, 1,  0, 0, 6, 0,  0, 0, 0, 9,  9, 9, 9, 9, &
         0, 0, 0, 0,  2, 2, 2, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 9, 9, 0, &
         0, 0, 0, 2,  2, 2, 0, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 5, 5, 0, &
         0, 0, 2, 2,  2, 2, 0, 0,  0, 0, 0, 0,  0, 3, 3, 5,  0, 5, 0, 0, &
         0, 0, 2, 2,  2, 2, 0, 0,  0, 0, 6, 0,  4, 3, 3, 5,  5, 5, 0, 0, &
         0, 0, 0, 0,  2, 0, 0, 0,  0, 0, 0, 0,  4, 4, 0, 5,  5, 0, 0, 0, &
         0, 0, 0, 0,  2, 0, 0, 0,  0, 0, 0, 4,  4, 4, 0, 5,  5, 0, 0, 0, &
         0, 0, 0, 0,  2, 2, 2, 0,  0, 0, 4, 4,  4, 4, 0, 0,  5, 5, 0, 0, &
         0, 0, 6, 0,  0, 0, 0, 0,  0, 0, 0, 4,  4, 0, 0, 0,  0, 5, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(lake_pixel_mask)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          0, 16,  0,  0, 9, &
         16, 14, 0, 13, 0, &
         0, 15,  1,  0, 12, &
         0, 15,  1, 12, 14, &
         1, 0,  3,  1, 0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.0,    1.0,     0.0,    0.0,    0.5625, &
         1.0,    0.875,   0.0,    0.8125, 0.0, &
         0.0,    0.9375,  0.0625, 0.0,    0.75, &
         0.0,    0.9375,  0.0625, 0.75,   0.875, &
         0.0625, 0.0,     0.1875, 0.0625, 0.0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_binary_lake_mask(nlat_surface,nlon_surface))
      expected_binary_lake_mask = transpose(reshape((/   &
          .False.,  .True., .False., .False.,  .True., &
         .True.,  .True., .False.,  .True., .False., &
         .False.,  .True., .False., .False.,  .True., &
         .False.,  .True., .False.,  .True.,  .True., &
         .False., .False., .False., .False., .False. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      allocate(cell_mask(nlat_surface,nlon_surface))
      do lake_number = 1,9
        mask = (lake_pixel_mask == lake_number)
        npixels = count(mask)
        allocate(lake_pixel_coords_list_lat(npixels))
        allocate(lake_pixel_coords_list_lon(npixels))
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              lake_pixel_coords_list_lat(counter) = i
              lake_pixel_coords_list_lon(counter) = j
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        cell_mask(:,:) = .false.
        do i=1,size(potential_lake_pixel_coords_list_lat)
            lat = corresponding_surface_cell_lat_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            lon = corresponding_surface_cell_lon_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            cell_mask(lat,lon) = .true.
        end do
        allocate(cell_coords_list_lat(count(cell_mask)))
        allocate(cell_coords_list_lon(count(cell_mask)))
        counter = 1
        do i = 1,nlat_surface
          do j = 1,nlon_surface
            if (cell_mask(i,j)) then
              cell_coords_list_lat(counter) = i
              cell_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        lakes(lake_number)%lake_input_pointer => &
          lakeinput(lake_number, &
          lake_pixel_coords_list_lat, &
          lake_pixel_coords_list_lon, &
          potential_lake_pixel_coords_list_lat, &
          potential_lake_pixel_coords_list_lon, &
          cell_coords_list_lat, &
          cell_coords_list_lon)
      end do
      call calculate_lake_fractions(lakes, &
                                    cell_pixel_counts, &
                                    non_lake_mask, &
                                    lake_pixel_counts_field, &
                                    lake_fractions_field, &
                                    binary_lake_mask, &
                                    corresponding_surface_cell_lat_index, &
                                    corresponding_surface_cell_lon_index, &
                                    nlat_lake,nlon_lake, &
                                    nlat_surface,nlon_surface)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,nlat_surface,nlon_surface)
      call assert_equals(lake_fractions_field, expected_lake_fractions_field, &
                                            nlat_surface,nlon_surface)
      call assert_equals(binary_lake_mask, expected_binary_lake_mask, &
                                            nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(lake_fractions_field)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest3

subroutine testLakeFractionCalculationTest4
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: expected_binary_lake_mask
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Multiple lakes
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 1,  0, 1, 1, 1,  1, 1, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 1, 1,  0, 1, 1, 1,  1, 0, 1, 1,  1, 0, 1, 0,  0, 1, 1, 0, &
         0, 1, 1, 1,  0, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  1, 0, 1, 1,  1, 0, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 0, 1,  0, 1, 1, 0,  0, 0, 1, 1, &
         0, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 1,  1, 0, 1, 1,  1, 1, 1, 0, &
         0, 0, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 0, &
         0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1,  1, 0, 1, 1,  0, 0, 0, 0, &
         0, 2, 2, 2,  2, 0, 1, 1,  1, 0, 1, 1,  1, 1, 0, 1,  1, 0, 0, 0, &
         0, 2, 2, 0,  1, 1, 1, 1,  1, 1, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0, &
         0, 0, 2, 0,  0, 1, 1, 1,  0, 1, 1, 0,  0, 1, 1, 1,  0, 0, 0, 0, &
         0, 0, 2, 1,  0, 1, 0, 1,  0, 0, 1, 0,  1, 1, 1, 0,  0, 1, 0, 0, &
         1, 1, 0, 1,  1, 1, 1, 1,  1, 0, 1, 0,  1, 1, 1, 0,  1, 1, 1, 0, &
         0, 1, 1, 0,  1, 1, 1, 1,  1, 1, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0, &
         0, 1, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0,  0, 1, 0, 1,  0, 0, 0, 0, &
         0, 0, 0, 0,  1, 1, 0, 1,  1, 1, 0, 1,  1, 1, 0, 0,  1, 1, 0, 0, &
         0, 0, 1, 1,  0, 1, 1, 1,  1, 1, 0, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 1, 1, 1,  1, 0, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(lake_pixel_mask)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          0, 16, 16, 0, 0, &
         0, 16, 16, 16, 16, &
         8, 15, 16, 16, 0, &
         11, 16,  0, 16, 0, &
         0, 16, 16, 0, 0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.0, 1.0, 1.0, 0.0, 0.0, &
         0.0, 1.0, 1.0, 1.0, 1.0, &
         0.5, 0.9375, 1.0, 1.0, 0.0, &
         0.6875, 1.0, 0.0, 1.0, 0.0, &
         0.0, 1.0, 1.0, 0.0, 0.0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_binary_lake_mask(nlat_surface,nlon_surface))
      expected_binary_lake_mask = transpose(reshape((/   &
          .False.,  .True.,  .True., .False., .False., &
         .False.,  .True.,  .True.,  .True.,  .True., &
         .True.,  .True.,  .True.,  .True., .False., &
         .True.,  .True.,  .False.,  .True., .False., &
         .False.,  .True.,  .True., .False., .False. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      allocate(cell_mask(nlat_surface,nlon_surface))
      do lake_number = 1,2
        mask = (lake_pixel_mask == lake_number)
        npixels = count(mask)
        allocate(lake_pixel_coords_list_lat(npixels))
        allocate(lake_pixel_coords_list_lon(npixels))
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              lake_pixel_coords_list_lat(counter) = i
              lake_pixel_coords_list_lon(counter) = j
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        cell_mask(:,:) = .false.
        do i=1,size(potential_lake_pixel_coords_list_lat)
            lat = corresponding_surface_cell_lat_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            lon = corresponding_surface_cell_lon_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            cell_mask(lat,lon) = .true.
        end do
        allocate(cell_coords_list_lat(count(cell_mask)))
        allocate(cell_coords_list_lon(count(cell_mask)))
        counter = 1
        do i = 1,nlat_surface
          do j = 1,nlon_surface
            if (cell_mask(i,j)) then
              cell_coords_list_lat(counter) = i
              cell_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        lakes(lake_number)%lake_input_pointer => &
          lakeinput(lake_number, &
          lake_pixel_coords_list_lat, &
          lake_pixel_coords_list_lon, &
          potential_lake_pixel_coords_list_lat, &
          potential_lake_pixel_coords_list_lon, &
          cell_coords_list_lat, &
          cell_coords_list_lon)
      end do
      call calculate_lake_fractions(lakes, &
                                    cell_pixel_counts, &
                                    non_lake_mask, &
                                    lake_pixel_counts_field, &
                                    lake_fractions_field, &
                                    binary_lake_mask, &
                                    corresponding_surface_cell_lat_index, &
                                    corresponding_surface_cell_lon_index, &
                                    nlat_lake,nlon_lake, &
                                    nlat_surface,nlon_surface)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,nlat_surface,nlon_surface)
      call assert_equals(lake_fractions_field, expected_lake_fractions_field, &
                                          nlat_surface,nlon_surface)
      call assert_equals(binary_lake_mask, expected_binary_lake_mask, &
                                            nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(lake_fractions_field)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest4

subroutine testLakeFractionCalculationTest5
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: expected_binary_lake_mask
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Multiple lakes
      nlat_lake = 15
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 2, 2, 2, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  2, 2, 2, 2, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 2, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 2, 2, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         1, 1, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(lake_pixel_mask)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         8,  8,  8,  8,  8, &
         16, 16, 16, 16, 16, &
         4,  4,  4,  4,  4, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          0, 0, 0, 0, 10, &
         0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, &
         4, 0, 0, 0, 0, &
         6, 0, 0, 0, 0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.0, 0.0, 0.0, 0.0, 0.625, &
         0.0, 0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 0.0, &
         1.0, 0.0, 0.0, 0.0, 0.0, &
         0.375, 0.0, 0.0, 0.0, 0.0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_binary_lake_mask(nlat_surface,nlon_surface))
      expected_binary_lake_mask = transpose(reshape((/   &
          .False., .False., .False., .False.,  .True., &
         .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., &
         .True., .False., .False., .False., .False., &
         .False.,  .False., .False., .False., .False. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      allocate(cell_mask(nlat_surface,nlon_surface))
      do lake_number = 1,2
        mask = (lake_pixel_mask == lake_number)
        npixels = count(mask)
        allocate(lake_pixel_coords_list_lat(npixels))
        allocate(lake_pixel_coords_list_lon(npixels))
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              lake_pixel_coords_list_lat(counter) = i
              lake_pixel_coords_list_lon(counter) = j
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        cell_mask(:,:) = .false.
        do i=1,size(potential_lake_pixel_coords_list_lat)
            lat = corresponding_surface_cell_lat_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            lon = corresponding_surface_cell_lon_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            cell_mask(lat,lon) = .true.
        end do
        allocate(cell_coords_list_lat(count(cell_mask)))
        allocate(cell_coords_list_lon(count(cell_mask)))
        counter = 1
        do i = 1,nlat_surface
          do j = 1,nlon_surface
            if (cell_mask(i,j)) then
              cell_coords_list_lat(counter) = i
              cell_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        lakes(lake_number)%lake_input_pointer => &
          lakeinput(lake_number, &
          lake_pixel_coords_list_lat, &
          lake_pixel_coords_list_lon, &
          potential_lake_pixel_coords_list_lat, &
          potential_lake_pixel_coords_list_lon, &
          cell_coords_list_lat, &
          cell_coords_list_lon)
      end do
      call calculate_lake_fractions(lakes, &
                                    cell_pixel_counts, &
                                    non_lake_mask, &
                                    lake_pixel_counts_field, &
                                    lake_fractions_field, &
                                    binary_lake_mask, &
                                    corresponding_surface_cell_lat_index, &
                                    corresponding_surface_cell_lon_index, &
                                    nlat_lake,nlon_lake, &
                                    nlat_surface,nlon_surface)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,nlat_surface,nlon_surface)
      call assert_equals(lake_fractions_field, expected_lake_fractions_field, &
                                            nlat_surface,nlon_surface)
      call assert_equals(binary_lake_mask, expected_binary_lake_mask, &
                                            nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(lake_fractions_field)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest5

subroutine testLakeFractionCalculationTest6
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: potential_lake_pixel_mask
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: expected_binary_lake_mask
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Partially filled single lake
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      lake_number = 1
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(potential_lake_pixel_mask(nlat_lake,nlon_lake))
      potential_lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(potential_lake_pixel_mask)))
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 0, 0, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, &
         0, 0, 0, 0, 5, &
         0, 0, 0, 0, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.0, 0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 0.3125, &
         0.0, 0.0, 0.0, 0.0, 1.0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_binary_lake_mask(nlat_surface,nlon_surface))
      expected_binary_lake_mask = transpose(reshape((/   &
          .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .True. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      mask = (potential_lake_pixel_mask == 1)
      npixels = count(mask)
      allocate(potential_lake_pixel_coords_list_lat(npixels))
      allocate(potential_lake_pixel_coords_list_lon(npixels))
      counter = 1
      do i = 1,nlat_lake
        do j = 1,nlon_lake
          if (mask(i,j)) then
            potential_lake_pixel_coords_list_lat(counter) = i
            potential_lake_pixel_coords_list_lon(counter) = j
            counter = counter + 1
          end if
        end do
      end do
      mask = (lake_pixel_mask == 1)
      npixels = count(mask)
      allocate(lake_pixel_coords_list_lat(npixels))
      allocate(lake_pixel_coords_list_lon(npixels))
      counter = 1
      do i = 1,nlat_lake
        do j = 1,nlon_lake
          if (mask(i,j)) then
            lake_pixel_coords_list_lat(counter) = i
            lake_pixel_coords_list_lon(counter) = j
            counter = counter + 1
          end if
        end do
      end do
      allocate(cell_mask(nlat_surface,nlon_surface))
      cell_mask(:,:) = .false.
      do i=1,size(potential_lake_pixel_coords_list_lat)
          lat = corresponding_surface_cell_lat_index( &
            potential_lake_pixel_coords_list_lat(i), &
            potential_lake_pixel_coords_list_lon(i))
          lon = corresponding_surface_cell_lon_index( &
            potential_lake_pixel_coords_list_lat(i), &
            potential_lake_pixel_coords_list_lon(i))
          cell_mask(lat,lon) = .true.
      end do
      allocate(cell_coords_list_lat(count(cell_mask)))
      allocate(cell_coords_list_lon(count(cell_mask)))
      counter = 1
      do i = 1,nlat_surface
        do j = 1,nlon_surface
          if (cell_mask(i,j)) then
            cell_coords_list_lat(counter) = i
            cell_coords_list_lon(counter) = j
            counter = counter + 1
          end if
        end do
      end do
      lakes(1)%lake_input_pointer => lakeinput(lake_number, &
                                               lake_pixel_coords_list_lat, &
                                               lake_pixel_coords_list_lon, &
                                               potential_lake_pixel_coords_list_lat, &
                                               potential_lake_pixel_coords_list_lon, &
                                               cell_coords_list_lat, &
                                               cell_coords_list_lon)
      call calculate_lake_fractions(lakes, &
                                    cell_pixel_counts, &
                                    non_lake_mask, &
                                    lake_pixel_counts_field, &
                                    lake_fractions_field, &
                                    binary_lake_mask, &
                                    corresponding_surface_cell_lat_index, &
                                    corresponding_surface_cell_lon_index, &
                                    nlat_lake,nlon_lake, &
                                    nlat_surface,nlon_surface)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,nlat_surface,nlon_surface)
      call assert_equals(lake_fractions_field, expected_lake_fractions_field, &
                                            nlat_surface,nlon_surface)
      call assert_equals(binary_lake_mask, expected_binary_lake_mask, &
                                            nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(potential_lake_pixel_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(lake_fractions_field)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest6

subroutine testLakeFractionCalculationTest7
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: potential_lake_pixel_mask
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: expected_binary_lake_mask
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Multiple partially filled lakes
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(potential_lake_pixel_mask(nlat_lake,nlon_lake))
      potential_lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 2, 2, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 7, 0, &
         0, 2, 2, 2,  2, 0, 0, 0,  0, 0, 3, 3,  0, 0, 0, 0,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 6,  6, 6, 6, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 4, 4, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 4, 4, 4,  4, 4, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 4, 4, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 5, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 5, 5, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 5, 0, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 1, 0, &
         0, 0, 0, 5,  5, 5, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  0, 5, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(potential_lake_pixel_mask)))
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 2, 2, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 7, 0, &
         0, 2, 2, 2,  2, 0, 0, 0,  0, 0, 3, 3,  0, 0, 0, 0,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 6,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 4, 4, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 4, 4, 0,  4, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 4, 4, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 5, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 5, 5, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 5,  5, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  1, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          5, 1, 2, 0, 1, &
         0, 0, 0, 0, 9, &
         0, 6, 1, 0, 0, &
         0, 1, 0, 0, 16, &
         1, 3, 0, 0, 2 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.3125, 0.0625, 0.125, 0.0, 0.0625, &
         0.0, 0.0, 0.0, 0.0, 0.5625, &
         0.0, 0.375, 0.0625, 0.0, 0.0, &
         0.0, 0.0625, 0.0, 0.0, 1.0, &
         0.0625, 0.1875, 0.0, 0.0, 0.125 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_binary_lake_mask(nlat_surface,nlon_surface))
      expected_binary_lake_mask = transpose(reshape((/   &
          .False., .False., .False., .False., .False., &
         .False., .False., .False., .False.,  .True., &
         .False., .False., .False., .False., .False., &
         .False., .False., .False., .False.,  .True., &
         .False., .False., .False., .False., .False. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      allocate(cell_mask(nlat_surface,nlon_surface))
      do lake_number = 1,7
        mask = (potential_lake_pixel_mask == lake_number)
        npixels = count(mask)
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        mask = (lake_pixel_mask == lake_number)
        npixels = count(mask)
        allocate(lake_pixel_coords_list_lat(npixels))
        allocate(lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              lake_pixel_coords_list_lat(counter) = i
              lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        cell_mask(:,:) = .false.
        do i=1,size(potential_lake_pixel_coords_list_lat)
            lat = corresponding_surface_cell_lat_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            lon = corresponding_surface_cell_lon_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            cell_mask(lat,lon) = .true.
        end do
        allocate(cell_coords_list_lat(count(cell_mask)))
        allocate(cell_coords_list_lon(count(cell_mask)))
        counter = 1
        do i = 1,nlat_surface
          do j = 1,nlon_surface
            if (cell_mask(i,j)) then
              cell_coords_list_lat(counter) = i
              cell_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        lakes(lake_number)%lake_input_pointer => &
          lakeinput(lake_number, &
                    lake_pixel_coords_list_lat, &
                    lake_pixel_coords_list_lon, &
                    potential_lake_pixel_coords_list_lat, &
                    potential_lake_pixel_coords_list_lon, &
                    cell_coords_list_lat, &
                    cell_coords_list_lon)
      end do
      call calculate_lake_fractions(lakes, &
                                    cell_pixel_counts, &
                                    non_lake_mask, &
                                    lake_pixel_counts_field, &
                                    lake_fractions_field, &
                                    binary_lake_mask, &
                                    corresponding_surface_cell_lat_index, &
                                    corresponding_surface_cell_lon_index, &
                                    nlat_lake,nlon_lake, &
                                    nlat_surface,nlon_surface)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,nlat_surface,nlon_surface)
      call assert_equals(lake_fractions_field, expected_lake_fractions_field, &
                         nlat_surface,nlon_surface)
      call assert_equals(binary_lake_mask, expected_binary_lake_mask, &
                         nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(potential_lake_pixel_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(lake_fractions_field)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest7

subroutine testLakeFractionCalculationTest8
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: potential_lake_pixel_mask
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: expected_binary_lake_mask
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Multiple partially filled lakes
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(potential_lake_pixel_mask(nlat_lake,nlon_lake))
      potential_lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 1, 0,  1, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 0,  0, 0, 7, 0, &
         0, 0, 1, 0,  1, 1, 0, 1,  1, 1, 1, 0,  0, 0, 8, 8,  0, 7, 7, 0, &
         0, 0, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0,  0, 0, 8, 8,  0, 7, 7, 0, &
         0, 0, 1, 1,  1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 8,  0, 7, 7, 0, &
         0, 0, 1, 1,  1, 0, 1, 1,  1, 1, 1, 0,  0, 0, 0, 0,  0, 7, 7, 0, &
         0, 0, 1, 1,  1, 2, 1, 1,  1, 1, 0, 0,  0, 0, 8, 8,  8, 8, 0, 9, &
         0, 0, 1, 0,  1, 2, 1, 1,  0, 0, 0, 0,  0, 0, 8, 8,  8, 8, 9, 9, &
         0, 0, 1, 0,  2, 2, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 9, 9, &
         0, 0, 1, 1,  2, 2, 1, 1,  0, 0, 6, 0,  0, 0, 0, 9,  9, 9, 9, 9, &
         0, 0, 0, 0,  2, 2, 2, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 9, 9, 0, &
         0, 0, 0, 2,  2, 2, 0, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 5, 5, 0, &
         0, 0, 2, 2,  2, 2, 0, 0,  0, 0, 0, 0,  0, 3, 3, 5,  0, 5, 0, 0, &
         0, 0, 2, 2,  2, 2, 0, 0,  0, 0, 6, 0,  4, 3, 3, 5,  5, 5, 0, 0, &
         0, 0, 0, 0,  2, 0, 0, 0,  0, 0, 0, 0,  4, 4, 0, 5,  5, 0, 0, 0, &
         0, 0, 0, 0,  2, 0, 0, 0,  0, 0, 0, 4,  4, 4, 0, 5,  5, 0, 0, 0, &
         0, 0, 0, 0,  2, 2, 2, 0,  0, 0, 4, 4,  4, 4, 0, 0,  5, 5, 0, 0, &
         0, 0, 6, 0,  0, 0, 0, 0,  0, 0, 0, 4,  4, 0, 0, 0,  0, 5, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(potential_lake_pixel_mask)))
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  1, 1, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 7, 0, &
         0, 0, 0, 0,  1, 1, 0, 0,  0, 0, 0, 0,  0, 0, 8, 8,  0, 0, 7, 0, &
         0, 0, 0, 0,  1, 1, 1, 1,  0, 1, 0, 0,  0, 0, 8, 8,  0, 7, 0, 0, &
         0, 0, 0, 0,  0, 0, 1, 0,  0, 0, 0, 0,  0, 0, 0, 8,  0, 7, 7, 0, &
         0, 0, 0, 0,  0, 0, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 7, 7, 0, &
         0, 0, 1, 0,  0, 0, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  8, 8, 0, 9, &
         0, 0, 1, 0,  0, 0, 1, 1,  0, 0, 0, 0,  0, 0, 8, 0,  8, 8, 9, 9, &
         0, 0, 0, 0,  2, 2, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 9, 9, &
         0, 0, 0, 0,  2, 2, 1, 1,  0, 0, 6, 0,  0, 0, 0, 9,  9, 0, 0, 0, &
         0, 0, 0, 0,  2, 2, 2, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 9, 9, 0, &
         0, 0, 0, 2,  2, 2, 0, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 5, 0, &
         0, 0, 2, 2,  0, 0, 0, 0,  0, 0, 0, 0,  0, 3, 3, 5,  0, 0, 0, 0, &
         0, 0, 2, 2,  0, 0, 0, 0,  0, 0, 6, 0,  4, 3, 3, 5,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  4, 4, 0, 5,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 4,  4, 4, 0, 5,  0, 0, 0, 0, &
         0, 0, 0, 0,  2, 2, 2, 0,  0, 0, 4, 0,  4, 0, 0, 0,  5, 0, 0, 0, &
         0, 0, 6, 0,  0, 0, 0, 0,  0, 0, 0, 0,  4, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          0, 16, 0, 10, 3, &
         0, 6,  0,  0, 4, &
         0, 11, 1,  0, 10, &
         4, 0,  1, 16, 0, &
         1, 3,  0,  1, 1 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.0, 1.0, 0.0, 0.625, 0.1875, &
         0.0, 0.375, 0.0, 0.0, 0.25, &
         0.0, 0.6875, 0.0625, 0.0, 0.625, &
         0.25, 0.0, 0.0625, 1.0, 0.0, &
         0.0625, 0.1875, 0.0, 0.0625, 0.0625 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_binary_lake_mask(nlat_surface,nlon_surface))
      expected_binary_lake_mask = transpose(reshape((/   &
          .False.,  .True., .False.,  .True., .False., &
         .False., .False., .False., .False., .False., &
         .False.,  .True., .False., .False.,  .True., &
         .False., .False., .False.,  .True., .False., &
         .False., .False., .False., .False., .False. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      allocate(cell_mask(nlat_surface,nlon_surface))
      do lake_number = 1,9
        mask = (potential_lake_pixel_mask == lake_number)
        npixels = count(mask)
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        mask = (lake_pixel_mask == lake_number)
        npixels = count(mask)
        allocate(lake_pixel_coords_list_lat(npixels))
        allocate(lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              lake_pixel_coords_list_lat(counter) = i
              lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        cell_mask(:,:) = .false.
        do i=1,size(potential_lake_pixel_coords_list_lat)
            lat = corresponding_surface_cell_lat_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            lon = corresponding_surface_cell_lon_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            cell_mask(lat,lon) = .true.
        end do
        allocate(cell_coords_list_lat(count(cell_mask)))
        allocate(cell_coords_list_lon(count(cell_mask)))
        counter = 1
        do i = 1,nlat_surface
          do j = 1,nlon_surface
            if (cell_mask(i,j)) then
              cell_coords_list_lat(counter) = i
              cell_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        lakes(lake_number)%lake_input_pointer => &
          lakeinput(lake_number, &
                    lake_pixel_coords_list_lat, &
                    lake_pixel_coords_list_lon, &
                    potential_lake_pixel_coords_list_lat, &
                    potential_lake_pixel_coords_list_lon, &
                    cell_coords_list_lat, &
                    cell_coords_list_lon)
      end do
      call calculate_lake_fractions(lakes, &
                                    cell_pixel_counts, &
                                    non_lake_mask, &
                                    lake_pixel_counts_field, &
                                    lake_fractions_field, &
                                    binary_lake_mask, &
                                    corresponding_surface_cell_lat_index, &
                                    corresponding_surface_cell_lon_index, &
                                    nlat_lake,nlon_lake, &
                                    nlat_surface,nlon_surface)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,nlat_surface,nlon_surface)
      call assert_equals(lake_fractions_field, expected_lake_fractions_field, &
                                            nlat_surface,nlon_surface)
      call assert_equals(binary_lake_mask, expected_binary_lake_mask, &
                                            nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(potential_lake_pixel_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(lake_fractions_field)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest8

subroutine testLakeFractionCalculationTest9
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: potential_lake_pixel_mask
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: expected_binary_lake_mask
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Multiple partially filled lakes
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(potential_lake_pixel_mask(nlat_lake,nlon_lake))
      potential_lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 1,  0, 1, 1, 1,  1, 1, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 1, 1,  0, 1, 1, 1,  1, 0, 1, 1,  1, 0, 1, 0,  0, 1, 1, 0, &
         0, 1, 1, 1,  0, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  1, 0, 1, 1,  1, 0, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 0, 1,  0, 1, 1, 0,  0, 0, 1, 1, &
         0, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 1,  1, 0, 1, 1,  1, 1, 1, 0, &
         0, 0, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 0, &
         0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1,  1, 0, 1, 1,  0, 0, 0, 0, &
         0, 2, 2, 2,  2, 0, 1, 1,  1, 0, 1, 1,  1, 1, 0, 1,  1, 0, 0, 0, &
         0, 2, 2, 0,  1, 1, 1, 1,  1, 1, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  0, 1, 1, 1,  0, 1, 1, 0,  0, 1, 1, 1,  0, 0, 0, 0, &
         0, 0, 0, 1,  0, 1, 0, 1,  0, 0, 1, 0,  1, 1, 1, 0,  0, 1, 0, 0, &
         1, 1, 0, 1,  1, 1, 1, 1,  1, 0, 1, 0,  1, 1, 1, 0,  1, 1, 1, 0, &
         0, 1, 1, 0,  1, 1, 1, 1,  1, 1, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0, &
         0, 1, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0,  0, 1, 0, 1,  0, 0, 0, 0, &
         0, 0, 0, 0,  1, 1, 0, 1,  1, 1, 0, 1,  1, 1, 0, 0,  1, 1, 0, 0, &
         0, 0, 1, 1,  0, 1, 1, 1,  1, 1, 0, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 1, 1, 1,  1, 0, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(potential_lake_pixel_mask)))
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 1,  0, 1, 1, 1,  1, 1, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 1, 1,  0, 1, 1, 1,  1, 0, 1, 1,  1, 0, 1, 0,  0, 1, 1, 0, &
         0, 1, 1, 1,  0, 1, 1, 1,  0, 1, 0, 1,  1, 1, 1, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  1, 0, 1, 1,  1, 0, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 0, 1,  0, 1, 1, 0,  0, 0, 1, 1, &
         0, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 1,  1, 0, 1, 1,  1, 1, 1, 0, &
         0, 0, 1, 1,  0, 0, 1, 0,  0, 0, 1, 1,  1, 1, 1, 1,  1, 1, 1, 0, &
         0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1,  1, 0, 1, 1,  0, 0, 0, 0, &
         0, 2, 2, 2,  0, 0, 1, 1,  1, 0, 1, 1,  0, 1, 0, 1,  1, 0, 0, 0, &
         0, 2, 0, 0,  1, 1, 1, 1,  1, 1, 1, 0,  1, 0, 0, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  0, 1, 1, 1,  0, 1, 1, 0,  0, 1, 1, 1,  0, 0, 0, 0, &
         0, 0, 0, 1,  0, 0, 0, 1,  0, 0, 1, 0,  1, 1, 1, 0,  0, 0, 0, 0, &
         1, 1, 0, 1,  0, 0, 0, 0,  1, 0, 1, 0,  1, 1, 1, 0,  0, 0, 1, 0, &
         0, 1, 1, 0,  0, 0, 1, 0,  1, 1, 1, 0,  1, 1, 1, 1,  0, 0, 0, 0, &
         0, 1, 1, 0,  1, 0, 1, 1,  1, 1, 0, 0,  0, 1, 0, 1,  0, 0, 0, 0, &
         0, 0, 0, 0,  1, 0, 0, 1,  1, 1, 0, 1,  1, 1, 0, 0,  1, 1, 0, 0, &
         0, 0, 1, 1,  0, 1, 1, 1,  1, 1, 0, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 1, 1, 1,  1, 0, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          0, 16,  16,  0,  0, &
         0, 0,  0,  16, 16, &
         4, 15, 16, 16, 0, &
         16, 0,  7,  16, 0, &
         0, 16, 16, 0,  0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.0, 1.0, 1.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 1.0, 1.0, &
         0.25, 0.9375, 1.0, 1.0, 0.0, &
         1.0, 0.0, 0.4375, 1.0, 0.0, &
         0.0, 1.0, 1.0, 0.0, 0.0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_binary_lake_mask(nlat_surface,nlon_surface))
      expected_binary_lake_mask = transpose(reshape((/   &
          .False.,  .True.,  .True., .False., .False., &
         .False., .False., .False.,  .True.,  .True., &
         .False.,  .True., .True.,  .True.,  .False., &
         .True., .False., .False.,  .True., .False., &
         .False.,  .True.,  .True., .False., .False. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      allocate(cell_mask(nlat_surface,nlon_surface))
      do lake_number = 1,2
        mask = (potential_lake_pixel_mask == lake_number)
        npixels = count(mask)
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        mask = (lake_pixel_mask == lake_number)
        npixels = count(mask)
        allocate(lake_pixel_coords_list_lat(npixels))
        allocate(lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              lake_pixel_coords_list_lat(counter) = i
              lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        cell_mask(:,:) = .false.
        do i=1,size(potential_lake_pixel_coords_list_lat)
            lat = corresponding_surface_cell_lat_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            lon = corresponding_surface_cell_lon_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            cell_mask(lat,lon) = .true.
        end do
        allocate(cell_coords_list_lat(count(cell_mask)))
        allocate(cell_coords_list_lon(count(cell_mask)))
        counter = 1
        do i = 1,nlat_surface
          do j = 1,nlon_surface
            if (cell_mask(i,j)) then
              cell_coords_list_lat(counter) = i
              cell_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        lakes(lake_number)%lake_input_pointer => &
          lakeinput(lake_number, &
                    lake_pixel_coords_list_lat, &
                    lake_pixel_coords_list_lon, &
                    potential_lake_pixel_coords_list_lat, &
                    potential_lake_pixel_coords_list_lon, &
                    cell_coords_list_lat, &
                    cell_coords_list_lon)
      end do
      call calculate_lake_fractions(lakes, &
                                    cell_pixel_counts, &
                                    non_lake_mask, &
                                    lake_pixel_counts_field, &
                                    lake_fractions_field, &
                                    binary_lake_mask, &
                                    corresponding_surface_cell_lat_index, &
                                    corresponding_surface_cell_lon_index, &
                                    nlat_lake,nlon_lake, &
                                    nlat_surface,nlon_surface)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,nlat_surface,nlon_surface)
      call assert_equals(lake_fractions_field, expected_lake_fractions_field, &
                                            nlat_surface,nlon_surface)
      call assert_equals(binary_lake_mask, expected_binary_lake_mask, &
                                            nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(potential_lake_pixel_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(lake_fractions_field)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest9

subroutine testLakeFractionCalculationTest10
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: potential_lake_pixel_mask
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: expected_binary_lake_mask
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Multiple partially filled lakes with uneven cell sizes
      nlat_lake = 15
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(potential_lake_pixel_mask(nlat_lake,nlon_lake))
      potential_lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 2, 2, 2, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  2, 2, 2, 2, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 2, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 2, 2, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         1, 1, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(potential_lake_pixel_mask)))
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 2, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  2, 0, 0, 2, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 2, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 2, 2, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         1, 1, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         16, 16, 16, 16,  8, &
         16, 16, 16, 16, 16, &
         4,  4,  4,  4,  4, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          0, 0, 0, 0, 0, &
         0, 0, 0, 0, 6, &
         0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, &
         8, 0, 0, 0, 0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.0, 0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 0.75, &
         0.0, 0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 0.0, &
         0.5, 0.0, 0.0, 0.0, 0.0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_binary_lake_mask(nlat_surface,nlon_surface))
      expected_binary_lake_mask = transpose(reshape((/   &
          .False., .False., .False., .False., .False., &
         .False., .False., .False., .False.,  .True., &
         .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., &
         .True., .False., .False., .False., .False. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      allocate(cell_mask(nlat_surface,nlon_surface))
      do lake_number = 1,2
        mask = (potential_lake_pixel_mask == lake_number)
        npixels = count(mask)
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        mask = (lake_pixel_mask == lake_number)
        npixels = count(mask)
        allocate(lake_pixel_coords_list_lat(npixels))
        allocate(lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              lake_pixel_coords_list_lat(counter) = i
              lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        cell_mask(:,:) = .false.
        do i=1,size(potential_lake_pixel_coords_list_lat)
            lat = corresponding_surface_cell_lat_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            lon = corresponding_surface_cell_lon_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            cell_mask(lat,lon) = .true.
        end do
        allocate(cell_coords_list_lat(count(cell_mask)))
        allocate(cell_coords_list_lon(count(cell_mask)))
        counter = 1
        do i = 1,nlat_surface
          do j = 1,nlon_surface
            if (cell_mask(i,j)) then
              cell_coords_list_lat(counter) = i
              cell_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        lakes(lake_number)%lake_input_pointer => &
          lakeinput(lake_number, &
                    lake_pixel_coords_list_lat, &
                    lake_pixel_coords_list_lon, &
                    potential_lake_pixel_coords_list_lat, &
                    potential_lake_pixel_coords_list_lon, &
                    cell_coords_list_lat, &
                    cell_coords_list_lon)
      end do
      call calculate_lake_fractions(lakes, &
                                    cell_pixel_counts, &
                                    non_lake_mask, &
                                    lake_pixel_counts_field, &
                                    lake_fractions_field, &
                                    binary_lake_mask, &
                                    corresponding_surface_cell_lat_index, &
                                    corresponding_surface_cell_lon_index, &
                                    nlat_lake,nlon_lake, &
                                    nlat_surface,nlon_surface)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,nlat_surface,nlon_surface)
      call assert_equals(lake_fractions_field, expected_lake_fractions_field, &
                                            nlat_surface,nlon_surface)
      call assert_equals(binary_lake_mask, expected_binary_lake_mask, &
                                            nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(potential_lake_pixel_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(lake_fractions_field)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest10

subroutine testLakeFractionCalculationTest11
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   type(lakefractioncalculationprognostics), pointer :: prognostics
   integer,dimension(:,:), pointer :: lake_pixel_mask
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   integer,dimension(:,:), pointer :: expected_immediate_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Single lake
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      lake_number = 1
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(lake_pixel_mask)))
      allocate(binary_lake_mask(nlat_surface,nlon_surface))
      binary_lake_mask = transpose(reshape((/   &
          .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .True., &
         .False., .False., .False., .False., .True. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, &
         0, 0, 0, 0, 16, &
         0, 0, 0, 0, 14 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_immediate_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_immediate_lake_pixel_counts_field = transpose(reshape((/   &
          0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, &
         0, 0, 0, 0, 14, &
         0, 0, 0, 0, 13 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.0, 0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 1.0, &
         0.0, 0.0, 0.0, 0.0, 0.875 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      mask = (lake_pixel_mask == 1)
      allocate(lake_pixel_coords_list_lat(0))
      allocate(lake_pixel_coords_list_lon(0))
      npixels = count(mask)
      allocate(potential_lake_pixel_coords_list_lat(npixels))
      allocate(potential_lake_pixel_coords_list_lon(npixels))
      counter = 1
      do i = 1,nlat_lake
        do j = 1,nlon_lake
          if (mask(i,j)) then
            potential_lake_pixel_coords_list_lat(counter) = i
            potential_lake_pixel_coords_list_lon(counter) = j
            counter = counter + 1
          end if
        end do
      end do
      allocate(cell_mask(nlat_surface,nlon_surface))
      cell_mask(:,:) = .false.
      do i=1,size(potential_lake_pixel_coords_list_lat)
          lat = corresponding_surface_cell_lat_index( &
            potential_lake_pixel_coords_list_lat(i), &
            potential_lake_pixel_coords_list_lon(i))
          lon = corresponding_surface_cell_lon_index( &
            potential_lake_pixel_coords_list_lat(i), &
            potential_lake_pixel_coords_list_lon(i))
          cell_mask(lat,lon) = .true.
      end do
      allocate(cell_coords_list_lat(count(cell_mask)))
      allocate(cell_coords_list_lon(count(cell_mask)))
      counter = 1
      do i = 1,nlat_surface
        do j = 1,nlon_surface
          if (cell_mask(i,j)) then
            cell_coords_list_lat(counter) = i
            cell_coords_list_lon(counter) = j
            counter = counter + 1
          end if
        end do
      end do
      lakes(1)%lake_input_pointer => lakeinput(lake_number, &
                                               lake_pixel_coords_list_lat, &
                                               lake_pixel_coords_list_lon, &
                                               potential_lake_pixel_coords_list_lat, &
                                               potential_lake_pixel_coords_list_lon, &
                                               cell_coords_list_lat, &
                                               cell_coords_list_lon)
      prognostics => setup_lake_for_fraction_calculation(lakes, &
                                                         cell_pixel_counts, &
                                                         non_lake_mask, &
                                                         binary_lake_mask, &
                                                         lake_pixel_mask, &
                                                         corresponding_surface_cell_lat_index, &
                                                         corresponding_surface_cell_lon_index, &
                                                         nlat_lake,nlon_lake, &
                                                         nlat_surface,nlon_surface)
      allocate(lake_pixel_counts_field(nlat_surface,nlon_surface))
      lake_pixel_counts_field(:,:) = 0
      do i = 1,size(potential_lake_pixel_coords_list_lat)
        call add_pixel_by_coords(potential_lake_pixel_coords_list_lat(i),&
                                 potential_lake_pixel_coords_list_lon(i), &
                                 lake_pixel_counts_field,prognostics)
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,&
                         nlat_surface,nlon_surface)
      call remove_pixel_by_coords(15,19,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(11,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(15,20,lake_pixel_counts_field, &
                                  prognostics)
      call assert_equals(lake_pixel_counts_field, &
                         expected_immediate_lake_pixel_counts_field, &
                         nlat_surface,nlon_surface)
      call add_pixel_by_coords(15,19,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(11,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(15,20,lake_pixel_counts_field, &
                               prognostics)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_immediate_lake_pixel_counts_field)
      deallocate(cell_mask)
      deallocate(mask)
      call clean_lake_fraction_calculation_prognostics(prognostics)
      deallocate(prognostics)
end subroutine testLakeFractionCalculationTest11

subroutine testLakeFractionCalculationTest12
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   type(lakefractioncalculationprognostics), pointer :: prognostics
   integer :: lake_number
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   integer,dimension(:,:), pointer :: expected_immediate_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Multiple lakes
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 2, 2, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 7, 0, &
         0, 2, 2, 2,  2, 0, 0, 0,  0, 0, 3, 3,  0, 0, 0, 0,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 6,  6, 6, 6, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 4, 4, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 4, 4, 4,  4, 4, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 4, 4, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 5, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 5, 5, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 5, 0, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 1, 0, &
         0, 0, 0, 5,  5, 5, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  0, 5, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(lake_pixel_mask)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(binary_lake_mask(nlat_surface,nlon_surface))
      binary_lake_mask = transpose(reshape((/   &
          .False., .False., .False., .False., .False., &
         .False., .False., .False., .False.,  .True., &
         .False.,  .True., .False., .False., .False., &
         .False., .False., .False., .False.,  .True., &
         .False.,  .True., .False., .False.,  .True. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          5, 1, 2, 0, 1, &
         0, 0, 0, 0, 11, &
         0, 9, 0, 0, 0, &
         0, 0, 0, 0, 16, &
         0, 8, 0, 0, 14 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_immediate_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_immediate_lake_pixel_counts_field = transpose(reshape((/   &
           3, 1, 0, 0, 0, &
         0, 0, 0, 0, 0, &
         0, 3, 0, 0, 0, &
         0, 0, 0, 0, 7, &
         0, 6, 0, 0, 1  /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.3125, 0.0625, 0.125, 0.0, 0.0625, &
         0.0, 0.0, 0.0, 0.0, 0.6875, &
         0.0, 0.5625, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 1.0, &
         0.0, 0.5, 0.0, 0.0, 0.875 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      allocate(cell_mask(nlat_surface,nlon_surface))
      do lake_number = 1,7
        mask = (lake_pixel_mask == lake_number)
        allocate(lake_pixel_coords_list_lat(0))
        allocate(lake_pixel_coords_list_lon(0))
        npixels = count(mask)
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        cell_mask(:,:) = .false.
        do i=1,size(potential_lake_pixel_coords_list_lat)
            lat = corresponding_surface_cell_lat_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            lon = corresponding_surface_cell_lon_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            cell_mask(lat,lon) = .true.
        end do
        allocate(cell_coords_list_lat(count(cell_mask)))
        allocate(cell_coords_list_lon(count(cell_mask)))
        counter = 1
        do i = 1,nlat_surface
          do j = 1,nlon_surface
            if (cell_mask(i,j)) then
              cell_coords_list_lat(counter) = i
              cell_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        lakes(lake_number)%lake_input_pointer => &
          lakeinput(lake_number, &
                    lake_pixel_coords_list_lat, &
                    lake_pixel_coords_list_lon, &
                    potential_lake_pixel_coords_list_lat, &
                    potential_lake_pixel_coords_list_lon, &
                    cell_coords_list_lat, &
                    cell_coords_list_lon)
      end do
      prognostics => setup_lake_for_fraction_calculation(lakes, &
                                                         cell_pixel_counts, &
                                                         non_lake_mask, &
                                                         binary_lake_mask, &
                                                         lake_pixel_mask, &
                                                         corresponding_surface_cell_lat_index, &
                                                         corresponding_surface_cell_lon_index, &
                                                         nlat_lake,nlon_lake, &
                                                         nlat_surface,nlon_surface)
      allocate(lake_pixel_counts_field(nlat_surface,nlon_surface))
      lake_pixel_counts_field(:,:) = 0
      do lake_number = 1,7
        mask = (lake_pixel_mask == lake_number)
        npixels = count(mask)
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        do i = 1,size(potential_lake_pixel_coords_list_lat)
          call add_pixel_by_coords(potential_lake_pixel_coords_list_lat(i), &
                                   potential_lake_pixel_coords_list_lon(i), &
                                   lake_pixel_counts_field,prognostics)
        end do
        deallocate(potential_lake_pixel_coords_list_lat)
        deallocate(potential_lake_pixel_coords_list_lon)
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,nlat_surface,nlon_surface)
      call remove_pixel_by_coords(2,2,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,3,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(10,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(16,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(19,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(8,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(5,19,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(6,17,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,17,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(2,19,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,12,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,11,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(11,9,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(12,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(12,7,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(10,7,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(11,10,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(11,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(12,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(12,19,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(11,19,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(14,16,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(15,16,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(15,15,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(14,15,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(14,17,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(15,17,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,19,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(19,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(20,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(19,17,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(19,16,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(19,15,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,15,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(17,15,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(13,20,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(14,20,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(15,20,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(6,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(7,17,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(7,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(5,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(5,17,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(5,16,lake_pixel_counts_field, &
                                  prognostics)
      call assert_equals(lake_pixel_counts_field, &
                         expected_immediate_lake_pixel_counts_field, &
                         nlat_surface,nlon_surface)
      call add_pixel_by_coords(2,2,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,3,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(10,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(16,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(19,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(8,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(5,19,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(6,17,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,17,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(2,19,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,12,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,11,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(11,9,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(12,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(12,7,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(10,7,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(11,10,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(11,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(12,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(12,19,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(11,19,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(14,16,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(15,16,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(15,15,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(14,15,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(14,17,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(15,17,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,19,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(19,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(20,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(19,17,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(19,16,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(19,15,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,15,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(17,15,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(13,20,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(14,20,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(15,20,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(6,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(7,17,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(7,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(5,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(5,17,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(5,16,lake_pixel_counts_field, &
                               prognostics)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_immediate_lake_pixel_counts_field)
      deallocate(cell_mask)
      deallocate(mask)
      call clean_lake_fraction_calculation_prognostics(prognostics)
      deallocate(prognostics)
end subroutine testLakeFractionCalculationTest12

subroutine testLakeFractionCalculationTest13
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   type(lakefractioncalculationprognostics), pointer :: prognostics
   integer :: lake_number
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field_two
   integer,dimension(:,:), pointer :: expected_intermediate_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Multiple lakes
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 1, 0,  1, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 0,  0, 0, 7, 0, &
         0, 0, 1, 0,  1, 1, 0, 1,  1, 1, 1, 0,  0, 0, 8, 8,  0, 7, 7, 0, &
         0, 0, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0,  0, 0, 8, 8,  0, 7, 7, 0, &
         0, 0, 1, 1,  1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 8,  0, 7, 7, 0, &
         0, 0, 1, 1,  1, 0, 1, 1,  1, 1, 1, 0,  0, 0, 0, 0,  0, 7, 7, 0, &
         0, 0, 1, 1,  1, 2, 1, 1,  1, 1, 0, 0,  0, 0, 8, 8,  8, 8, 0, 9, &
         0, 0, 1, 0,  1, 2, 1, 1,  0, 0, 0, 0,  0, 0, 8, 8,  8, 8, 9, 9, &
         0, 0, 1, 0,  2, 2, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 9, 9, &
         0, 0, 1, 1,  2, 2, 1, 1,  0, 0, 6, 0,  0, 0, 0, 9,  9, 9, 9, 9, &
         0, 0, 0, 0,  2, 2, 2, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 9, 9, 0, &
         0, 0, 0, 2,  2, 2, 0, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 5, 5, 0, &
         0, 0, 2, 2,  2, 2, 0, 0,  0, 0, 0, 0,  0, 3, 3, 5,  0, 5, 0, 0, &
         0, 0, 2, 2,  2, 2, 0, 0,  0, 0, 6, 0,  4, 3, 3, 5,  5, 5, 0, 0, &
         0, 0, 0, 0,  2, 0, 0, 0,  0, 0, 0, 0,  4, 4, 0, 5,  5, 0, 0, 0, &
         0, 0, 0, 0,  2, 0, 0, 0,  0, 0, 0, 4,  4, 4, 0, 5,  5, 0, 0, 0, &
         0, 0, 0, 0,  2, 2, 2, 0,  0, 0, 4, 4,  4, 4, 0, 0,  5, 5, 0, 0, &
         0, 0, 6, 0,  0, 0, 0, 0,  0, 0, 0, 4,  4, 0, 0, 0,  0, 5, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(lake_pixel_mask)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(binary_lake_mask(nlat_surface,nlon_surface))
      binary_lake_mask = transpose(reshape((/   &
          .False.,  .True., .False., .False.,  .True., &
         .True.,  .True., .False.,  .True., .False., &
         .False.,  .True., .False., .False.,  .True., &
         .False.,  .True., .False.,  .True.,  .True., &
         .False., .False., .False., .False., .False. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          0, 16,  0,  0, 9, &
         16, 16, 0, 13, 0, &
         0, 16,  1,  0, 14, &
         0, 13,  1, 16, 8, &
         1, 0,  1,  3, 0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_pixel_counts_field_two(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field_two = transpose(reshape((/   &
          0, 16,  0,  0, 9, &
         16, 16, 0, 13, 0, &
         0, 16,  1,  0, 14, &
         0, 13,  1, 16, 8, &
         1, 0,  1,  3, 0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_intermediate_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_intermediate_lake_pixel_counts_field = transpose(reshape((/   &
          0, 9,  0,  0, 5, &
         14, 1, 0,   9, 0, &
         0, 16,  0,  0, 8, &
         0, 10,  1, 10, 5, &
         1, 0,  0,  0, 0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.0,    1.0,     0.0,    0.0,    0.5625, &
         1.0,    0.875,   0.0,    0.8125, 0.0, &
         0.0,    1.0,     0.0625, 0.0,    0.75, &
         0.0,    0.9375,  0.0625, 0.75,   0.875, &
         0.0625, 0.0,     0.1875, 0.0625, 0.0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      allocate(cell_mask(nlat_surface,nlon_surface))
      do lake_number = 1,9
        mask = (lake_pixel_mask == lake_number)
        allocate(lake_pixel_coords_list_lat(0))
        allocate(lake_pixel_coords_list_lon(0))
        npixels = count(mask)
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        cell_mask(:,:) = .false.
        do i=1,size(potential_lake_pixel_coords_list_lat)
            lat = corresponding_surface_cell_lat_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            lon = corresponding_surface_cell_lon_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            cell_mask(lat,lon) = .true.
        end do
        allocate(cell_coords_list_lat(count(cell_mask)))
        allocate(cell_coords_list_lon(count(cell_mask)))
        counter = 1
        do i = 1,nlat_surface
          do j = 1,nlon_surface
            if (cell_mask(i,j)) then
              cell_coords_list_lat(counter) = i
              cell_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        lakes(lake_number)%lake_input_pointer => &
          lakeinput(lake_number, &
                    lake_pixel_coords_list_lat, &
                    lake_pixel_coords_list_lon, &
                    potential_lake_pixel_coords_list_lat, &
                    potential_lake_pixel_coords_list_lon, &
                    cell_coords_list_lat, &
                    cell_coords_list_lon)
      end do
      prognostics => setup_lake_for_fraction_calculation(lakes, &
                                                         cell_pixel_counts, &
                                                         non_lake_mask, &
                                                         binary_lake_mask, &
                                                         lake_pixel_mask, &
                                                         corresponding_surface_cell_lat_index, &
                                                         corresponding_surface_cell_lon_index, &
                                                         nlat_lake,nlon_lake, &
                                                         nlat_surface,nlon_surface)
      allocate(lake_pixel_counts_field(nlat_surface,nlon_surface))
      lake_pixel_counts_field(:,:) = 0
      do lake_number = 1,9
        mask = (lake_pixel_mask == lake_number)
        npixels = count(mask)
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        do i = 1,size(potential_lake_pixel_coords_list_lat)
          call add_pixel_by_coords(potential_lake_pixel_coords_list_lat(i), &
                                   potential_lake_pixel_coords_list_lon(i), &
                                   lake_pixel_counts_field,prognostics)
        end do
        deallocate(potential_lake_pixel_coords_list_lat)
        deallocate(potential_lake_pixel_coords_list_lon)
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,&
                         nlat_surface,nlon_surface)
      call remove_pixel_by_coords(2,3,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,3,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(4,3,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,9,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(4,9,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(4,10,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,10,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,11,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(2,11,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(5,5,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(5,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(5,7,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(5,8,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(6,8,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(6,7,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(6,5,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(7,5,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(7,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(7,7,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(7,8,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(8,8,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(8,7,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(8,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(8,5,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(17,5,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(17,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(17,7,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(17,11,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(17,12,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,12,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,13,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(17,13,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(17,14,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(14,14,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(14,15,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(13,15,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(13,14,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(17,17,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(17,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(8,19,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(8,20,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(9,20,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(9,19,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(10,16,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(10,17,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(7,17,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(8,17,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(8,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(7,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(6,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(5,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(5,19,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(6,19,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(10,11,lake_pixel_counts_field, &
                                  prognostics)
      call assert_equals(lake_pixel_counts_field, &
                         expected_intermediate_lake_pixel_counts_field,&
                         nlat_surface,nlon_surface)
      call add_pixel_by_coords(2,3,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,3,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(4,3,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,9,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(4,9,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(4,10,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,10,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,11,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(2,11,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(5,5,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(5,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(5,7,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(5,8,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(6,8,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(6,7,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(6,5,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(7,5,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(7,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(7,7,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(7,8,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(8,8,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(8,7,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(8,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(8,5,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(17,5,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(17,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(17,7,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(17,11,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(17,12,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,12,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,13,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(17,13,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(17,14,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(14,14,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(14,15,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(13,15,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(13,14,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(17,17,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(17,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(8,19,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(8,20,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(9,20,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(9,19,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(10,16,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(10,17,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(7,17,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(8,17,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(8,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(7,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(6,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(5,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(5,19,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(6,19,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(10,11,lake_pixel_counts_field, &
                               prognostics)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field_two,&
                         nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_pixel_counts_field_two)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_intermediate_lake_pixel_counts_field)
      deallocate(cell_mask)
      deallocate(mask)
      call clean_lake_fraction_calculation_prognostics(prognostics)
      deallocate(prognostics)
end subroutine testLakeFractionCalculationTest13

subroutine testLakeFractionCalculationTest14
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   type(lakefractioncalculationprognostics), pointer :: prognostics
   integer :: lake_number
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Multiple lakes
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 1,  0, 1, 1, 1,  1, 1, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 1, 1,  0, 1, 1, 1,  1, 0, 1, 1,  1, 0, 1, 0,  0, 1, 1, 0, &
         0, 1, 1, 1,  0, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  1, 0, 1, 1,  1, 0, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 0, 1,  0, 1, 1, 0,  0, 0, 1, 1, &
         0, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 1,  1, 0, 1, 1,  1, 1, 1, 0, &
         0, 0, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 0, &
         0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1,  1, 0, 1, 1,  0, 0, 0, 0, &
         0, 2, 2, 2,  2, 0, 1, 1,  1, 0, 1, 1,  1, 1, 0, 1,  1, 0, 0, 0, &
         0, 2, 2, 0,  1, 1, 1, 1,  1, 1, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0, &
         0, 0, 2, 0,  0, 1, 1, 1,  0, 1, 1, 0,  0, 1, 1, 1,  0, 0, 0, 0, &
         0, 0, 2, 1,  0, 1, 0, 1,  0, 0, 1, 0,  1, 1, 1, 0,  0, 1, 0, 0, &
         1, 1, 0, 1,  1, 1, 1, 1,  1, 0, 1, 0,  1, 1, 1, 0,  1, 1, 1, 0, &
         0, 1, 1, 0,  1, 1, 1, 1,  1, 1, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0, &
         0, 1, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0,  0, 1, 0, 1,  0, 0, 0, 0, &
         0, 0, 0, 0,  1, 1, 0, 1,  1, 1, 0, 1,  1, 1, 0, 0,  1, 1, 0, 0, &
         0, 0, 1, 1,  0, 1, 1, 1,  1, 1, 0, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 1, 1, 1,  1, 0, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(lake_pixel_mask)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(binary_lake_mask(nlat_surface,nlon_surface))
      binary_lake_mask = transpose(reshape((/   &
          .False.,  .True.,  .True., .False., .False., &
         .False.,  .True.,  .True.,  .True.,  .True., &
         .True.,  .True.,  .True.,  .True., .False., &
         .True.,  .True.,  .False.,  .True., .False., &
         .False.,  .True.,  .True., .False., .False. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          0, 16, 16, 0, 0, &
         0, 16, 16, 16, 16, &
         6, 16, 16, 16, 0, &
         16, 16,  0, 16, 0, &
         0, 16, 12, 0, 0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.0, 1.0, 1.0, 0.0, 0.0, &
         0.0, 1.0, 1.0, 1.0, 1.0, &
         0.5, 0.9375, 1.0, 1.0, 0.0, &
         0.6875, 1.0, 0.0, 1.0, 0.0, &
         0.0, 1.0, 1.0, 0.0, 0.0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      allocate(cell_mask(nlat_surface,nlon_surface))
      do lake_number = 1,2
        mask = (lake_pixel_mask == lake_number)
        allocate(lake_pixel_coords_list_lat(0))
        allocate(lake_pixel_coords_list_lon(0))
        npixels = count(mask)
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        cell_mask(:,:) = .false.
        do i=1,size(potential_lake_pixel_coords_list_lat)
            lat = corresponding_surface_cell_lat_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            lon = corresponding_surface_cell_lon_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            cell_mask(lat,lon) = .true.
        end do
        allocate(cell_coords_list_lat(count(cell_mask)))
        allocate(cell_coords_list_lon(count(cell_mask)))
        counter = 1
        do i = 1,nlat_surface
          do j = 1,nlon_surface
            if (cell_mask(i,j)) then
              cell_coords_list_lat(counter) = i
              cell_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        lakes(lake_number)%lake_input_pointer => &
          lakeinput(lake_number, &
                    lake_pixel_coords_list_lat, &
                    lake_pixel_coords_list_lon, &
                    potential_lake_pixel_coords_list_lat, &
                    potential_lake_pixel_coords_list_lon, &
                    cell_coords_list_lat, &
                    cell_coords_list_lon)
      end do
      prognostics => setup_lake_for_fraction_calculation(lakes, &
                                                         cell_pixel_counts, &
                                                         non_lake_mask, &
                                                         binary_lake_mask, &
                                                         lake_pixel_mask, &
                                                         corresponding_surface_cell_lat_index, &
                                                         corresponding_surface_cell_lon_index, &
                                                         nlat_lake,nlon_lake, &
                                                         nlat_surface,nlon_surface)
      allocate(lake_pixel_counts_field(nlat_surface,nlon_surface))
      lake_pixel_counts_field(:,:) = 0
      do lake_number = 1,2
        mask = (lake_pixel_mask == lake_number)
        npixels = count(mask)
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        do i = 1,size(potential_lake_pixel_coords_list_lat)
          call add_pixel_by_coords(potential_lake_pixel_coords_list_lat(i), &
                                   potential_lake_pixel_coords_list_lon(i), &
                                   lake_pixel_counts_field,prognostics)
        end do
        deallocate(potential_lake_pixel_coords_list_lat)
        deallocate(potential_lake_pixel_coords_list_lon)
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(cell_mask)
      deallocate(mask)
      call clean_lake_fraction_calculation_prognostics(prognostics)
      deallocate(prognostics)
end subroutine testLakeFractionCalculationTest14

subroutine testLakeFractionCalculationTest15
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   type(lakefractioncalculationprognostics), pointer :: prognostics
   integer :: lake_number
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   integer,dimension(:,:), pointer :: expected_intermediate_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Multiple lakes
      nlat_lake = 15
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 2, 2, 2, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  2, 2, 2, 2, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 2, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 2, 2, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         1, 1, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(lake_pixel_mask)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         8,  8,  8,  8,  8, &
         16, 16, 16, 16, 16, &
         4,  4,  4,  4,  4, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(binary_lake_mask(nlat_surface,nlon_surface))
      binary_lake_mask = transpose(reshape((/   &
          .False., .False., .False., .False.,  .True., &
         .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., &
         .True., .False., .False., .False., .False., &
         .False.,  .False., .False., .False., .False. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          0, 0, 0, 0, 10, &
         0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, &
         4, 0, 0, 0, 0, &
         6, 0, 0, 0, 0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_intermediate_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_intermediate_lake_pixel_counts_field = transpose(reshape((/   &
          0, 0, 0, 0, 3, &
         0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, &
         4, 0, 0, 0, 0, &
         3, 0, 0, 0, 0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.0, 0.0, 0.0, 0.0, 0.625, &
         0.0, 0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 0.0, &
         1.0, 0.0, 0.0, 0.0, 0.0, &
         0.375, 0.0, 0.0, 0.0, 0.0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      allocate(cell_mask(nlat_surface,nlon_surface))
      do lake_number = 1,2
        mask = (lake_pixel_mask == lake_number)
        allocate(lake_pixel_coords_list_lat(0))
        allocate(lake_pixel_coords_list_lon(0))
        npixels = count(mask)
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        cell_mask(:,:) = .false.
        do i=1,size(potential_lake_pixel_coords_list_lat)
            lat = corresponding_surface_cell_lat_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            lon = corresponding_surface_cell_lon_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            cell_mask(lat,lon) = .true.
        end do
        allocate(cell_coords_list_lat(count(cell_mask)))
        allocate(cell_coords_list_lon(count(cell_mask)))
        counter = 1
        do i = 1,nlat_surface
          do j = 1,nlon_surface
            if (cell_mask(i,j)) then
              cell_coords_list_lat(counter) = i
              cell_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        lakes(lake_number)%lake_input_pointer => &
          lakeinput(lake_number, &
                    lake_pixel_coords_list_lat, &
                    lake_pixel_coords_list_lon, &
                    potential_lake_pixel_coords_list_lat, &
                    potential_lake_pixel_coords_list_lon, &
                    cell_coords_list_lat, &
                    cell_coords_list_lon)
      end do
      prognostics => setup_lake_for_fraction_calculation(lakes, &
                                                         cell_pixel_counts, &
                                                         non_lake_mask, &
                                                         binary_lake_mask, &
                                                         lake_pixel_mask, &
                                                         corresponding_surface_cell_lat_index, &
                                                         corresponding_surface_cell_lon_index, &
                                                         nlat_lake,nlon_lake, &
                                                         nlat_surface,nlon_surface)
      allocate(lake_pixel_counts_field(nlat_surface,nlon_surface))
      lake_pixel_counts_field(:,:) = 0
      do lake_number = 1,2
        mask = (lake_pixel_mask == lake_number)
        npixels = count(mask)
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        do i = 1,size(potential_lake_pixel_coords_list_lat)
          call add_pixel_by_coords(potential_lake_pixel_coords_list_lat(i), &
                                   potential_lake_pixel_coords_list_lon(i), &
                                   lake_pixel_counts_field,prognostics)
        end do
        deallocate(potential_lake_pixel_coords_list_lat)
        deallocate(potential_lake_pixel_coords_list_lon)
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field, &
                         nlat_surface,nlon_surface)
      call remove_pixel_by_coords(11,2,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(11,4,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(11,3,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(4,20,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(4,19,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(4,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,19,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,20,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(4,17,lake_pixel_counts_field, &
                                  prognostics)
      call assert_equals(lake_pixel_counts_field, &
                         expected_intermediate_lake_pixel_counts_field, &
                         nlat_surface,nlon_surface)
      call add_pixel_by_coords(11,2,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(11,4,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(11,3,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(4,20,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(4,19,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(4,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,19,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,20,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(4,17,lake_pixel_counts_field, &
                               prognostics)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,&
                         nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_intermediate_lake_pixel_counts_field)
      deallocate(cell_mask)
      deallocate(mask)
      call clean_lake_fraction_calculation_prognostics(prognostics)
      deallocate(prognostics)
end subroutine testLakeFractionCalculationTest15

subroutine testLakeFractionCalculationTest16
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   type(lakefractioncalculationprognostics), pointer :: prognostics
   integer :: lake_number
   integer,dimension(:,:), pointer :: potential_lake_pixel_mask
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Partially filled single lake
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      lake_number = 1
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(potential_lake_pixel_mask(nlat_lake,nlon_lake))
      potential_lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(potential_lake_pixel_mask)))
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 0, 0, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(binary_lake_mask(nlat_surface,nlon_surface))
      binary_lake_mask = transpose(reshape((/   &
          .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .True. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, &
         0, 0, 0, 0, 1, &
         0, 0, 0, 0, 0, &
         0, 0, 0, 4, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.0, 0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 0.3125, &
         0.0, 0.0, 0.0, 0.0, 1.0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      mask = (potential_lake_pixel_mask == 1)
      allocate(lake_pixel_coords_list_lat(0))
      allocate(lake_pixel_coords_list_lon(0))
      npixels = count(mask)
      allocate(potential_lake_pixel_coords_list_lat(npixels))
      allocate(potential_lake_pixel_coords_list_lon(npixels))
      counter = 1
      do i = 1,nlat_lake
        do j = 1,nlon_lake
          if (mask(i,j)) then
            potential_lake_pixel_coords_list_lat(counter) = i
            potential_lake_pixel_coords_list_lon(counter) = j
            counter = counter + 1
          end if
        end do
      end do
      allocate(cell_mask(nlat_surface,nlon_surface))
      cell_mask(:,:) = .false.
      do i=1,size(potential_lake_pixel_coords_list_lat)
          lat = corresponding_surface_cell_lat_index( &
            potential_lake_pixel_coords_list_lat(i), &
            potential_lake_pixel_coords_list_lon(i))
          lon = corresponding_surface_cell_lon_index( &
            potential_lake_pixel_coords_list_lat(i), &
            potential_lake_pixel_coords_list_lon(i))
          cell_mask(lat,lon) = .true.
      end do
      allocate(cell_coords_list_lat(count(cell_mask)))
      allocate(cell_coords_list_lon(count(cell_mask)))
      counter = 1
      do i = 1,nlat_surface
        do j = 1,nlon_surface
          if (cell_mask(i,j)) then
            cell_coords_list_lat(counter) = i
            cell_coords_list_lon(counter) = j
            counter = counter + 1
          end if
        end do
      end do
      lakes(1)%lake_input_pointer => lakeinput(lake_number, &
                                               lake_pixel_coords_list_lat, &
                                               lake_pixel_coords_list_lon, &
                                               potential_lake_pixel_coords_list_lat, &
                                               potential_lake_pixel_coords_list_lon, &
                                               cell_coords_list_lat, &
                                               cell_coords_list_lon)
      prognostics => setup_lake_for_fraction_calculation(lakes, &
                                                         cell_pixel_counts, &
                                                         non_lake_mask, &
                                                         binary_lake_mask, &
                                                         potential_lake_pixel_mask, &
                                                         corresponding_surface_cell_lat_index, &
                                                         corresponding_surface_cell_lon_index, &
                                                         nlat_lake,nlon_lake, &
                                                         nlat_surface,nlon_surface)
      allocate(lake_pixel_counts_field(nlat_surface,nlon_surface))
      lake_pixel_counts_field(:,:) = 0
      do i = 1,nlat_lake
        do j = 1,nlon_lake
          if (lake_pixel_mask(i,j) == 1) then
            call add_pixel_by_coords(i,j,lake_pixel_counts_field,prognostics)
          end if
        end do
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,&
                         nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(cell_mask)
      deallocate(mask)
      call clean_lake_fraction_calculation_prognostics(prognostics)
      deallocate(prognostics)
end subroutine testLakeFractionCalculationTest16

subroutine testLakeFractionCalculationTest17
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   type(lakefractioncalculationprognostics), pointer :: prognostics
   integer :: lake_number
   integer,dimension(:,:), pointer :: potential_lake_pixel_mask
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Multiple partially filled lakes
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(potential_lake_pixel_mask(nlat_lake,nlon_lake))
      potential_lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 2, 2, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 7, 0, &
         0, 2, 2, 2,  2, 0, 0, 0,  0, 0, 3, 3,  0, 0, 0, 0,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 6,  6, 6, 6, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 4, 4, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 4, 4, 4,  4, 4, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 4, 4, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 5, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 5, 5, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 5, 0, 0,  0, 0, 0, 0,  0, 0, 1, 0,  0, 1, 1, 0, &
         0, 0, 0, 5,  5, 5, 0, 0,  0, 0, 0, 0,  0, 0, 1, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  0, 5, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(potential_lake_pixel_mask)))
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 2, 2, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 7, 0, &
         0, 2, 2, 2,  2, 0, 0, 0,  0, 0, 3, 3,  0, 0, 0, 0,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 6,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  6, 6, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 4, 4, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 4, 4, 0,  4, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 4, 4, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  1, 1, 1, 1, &
         0, 0, 0, 0,  0, 5, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 5, 5, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 5,  5, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  1, 1, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 1, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(binary_lake_mask(nlat_surface,nlon_surface))
      binary_lake_mask = transpose(reshape((/   &
          .False., .False., .False., .False., .False., &
         .False., .False., .False., .False.,  .True., &
         .False., .False., .False., .False., .False., &
         .False., .False., .False., .False.,  .True., &
         .False., .False., .False., .False., .False. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          5, 1, 2, 0, 1, &
         0, 0, 0, 0, 9, &
         0, 6, 1, 0, 0, &
         0, 1, 0, 0, 16, &
         1, 3, 0, 0, 2 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.3125, 0.0625, 0.125, 0.0, 0.0625, &
         0.0, 0.0, 0.0, 0.0, 0.5625, &
         0.0, 0.375, 0.0625, 0.0, 0.0, &
         0.0, 0.0625, 0.0, 0.0, 1.0, &
         0.0625, 0.1875, 0.0, 0.0, 0.125 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      allocate(cell_mask(nlat_surface,nlon_surface))
      do lake_number = 1,7
        mask = (potential_lake_pixel_mask == lake_number)
        allocate(lake_pixel_coords_list_lat(0))
        allocate(lake_pixel_coords_list_lon(0))
        npixels = count(mask)
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        cell_mask(:,:) = .false.
        do i=1,size(potential_lake_pixel_coords_list_lat)
            lat = corresponding_surface_cell_lat_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            lon = corresponding_surface_cell_lon_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            cell_mask(lat,lon) = .true.
        end do
        allocate(cell_coords_list_lat(count(cell_mask)))
        allocate(cell_coords_list_lon(count(cell_mask)))
        counter = 1
        do i = 1,nlat_surface
          do j = 1,nlon_surface
            if (cell_mask(i,j)) then
              cell_coords_list_lat(counter) = i
              cell_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        lakes(lake_number)%lake_input_pointer => &
          lakeinput(lake_number, &
                    lake_pixel_coords_list_lat, &
                    lake_pixel_coords_list_lon, &
                    potential_lake_pixel_coords_list_lat, &
                    potential_lake_pixel_coords_list_lon, &
                    cell_coords_list_lat, &
                    cell_coords_list_lon)
      end do
      prognostics => setup_lake_for_fraction_calculation(lakes, &
                                                         cell_pixel_counts, &
                                                         non_lake_mask, &
                                                         binary_lake_mask, &
                                                         potential_lake_pixel_mask, &
                                                         corresponding_surface_cell_lat_index, &
                                                         corresponding_surface_cell_lon_index, &
                                                         nlat_lake,nlon_lake, &
                                                         nlat_surface,nlon_surface)
      allocate(lake_pixel_counts_field(nlat_surface,nlon_surface))
      lake_pixel_counts_field(:,:) = 0
      do lake_number = 1,7
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (lake_pixel_mask(i,j) == lake_number) then
              call add_pixel_by_coords(i,j,lake_pixel_counts_field,prognostics)
            end if
          end do
        end do
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field, &
                         nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(cell_mask)
      deallocate(mask)
      call clean_lake_fraction_calculation_prognostics(prognostics)
      deallocate(prognostics)
end subroutine testLakeFractionCalculationTest17

subroutine testLakeFractionCalculationTest18
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   type(lakefractioncalculationprognostics), pointer :: prognostics
   integer :: lake_number
   integer,dimension(:,:), pointer :: potential_lake_pixel_mask
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Multiple partially filled lakes
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(potential_lake_pixel_mask(nlat_lake,nlon_lake))
      potential_lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 1, 0,  1, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 0,  0, 0, 7, 0, &
         0, 0, 1, 0,  1, 1, 0, 1,  1, 1, 1, 0,  0, 0, 8, 8,  0, 7, 7, 0, &
         0, 0, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0,  0, 0, 8, 8,  0, 7, 7, 0, &
         0, 0, 1, 1,  1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 8,  0, 7, 7, 0, &
         0, 0, 1, 1,  1, 0, 1, 1,  1, 1, 1, 0,  0, 0, 0, 0,  0, 7, 7, 0, &
         0, 0, 1, 1,  1, 2, 1, 1,  1, 1, 0, 0,  0, 0, 8, 8,  8, 8, 0, 9, &
         0, 0, 1, 0,  1, 2, 1, 1,  0, 0, 0, 0,  0, 0, 8, 8,  8, 8, 9, 9, &
         0, 0, 1, 0,  2, 2, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 9, 9, &
         0, 0, 1, 1,  2, 2, 1, 1,  0, 0, 6, 0,  0, 0, 0, 9,  9, 9, 9, 9, &
         0, 0, 0, 0,  2, 2, 2, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 9, 9, 0, &
         0, 0, 0, 2,  2, 2, 0, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 5, 5, 0, &
         0, 0, 2, 2,  2, 2, 0, 0,  0, 0, 0, 0,  0, 3, 3, 5,  0, 5, 0, 0, &
         0, 0, 2, 2,  2, 2, 0, 0,  0, 0, 6, 0,  4, 3, 3, 5,  5, 5, 0, 0, &
         0, 0, 0, 0,  2, 0, 0, 0,  0, 0, 0, 0,  4, 4, 0, 5,  5, 0, 0, 0, &
         0, 0, 0, 0,  2, 0, 0, 0,  0, 0, 0, 4,  4, 4, 0, 5,  5, 0, 0, 0, &
         0, 0, 0, 0,  2, 2, 2, 0,  0, 0, 4, 4,  4, 4, 0, 0,  5, 5, 0, 0, &
         0, 0, 6, 0,  0, 0, 0, 0,  0, 0, 0, 4,  4, 0, 0, 0,  0, 5, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(potential_lake_pixel_mask)))
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  1, 1, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 7, 0, &
         0, 0, 0, 0,  1, 1, 0, 0,  0, 0, 0, 0,  0, 0, 8, 8,  0, 0, 7, 0, &
         0, 0, 0, 0,  1, 1, 1, 1,  0, 1, 0, 0,  0, 0, 8, 8,  0, 7, 0, 0, &
         0, 0, 0, 0,  0, 0, 1, 0,  0, 0, 0, 0,  0, 0, 0, 8,  0, 7, 7, 0, &
         0, 0, 0, 0,  0, 0, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 7, 7, 0, &
         0, 0, 1, 0,  0, 0, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  8, 8, 0, 9, &
         0, 0, 1, 0,  0, 0, 1, 1,  0, 0, 0, 0,  0, 0, 8, 0,  8, 8, 9, 9, &
         0, 0, 0, 0,  2, 2, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 9, 9, &
         0, 0, 0, 0,  2, 2, 1, 1,  0, 0, 6, 0,  0, 0, 0, 9,  9, 0, 0, 0, &
         0, 0, 0, 0,  2, 2, 2, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 9, 9, 0, &
         0, 0, 0, 2,  2, 2, 0, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 5, 0, &
         0, 0, 2, 2,  0, 0, 0, 0,  0, 0, 0, 0,  0, 3, 3, 5,  0, 0, 0, 0, &
         0, 0, 2, 2,  0, 0, 0, 0,  0, 0, 6, 0,  4, 3, 3, 5,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  4, 4, 0, 5,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 4,  4, 4, 0, 5,  0, 0, 0, 0, &
         0, 0, 0, 0,  2, 2, 2, 0,  0, 0, 4, 0,  4, 0, 0, 0,  5, 0, 0, 0, &
         0, 0, 6, 0,  0, 0, 0, 0,  0, 0, 0, 0,  4, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(binary_lake_mask(nlat_surface,nlon_surface))
      binary_lake_mask = transpose(reshape((/   &
          .False.,  .True., .False.,  .True., .False., &
         .False., .False., .False., .False., .False., &
         .False.,  .True., .False., .False.,  .True., &
         .False., .False., .False.,  .True., .False., &
         .False., .False., .False., .False., .False. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          0, 16, 0, 10, 3, &
         0, 0,  0,  0, 4, &
         1, 16, 1,  0, 11, &
         4, 0,  1, 16, 0, &
         1, 3,  0,  1, 0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.0, 1.0, 0.0, 0.625, 0.1875, &
         0.0, 0.375, 0.0, 0.0, 0.25, &
         0.0, 0.6875, 0.0625, 0.0, 0.625, &
         0.25, 0.0, 0.0625, 1.0, 0.0, &
         0.0625, 0.1875, 0.0, 0.0625, 0.0625 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      allocate(cell_mask(nlat_surface,nlon_surface))
      do lake_number = 1,9
        mask = (potential_lake_pixel_mask == lake_number)
        allocate(lake_pixel_coords_list_lat(0))
        allocate(lake_pixel_coords_list_lon(0))
        npixels = count(mask)
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        cell_mask(:,:) = .false.
        do i=1,size(potential_lake_pixel_coords_list_lat)
            lat = corresponding_surface_cell_lat_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            lon = corresponding_surface_cell_lon_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            cell_mask(lat,lon) = .true.
        end do
        allocate(cell_coords_list_lat(count(cell_mask)))
        allocate(cell_coords_list_lon(count(cell_mask)))
        counter = 1
        do i = 1,nlat_surface
          do j = 1,nlon_surface
            if (cell_mask(i,j)) then
              cell_coords_list_lat(counter) = i
              cell_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        lakes(lake_number)%lake_input_pointer => &
          lakeinput(lake_number, &
                    lake_pixel_coords_list_lat, &
                    lake_pixel_coords_list_lon, &
                    potential_lake_pixel_coords_list_lat, &
                    potential_lake_pixel_coords_list_lon, &
                    cell_coords_list_lat, &
                    cell_coords_list_lon)
      end do
      prognostics => setup_lake_for_fraction_calculation(lakes, &
                                                         cell_pixel_counts, &
                                                         non_lake_mask, &
                                                         binary_lake_mask, &
                                                         potential_lake_pixel_mask, &
                                                         corresponding_surface_cell_lat_index, &
                                                         corresponding_surface_cell_lon_index, &
                                                         nlat_lake,nlon_lake, &
                                                         nlat_surface,nlon_surface)
      allocate(lake_pixel_counts_field(nlat_surface,nlon_surface))
      lake_pixel_counts_field(:,:) = 0
      do lake_number = 1,9
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (lake_pixel_mask(i,j) == lake_number) then
              call add_pixel_by_coords(i,j,lake_pixel_counts_field,prognostics)
            end if
          end do
        end do
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field, &
                         nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(cell_mask)
      deallocate(mask)
      call clean_lake_fraction_calculation_prognostics(prognostics)
      deallocate(prognostics)
end subroutine testLakeFractionCalculationTest18

subroutine testLakeFractionCalculationTest19
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   type(lakefractioncalculationprognostics), pointer :: prognostics
   integer :: lake_number
   integer,dimension(:,:), pointer :: potential_lake_pixel_mask
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   integer,dimension(:,:), pointer :: expected_intermediate_lake_pixel_counts_field
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field_after_cycle
   integer,dimension(:,:), pointer :: expected_intermediate_lake_pixel_counts_field_two
   integer,dimension(:,:), pointer :: expected_intermediate_lake_pixel_counts_field_three
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field_after_second_cycle
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field_after_third_cycle
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Multiple partially filled lakes
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(potential_lake_pixel_mask(nlat_lake,nlon_lake))
      potential_lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 1,  0, 1, 1, 1,  1, 1, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 1, 1,  0, 1, 1, 1,  1, 0, 1, 1,  1, 0, 1, 0,  0, 1, 1, 0, &
         0, 1, 1, 1,  0, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  1, 0, 1, 1,  1, 0, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 0, 1,  0, 1, 1, 0,  0, 0, 1, 1, &
         0, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 1,  1, 0, 1, 1,  1, 1, 1, 0, &
         0, 0, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 0, &
         0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1,  1, 0, 1, 1,  0, 0, 0, 0, &
         0, 2, 2, 2,  2, 0, 1, 1,  1, 0, 1, 1,  1, 1, 0, 1,  1, 0, 0, 0, &
         0, 2, 2, 0,  1, 1, 1, 1,  1, 1, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  0, 1, 1, 1,  0, 1, 1, 0,  0, 1, 1, 1,  0, 0, 0, 0, &
         0, 0, 0, 1,  0, 1, 0, 1,  0, 0, 1, 0,  1, 1, 1, 0,  0, 1, 0, 0, &
         1, 1, 0, 1,  1, 1, 1, 1,  1, 0, 1, 0,  1, 1, 1, 0,  1, 1, 1, 0, &
         0, 1, 1, 0,  1, 1, 1, 1,  1, 1, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0, &
         0, 1, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0,  0, 1, 0, 1,  0, 0, 0, 0, &
         0, 0, 0, 0,  1, 1, 0, 1,  1, 1, 0, 1,  1, 1, 0, 0,  1, 1, 0, 0, &
         0, 0, 1, 1,  0, 1, 1, 1,  1, 1, 0, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 1, 1, 1,  1, 0, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(potential_lake_pixel_mask)))
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 1,  0, 1, 1, 1,  1, 1, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 1, 1,  0, 1, 1, 1,  1, 0, 1, 1,  1, 0, 1, 0,  0, 1, 1, 0, &
         0, 1, 1, 1,  0, 1, 1, 1,  0, 1, 0, 1,  1, 1, 1, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  1, 0, 1, 1,  1, 0, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 0, 1,  0, 1, 1, 0,  0, 0, 1, 1, &
         0, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 1,  1, 0, 1, 1,  1, 1, 1, 0, &
         0, 0, 1, 1,  0, 0, 1, 0,  0, 0, 1, 1,  1, 1, 1, 1,  1, 1, 1, 0, &
         0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1,  1, 0, 1, 1,  0, 0, 0, 0, &
         0, 2, 2, 2,  0, 0, 1, 1,  1, 0, 1, 1,  0, 1, 0, 1,  1, 0, 0, 0, &
         0, 2, 0, 0,  1, 1, 1, 1,  1, 1, 1, 0,  1, 0, 0, 1,  1, 1, 0, 0, &
         0, 0, 0, 0,  0, 1, 1, 1,  0, 1, 1, 0,  0, 1, 1, 1,  0, 0, 0, 0, &
         0, 0, 0, 1,  0, 0, 0, 1,  0, 0, 1, 0,  1, 1, 1, 0,  0, 0, 0, 0, &
         1, 1, 0, 1,  0, 0, 0, 0,  1, 0, 1, 0,  1, 1, 1, 0,  0, 0, 1, 0, &
         0, 1, 1, 0,  0, 0, 1, 0,  1, 1, 1, 0,  1, 1, 1, 1,  0, 0, 0, 0, &
         0, 1, 1, 0,  1, 0, 1, 1,  1, 1, 0, 0,  0, 1, 0, 1,  0, 0, 0, 0, &
         0, 0, 0, 0,  1, 0, 0, 1,  1, 1, 0, 1,  1, 1, 0, 0,  1, 1, 0, 0, &
         0, 0, 1, 1,  0, 1, 1, 1,  1, 1, 0, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
         0, 0, 0, 0,  0, 1, 1, 1,  1, 0, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(binary_lake_mask(nlat_surface,nlon_surface))
      binary_lake_mask = transpose(reshape((/   &
          .False.,  .True.,  .True., .False., .False., &
         .False., .False., .False.,  .True.,  .True., &
         .False.,  .True., .True.,  .True.,  .False., &
         .True., .False., .False.,  .True., .False., &
         .False.,  .True.,  .True., .False., .False. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          0, 16,  16,  0,  0, &
         0, 0,  0,  16, 16, &
         3, 16, 16, 16, 0, &
         16, 0,  0,  16, 0, &
         2, 16, 16, 1,  4 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_intermediate_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_intermediate_lake_pixel_counts_field = transpose(reshape((/   &
          0, 11,  15,  0,  0, &
         0, 0,  0,  14, 15, &
         0, 12, 14, 15, 0, &
         10, 0,  0,  14, 0, &
         0, 10, 12, 0,  0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_pixel_counts_field_after_cycle(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field_after_cycle = transpose(reshape((/   &
          0, 16,  16,  0,  0, &
         0, 2,  3,  16, 16, &
         1, 16, 16, 16, 2, &
         16, 0,  0,  16, 1, &
         0, 16, 16, 1,  0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_intermediate_lake_pixel_counts_field_two(nlat_surface,nlon_surface))
      expected_intermediate_lake_pixel_counts_field_two = transpose(reshape((/   &
          0, 16,  14,  0,  0, &
         0, 0,  0,  13, 16, &
         0, 10, 12, 16, 0, &
         11, 0,  0,  13, 0, &
         0, 10, 11, 0,  0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_intermediate_lake_pixel_counts_field_three(nlat_surface,nlon_surface))
      expected_intermediate_lake_pixel_counts_field_three = transpose(reshape((/   &
          0, 14,  16,  0,  0, &
         0, 0,  0,  14, 16, &
         0, 10, 11, 16, 0, &
         11, 0,  0,  15, 0, &
         0, 10, 9, 0,  0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_pixel_counts_field_after_second_cycle(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field_after_second_cycle = transpose(reshape((/   &
          0, 16,  16,  0,  0, &
         1, 0,  3,  16, 16, &
         1, 16, 16, 16, 1, &
         16, 0,  1,  16, 0, &
         0, 16, 16, 3,  0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_pixel_counts_field_after_third_cycle(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field_after_third_cycle = transpose(reshape((/   &
          0, 16,  16,  0,  0, &
         0, 0,  2,  16, 16, &
         1, 16, 16, 16, 0, &
         16, 0,  2,  16, 1, &
         2, 16, 16, 1,  1 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.0, 1.0, 1.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 1.0, 1.0, &
         0.25, 0.9375, 1.0, 1.0, 0.0, &
         1.0, 0.0, 0.4375, 1.0, 0.0, &
         0.0, 1.0, 1.0, 0.0, 0.0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      allocate(cell_mask(nlat_surface,nlon_surface))
      do lake_number = 1,2
        mask = (potential_lake_pixel_mask == lake_number)
        allocate(lake_pixel_coords_list_lat(0))
        allocate(lake_pixel_coords_list_lon(0))
        npixels = count(mask)
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        cell_mask(:,:) = .false.
        do i=1,size(potential_lake_pixel_coords_list_lat)
            lat = corresponding_surface_cell_lat_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            lon = corresponding_surface_cell_lon_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            cell_mask(lat,lon) = .true.
        end do
        allocate(cell_coords_list_lat(count(cell_mask)))
        allocate(cell_coords_list_lon(count(cell_mask)))
        counter = 1
        do i = 1,nlat_surface
          do j = 1,nlon_surface
            if (cell_mask(i,j)) then
              cell_coords_list_lat(counter) = i
              cell_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        lakes(lake_number)%lake_input_pointer => &
          lakeinput(lake_number, &
                    lake_pixel_coords_list_lat, &
                    lake_pixel_coords_list_lon, &
                    potential_lake_pixel_coords_list_lat, &
                    potential_lake_pixel_coords_list_lon, &
                    cell_coords_list_lat, &
                    cell_coords_list_lon)
      end do
      prognostics => setup_lake_for_fraction_calculation(lakes, &
                                                         cell_pixel_counts, &
                                                         non_lake_mask, &
                                                         binary_lake_mask, &
                                                         potential_lake_pixel_mask, &
                                                         corresponding_surface_cell_lat_index, &
                                                         corresponding_surface_cell_lon_index, &
                                                         nlat_lake,nlon_lake, &
                                                         nlat_surface,nlon_surface)
      allocate(lake_pixel_counts_field(nlat_surface,nlon_surface))
      lake_pixel_counts_field(:,:) = 0
      do lake_number = 1,2
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (lake_pixel_mask(i,j) == lake_number) then
              call add_pixel_by_coords(i,j,lake_pixel_counts_field,prognostics)
            end if
          end do
        end do
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,&
                         nlat_surface,nlon_surface)
      call remove_pixel_by_coords(2,4,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(2,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(4,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(5,7,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(4,8,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,11,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,12,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,13,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,19,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(10,4,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(10,3,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(10,2,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(11,2,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(15,3,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(16,3,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(17,17,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(19,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(14,19,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(13,13,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(13,11,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(14,11,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(15,11,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(15,10,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(16,10,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(17,9,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(17,8,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,8,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,9,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,10,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(19,8,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(20,7,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(19,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(11,9,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(10,8,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(9,7,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(9,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(9,12,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(8,12,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(7,13,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(6,12,lake_pixel_counts_field, &
                                  prognostics)
      call assert_equals(lake_pixel_counts_field, &
                         expected_intermediate_lake_pixel_counts_field, &
                         nlat_surface,nlon_surface)
      call add_pixel_by_coords(2,4,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(2,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(4,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(5,7,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(4,8,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,11,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,12,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,13,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,19,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(10,4,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(10,3,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(10,2,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(11,2,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(15,3,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(16,3,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(17,17,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(19,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(14,19,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(13,13,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(13,11,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(14,11,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(15,11,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(15,10,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(16,10,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(17,9,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(17,8,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,8,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,9,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,10,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(19,8,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(20,7,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(19,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(11,9,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(10,8,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(9,7,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(9,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(9,12,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(8,12,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(7,13,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(6,12,lake_pixel_counts_field, &
                               prognostics)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field_after_cycle, &
                         nlat_surface,nlon_surface)
      call remove_pixel_by_coords(2,4,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(2,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(4,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(5,7,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(4,8,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,11,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,12,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,13,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,19,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(10,4,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(10,3,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(10,2,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(11,2,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(15,3,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(16,3,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(17,17,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(19,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(14,19,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(13,13,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(13,11,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(14,11,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(15,11,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(15,10,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(16,10,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(17,9,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(17,8,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,8,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,9,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,10,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(19,8,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(20,7,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(19,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(11,9,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(10,8,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(9,7,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(9,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(9,12,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(8,12,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(7,13,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(6,12,lake_pixel_counts_field, &
                                  prognostics)
      call assert_equals(lake_pixel_counts_field, &
                         expected_intermediate_lake_pixel_counts_field_two, &
                         nlat_surface,nlon_surface)
      call add_pixel_by_coords(2,4,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(2,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(4,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(5,7,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(4,8,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,11,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,12,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,13,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,19,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(10,4,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(10,3,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(10,2,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(11,2,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(15,3,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(16,3,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(17,17,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(19,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(14,19,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(13,13,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(13,11,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(14,11,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(15,11,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(15,10,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(16,10,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(17,9,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(17,8,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,8,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,9,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,10,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(19,8,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(20,7,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(19,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(11,9,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(10,8,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(9,7,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(9,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(9,12,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(8,12,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(7,13,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(6,12,lake_pixel_counts_field, &
                               prognostics)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field_after_second_cycle, &
                         nlat_surface,nlon_surface)
      call remove_pixel_by_coords(2,4,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(2,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(4,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(5,7,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(4,8,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,11,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,12,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,13,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,19,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(10,4,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(10,3,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(10,2,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(11,2,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(15,3,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(16,3,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(17,17,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(19,18,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(14,19,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(13,13,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(13,11,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(14,11,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(15,11,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(15,10,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(16,10,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(17,9,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(17,8,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,8,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,9,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,10,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(19,8,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(20,7,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(19,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(18,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(11,9,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(10,8,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(9,7,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(9,6,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(9,12,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(8,12,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(7,13,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(6,12,lake_pixel_counts_field, &
                                  prognostics)
      call assert_equals(lake_pixel_counts_field, &
                         expected_intermediate_lake_pixel_counts_field_three, &
                         nlat_surface,nlon_surface)
      call add_pixel_by_coords(2,4,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(2,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(4,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(5,7,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(4,8,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,11,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,12,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,13,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,19,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(10,4,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(10,3,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(10,2,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(11,2,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(15,3,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(16,3,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(17,17,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(19,18,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(14,19,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(13,13,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(13,11,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(14,11,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(15,11,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(15,10,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(16,10,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(17,9,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(17,8,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,8,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,9,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,10,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(19,8,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(20,7,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(19,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(18,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(11,9,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(10,8,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(9,7,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(9,6,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(9,12,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(8,12,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(7,13,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(6,12,lake_pixel_counts_field, &
                               prognostics)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field_after_third_cycle, &
                         nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_lake_pixel_counts_field_after_cycle)
      deallocate(expected_lake_pixel_counts_field_after_second_cycle)
      deallocate(expected_lake_pixel_counts_field_after_third_cycle)
      deallocate(expected_intermediate_lake_pixel_counts_field)
      deallocate(expected_intermediate_lake_pixel_counts_field_two)
      deallocate(expected_intermediate_lake_pixel_counts_field_three)
      deallocate(cell_mask)
      deallocate(mask)
      call clean_lake_fraction_calculation_prognostics(prognostics)
      deallocate(prognostics)
end subroutine testLakeFractionCalculationTest19

subroutine testLakeFractionCalculationTest20
   use calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   type(lakefractioncalculationprognostics), pointer :: prognostics
   integer :: lake_number
   integer,dimension(:,:), pointer :: potential_lake_pixel_mask
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
   integer,dimension(:), pointer :: lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
   integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
   integer,dimension(:), pointer :: cell_coords_list_lat
   integer,dimension(:), pointer :: cell_coords_list_lon
   logical,dimension(:,:), pointer :: non_lake_mask
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   integer,dimension(:,:), pointer :: expected_intermediate_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Multiple partially filled lakes with uneven cell sizes
      nlat_lake = 15
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      allocate(non_lake_mask(nlat_surface,nlon_surface))
      non_lake_mask(:,:) = .False.
      allocate(potential_lake_pixel_mask(nlat_lake,nlon_lake))
      potential_lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 2, 2, 2, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  2, 2, 2, 2, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 2, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 2, 2, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         1, 1, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(lakes(maxval(potential_lake_pixel_mask)))
      allocate(lake_pixel_mask(nlat_lake,nlon_lake))
      lake_pixel_mask = transpose(reshape((/   &
          0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 2, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  2, 0, 0, 2, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 2, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 2, 2, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 1, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         1, 1, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
         0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(cell_pixel_counts(nlat_surface,nlon_surface))
      cell_pixel_counts = transpose(reshape((/   &
          16, 16, 16, 16, 16, &
         8,  8,  8,  8,  8, &
         16, 16, 16, 16, 16, &
         4,  4,  4,  4,  4, &
         16, 16, 16, 16, 16 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lat_index = transpose(reshape((/   &
          1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
         4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
         5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
      corresponding_surface_cell_lon_index = transpose(reshape((/   &
          1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
         1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
         (/nlon_lake,nlat_lake/)))
      allocate(binary_lake_mask(nlat_surface,nlon_surface))
      binary_lake_mask = transpose(reshape((/   &
          .False., .False., .False., .False., .False., &
         .False., .False., .False., .False.,  .True., &
         .False., .False., .False., .False., .False., &
         .False., .False., .False., .False., .False., &
         .True., .False., .False., .False., .False. /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field = transpose(reshape((/   &
          0, 0, 0, 0, 0, &
         0, 0, 0, 0, 6, &
         0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, &
         8, 0, 0, 0, 0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_intermediate_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_intermediate_lake_pixel_counts_field = transpose(reshape((/   &
          0, 0, 0, 0, 0, &
         0, 0, 0, 0, 2, &
         0, 0, 0, 0, 0, &
         0, 0, 0, 0, 0, &
         2, 0, 0, 0, 0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
      expected_lake_fractions_field = transpose(reshape((/   &
          0.0, 0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 0.75, &
         0.0, 0.0, 0.0, 0.0, 0.0, &
         0.0, 0.0, 0.0, 0.0, 0.0, &
         0.5, 0.0, 0.0, 0.0, 0.0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(mask(nlat_lake,nlon_lake))
      allocate(cell_mask(nlat_surface,nlon_surface))
      do lake_number = 1,2
        mask = (potential_lake_pixel_mask == lake_number)
        allocate(lake_pixel_coords_list_lat(0))
        allocate(lake_pixel_coords_list_lon(0))
        npixels = count(mask)
        allocate(potential_lake_pixel_coords_list_lat(npixels))
        allocate(potential_lake_pixel_coords_list_lon(npixels))
        counter = 1
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (mask(i,j)) then
              potential_lake_pixel_coords_list_lat(counter) = i
              potential_lake_pixel_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        cell_mask(:,:) = .false.
        do i=1,size(potential_lake_pixel_coords_list_lat)
            lat = corresponding_surface_cell_lat_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            lon = corresponding_surface_cell_lon_index( &
              potential_lake_pixel_coords_list_lat(i), &
              potential_lake_pixel_coords_list_lon(i))
            cell_mask(lat,lon) = .true.
        end do
        allocate(cell_coords_list_lat(count(cell_mask)))
        allocate(cell_coords_list_lon(count(cell_mask)))
        counter = 1
        do i = 1,nlat_surface
          do j = 1,nlon_surface
            if (cell_mask(i,j)) then
              cell_coords_list_lat(counter) = i
              cell_coords_list_lon(counter) = j
              counter = counter + 1
            end if
          end do
        end do
        lakes(lake_number)%lake_input_pointer => &
          lakeinput(lake_number, &
                    lake_pixel_coords_list_lat, &
                    lake_pixel_coords_list_lon, &
                    potential_lake_pixel_coords_list_lat, &
                    potential_lake_pixel_coords_list_lon, &
                    cell_coords_list_lat, &
                    cell_coords_list_lon)
      end do
      prognostics => setup_lake_for_fraction_calculation(lakes, &
                                                         cell_pixel_counts, &
                                                         non_lake_mask, &
                                                         binary_lake_mask, &
                                                         potential_lake_pixel_mask, &
                                                         corresponding_surface_cell_lat_index, &
                                                         corresponding_surface_cell_lon_index, &
                                                         nlat_lake,nlon_lake, &
                                                         nlat_surface,nlon_surface)
      allocate(lake_pixel_counts_field(nlat_surface,nlon_surface))
      lake_pixel_counts_field(:,:) = 0
      do lake_number = 1,2
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (lake_pixel_mask(i,j) == lake_number) then
              call add_pixel_by_coords(i,j,lake_pixel_counts_field,prognostics)
            end if
          end do
        end do
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field, &
                         nlat_surface,nlon_surface)
      call remove_pixel_by_coords(12,1,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(12,2,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(12,3,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(13,3,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(13,2,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(13,1,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(6,19,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(6,20,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(4,20,lake_pixel_counts_field, &
                                  prognostics)
      call remove_pixel_by_coords(3,20,lake_pixel_counts_field, &
                                  prognostics)
      call assert_equals(lake_pixel_counts_field, &
                         expected_intermediate_lake_pixel_counts_field, &
                         nlat_surface,nlon_surface)
      call add_pixel_by_coords(12,1,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(12,2,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(12,3,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(13,3,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(13,2,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(13,1,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(6,19,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(6,20,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(4,20,lake_pixel_counts_field, &
                               prognostics)
      call add_pixel_by_coords(3,20,lake_pixel_counts_field, &
                               prognostics)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,&
                         nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(binary_lake_mask)
      deallocate(non_lake_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_intermediate_lake_pixel_counts_field)
      deallocate(cell_mask)
      deallocate(mask)
      call clean_lake_fraction_calculation_prognostics(prognostics)
      deallocate(prognostics)
end subroutine testLakeFractionCalculationTest20

subroutine testLakeFractionCalculationTest21
  use calculate_lake_fractions_mod
  type(lakeinputpointer), dimension(:), pointer :: lakes
  type(lakefractioncalculationprognostics), pointer :: prognostics
  integer :: lake_number
  integer,dimension(:,:), pointer :: potential_lake_pixel_mask
  integer,dimension(:,:), pointer :: lake_pixel_mask
  integer,dimension(:,:), pointer :: cell_pixel_counts
  integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
  integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
  integer,dimension(:,:), pointer :: lake_pixel_counts_field
  real(dp),dimension(:,:), pointer :: lake_fractions_field
  integer,dimension(:), pointer :: lake_pixel_coords_list_lat
  integer,dimension(:), pointer :: lake_pixel_coords_list_lon
  integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lat
  integer,dimension(:), pointer :: potential_lake_pixel_coords_list_lon
  integer,dimension(:), pointer :: cell_coords_list_lat
  integer,dimension(:), pointer :: cell_coords_list_lon
  logical,dimension(:,:), pointer :: non_lake_mask
  logical,dimension(:,:), pointer :: binary_lake_mask
  integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
  integer,dimension(:,:), pointer :: expected_intermediate_lake_pixel_counts_field
  real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
  logical,dimension(:,:), pointer :: cell_mask
  logical, dimension(:,:), pointer :: mask
  integer :: nlat_lake, nlon_lake
  integer :: nlat_surface, nlon_surface
  integer :: npixels, counter
  integer :: lat,lon
  integer :: i,j
    !Two lake example taken from lake tests 22
    nlat_lake = 20
    nlon_lake = 20
    nlat_surface = 3
    nlon_surface = 3
    allocate(non_lake_mask(nlat_surface,nlon_surface))
    non_lake_mask(:,:) = .False.
    allocate(potential_lake_pixel_mask(nlat_lake,nlon_lake))
    potential_lake_pixel_mask = transpose(reshape((/   &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, &
       0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, &
       0, 2, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, &
       0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, &
       0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, &
       0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, &
       0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, &
       0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, &
       0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, &
       0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, &
       0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, &
       0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, &
       0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, &
       0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 /), &
       (/nlon_lake,nlat_lake/)))
    allocate(lake_pixel_mask(nlat_lake,nlon_lake))
    lake_pixel_mask = transpose(reshape((/   &
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, &
       0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, &
       0, 2, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, &
       0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, &
       0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, &
       0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, &
       0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, &
       0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, &
       0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, &
       0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, &
       0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, &
       0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, &
       0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, &
       0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 /), &
       (/nlon_lake,nlat_lake/)))
    allocate(cell_pixel_counts(nlat_surface,nlon_surface))
    cell_pixel_counts = transpose(reshape((/   &
        36, 48, 36, &
       48, 64, 48, &
       36, 48, 36  /), &
       (/nlon_surface,nlat_surface/)))
    allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
    corresponding_surface_cell_lat_index = transpose(reshape((/   &
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, &
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, &
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, &
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, &
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, &
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, &
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, &
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, &
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, &
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, &
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, &
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, &
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, &
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, &
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, &
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, &
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, &
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, &
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, &
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 /), &
       (/nlon_lake,nlat_lake/)))
    allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
    corresponding_surface_cell_lon_index = transpose(reshape((/   &
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, &
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, &
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, &
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, &
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, &
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, &
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, &
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, &
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, &
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, &
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, &
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, &
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, &
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, &
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, &
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, &
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, &
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, &
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, &
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3 /), &
       (/nlon_lake,nlat_lake/)))
    allocate(binary_lake_mask(nlat_surface,nlon_surface))
    binary_lake_mask = transpose(reshape((/   &
        .False., .True., .False., &
       .False., .False., .False., &
       .False.,  .True., .False.  /), &
       (/nlon_surface,nlat_surface/)))
    allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
    expected_lake_pixel_counts_field = transpose(reshape((/   &
        2,  48, 1, &
       0,   19, 21, &
       8,  48, 0 /), &
       (/nlon_surface,nlat_surface/)))
    allocate(mask(nlat_lake,nlon_lake))
    allocate(cell_mask(nlat_surface,nlon_surface))
    allocate(lakes(2))
    do lake_number = 1,2
      mask = (potential_lake_pixel_mask == lake_number)
      allocate(lake_pixel_coords_list_lat(0))
      allocate(lake_pixel_coords_list_lon(0))
      npixels = count(mask)
      allocate(potential_lake_pixel_coords_list_lat(npixels))
      allocate(potential_lake_pixel_coords_list_lon(npixels))
      counter = 1
      do i = 1,nlat_lake
        do j = 1,nlon_lake
          if (mask(i,j)) then
            potential_lake_pixel_coords_list_lat(counter) = i
            potential_lake_pixel_coords_list_lon(counter) = j
            counter = counter + 1
          end if
        end do
      end do
      cell_mask(:,:) = .false.
      do i=1,size(potential_lake_pixel_coords_list_lat)
          lat = corresponding_surface_cell_lat_index( &
            potential_lake_pixel_coords_list_lat(i), &
            potential_lake_pixel_coords_list_lon(i))
          lon = corresponding_surface_cell_lon_index( &
            potential_lake_pixel_coords_list_lat(i), &
            potential_lake_pixel_coords_list_lon(i))
          cell_mask(lat,lon) = .true.
      end do
      allocate(cell_coords_list_lat(count(cell_mask)))
      allocate(cell_coords_list_lon(count(cell_mask)))
      counter = 1
      do i = 1,nlat_surface
        do j = 1,nlon_surface
          if (cell_mask(i,j)) then
            cell_coords_list_lat(counter) = i
            cell_coords_list_lon(counter) = j
            counter = counter + 1
          end if
        end do
      end do
      lakes(lake_number)%lake_input_pointer => &
        lakeinput(lake_number, &
                  lake_pixel_coords_list_lat, &
                  lake_pixel_coords_list_lon, &
                  potential_lake_pixel_coords_list_lat, &
                  potential_lake_pixel_coords_list_lon, &
                  cell_coords_list_lat, &
                  cell_coords_list_lon)
    end do
    prognostics => setup_lake_for_fraction_calculation(lakes, &
                                                       cell_pixel_counts, &
                                                       non_lake_mask, &
                                                       binary_lake_mask, &
                                                       potential_lake_pixel_mask, &
                                                       corresponding_surface_cell_lat_index, &
                                                       corresponding_surface_cell_lon_index, &
                                                       nlat_lake,nlon_lake, &
                                                       nlat_surface,nlon_surface)
    allocate(lake_pixel_counts_field(nlat_surface,nlon_surface))
    lake_pixel_counts_field(:,:) = 0
    do lake_number = 1,2
      do i = 1,nlat_lake
        do j = 1,nlon_lake
          if (lake_pixel_mask(i,j) == lake_number) then
            call add_pixel_by_coords(i,j,lake_pixel_counts_field,prognostics)
          end if
        end do
      end do
    end do
    call assert_equals(lake_pixel_counts_field, &
                       expected_lake_pixel_counts_field, &
                       nlat_surface,nlon_surface)
    do i = 1,size(lakes)
        call clean_lake_input(lakes(i)%lake_input_pointer)
        deallocate(lakes(i)%lake_input_pointer)
    end do
    deallocate(lakes)
    deallocate(lake_pixel_mask)
    deallocate(binary_lake_mask)
    deallocate(non_lake_mask)
    deallocate(cell_pixel_counts)
    deallocate(corresponding_surface_cell_lat_index)
    deallocate(corresponding_surface_cell_lon_index)
    deallocate(lake_pixel_counts_field)
    deallocate(expected_lake_pixel_counts_field)
    deallocate(cell_mask)
    deallocate(mask)
    call clean_lake_fraction_calculation_prognostics(prognostics)
    deallocate(prognostics)
end subroutine testLakeFractionCalculationTest21

! subroutine testLakeFractionCalculationTest22
!   integer,dimension(:,:), allocatable :: potential_lake_pixel_mask
!   integer,dimension(:,:), allocatable :: lake_pixel_mask
!   integer,dimension(:,:), allocatable :: cell_pixel_counts
!   integer,dimension(:,:), allocatable :: corresponding_surface_cell_lat_index
!   integer,dimension(:,:), allocatable :: corresponding_surface_cell_lon_index
!   integer,dimension(:,:), allocatable :: expected_lake_pixel_counts_field
!   real,dimension(:,:), allocatable :: expected_lake_fractions_field
!   logical,dimension(:,:), allocatable :: expected_binary_lake_mask
!   logical,dimension(:,:), allocatable :: non_lake_mask
!   logical,dimension(:,:), allocatable :: cell_mask
!   integer :: nlat_lake, nlon_lake
!   integer :: nlat_surface, nlon_surface
!     !Multiple partially filled lakes and two non lake points
!     nlat_lake = 20
!     nlon_lake = 20
!     nlat_surface = 5
!     nlon_surface = 5
!     PROCESS BY HAND=>lakes::Vector{LakeInput} = LakeInput[]
!     allocate(potential_lake_pixel_mask(nlat_lake,nlon_lake))
!     potential_lake_pixel_mask = transpose(reshape((/   &
!        0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
!        0, 0, 0, 1,  0, 1, 1, 1,  1, 1, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
!        0, 0, 1, 1,  0, 1, 1, 1,  1, 0, 1, 1,  1, 0, 1, 0,  0, 1, 1, 0, &
!        0, 1, 1, 1,  0, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 0, &
!        0, 0, 0, 0,  1, 0, 1, 1,  1, 0, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
!        0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 0, 1,  0, 1, 1, 0,  0, 0, 1, 1, &
!        0, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 1,  1, 0, 1, 1,  1, 1, 1, 0, &
!        0, 0, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 0, &
!        0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1,  1, 0, 1, 1,  0, 0, 0, 0, &
!        0, 2, 2, 2,  2, 0, 1, 1,  1, 0, 1, 1,  1, 1, 0, 1,  1, 0, 0, 0, &
!        0, 2, 2, 0,  1, 1, 1, 1,  1, 1, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0, &
!        0, 0, 0, 0,  0, 1, 1, 1,  0, 1, 1, 0,  0, 1, 1, 1,  0, 0, 0, 0, &
!        0, 0, 0, 1,  0, 1, 0, 1,  0, 0, 1, 0,  1, 1, 1, 0,  0, 1, 0, 0, &
!        1, 1, 0, 1,  1, 1, 1, 1,  1, 0, 1, 0,  1, 1, 1, 0,  1, 1, 1, 0, &
!        0, 1, 1, 0,  1, 1, 1, 1,  1, 1, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0, &
!        0, 1, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0,  0, 1, 0, 1,  0, 0, 0, 0, &
!        0, 0, 0, 0,  1, 1, 0, 1,  1, 1, 0, 1,  1, 1, 0, 0,  1, 1, 0, 0, &
!        0, 0, 1, 1,  0, 1, 1, 1,  1, 1, 0, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
!        0, 0, 0, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
!        0, 0, 0, 0,  0, 1, 1, 1,  1, 0, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0 /), &
!        (/nlon_lake,nlat_lake/)))
!     allocate(lake_pixel_mask(nlat_lake,nlon_lake))
!     lake_pixel_mask = transpose(reshape((/   &
!        0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
!        0, 0, 0, 1,  0, 1, 1, 1,  1, 1, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
!        0, 0, 1, 1,  0, 1, 1, 1,  1, 0, 1, 1,  1, 0, 1, 0,  0, 1, 1, 0, &
!        0, 1, 1, 1,  0, 1, 1, 1,  0, 1, 0, 1,  1, 1, 1, 1,  1, 1, 0, 0, &
!        0, 0, 0, 0,  1, 0, 1, 1,  1, 0, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
!        0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 0, 1,  0, 1, 1, 0,  0, 0, 1, 1, &
!        0, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 1,  1, 0, 1, 1,  1, 1, 1, 0, &
!        0, 0, 1, 1,  0, 0, 1, 0,  0, 0, 1, 1,  1, 1, 1, 1,  1, 1, 1, 0, &
!        0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1,  1, 0, 1, 1,  0, 0, 0, 0, &
!        0, 2, 2, 2,  0, 0, 1, 1,  1, 0, 1, 1,  0, 1, 0, 1,  1, 0, 0, 0, &
!        0, 2, 0, 0,  1, 1, 1, 1,  1, 1, 1, 0,  1, 0, 0, 1,  1, 1, 0, 0, &
!        0, 0, 0, 0,  0, 1, 1, 1,  0, 1, 1, 0,  0, 1, 1, 1,  0, 0, 0, 0, &
!        0, 0, 0, 1,  0, 0, 0, 1,  0, 0, 1, 0,  1, 1, 1, 0,  0, 0, 0, 0, &
!        1, 1, 0, 1,  0, 0, 0, 0,  1, 0, 1, 0,  1, 1, 1, 0,  0, 0, 1, 0, &
!        0, 1, 1, 0,  0, 0, 1, 0,  1, 1, 1, 0,  1, 1, 1, 1,  0, 0, 0, 0, &
!        0, 1, 1, 0,  1, 0, 1, 1,  1, 1, 0, 0,  0, 1, 0, 1,  0, 0, 0, 0, &
!        0, 0, 0, 0,  1, 0, 0, 1,  1, 1, 0, 1,  1, 1, 0, 0,  1, 1, 0, 0, &
!        0, 0, 1, 1,  0, 1, 1, 1,  1, 1, 0, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
!        0, 0, 0, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
!        0, 0, 0, 0,  0, 1, 1, 1,  1, 0, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0 /), &
!        (/nlon_lake,nlat_lake/)))
!     allocate(cell_pixel_counts(nlat_surface,nlon_surface))
!     cell_pixel_counts = transpose(reshape((/   &
!        16, 16, 16, 16, 16, &
!        16, 16, 16, 16, 16, &
!        16, 16, 16, 16, 16, &
!        16, 16, 16, 16, 16, &
!        16, 16, 16, 16, 16 /), &
!        (/nlon_surface,nlat_surface/)))
!     allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
!     corresponding_surface_cell_lat_index = transpose(reshape((/   &
!        1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
!        1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
!        1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
!        1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
!        2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
!        2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
!        2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
!        2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
!        3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
!        3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
!        3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
!        3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
!        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
!        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
!        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
!        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
!        5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
!        5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
!        5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
!        5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
!        (/nlon_lake,nlat_lake/)))
!     allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
!     corresponding_surface_cell_lon_index = transpose(reshape((/   &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
!        (/nlon_lake,nlat_lake/)))
!     allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
!     expected_lake_pixel_counts_field = transpose(reshape((/   &
!        0, 0,  16,  0,  0, &
!        0, 0,  16,  16, 16, &
!        4, 15, 16, 16, 0, &
!        16, 0, 16,  16, 0, &
!        0,  0, 16, 7,  0 /), &
!        (/nlon_surface,nlat_surface/)))
!     allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
!     expected_lake_fractions_field = transpose(reshape((/   &
!        0.0, 0.0, 1.0, 0.0, 0.0, &
!        0.0, 0.0, 1.0, 1.0, 1.0, &
!        0.25, 0.9375, 1.0, 1.0, 0.0, &
!        1.0, 0.0, 1.0, 1.0, 0.0, &
!        0.0, 0.0, 1.0, 0.4375, 0.0 /), &
!        (/nlon_surface,nlat_surface/)))
!     allocate(expected_binary_lake_mask(nlat_surface,nlon_surface))
!     expected_binary_lake_mask = transpose(reshape((/   &
!        .False., .False.,  .True., .False., .False., &
!        .False., .False.,  .True.,  .True.,  .True., &
!        .False.,  .True.,  .True.,  .True.,  .False., &
!        .True., .False.,  .True.,  .True., .False., &
!        .False., .False.,  .True., .False., .False. /), &
!        (/nlon_surface,nlat_surface/)))
!     allocate(non_lake_mask(nlat_surface,nlon_surface))
!     non_lake_mask = transpose(reshape((/   &
!        .False.,  .True., .False., .False., .False., &
!        .False., .False., .False., .False., .False., &
!        .False., .False., .False., .False., .False., &
!        .False., .False., .False., .False., .False., &
!        .False.,  .True., .False., .False., .False. /), &
!        (/nlon_surface,nlat_surface/)))
!     PROCESS BY HAND=>grid_specific_lake_model_parameters::LatLonLakeModelParameters =    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
!     PROCESS BY HAND=>corresponding_surface_cell_lon_index)
!     PROCESS BY HAND=>do lake_number = 1,2 #
!     PROCESS BY HAND=>
!     PROCESS BY HAND=>lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
!     PROCESS BY HAND=>lake_pixel_mask.data)
!     PROCESS BY HAND=>potential_lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
!     PROCESS BY HAND=>potential_lake_pixel_mask.data)
!     allocate(cell_mask(nlat_lake,nlon_lake))
!     cell_mask(:,:) = .false.
!     PROCESS BY HAND=>dopixel in potential_lake_pixel_coords_list !
!     PROCESS BY HAND=>
!     PROCESS BY HAND=>cell_coords::CartesianIndex =        get_corresponding_surface_model_grid_cell(pixel,
!     PROCESS BY HAND=>grid_specific_lake_model_parameters)
!     PROCESS BY HAND=>set!(cell_mask,cell_coords,true)
!     PROCESS BY HAND=>end
!     PROCESS BY HAND=>cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
!     PROCESS BY HAND=>input = LakeInput(lake_number, &
!         lake_pixel_coords_list, &
!         potential_lake_pixel_coords_list, &
!         cell_coords_list)
!     PROCESS BY HAND=>push!(lakes,input)
!     PROCESS BY HAND=>end
!     PROCESS BY HAND=>lake_pixel_counts_field::Field{Int64},
!     PROCESS BY HAND=>lake_fractions_field::Field{Float64},
!     PROCESS BY HAND=>binary_lake_mask::Field{Bool} =    calculate_lake_fractions(lakes,
!     PROCESS BY HAND=>cell_pixel_counts,
!     PROCESS BY HAND=>non_lake_mask,
!     PROCESS BY HAND=>grid_specific_lake_model_parameters,
!     PROCESS BY HAND=>lake_grid,
!     PROCESS BY HAND=>surface_grid)
!     PROCESS BY HAND=>?call assert_equals(( lake_pixel_counts_field == expected_lake_pixel_counts_field)
!     PROCESS BY HAND=>?call assert_equals(( lake_fractions_field == expected_lake_fractions_field)
!     PROCESS BY HAND=>?call assert_equals(( binary_lake_mask == expected_binary_lake_mask)
!     PROCESS BY HAND=>end
!     PROCESS BY HAND=>
! end subroutine testLakeFractionCalculationTest22

! subroutine testLakeFractionCalculationTest23
!   integer,dimension(:,:), allocatable :: potential_lake_pixel_mask
!   integer,dimension(:,:), allocatable :: lake_pixel_mask
!   integer,dimension(:,:), allocatable :: cell_pixel_counts
!   integer,dimension(:,:), allocatable :: corresponding_surface_cell_lat_index
!   integer,dimension(:,:), allocatable :: corresponding_surface_cell_lon_index
!   logical,dimension(:,:), allocatable :: binary_lake_mask
!   integer,dimension(:,:), allocatable :: expected_lake_pixel_counts_field
!   integer,dimension(:,:), allocatable :: expected_intermediate_lake_pixel_counts_field
!   integer,dimension(:,:), allocatable :: expected_lake_pixel_counts_field_after_cycle
!   integer,dimension(:,:), allocatable :: expected_intermediate_lake_pixel_counts_field_two
!   integer,dimension(:,:), allocatable :: expected_intermediate_lake_pixel_counts_field_three
!   integer,dimension(:,:), allocatable :: expected_lake_pixel_counts_field_after_second_cycle
!   integer,dimension(:,:), allocatable :: expected_lake_pixel_counts_field_after_third_cycle
!   real,dimension(:,:), allocatable :: expected_lake_fractions_field
!   logical,dimension(:,:), allocatable :: non_lake_mask
!   logical,dimension(:,:), allocatable :: cell_mask
!   integer :: nlat_lake, nlon_lake
!   integer :: nlat_surface, nlon_surface
!     !Multiple partially filled lakes and two non lake points
!     PROCESS BY HAND=>lake_grid::Grid = LatLonGrid(20,20,true)
!     PROCESS BY HAND=>surface_grid::Grid = LatLonGrid(5,5,true)
!     nlat_lake = 20
!     nlon_lake = 20
!     nlat_surface = 5
!     nlon_surface = 5
!     PROCESS BY HAND=>lakes::Vector{LakeInput} = LakeInput[]
!     allocate(potential_lake_pixel_mask(nlat_lake,nlon_lake))
!     potential_lake_pixel_mask = transpose(reshape((/   &
!        0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
!        0, 0, 0, 1,  0, 1, 1, 1,  1, 1, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
!        0, 0, 1, 1,  0, 1, 1, 1,  1, 0, 1, 1,  1, 0, 1, 0,  0, 1, 1, 0, &
!        0, 1, 1, 1,  0, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 0, &
!        0, 0, 0, 0,  1, 0, 1, 1,  1, 0, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
!        0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 0, 1,  0, 1, 1, 0,  0, 0, 1, 1, &
!        0, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 1,  1, 0, 1, 1,  1, 1, 1, 0, &
!        0, 0, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 0, &
!        0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1,  1, 0, 1, 1,  0, 0, 0, 0, &
!        0, 2, 2, 2,  2, 0, 1, 1,  1, 0, 1, 1,  1, 1, 0, 1,  1, 0, 0, 0, &
!        0, 2, 2, 0,  1, 1, 1, 1,  1, 1, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0, &
!        0, 0, 0, 0,  0, 1, 1, 1,  0, 1, 1, 0,  0, 1, 1, 1,  0, 0, 0, 0, &
!        0, 0, 0, 1,  0, 1, 0, 1,  0, 0, 1, 0,  1, 1, 1, 0,  0, 1, 0, 0, &
!        1, 1, 0, 1,  1, 1, 1, 1,  1, 0, 1, 0,  1, 1, 1, 0,  1, 1, 1, 0, &
!        0, 1, 1, 0,  1, 1, 1, 1,  1, 1, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0, &
!        0, 1, 1, 0,  1, 1, 1, 1,  1, 1, 0, 0,  0, 1, 0, 1,  0, 0, 0, 0, &
!        0, 0, 0, 0,  1, 1, 0, 1,  1, 1, 0, 1,  1, 1, 0, 0,  1, 1, 0, 0, &
!        0, 0, 1, 1,  0, 1, 1, 1,  1, 1, 0, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
!        0, 0, 0, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
!        0, 0, 0, 0,  0, 1, 1, 1,  1, 0, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0 /), &
!        (/nlon_lake,nlat_lake/)))
!     allocate(lake_pixel_mask(nlat_lake,nlon_lake))
!     lake_pixel_mask = transpose(reshape((/   &
!        0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
!        0, 0, 0, 1,  0, 1, 1, 1,  1, 1, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0, &
!        0, 0, 1, 1,  0, 1, 1, 1,  1, 0, 1, 1,  1, 0, 1, 0,  0, 1, 1, 0, &
!        0, 1, 1, 1,  0, 1, 1, 1,  0, 1, 0, 1,  1, 1, 1, 1,  1, 1, 0, 0, &
!        0, 0, 0, 0,  1, 0, 1, 1,  1, 0, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
!        0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 0, 1,  0, 1, 1, 0,  0, 0, 1, 1, &
!        0, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 1,  1, 0, 1, 1,  1, 1, 1, 0, &
!        0, 0, 1, 1,  0, 0, 1, 0,  0, 0, 1, 1,  1, 1, 1, 1,  1, 1, 1, 0, &
!        0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1,  1, 0, 1, 1,  0, 0, 0, 0, &
!        0, 2, 2, 2,  0, 0, 1, 1,  1, 0, 1, 1,  0, 1, 0, 1,  1, 0, 0, 0, &
!        0, 2, 0, 0,  1, 1, 1, 1,  1, 1, 1, 0,  1, 0, 0, 1,  1, 1, 0, 0, &
!        0, 0, 0, 0,  0, 1, 1, 1,  0, 1, 1, 0,  0, 1, 1, 1,  0, 0, 0, 0, &
!        0, 0, 0, 1,  0, 0, 0, 1,  0, 0, 1, 0,  1, 1, 1, 0,  0, 0, 0, 0, &
!        1, 1, 0, 1,  0, 0, 0, 0,  1, 0, 1, 0,  1, 1, 1, 0,  0, 0, 1, 0, &
!        0, 1, 1, 0,  0, 0, 1, 0,  1, 1, 1, 0,  1, 1, 1, 1,  0, 0, 0, 0, &
!        0, 1, 1, 0,  1, 0, 1, 1,  1, 1, 0, 0,  0, 1, 0, 1,  0, 0, 0, 0, &
!        0, 0, 0, 0,  1, 0, 0, 1,  1, 1, 0, 1,  1, 1, 0, 0,  1, 1, 0, 0, &
!        0, 0, 1, 1,  0, 1, 1, 1,  1, 1, 0, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
!        0, 0, 0, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 0, 0,  0, 1, 1, 0, &
!        0, 0, 0, 0,  0, 1, 1, 1,  1, 0, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0 /), &
!        (/nlon_lake,nlat_lake/)))
!     allocate(cell_pixel_counts(nlat_surface,nlon_surface))
!     cell_pixel_counts = transpose(reshape((/   &
!        16, 16, 16, 16, 16, &
!        16, 16, 16, 16, 16, &
!        16, 16, 16, 16, 16, &
!        16, 16, 16, 16, 16, &
!        16, 16, 16, 16, 16 /), &
!        (/nlon_surface,nlat_surface/)))
!     allocate(corresponding_surface_cell_lat_index(nlat_lake,nlon_lake))
!     corresponding_surface_cell_lat_index = transpose(reshape((/   &
!        1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
!        1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
!        1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
!        1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, &
!        2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
!        2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
!        2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
!        2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2, &
!        3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
!        3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
!        3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
!        3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3,  3, 3, 3, 3, &
!        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
!        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
!        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
!        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, &
!        5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
!        5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
!        5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5, &
!        5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5 /), &
!        (/nlon_lake,nlat_lake/)))
!     allocate(corresponding_surface_cell_lon_index(nlat_lake,nlon_lake))
!     corresponding_surface_cell_lon_index = transpose(reshape((/   &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5, &
!        1, 1, 1, 1,  2, 2, 2, 2,  3, 3, 3, 3,  4, 4, 4, 4,  5, 5, 5, 5 /), &
!        (/nlon_lake,nlat_lake/)))
!     allocate(binary_lake_mask(nlat_surface,nlon_surface))
!     binary_lake_mask = transpose(reshape((/   &
!        .False., .False.,  .True., .False., .False., &
!        .False., .False., .False.,  .True.,  .True., &
!        .False.,  .True., .True.,  .True.,  .False., &
!        .True., .False., .False.,  .True., .False., &
!        .False.,  .True., .False., .False., .False. /), &
!        (/nlon_surface,nlat_surface/)))
!     allocate(expected_lake_pixel_counts_field(nlat_surface,nlon_surface))
!     expected_lake_pixel_counts_field = transpose(reshape((/   &
!        0, 1,  16,  6,  4, &
!        0, 0,  3,  16, 16, &
!        3, 16, 16, 16, 3, &
!        16, 1,  7,  16, 1, &
!        1, 16,  1, 5,  6 /), &
!        (/nlon_surface,nlat_surface/)))
!     allocate(expected_intermediate_lake_pixel_counts_field(nlat_surface,nlon_surface))
!     expected_intermediate_lake_pixel_counts_field = transpose(reshape((/   &
!        0, 0,  15,  0,  0, &
!        0, 0,  0,  16, 16, &
!        0, 15, 16, 16, 0, &
!        16, 0,  0,  16, 0, &
!        0, 16, 0, 0,  0 /), &
!        (/nlon_surface,nlat_surface/)))
!     allocate(expected_lake_pixel_counts_field_after_cycle(nlat_surface,nlon_surface))
!     expected_lake_pixel_counts_field_after_cycle = transpose(reshape((/   &
!        0, 5,  16,  4,  4, &
!        0, 3,  2,  16, 16, &
!        3, 16, 16, 16, 0, &
!        16, 1,  7,  16, 1, &
!        0, 16,  5, 3,  4  /), &
!        (/nlon_surface,nlat_surface/)))
!     allocate(expected_intermediate_lake_pixel_counts_field_two(nlat_surface,nlon_surface))
!     expected_intermediate_lake_pixel_counts_field_two = transpose(reshape((/   &
!        0, 0,  15,  0,  0, &
!        0, 0,  0,  16, 16, &
!        0, 15, 16, 16, 0, &
!        16, 0,  0,  16, 0, &
!        0, 16, 0, 0,  0 /), &
!        (/nlon_surface,nlat_surface/)))
!     allocate(expected_intermediate_lake_pixel_counts_field_three(nlat_surface,nlon_surface))
!     expected_intermediate_lake_pixel_counts_field_three = transpose(reshape((/   &
!        0, 0,  16,  0,  0, &
!        0, 0,  0,  16, 16, &
!        0, 14, 16, 16, 0, &
!        16, 0,  0,  16, 0, &
!        0, 16, 0, 0,  0 /), &
!        (/nlon_surface,nlat_surface/)))
!     allocate(expected_lake_pixel_counts_field_after_second_cycle(nlat_surface,nlon_surface))
!     expected_lake_pixel_counts_field_after_second_cycle = transpose(reshape((/   &
!        0, 5,  16,  4,  4, &
!        0, 3,  3,  16, 16, &
!        3, 16, 16, 16, 0, &
!        16, 1,  7,  16, 1, &
!        2, 16,  3, 2,  4  /), &
!        (/nlon_surface,nlat_surface/)))
!     allocate(expected_lake_pixel_counts_field_after_third_cycle(nlat_surface,nlon_surface))
!     expected_lake_pixel_counts_field_after_third_cycle = transpose(reshape((/   &
!        1, 5,  16,  5,  4, &
!        0, 3,  3,  16, 16, &
!        3, 16, 16, 16, 0, &
!        16, 2,  7,  16, 1, &
!        0, 16,  4, 0,  4  /), &
!        (/nlon_surface,nlat_surface/)))
!     allocate(expected_lake_fractions_field(nlat_surface,nlon_surface))
!     expected_lake_fractions_field = transpose(reshape((/   &
!        0.0, 1.0, 1.0, 0.0, 0.0, &
!        0.0, 0.0, 0.0, 1.0, 1.0, &
!        0.25, 0.9375, 1.0, 1.0, 0.0, &
!        1.0, 0.0, 0.4375, 1.0, 0.0, &
!        0.0, 1.0, 1.0, 0.0, 0.0 /), &
!        (/nlon_surface,nlat_surface/)))
!     allocate(non_lake_mask(nlat_surface,nlon_surface))
!     non_lake_mask = transpose(reshape((/   &
!        .False.,  .True., .False., .False., .False., &
!        .False., .False., .False., .False., .False., &
!        .False., .False., .False., .False., .False., &
!        .False., .False., .False., .False., .False., &
!        .False., .False.,  .True., .False., .False. /), &
!        (/nlon_surface,nlat_surface/)))
!     PROCESS BY HAND=>grid_specific_lake_model_parameters::LatLonLakeModelParameters =    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
!     PROCESS BY HAND=>corresponding_surface_cell_lon_index)
!     PROCESS BY HAND=>do lake_number = 1,2 #
!     PROCESS BY HAND=>
!     PROCESS BY HAND=>lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
!     PROCESS BY HAND=>lake_pixel_mask.data)
!     PROCESS BY HAND=>potential_lake_pixel_coords_list::Vector{CartesianIndex} =      findall(x -> x == lake_number,potential_lake_pixel_mask.data)
!     allocate(cell_mask(nlat_lake,nlon_lake))
!     cell_mask(:,:) = .false.
!     PROCESS BY HAND=>dopixel in potential_lake_pixel_coords_list !
!     PROCESS BY HAND=>
!     PROCESS BY HAND=>cell_coords::CartesianIndex =        get_corresponding_surface_model_grid_cell(pixel,
!     PROCESS BY HAND=>grid_specific_lake_model_parameters)
!     PROCESS BY HAND=>set!(cell_mask,cell_coords,true)
!     PROCESS BY HAND=>end
!     PROCESS BY HAND=>cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
!     PROCESS BY HAND=>input = LakeInput(lake_number, &
!         CartesianIndex[], &
!         potential_lake_pixel_coords_list, &
!         cell_coords_list)
!     PROCESS BY HAND=>push!(lakes,input)
!     PROCESS BY HAND=>end
!     PROCESS BY HAND=>prognostics::LakeFractionCalculationPrognostics =    setup_lake_for_fraction_calculation(lakes,
!     PROCESS BY HAND=>cell_pixel_counts,
!     PROCESS BY HAND=>non_lake_mask,
!     PROCESS BY HAND=>binary_lake_mask,
!     PROCESS BY HAND=>potential_lake_pixel_mask,
!     PROCESS BY HAND=>grid_specific_lake_model_parameters,
!     PROCESS BY HAND=>lake_grid,
!     PROCESS BY HAND=>surface_grid)
!     PROCESS BY HAND=>lake_pixel_counts_field::Field{Int64} = Field{Int64}(surface_grid,0)
!     PROCESS BY HAND=>do lake_number = 1,2 #
!     PROCESS BY HAND=>
!     PROCESS BY HAND=>lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
!     PROCESS BY HAND=>lake_pixel_mask.data)
!     PROCESS BY HAND=>docoords in lake_pixel_coords_list !
!     PROCESS BY HAND=>
!     PROCESS BY HAND=>call add_pixel_by_coords(coords, &
!        lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>end
!     PROCESS BY HAND=>end
!     PROCESS BY HAND=>?call assert_equals(( lake_pixel_counts_field == expected_lake_pixel_counts_field)
!     !@test lake_fractions_field == expected_lake_fractions_field
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(2,4),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(2,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(3,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(4,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(4,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(3,13),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(10,4),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(10,3),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(10,2),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(15,3),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(16,3),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(14,19),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(13,13),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(13,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(14,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(15,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(15,10),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(16,10),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(17,9),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(17,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(18,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(18,9),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(18,10),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(19,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(20,7),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(18,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(10,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(9,7),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(9,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(9,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(8,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(7,13),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(6,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>?call assert_equals(( lake_pixel_counts_field == expected_intermediate_lake_pixel_counts_field)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(2,4),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(2,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(3,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(4,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(4,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(3,13),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(10,4),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(10,3),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(10,2),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(15,3),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(16,3),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(14,19),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(13,13),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(13,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(14,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(15,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(15,10),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(16,10),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(17,9),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(17,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(18,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(18,9),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(18,10),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(19,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(20,7),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(18,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(10,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(9,7),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(9,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(9,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(8,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(7,13),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(6,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>?call assert_equals(( lake_pixel_counts_field == expected_lake_pixel_counts_field_after_cycle)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(2,4),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(2,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(3,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(4,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(4,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(3,13),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(10,4),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(10,3),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(10,2),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(15,3),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(16,3),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(14,19),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(13,13),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(13,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(14,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(15,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(15,10),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(16,10),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(17,9),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(17,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(18,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(18,9),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(18,10),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(19,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(20,7),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(18,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(10,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(9,7),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(9,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(9,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(8,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(7,13),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(6,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>?call assert_equals(( lake_pixel_counts_field == expected_intermediate_lake_pixel_counts_field_two)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(2,4),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(2,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(3,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(4,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(4,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(3,13),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(10,4),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(10,3),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(10,2),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(15,3),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(16,3),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(14,19),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(13,13),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(13,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(14,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(15,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(15,10),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(16,10),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(17,9),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(17,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(18,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(18,9),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(18,10),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(19,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(20,7),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(18,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(10,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(9,7),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(9,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(9,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(8,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(7,13),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(6,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>?call assert_equals(( lake_pixel_counts_field == expected_lake_pixel_counts_field_after_second_cycle)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(2,4),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(2,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(3,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(4,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(4,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(3,13),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(10,4),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(10,3),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(10,2),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(15,3),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(16,3),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(14,19),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(13,13),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(13,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(14,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(15,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(15,10),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(16,10),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(17,9),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(17,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(18,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(18,9),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(18,10),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(19,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(20,7),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(18,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(10,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(9,7),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(9,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(9,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(8,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(7,13),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call remove_pixel_by_coords(CartesianIndex(6,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>?call assert_equals(( lake_pixel_counts_field == expected_intermediate_lake_pixel_counts_field_three)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(2,4),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(2,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(3,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(4,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(4,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(3,13),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(10,4),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(10,3),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(10,2),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(15,3),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(16,3),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(14,19),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(13,13),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(13,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(14,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(15,11),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(15,10),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(16,10),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(17,9),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(17,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(18,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(18,9),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(18,10),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(19,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(20,7),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(18,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(10,8),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(9,7),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(9,6),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(9,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(8,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(7,13),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>call add_pixel_by_coords(CartesianIndex(6,12),lake_pixel_counts_field,prognostics)
!     PROCESS BY HAND=>?call assert_equals(( lake_pixel_counts_field == expected_lake_pixel_counts_field_after_third_cycle)
! end subroutine testLakeFractionCalculationTest23


end module calculate_lake_fractions_test_mod
