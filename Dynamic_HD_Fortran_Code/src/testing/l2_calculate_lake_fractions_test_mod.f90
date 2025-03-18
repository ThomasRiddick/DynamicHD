module l2_calculate_lake_fractions_test_mod

use fruit
implicit none

contains

subroutine testLakeFractionCalculationTest1
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
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
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
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
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest1

subroutine testLakeFractionCalculationTest2
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
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
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
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
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest2

subroutine testLakeFractionCalculationTest3
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
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
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
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
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest3

subroutine testLakeFractionCalculationTest4
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
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
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
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
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest4

subroutine testLakeFractionCalculationTest5
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
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
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
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
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest5

subroutine testLakeFractionCalculationTest6
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: potential_lake_pixel_mask
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
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
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
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
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest6

subroutine testLakeFractionCalculationTest7
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: potential_lake_pixel_mask
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
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
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
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
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest7

subroutine testLakeFractionCalculationTest8
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: potential_lake_pixel_mask
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
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
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
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
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest8

subroutine testLakeFractionCalculationTest9
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: potential_lake_pixel_mask
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
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
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
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
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest9

subroutine testLakeFractionCalculationTest10
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   integer,dimension(:,:), pointer :: potential_lake_pixel_mask
   integer,dimension(:,:), pointer :: lake_pixel_mask
   integer,dimension(:,:), pointer :: cell_pixel_counts
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lat_index
   integer,dimension(:,:), pointer :: corresponding_surface_cell_lon_index
   integer,dimension(:,:), pointer :: lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: lake_fractions_field
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
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
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
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_binary_lake_mask)
      deallocate(cell_mask)
      deallocate(mask)
end subroutine testLakeFractionCalculationTest10

subroutine testLakeFractionCalculationTest11
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   type(lakepropertiespointer), dimension(:), pointer :: lake_properties
   integer, dimension(:,:), pointer :: pixel_numbers
   type(pixelpointer), dimension(:), pointer :: pixels
   type(pixel), pointer :: working_pixel
   integer,dimension(:,:), pointer :: lake_pixel_mask
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
   integer :: pixel_number
   integer :: lat,lon
   integer :: i,j

      !Single lake
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      lake_number = 1
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
      call setup_lake_for_fraction_calculation(lakes, &
                                               cell_pixel_counts, &
                                               binary_lake_mask, &
                                               lake_properties, &
                                               pixel_numbers, &
                                               pixels, &
                                               lake_pixel_counts_field, &
                                               corresponding_surface_cell_lat_index, &
                                               corresponding_surface_cell_lon_index, &
                                               nlat_lake,nlon_lake, &
                                               nlat_surface,nlon_surface)
      do i = 1,size(potential_lake_pixel_coords_list_lat)
        pixel_number = pixel_numbers(potential_lake_pixel_coords_list_lat(i),&
                                     potential_lake_pixel_coords_list_lon(i))
        working_pixel => pixels(pixel_number)%pixel_pointer
        call add_pixel(lake_properties(1)%lake_properties_pointer,&
                       working_pixel,pixels,lake_pixel_counts_field)
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,&
                         nlat_surface,nlon_surface)
      pixel_number = pixel_numbers(15,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake_properties(1)%lake_properties_pointer,&
                        working_pixel,pixels,lake_pixel_counts_field)
      pixel_number = pixel_numbers(11,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake_properties(1)%lake_properties_pointer,&
                        working_pixel,pixels,lake_pixel_counts_field)
      pixel_number = pixel_numbers(15,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake_properties(1)%lake_properties_pointer,&
                        working_pixel,pixels,lake_pixel_counts_field)
      call assert_equals(lake_pixel_counts_field, &
                         expected_immediate_lake_pixel_counts_field, &
                         nlat_surface,nlon_surface)
      pixel_number = pixel_numbers(15,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake_properties(1)%lake_properties_pointer,&
                     working_pixel,pixels,lake_pixel_counts_field)
      pixel_number = pixel_numbers(11,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake_properties(1)%lake_properties_pointer,&
                     working_pixel,pixels,lake_pixel_counts_field)
      pixel_number = pixel_numbers(15,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake_properties(1)%lake_properties_pointer,&
                     working_pixel,pixels,lake_pixel_counts_field)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(binary_lake_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_immediate_lake_pixel_counts_field)
      deallocate(cell_mask)
      deallocate(mask)
      deallocate(pixel_numbers)
      do i = 1,size(pixels)
        deallocate(pixels(i)%pixel_pointer)
      end do
      deallocate(pixels)
      do i = 1,size(lake_properties)
        do j = 1,size(lake_properties(i)%lake_properties_pointer%cell_list)
          call clean_lake_cell(lake_properties(i)%lake_properties_pointer%&
                               &cell_list(j)%lake_cell_pointer)
          deallocate(lake_properties(i)%lake_properties_pointer%&
                     &cell_list(j)%lake_cell_pointer)
        end do
        deallocate(lake_properties(i)%lake_properties_pointer%&
                   &cell_list)
        deallocate(lake_properties(i)%lake_properties_pointer)
      end do
      deallocate(lake_properties)
end subroutine testLakeFractionCalculationTest11

subroutine testLakeFractionCalculationTest12
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   type(lakepropertiespointer), dimension(:), pointer :: lake_properties
   type(lakeproperties), pointer :: lake
   integer, dimension(:,:), pointer :: pixel_numbers
   type(pixelpointer), dimension(:), pointer :: pixels
   type(pixel), pointer :: working_pixel
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
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   integer,dimension(:,:), pointer :: expected_immediate_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: pixel_number
   integer :: lat,lon
   integer :: i,j

      !Multiple lakes
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
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
      call setup_lake_for_fraction_calculation(lakes, &
                                               cell_pixel_counts, &
                                               binary_lake_mask, &
                                               lake_properties, &
                                               pixel_numbers, &
                                               pixels, &
                                               lake_pixel_counts_field, &
                                               corresponding_surface_cell_lat_index, &
                                               corresponding_surface_cell_lon_index, &
                                               nlat_lake,nlon_lake, &
                                               nlat_surface,nlon_surface)
      do lake_number = 1,7
        lake => lake_properties(lake_number)%lake_properties_pointer
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
          pixel_number = pixel_numbers(potential_lake_pixel_coords_list_lat(i),&
                                       potential_lake_pixel_coords_list_lon(i))
          working_pixel => pixels(pixel_number)%pixel_pointer
          call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
        end do
        deallocate(potential_lake_pixel_coords_list_lat)
        deallocate(potential_lake_pixel_coords_list_lon)
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,nlat_surface,nlon_surface)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(2,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(3,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(10,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(5)%lake_properties_pointer
      pixel_number = pixel_numbers(16,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(5)%lake_properties_pointer
      pixel_number = pixel_numbers(19,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(8,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(5,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(6,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(3,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(7)%lake_properties_pointer
      pixel_number = pixel_numbers(2,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(3,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(3)%lake_properties_pointer
      pixel_number = pixel_numbers(3,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(3)%lake_properties_pointer
      pixel_number = pixel_numbers(3,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(11,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(12,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(12,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(10,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(11,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(11,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(12,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(12,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(11,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,16)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,16)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,15)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,15)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(20,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,16)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,15)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,15)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,15)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(6,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(7,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(7,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(5,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(5,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(5,16)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      call assert_equals(lake_pixel_counts_field, &
                         expected_immediate_lake_pixel_counts_field, &
                         nlat_surface,nlon_surface)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(2,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(3,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(10,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(5)%lake_properties_pointer
      pixel_number = pixel_numbers(16,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(5)%lake_properties_pointer
      pixel_number = pixel_numbers(19,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(8,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(5,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(6,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(3,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(7)%lake_properties_pointer
      pixel_number = pixel_numbers(2,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(3,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(3)%lake_properties_pointer
      pixel_number = pixel_numbers(3,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(3)%lake_properties_pointer
      pixel_number = pixel_numbers(3,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(11,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(12,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(12,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(10,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(11,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(11,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(12,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(12,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(11,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,16)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,16)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,15)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,15)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(20,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,16)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,15)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,15)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,15)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(6,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(7,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(7,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(5,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(5,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(5,16)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(binary_lake_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_immediate_lake_pixel_counts_field)
      deallocate(cell_mask)
      deallocate(mask)
      deallocate(pixel_numbers)
      do i = 1,size(pixels)
        deallocate(pixels(i)%pixel_pointer)
      end do
      deallocate(pixels)
      do i = 1,size(lake_properties)
        do j = 1,size(lake_properties(i)%lake_properties_pointer%cell_list)
          call clean_lake_cell(lake_properties(i)%lake_properties_pointer%&
                               &cell_list(j)%lake_cell_pointer)
          deallocate(lake_properties(i)%lake_properties_pointer%&
                     &cell_list(j)%lake_cell_pointer)
        end do
        deallocate(lake_properties(i)%lake_properties_pointer%&
                   &cell_list)
        deallocate(lake_properties(i)%lake_properties_pointer)
      end do
      deallocate(lake_properties)
end subroutine testLakeFractionCalculationTest12

subroutine testLakeFractionCalculationTest13
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   type(lakepropertiespointer), dimension(:), pointer :: lake_properties
   type(lakeproperties), pointer :: lake
   integer, dimension(:,:), pointer :: pixel_numbers
   type(pixelpointer), dimension(:), pointer :: pixels
   type(pixel), pointer :: working_pixel
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
   integer :: pixel_number
   integer :: lat,lon
   integer :: i,j

      !Multiple lakes
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
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
         0, 13,  2, 16, 8, &
         1, 0,  0,  3, 0 /), &
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
          0, 12,  0,  0, 5, &
         11, 1, 0,   9, 0, &
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
      call setup_lake_for_fraction_calculation(lakes, &
                                               cell_pixel_counts, &
                                               binary_lake_mask, &
                                               lake_properties, &
                                               pixel_numbers, &
                                               pixels, &
                                               lake_pixel_counts_field, &
                                               corresponding_surface_cell_lat_index, &
                                               corresponding_surface_cell_lon_index, &
                                               nlat_lake,nlon_lake, &
                                               nlat_surface,nlon_surface)
      do lake_number = 1,9
        lake => lake_properties(lake_number)%lake_properties_pointer
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
          pixel_number = pixel_numbers(potential_lake_pixel_coords_list_lat(i),&
                                       potential_lake_pixel_coords_list_lon(i))
          working_pixel => pixels(pixel_number)%pixel_pointer
          call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
        end do
        deallocate(potential_lake_pixel_coords_list_lat)
        deallocate(potential_lake_pixel_coords_list_lon)
      end do
      do i = 1,nlat_surface
        write(*,*) (lake_pixel_counts_field(i,j), j =1,nlon_surface)
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,&
                         nlat_surface,nlon_surface)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(2,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(4,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(4,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(4,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(2,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(5,5)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(5,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(5,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(5,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(6,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(6,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(6,5)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(7,5)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(7,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(7,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(7,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(8,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(8,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(8,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(8,5)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(17,5)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(17,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(17,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(17,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(17,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(18,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(18,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(17,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(17,14)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(3)%lake_properties_pointer
      pixel_number = pixel_numbers(14,14)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(3)%lake_properties_pointer
      pixel_number = pixel_numbers(14,15)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(3)%lake_properties_pointer
      pixel_number = pixel_numbers(13,15)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(3)%lake_properties_pointer
      pixel_number = pixel_numbers(13,14)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(5)%lake_properties_pointer
      pixel_number = pixel_numbers(17,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(5)%lake_properties_pointer
      pixel_number = pixel_numbers(17,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(5)%lake_properties_pointer
      pixel_number = pixel_numbers(18,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(9)%lake_properties_pointer
      pixel_number = pixel_numbers(8,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(9)%lake_properties_pointer
      pixel_number = pixel_numbers(8,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(9)%lake_properties_pointer
      pixel_number = pixel_numbers(9,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(9)%lake_properties_pointer
      pixel_number = pixel_numbers(9,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(9)%lake_properties_pointer
      pixel_number = pixel_numbers(10,16)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(9)%lake_properties_pointer
      pixel_number = pixel_numbers(10,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(8)%lake_properties_pointer
      pixel_number = pixel_numbers(7,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(8)%lake_properties_pointer
      pixel_number = pixel_numbers(8,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(8)%lake_properties_pointer
      pixel_number = pixel_numbers(8,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(8)%lake_properties_pointer
      pixel_number = pixel_numbers(7,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(7)%lake_properties_pointer
      pixel_number = pixel_numbers(6,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(7)%lake_properties_pointer
      pixel_number = pixel_numbers(5,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(7)%lake_properties_pointer
      pixel_number = pixel_numbers(5,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(7)%lake_properties_pointer
      pixel_number = pixel_numbers(6,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(10,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      call assert_equals(lake_pixel_counts_field, &
                         expected_intermediate_lake_pixel_counts_field,&
                         nlat_surface,nlon_surface)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(2,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(4,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(4,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(4,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(2,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(5,5)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(5,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(5,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(5,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(6,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(6,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(6,5)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(7,5)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(7,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(7,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(7,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(8,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(8,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(8,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(8,5)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(17,5)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(17,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(17,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(17,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(17,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(18,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(18,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(17,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(4)%lake_properties_pointer
      pixel_number = pixel_numbers(17,14)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(3)%lake_properties_pointer
      pixel_number = pixel_numbers(14,14)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(3)%lake_properties_pointer
      pixel_number = pixel_numbers(14,15)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(3)%lake_properties_pointer
      pixel_number = pixel_numbers(13,15)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(3)%lake_properties_pointer
      pixel_number = pixel_numbers(13,14)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(5)%lake_properties_pointer
      pixel_number = pixel_numbers(17,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(5)%lake_properties_pointer
      pixel_number = pixel_numbers(17,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(5)%lake_properties_pointer
      pixel_number = pixel_numbers(18,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(9)%lake_properties_pointer
      pixel_number = pixel_numbers(8,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(9)%lake_properties_pointer
      pixel_number = pixel_numbers(8,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(9)%lake_properties_pointer
      pixel_number = pixel_numbers(9,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(9)%lake_properties_pointer
      pixel_number = pixel_numbers(9,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(9)%lake_properties_pointer
      pixel_number = pixel_numbers(10,16)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(9)%lake_properties_pointer
      pixel_number = pixel_numbers(10,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(8)%lake_properties_pointer
      pixel_number = pixel_numbers(7,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(8)%lake_properties_pointer
      pixel_number = pixel_numbers(8,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(8)%lake_properties_pointer
      pixel_number = pixel_numbers(8,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(8)%lake_properties_pointer
      pixel_number = pixel_numbers(7,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(7)%lake_properties_pointer
      pixel_number = pixel_numbers(6,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(7)%lake_properties_pointer
      pixel_number = pixel_numbers(5,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(7)%lake_properties_pointer
      pixel_number = pixel_numbers(5,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(7)%lake_properties_pointer
      pixel_number = pixel_numbers(6,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(6)%lake_properties_pointer
      pixel_number = pixel_numbers(10,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field_two,&
                         nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(binary_lake_mask)
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
      deallocate(pixel_numbers)
      do i = 1,size(pixels)
        deallocate(pixels(i)%pixel_pointer)
      end do
      deallocate(pixels)
      do i = 1,size(lake_properties)
        do j = 1,size(lake_properties(i)%lake_properties_pointer%cell_list)
          call clean_lake_cell(lake_properties(i)%lake_properties_pointer%&
                               &cell_list(j)%lake_cell_pointer)
          deallocate(lake_properties(i)%lake_properties_pointer%&
                     &cell_list(j)%lake_cell_pointer)
        end do
        deallocate(lake_properties(i)%lake_properties_pointer%&
                   &cell_list)
        deallocate(lake_properties(i)%lake_properties_pointer)
      end do
      deallocate(lake_properties)
end subroutine testLakeFractionCalculationTest13

subroutine testLakeFractionCalculationTest14
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   type(lakepropertiespointer), dimension(:), pointer :: lake_properties
   type(lakeproperties), pointer :: lake
   integer, dimension(:,:), pointer :: pixel_numbers
   type(pixelpointer), dimension(:), pointer :: pixels
   type(pixel), pointer :: working_pixel
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
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: pixel_number
   integer :: npixels, counter
   integer :: lat,lon
   integer :: i,j

      !Multiple lakes
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
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
         0, 16, 16, 16, 12, &
         6, 16, 16, 16, 0, &
         16, 16,  0, 16, 0, &
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
      call setup_lake_for_fraction_calculation(lakes, &
                                               cell_pixel_counts, &
                                               binary_lake_mask, &
                                               lake_properties, &
                                               pixel_numbers, &
                                               pixels, &
                                               lake_pixel_counts_field, &
                                               corresponding_surface_cell_lat_index, &
                                               corresponding_surface_cell_lon_index, &
                                               nlat_lake,nlon_lake, &
                                               nlat_surface,nlon_surface)
      do lake_number = 1,2
        lake => lake_properties(lake_number)%lake_properties_pointer
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
          pixel_number = pixel_numbers(potential_lake_pixel_coords_list_lat(i),&
                                       potential_lake_pixel_coords_list_lon(i))
          working_pixel => pixels(pixel_number)%pixel_pointer
          call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
        end do
        deallocate(potential_lake_pixel_coords_list_lat)
        deallocate(potential_lake_pixel_coords_list_lon)
      end do
      do i=1,nlat_surface
        write(*,*) (lake_pixel_counts_field(i,j),j=1,nlon_surface)
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(binary_lake_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(cell_mask)
      deallocate(mask)
      deallocate(pixel_numbers)
      do i = 1,size(pixels)
        deallocate(pixels(i)%pixel_pointer)
      end do
      deallocate(pixels)
      do i = 1,size(lake_properties)
        do j = 1,size(lake_properties(i)%lake_properties_pointer%cell_list)
          call clean_lake_cell(lake_properties(i)%lake_properties_pointer%&
                               &cell_list(j)%lake_cell_pointer)
          deallocate(lake_properties(i)%lake_properties_pointer%&
                     &cell_list(j)%lake_cell_pointer)
        end do
        deallocate(lake_properties(i)%lake_properties_pointer%&
                   &cell_list)
        deallocate(lake_properties(i)%lake_properties_pointer)
      end do
      deallocate(lake_properties)
end subroutine testLakeFractionCalculationTest14

subroutine testLakeFractionCalculationTest15
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   type(lakepropertiespointer), dimension(:), pointer :: lake_properties
   type(lakeproperties), pointer :: lake
   integer, dimension(:,:), pointer :: pixel_numbers
   type(pixelpointer), dimension(:), pointer :: pixels
   type(pixel), pointer :: working_pixel
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
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   integer,dimension(:,:), pointer :: expected_intermediate_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: pixel_number
   integer :: lat,lon
   integer :: i,j

      !Multiple lakes
      nlat_lake = 15
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
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
      call setup_lake_for_fraction_calculation(lakes, &
                                               cell_pixel_counts, &
                                               binary_lake_mask, &
                                               lake_properties, &
                                               pixel_numbers, &
                                               pixels, &
                                               lake_pixel_counts_field, &
                                               corresponding_surface_cell_lat_index, &
                                               corresponding_surface_cell_lon_index, &
                                               nlat_lake,nlon_lake, &
                                               nlat_surface,nlon_surface)
      do lake_number = 1,2
        lake => lake_properties(lake_number)%lake_properties_pointer
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
          pixel_number = pixel_numbers(potential_lake_pixel_coords_list_lat(i),&
                                       potential_lake_pixel_coords_list_lon(i))
          working_pixel => pixels(pixel_number)%pixel_pointer
          call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
        end do
        deallocate(potential_lake_pixel_coords_list_lat)
        deallocate(potential_lake_pixel_coords_list_lon)
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field, &
                         nlat_surface,nlon_surface)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(11,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(11,4)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(11,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(4,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(4,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(4,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(3,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(3,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(3,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(4,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      call assert_equals(lake_pixel_counts_field, &
                         expected_intermediate_lake_pixel_counts_field, &
                         nlat_surface,nlon_surface)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(11,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(11,4)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(11,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(4,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(4,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(4,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(3,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(3,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(3,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(4,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,&
                         nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(binary_lake_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_intermediate_lake_pixel_counts_field)
      deallocate(cell_mask)
      deallocate(mask)
      deallocate(pixel_numbers)
      do i = 1,size(pixels)
        deallocate(pixels(i)%pixel_pointer)
      end do
      deallocate(pixels)
      do i = 1,size(lake_properties)
        do j = 1,size(lake_properties(i)%lake_properties_pointer%cell_list)
          call clean_lake_cell(lake_properties(i)%lake_properties_pointer%&
                               &cell_list(j)%lake_cell_pointer)
          deallocate(lake_properties(i)%lake_properties_pointer%&
                     &cell_list(j)%lake_cell_pointer)
        end do
        deallocate(lake_properties(i)%lake_properties_pointer%&
                   &cell_list)
        deallocate(lake_properties(i)%lake_properties_pointer)
      end do
      deallocate(lake_properties)
end subroutine testLakeFractionCalculationTest15

subroutine testLakeFractionCalculationTest16
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   type(lakepropertiespointer), dimension(:), pointer :: lake_properties
   integer, dimension(:,:), pointer :: pixel_numbers
   type(pixelpointer), dimension(:), pointer :: pixels
   type(pixel), pointer :: working_pixel
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
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: pixel_number
   integer :: lat,lon
   integer :: i,j

      !Partially filled single lake
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
      lake_number = 1
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
         0, 0, 0, 0, 2, &
         0, 0, 0, 0, 2, &
         0, 0, 0, 1, 16 /), &
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
      call setup_lake_for_fraction_calculation(lakes, &
                                               cell_pixel_counts, &
                                               binary_lake_mask, &
                                               lake_properties, &
                                               pixel_numbers, &
                                               pixels, &
                                               lake_pixel_counts_field, &
                                               corresponding_surface_cell_lat_index, &
                                               corresponding_surface_cell_lon_index, &
                                               nlat_lake,nlon_lake, &
                                               nlat_surface,nlon_surface)
      do i = 1,nlat_lake
        do j = 1,nlon_lake
          if (lake_pixel_mask(i,j) == 1) then
            pixel_number = pixel_numbers(i,j)
            working_pixel => pixels(pixel_number)%pixel_pointer
            call add_pixel(lake_properties(1)%lake_properties_pointer,&
                           working_pixel,pixels,lake_pixel_counts_field)
          end if
        end do
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,&
                         nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(potential_lake_pixel_mask)
      deallocate(binary_lake_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(cell_mask)
      deallocate(mask)
      deallocate(pixel_numbers)
      do i = 1,size(pixels)
        deallocate(pixels(i)%pixel_pointer)
      end do
      deallocate(pixels)
      do i = 1,size(lake_properties)
        do j = 1,size(lake_properties(i)%lake_properties_pointer%cell_list)
          call clean_lake_cell(lake_properties(i)%lake_properties_pointer%&
                               &cell_list(j)%lake_cell_pointer)
          deallocate(lake_properties(i)%lake_properties_pointer%&
                     &cell_list(j)%lake_cell_pointer)
        end do
        deallocate(lake_properties(i)%lake_properties_pointer%&
                   &cell_list)
        deallocate(lake_properties(i)%lake_properties_pointer)
      end do
      deallocate(lake_properties)
end subroutine testLakeFractionCalculationTest16

subroutine testLakeFractionCalculationTest17
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   type(lakepropertiespointer), dimension(:), pointer :: lake_properties
   type(lakeproperties), pointer :: lake
   integer, dimension(:,:), pointer :: pixel_numbers
   type(pixelpointer), dimension(:), pointer :: pixels
   type(pixel), pointer :: working_pixel
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
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: pixel_number
   integer :: lat,lon
   integer :: i,j

      !Multiple partially filled lakes
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
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
      call setup_lake_for_fraction_calculation(lakes, &
                                               cell_pixel_counts, &
                                               binary_lake_mask, &
                                               lake_properties, &
                                               pixel_numbers, &
                                               pixels, &
                                               lake_pixel_counts_field, &
                                               corresponding_surface_cell_lat_index, &
                                               corresponding_surface_cell_lon_index, &
                                               nlat_lake,nlon_lake, &
                                               nlat_surface,nlon_surface)
      do lake_number = 1,7
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (lake_pixel_mask(i,j) == lake_number) then
              pixel_number = pixel_numbers(i,j)
              working_pixel => pixels(pixel_number)%pixel_pointer
              call add_pixel(lake_properties(lake_number)%lake_properties_pointer,&
                             working_pixel,pixels,lake_pixel_counts_field)
            end if
          end do
        end do
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field, &
                         nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(potential_lake_pixel_mask)
      deallocate(binary_lake_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(cell_mask)
      deallocate(mask)
      deallocate(pixel_numbers)
      do i = 1,size(pixels)
        deallocate(pixels(i)%pixel_pointer)
      end do
      deallocate(pixels)
      do i = 1,size(lake_properties)
        do j = 1,size(lake_properties(i)%lake_properties_pointer%cell_list)
          call clean_lake_cell(lake_properties(i)%lake_properties_pointer%&
                               &cell_list(j)%lake_cell_pointer)
          deallocate(lake_properties(i)%lake_properties_pointer%&
                     &cell_list(j)%lake_cell_pointer)
        end do
        deallocate(lake_properties(i)%lake_properties_pointer%&
                   &cell_list)
        deallocate(lake_properties(i)%lake_properties_pointer)
      end do
      deallocate(lake_properties)
end subroutine testLakeFractionCalculationTest17

subroutine testLakeFractionCalculationTest18
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   type(lakepropertiespointer), dimension(:), pointer :: lake_properties
   type(lakeproperties), pointer :: lake
   integer, dimension(:,:), pointer :: pixel_numbers
   type(pixelpointer), dimension(:), pointer :: pixels
   type(pixel), pointer :: working_pixel
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
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: pixel_number
   integer :: lat,lon
   integer :: i,j

      !Multiple partially filled lakes
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
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
         4, 0,  2, 16, 0, &
         1, 3,  0,  0, 0 /), &
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
      call setup_lake_for_fraction_calculation(lakes, &
                                               cell_pixel_counts, &
                                               binary_lake_mask, &
                                               lake_properties, &
                                               pixel_numbers, &
                                               pixels, &
                                               lake_pixel_counts_field, &
                                               corresponding_surface_cell_lat_index, &
                                               corresponding_surface_cell_lon_index, &
                                               nlat_lake,nlon_lake, &
                                               nlat_surface,nlon_surface)
      do lake_number = 1,9
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (lake_pixel_mask(i,j) == lake_number) then
              pixel_number = pixel_numbers(i,j)
              working_pixel => pixels(pixel_number)%pixel_pointer
              call add_pixel(lake_properties(lake_number)%lake_properties_pointer,&
                             working_pixel,pixels,lake_pixel_counts_field)
            end if
          end do
        end do
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field, &
                         nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(potential_lake_pixel_mask)
      deallocate(binary_lake_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(cell_mask)
      deallocate(mask)
      deallocate(pixel_numbers)
      do i = 1,size(pixels)
        deallocate(pixels(i)%pixel_pointer)
      end do
      deallocate(pixels)
      do i = 1,size(lake_properties)
        do j = 1,size(lake_properties(i)%lake_properties_pointer%cell_list)
          call clean_lake_cell(lake_properties(i)%lake_properties_pointer%&
                               &cell_list(j)%lake_cell_pointer)
          deallocate(lake_properties(i)%lake_properties_pointer%&
                     &cell_list(j)%lake_cell_pointer)
        end do
        deallocate(lake_properties(i)%lake_properties_pointer%&
                   &cell_list)
        deallocate(lake_properties(i)%lake_properties_pointer)
      end do
      deallocate(lake_properties)
end subroutine testLakeFractionCalculationTest18

subroutine testLakeFractionCalculationTest19
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   type(lakepropertiespointer), dimension(:), pointer :: lake_properties
   type(lakeproperties), pointer :: lake
   integer, dimension(:,:), pointer :: pixel_numbers
   type(pixelpointer), dimension(:), pointer :: pixels
   type(pixel), pointer :: working_pixel
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
   integer :: pixel_number
   integer :: lat,lon
   integer :: i,j

      !Multiple partially filled lakes
      nlat_lake = 20
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
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
         16, 0,  0,  16, 1, &
         0, 16, 16, 0,  6 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_intermediate_lake_pixel_counts_field(nlat_surface,nlon_surface))
      expected_intermediate_lake_pixel_counts_field = transpose(reshape((/   &
          0, 14,  12,  0,  0, &
         0, 0,  0,  14, 14, &
         0, 12, 11, 16, 0, &
         13, 0,  0,  15, 0, &
         0, 10, 11, 0,  0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_pixel_counts_field_after_cycle(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field_after_cycle = transpose(reshape((/   &
          1, 16,  16,  0,  0, &
         0, 2,  2,  16, 16, &
         1, 16, 16, 16, 0, &
         16, 0,  0,  16, 0, &
         1, 16, 16, 1,  2 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_intermediate_lake_pixel_counts_field_two(nlat_surface,nlon_surface))
      expected_intermediate_lake_pixel_counts_field_two = transpose(reshape((/   &
          0, 15,  12,  0,  0, &
         0, 0,  0,  15, 15, &
         0, 10, 11, 16, 0, &
         13, 0,  0,  15, 0, &
         0, 10, 10, 0,  0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_intermediate_lake_pixel_counts_field_three(nlat_surface,nlon_surface))
      expected_intermediate_lake_pixel_counts_field_three = transpose(reshape((/   &
          0, 14,  13,  0,  0, &
         0, 0,  0,  15, 16, &
         0, 10, 12, 16, 0, &
         12, 0,  0,  15, 0, &
         0, 7, 12, 0,  0 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_pixel_counts_field_after_second_cycle(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field_after_second_cycle = transpose(reshape((/   &
          2, 16,  16,  0,  0, &
         0, 1,  2,  16, 16, &
         1, 16, 16, 16, 0, &
         16, 0,  1,  16, 1, &
         1, 16, 16, 0,  1 /), &
         (/nlon_surface,nlat_surface/)))
      allocate(expected_lake_pixel_counts_field_after_third_cycle(nlat_surface,nlon_surface))
      expected_lake_pixel_counts_field_after_third_cycle = transpose(reshape((/   &
          0, 16,  16,  0,  0, &
         0, 0,  2,  16, 16, &
         1, 16, 16, 16, 0, &
         16, 0,  4,  16, 0, &
         0, 16, 16, 1,  2 /), &
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
      call setup_lake_for_fraction_calculation(lakes, &
                                               cell_pixel_counts, &
                                               binary_lake_mask, &
                                               lake_properties, &
                                               pixel_numbers, &
                                               pixels, &
                                               lake_pixel_counts_field, &
                                               corresponding_surface_cell_lat_index, &
                                               corresponding_surface_cell_lon_index, &
                                               nlat_lake,nlon_lake, &
                                               nlat_surface,nlon_surface)
      do lake_number = 1,2
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (lake_pixel_mask(i,j) == lake_number) then
              pixel_number = pixel_numbers(i,j)
              working_pixel => pixels(pixel_number)%pixel_pointer
              call add_pixel(lake_properties(lake_number)%lake_properties_pointer,&
                             working_pixel,pixels,lake_pixel_counts_field)
            end if
          end do
        end do
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,&
                         nlat_surface,nlon_surface)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(2,4)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(2,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(4,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(5,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(4,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(10,4)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(10,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(10,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(11,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(16,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(16,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(20,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(11,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(10,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(9,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(9,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(9,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(8,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(7,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(6,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      call assert_equals(lake_pixel_counts_field, &
                         expected_intermediate_lake_pixel_counts_field, &
                         nlat_surface,nlon_surface)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(2,4)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(2,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(4,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(5,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(4,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(10,4)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(10,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(10,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(11,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(16,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(16,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(20,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(11,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(10,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(9,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(9,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(9,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(8,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(7,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(6,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field_after_cycle, &
                         nlat_surface,nlon_surface)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(2,4)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(2,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(4,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(5,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(4,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(10,4)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(10,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(10,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(11,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(16,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(16,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(20,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(11,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(10,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(9,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(9,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(9,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(8,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(7,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(6,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      call assert_equals(lake_pixel_counts_field, &
                         expected_intermediate_lake_pixel_counts_field_two, &
                         nlat_surface,nlon_surface)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(2,4)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(2,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(4,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(5,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(4,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(10,4)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(10,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(10,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(11,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(16,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(16,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(20,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(11,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(10,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(9,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(9,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(9,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(8,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(7,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(6,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field_after_second_cycle, &
                         nlat_surface,nlon_surface)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(2,4)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(2,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(4,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(5,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(4,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(10,4)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(10,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(10,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(11,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(16,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(16,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(20,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(11,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(10,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(9,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(9,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(9,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(8,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(7,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(6,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      call assert_equals(lake_pixel_counts_field, &
                         expected_intermediate_lake_pixel_counts_field_three, &
                         nlat_surface,nlon_surface)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(2,4)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(2,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(4,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(5,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(4,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(3,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(10,4)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(10,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(10,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(11,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(16,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,17)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,18)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(14,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,11)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(15,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(16,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(17,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,10)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(20,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(19,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(18,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(11,9)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(10,8)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(9,7)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(9,6)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(9,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(8,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(7,13)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(6,12)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field_after_third_cycle, &
                         nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(potential_lake_pixel_mask)
      deallocate(binary_lake_mask)
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
      deallocate(pixel_numbers)
      do i = 1,size(pixels)
        deallocate(pixels(i)%pixel_pointer)
      end do
      deallocate(pixels)
      do i = 1,size(lake_properties)
        do j = 1,size(lake_properties(i)%lake_properties_pointer%cell_list)
          call clean_lake_cell(lake_properties(i)%lake_properties_pointer%&
                               &cell_list(j)%lake_cell_pointer)
          deallocate(lake_properties(i)%lake_properties_pointer%&
                     &cell_list(j)%lake_cell_pointer)
        end do
        deallocate(lake_properties(i)%lake_properties_pointer%&
                   &cell_list)
        deallocate(lake_properties(i)%lake_properties_pointer)
      end do
      deallocate(lake_properties)
end subroutine testLakeFractionCalculationTest19

subroutine testLakeFractionCalculationTest20
   use l2_calculate_lake_fractions_mod
   type(lakeinputpointer), dimension(:), pointer :: lakes
   integer :: lake_number
   type(lakepropertiespointer), dimension(:), pointer :: lake_properties
   type(lakeproperties), pointer :: lake
   integer, dimension(:,:), pointer :: pixel_numbers
   type(pixelpointer), dimension(:), pointer :: pixels
   type(pixel), pointer :: working_pixel
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
   logical,dimension(:,:), pointer :: binary_lake_mask
   integer,dimension(:,:), pointer :: expected_lake_pixel_counts_field
   integer,dimension(:,:), pointer :: expected_intermediate_lake_pixel_counts_field
   real(dp),dimension(:,:), pointer :: expected_lake_fractions_field
   logical,dimension(:,:), pointer :: cell_mask
   logical, dimension(:,:), pointer :: mask
   integer :: nlat_lake, nlon_lake
   integer :: nlat_surface, nlon_surface
   integer :: npixels, counter
   integer :: pixel_number
   integer :: lat,lon
   integer :: i,j

      !Multiple partially filled lakes with uneven cell sizes
      nlat_lake = 15
      nlon_lake = 20
      nlat_surface = 5
      nlon_surface = 5
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
      call setup_lake_for_fraction_calculation(lakes, &
                                               cell_pixel_counts, &
                                               binary_lake_mask, &
                                               lake_properties, &
                                               pixel_numbers, &
                                               pixels, &
                                               lake_pixel_counts_field, &
                                               corresponding_surface_cell_lat_index, &
                                               corresponding_surface_cell_lon_index, &
                                               nlat_lake,nlon_lake, &
                                               nlat_surface,nlon_surface)
      do lake_number = 1,2
        do i = 1,nlat_lake
          do j = 1,nlon_lake
            if (lake_pixel_mask(i,j) == lake_number) then
              pixel_number = pixel_numbers(i,j)
              working_pixel => pixels(pixel_number)%pixel_pointer
              call add_pixel(lake_properties(lake_number)%lake_properties_pointer,&
                             working_pixel,pixels,lake_pixel_counts_field)
            end if
          end do
        end do
      end do
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field, &
                         nlat_surface,nlon_surface)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(12,1)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(12,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(12,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,1)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(6,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(6,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(4,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(3,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call remove_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      call assert_equals(lake_pixel_counts_field, &
                         expected_intermediate_lake_pixel_counts_field, &
                         nlat_surface,nlon_surface)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(12,1)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(12,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(12,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,3)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,2)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(1)%lake_properties_pointer
      pixel_number = pixel_numbers(13,1)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(6,19)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(6,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(4,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      lake => lake_properties(2)%lake_properties_pointer
      pixel_number = pixel_numbers(3,20)
      working_pixel => pixels(pixel_number)%pixel_pointer
      call add_pixel(lake,working_pixel,pixels,lake_pixel_counts_field)
      call assert_equals(lake_pixel_counts_field, &
                         expected_lake_pixel_counts_field,&
                         nlat_surface,nlon_surface)
      do i = 1,size(lakes)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &potential_lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &lake_pixel_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lat)
        deallocate(lakes(i)%lake_input_pointer%&
                   &cell_coords_list_lon)
        deallocate(lakes(i)%lake_input_pointer)
      end do
      deallocate(lakes)
      deallocate(lake_pixel_mask)
      deallocate(potential_lake_pixel_mask)
      deallocate(binary_lake_mask)
      deallocate(cell_pixel_counts)
      deallocate(corresponding_surface_cell_lat_index)
      deallocate(corresponding_surface_cell_lon_index)
      deallocate(lake_pixel_counts_field)
      deallocate(expected_lake_pixel_counts_field)
      deallocate(expected_lake_fractions_field)
      deallocate(expected_intermediate_lake_pixel_counts_field)
      deallocate(cell_mask)
      deallocate(mask)
      deallocate(pixel_numbers)
      do i = 1,size(pixels)
        deallocate(pixels(i)%pixel_pointer)
      end do
      deallocate(pixels)
      do i = 1,size(lake_properties)
        do j = 1,size(lake_properties(i)%lake_properties_pointer%cell_list)
          call clean_lake_cell(lake_properties(i)%lake_properties_pointer%&
                               &cell_list(j)%lake_cell_pointer)
          deallocate(lake_properties(i)%lake_properties_pointer%&
                     &cell_list(j)%lake_cell_pointer)
        end do
        deallocate(lake_properties(i)%lake_properties_pointer%&
                   &cell_list)
        deallocate(lake_properties(i)%lake_properties_pointer)
      end do
      deallocate(lake_properties)
end subroutine testLakeFractionCalculationTest20

end module l2_calculate_lake_fractions_test_mod
