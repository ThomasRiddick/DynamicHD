program icon_to_latlon_landsea_downscaler_simple_interface

  use netcdf
  use icon_to_latlon_landsea_downscaler
  use precision_mod
  implicit none

  integer, parameter :: MAX_NAME_LENGTH = 1000
  integer, dimension(:), allocatable :: coarse_landsea_mask
  integer, dimension(:,:), allocatable :: coarse_to_fine_cell_numbers_mapping_raw
  integer, dimension(:,:), allocatable :: coarse_to_fine_cell_numbers_mapping
  logical, dimension(:,:), pointer :: fine_landsea_mask
  integer, dimension(:,:), pointer :: fine_landsea_mask_integer
  integer, dimension(:,:), pointer :: fine_landsea_mask_integer_processed
  real(kind=double_precision), dimension(:), allocatable :: pixel_center_lats
  real(kind=double_precision), dimension(:), allocatable :: pixel_center_lons
  integer :: varid, ncid, dimid, cells_varid, lat_dimid, lon_dimid
  integer :: lat_varid, lon_varid
  integer, dimension(2) :: dimids
  integer :: nlat,nlon
  integer :: num_args
  integer :: ncells_coarse
  integer :: i,j

  character(len=MAX_NAME_LENGTH) :: input_coarse_to_fine_cell_numbers_mapping_filename
  character(len=MAX_NAME_LENGTH) :: input_coarse_landsea_mask_filename
  character(len=MAX_NAME_LENGTH) :: output_fine_landsea_mask_filename
  character(len=MAX_NAME_LENGTH) :: input_coarse_to_fine_cell_numbers_mapping_fieldname
  character(len=MAX_NAME_LENGTH) :: input_coarse_landsea_mask_fieldname
  character(len=MAX_NAME_LENGTH) :: output_fine_landsea_mask_fieldname

    num_args = command_argument_count()
    if (num_args /= 6) then
      write(*,*) "Wrong number of command line arguments given"
      stop
    end if
    call get_command_argument(1,value=input_coarse_to_fine_cell_numbers_mapping_filename)
    call get_command_argument(2,value=input_coarse_landsea_mask_filename)
    call get_command_argument(3,value=output_fine_landsea_mask_filename)
    call get_command_argument(4,value=input_coarse_to_fine_cell_numbers_mapping_fieldname)
    call get_command_argument(5,value=input_coarse_landsea_mask_fieldname)
    call get_command_argument(6,value=output_fine_landsea_mask_fieldname)

    write(*,*) "Reading Coarse to Fine Cell Numbers Mapping"
    call check_return_code(nf90_open(input_coarse_to_fine_cell_numbers_mapping_filename,&
                                     nf90_nowrite,ncid))
    call check_return_code(nf90_inq_varid(ncid,input_coarse_to_fine_cell_numbers_mapping_fieldname,varid))
    call check_return_code(nf90_inquire_variable(ncid,varid,dimids=dimids))
    call check_return_code(nf90_inquire_dimension(ncid,dimids(1),len=nlon))
    call check_return_code(nf90_inquire_dimension(ncid,dimids(2),len=nlat))
    allocate(coarse_to_fine_cell_numbers_mapping_raw(nlon,nlat))
    call check_return_code(nf90_get_var(ncid,varid,coarse_to_fine_cell_numbers_mapping_raw))
    allocate(pixel_center_lats(nlat))
    call check_return_code(nf90_inq_varid(ncid,"lat",varid))
    call check_return_code(nf90_get_var(ncid,varid,pixel_center_lats))
    allocate(pixel_center_lons(nlon))
    call check_return_code(nf90_inq_varid(ncid,"lon",varid))
    call check_return_code(nf90_get_var(ncid, varid,pixel_center_lons))
    call check_return_code(nf90_close(ncid))

    allocate(coarse_to_fine_cell_numbers_mapping(nlat,nlon))
    do j = 1,nlon
      do i = 1,nlat
        coarse_to_fine_cell_numbers_mapping(i,j) = coarse_to_fine_cell_numbers_mapping_raw(j,i)
      end do
    end do

    write(*,*) "Reading Icosohedral Coarse Landsea Mask"
    call check_return_code(nf90_open(input_coarse_landsea_mask_filename,nf90_nowrite,ncid))

    call check_return_code(nf90_inq_dimid(ncid,'cell',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=ncells_coarse))

    allocate(coarse_landsea_mask(ncells_coarse))
    call check_return_code(nf90_inq_varid(ncid,input_coarse_landsea_mask_fieldname,varid))
    call check_return_code(nf90_get_var(ncid,varid,coarse_landsea_mask))

    call check_return_code(nf90_close(ncid))

    fine_landsea_mask => downscale_coarse_icon_landsea_mask_to_fine_latlon_grid((coarse_landsea_mask == 1), &
                                                                                coarse_to_fine_cell_numbers_mapping)
    allocate(fine_landsea_mask_integer(nlat,nlon))
    where (fine_landsea_mask)
      fine_landsea_mask_integer = 1
    else where
      fine_landsea_mask_integer = 0
    end where

    allocate(fine_landsea_mask_integer_processed(nlon,nlat))
    do j = 1,nlon
      do i = 1,nlat
        fine_landsea_mask_integer_processed(j,i) = fine_landsea_mask_integer(i,j)
      end do
    end do

    call check_return_code(nf90_create(output_fine_landsea_mask_filename,nf90_netcdf4,ncid))
    call check_return_code(nf90_def_dim(ncid,"lat",nlat,lat_dimid))
    call check_return_code(nf90_def_dim(ncid,"lon",nlon,lon_dimid))
    call check_return_code(nf90_def_var(ncid,"lat",nf90_double,lat_dimid,lat_varid))
    call check_return_code(nf90_def_var(ncid,"lon",nf90_double,lon_dimid,lon_varid))
    dimids = (/lat_dimid,lon_dimid/)
    call check_return_code(nf90_def_var(ncid,output_fine_landsea_mask_fieldname,nf90_int,&
                                        dimids,cells_varid))
    call check_return_code(nf90_enddef(ncid))
    call check_return_code(nf90_put_var(ncid,cells_varid,fine_landsea_mask_integer))
    call check_return_code(nf90_put_var(ncid,lat_varid,pixel_center_lats))
    call check_return_code(nf90_put_var(ncid,lon_varid,pixel_center_lons))
    call check_return_code(nf90_close(ncid))
    deallocate(fine_landsea_mask)
    deallocate(coarse_landsea_mask)
    deallocate(coarse_to_fine_cell_numbers_mapping)
contains

  subroutine check_return_code(return_code)
    integer, intent(in) :: return_code
      if(return_code /= nf90_noerr) then
        print *,trim(nf90_strerror(return_code))
        stop
      end if
  end subroutine check_return_code

end program icon_to_latlon_landsea_downscaler_simple_interface

