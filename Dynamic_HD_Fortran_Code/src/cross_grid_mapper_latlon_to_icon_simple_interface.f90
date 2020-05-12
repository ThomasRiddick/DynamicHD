program cross_grid_mapper_latlon_to_icon_simple_interface

  use netcdf
  use cross_grid_mapper
  use check_return_code_netcdf_mod
  implicit none

  integer :: ncid
  integer :: varid, dimid, cells_varid
  integer :: lat_varid, lon_varid
  integer :: lat_dimid, lon_dimid
  integer :: nlat,nlon
  integer :: ncells_coarse
  real(kind=double_precision), dimension(:), allocatable :: pixel_center_lats
  real(kind=double_precision), dimension(:), allocatable :: pixel_center_lons
  real(kind=double_precision), dimension(:), allocatable :: cell_lats
  real(kind=double_precision), dimension(:), allocatable :: cell_lons
  real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lats
  real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lons
  real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lats_raw
  real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lons_raw
  integer, dimension(:,:), pointer :: cell_neighbors
  integer, dimension(:), allocatable :: output_coarse_next_cell_index
  integer, dimension(:,:), allocatable :: output_cell_numbers_processed
  integer, dimension(:,:), pointer :: output_cell_numbers
  integer, dimension(2) :: dimids
  integer, parameter :: MAX_NAME_LENGTH = 1000
  real(kind=double_precision), parameter :: ABS_TOL_FOR_DEG = 0.00001
  real(kind=double_precision), parameter :: PI = 4.0*atan(1.0)
  character(len=MAX_NAME_LENGTH) :: arg
  character(len=MAX_NAME_LENGTH) :: coarse_grid_params_filename
  character(len=MAX_NAME_LENGTH) :: input_fine_orography_filename
  character(len=MAX_NAME_LENGTH) :: input_fine_orography_fieldname
  character(len=MAX_NAME_LENGTH) :: output_cell_numbers_filename
  character(len=MAX_NAME_LENGTH) :: output_cell_numbers_fieldname
  integer :: i,j
  integer :: num_args

    num_args = command_argument_count()
    if (num_args == 1) then
      call get_command_argument(1,value=arg)
      if (arg == '-h' .or. arg == '--help') then
        call print_help()
        stop
      end if
    end if
    if (num_args /= 5) then
      write(*,*) "Wrong number of command line arguments given"
      call print_usage()
      stop
    end if
    call get_command_argument(1,value=coarse_grid_params_filename)
    call get_command_argument(2,value=input_fine_orography_filename)
    call get_command_argument(3,value=input_fine_orography_fieldname)
    call get_command_argument(4,value=output_cell_numbers_filename)
    call get_command_argument(5,value=output_cell_numbers_fieldname)

    write(*,*) "Reading Fine Lat-Lon Orography (for cells bounds)"
    call check_return_code(nf90_open(input_fine_orography_filename,nf90_nowrite,ncid))
    call check_return_code(nf90_inq_varid(ncid,input_fine_orography_fieldname,varid))
    call check_return_code(nf90_inquire_variable(ncid,varid,dimids=dimids))
    call check_return_code(nf90_inquire_dimension(ncid,dimids(1),len=nlon))
    call check_return_code(nf90_inquire_dimension(ncid,dimids(2),len=nlat))
    allocate(pixel_center_lats(nlat))
    call check_return_code(nf90_inq_varid(ncid,"lat",varid))
    call check_return_code(nf90_get_var(ncid, varid,pixel_center_lats))
    allocate(pixel_center_lons(nlon))
    call check_return_code(nf90_inq_varid(ncid,"lon",varid))
    call check_return_code(nf90_get_var(ncid, varid,pixel_center_lons))
    call check_return_code(nf90_close(ncid))

    write(*,*) "Reading Icosohedral Grid Parameters"
    call check_return_code(nf90_open(coarse_grid_params_filename,nf90_nowrite,ncid))

    call check_return_code(nf90_inq_dimid(ncid,'cell',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=ncells_coarse))

    allocate(cell_lats(ncells_coarse))
    call check_return_code(nf90_inq_varid(ncid,"clat",varid))
    call check_return_code(nf90_get_var(ncid,varid,cell_lats))

    allocate(cell_lons(ncells_coarse))
    call check_return_code(nf90_inq_varid(ncid,"clon",varid))
    call check_return_code(nf90_get_var(ncid,varid,cell_lons))

    allocate(cell_vertices_lats_raw(3,ncells_coarse))
    call check_return_code(nf90_inq_varid(ncid,"clat_vertices",varid))
    call check_return_code(nf90_get_var(ncid,varid,cell_vertices_lats_raw))

    allocate(cell_vertices_lons_raw(3,ncells_coarse))
    call check_return_code(nf90_inq_varid(ncid,"clon_vertices",varid))
    call check_return_code(nf90_get_var(ncid,varid,cell_vertices_lons_raw))

    allocate(cell_neighbors(ncells_coarse,3))
    call check_return_code(nf90_inq_varid(ncid,"neighbor_cell_index",varid))
    call check_return_code(nf90_get_var(ncid,varid,cell_neighbors))

    call check_return_code(nf90_close(ncid))

    allocate(cell_vertices_lats(ncells_coarse,3))
    allocate(cell_vertices_lons(ncells_coarse,3))
    do j = 1,3
      do i=1,ncells_coarse
        cell_vertices_lats(i,j) = cell_vertices_lats_raw(j,i)*(180.0/PI)
        if(cell_vertices_lats(i,j) > 90.0 - ABS_TOL_FOR_DEG) cell_vertices_lats(i,j) = 90.0
        if(cell_vertices_lats(i,j) < -90.0 + ABS_TOL_FOR_DEG) cell_vertices_lats(i,j) = -90.0
        cell_vertices_lons(i,j) = cell_vertices_lons_raw(j,i)*(180.0/PI)
      end do
    end do
    allocate(output_coarse_next_cell_index(ncells_coarse))
    call cross_grid_mapper_latlon_to_icon(pixel_center_lats,&
                                          pixel_center_lons,&
                                          cell_vertices_lats, &
                                          cell_vertices_lons, &
                                          cell_neighbors, &
                                          output_cell_numbers, &
                                          .true.)
    write(*,*) "Writing Cell Numbers File"
    allocate(output_cell_numbers_processed(nlon,nlat))
    output_cell_numbers_processed = transpose(output_cell_numbers)
    call check_return_code(nf90_create(output_cell_numbers_filename,nf90_netcdf4,ncid))
    call check_return_code(nf90_def_dim(ncid,"lat",nlat,lat_dimid))
    call check_return_code(nf90_def_dim(ncid,"lon",nlon,lon_dimid))
    call check_return_code(nf90_def_var(ncid,"lat",nf90_double,lat_dimid,lat_varid))
    call check_return_code(nf90_def_var(ncid,"lon",nf90_double,lon_dimid,lon_varid))
    dimids = (/lon_dimid,lat_dimid/)
    call check_return_code(nf90_def_var(ncid,output_cell_numbers_fieldname,nf90_int,&
                                        dimids,cells_varid))
    call check_return_code(nf90_enddef(ncid))
    call check_return_code(nf90_put_var(ncid,cells_varid,output_cell_numbers_processed))
    call check_return_code(nf90_put_var(ncid,lat_varid,pixel_center_lats))
    call check_return_code(nf90_put_var(ncid,lon_varid,pixel_center_lons))
    call check_return_code(nf90_close(ncid))

  contains

    subroutine print_usage()
    write(*,*) "Usage: "
    write(*,*) "Lat-Lon to Icon Cross Grid Mapper"
    write(*,*) "./LatLon_To_Icon_Cross_Grid_Mapper_Simple_Interface  [coarse_grid_params_filename]" // &
               " [input_fine_orography_filename] [input_fine_orography_fieldname]" // &
               " [output_cell_numbers_filename] [output_cell_numbers_fieldname]"
  end subroutine print_usage

  subroutine print_help()
    call print_usage()
    write(*,*) "Generate a mapping from an latitude-longitude grid to a ICON icosohedral grid"
    write(*,*) "Arguments:"
    write(*,*) "coarse_grid_params_filename - Full path to the grid description file for the ICON icosohedral"
    write(*,*) " grid"
    write(*,*) "input_fine_orography_filename - Full path to an orography input file on the fine"
    write(*,*) " latitude-longitude grid - this is only used to determine the parameters of the"
    write(*,*) " latitude-longitude grid itself and not for the actual height data"
    write(*,*) "input_fine_orography_fieldname  - Field name of the orography within the specified file"
    write(*,*) "output_cell_numbers_filename - Full path for a file of the mapping to lat-lon cells"
    write(*,*) " to ICON icosohedral files"
    write(*,*) "output_cell_numbers_fieldname  - Field name for the mapping within the specified file"
  end subroutine print_help

end program cross_grid_mapper_latlon_to_icon_simple_interface
