program cotat_plus_latlon_to_icon_simple_interface

  use netcdf
  use cotat_plus
  implicit none

  integer, parameter :: MAX_NAME_LENGTH = 1000
  real(kind=double_precision), parameter :: ABS_TOL_FOR_DEG = 0.00001
  real(kind=double_precision), parameter :: PI = 4.0*atan(1.0)
  integer :: ncid
  integer :: varid, dimid, ncells_dimid, vertices_dimid, cells_varid
  integer :: lat_varid, lon_varid
  integer :: lat_dimid, lon_dimid
  integer :: rdirs_varid, clat_varid, clon_varid, clat_vert_varid, clon_vert_varid
  real, dimension(:,:), allocatable :: input_fine_rdirs
  integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
  real, dimension(:,:), allocatable :: input_fine_rdirs_raw
  integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow_raw
  integer, dimension(:), allocatable :: output_coarse_next_cell_index
  integer, dimension(:,:), pointer :: output_cell_numbers
  integer, dimension(:,:), allocatable :: output_cell_numbers_processed
  real(kind=double_precision), dimension(:), allocatable :: pixel_center_lats
  real(kind=double_precision), dimension(:), allocatable :: pixel_center_lons
  real(kind=double_precision), dimension(:), allocatable :: cell_lats
  real(kind=double_precision), dimension(:), allocatable :: cell_lons
  real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lats
  real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lons
  real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lats_raw
  real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lons_raw
  character(len=MAX_NAME_LENGTH) :: cotat_parameters_filename
  character(len=MAX_NAME_LENGTH) :: input_fine_rdirs_filename
  character(len=MAX_NAME_LENGTH) :: input_fine_total_cumulative_flow_filename
  character(len=MAX_NAME_LENGTH) :: coarse_grid_params_filename
  character(len=MAX_NAME_LENGTH) :: output_rdirs_filename
  character(len=MAX_NAME_LENGTH) :: output_cell_numbers_filename
  character(len=MAX_NAME_LENGTH) :: input_fine_rdirs_fieldname
  character(len=MAX_NAME_LENGTH) :: input_fine_total_cumulative_flow_fieldname
  character(len=MAX_NAME_LENGTH) :: output_rdirs_fieldname
  character(len=MAX_NAME_LENGTH) :: output_cell_numbers_fieldname
  integer, dimension(:,:), pointer :: cell_neighbors
  integer, dimension(2) :: dimids
  integer :: ncells_coarse
  integer :: num_args
  integer :: nlat,nlon
  integer :: i,j
  logical :: write_cell_numbers

    num_args = command_argument_count()
    if (num_args /= 8 .and. num_args /= 10) then
      write(*,*) "Wrong number of command line arguments given"
      stop
    end if
    call get_command_argument(1,value=input_fine_rdirs_filename)
    call get_command_argument(2,value=input_fine_total_cumulative_flow_filename)
    call get_command_argument(3,value=coarse_grid_params_filename)
    call get_command_argument(4,value=output_rdirs_filename)
    call get_command_argument(5,value=input_fine_rdirs_fieldname)
    call get_command_argument(6,value=input_fine_total_cumulative_flow_fieldname)
    call get_command_argument(7,value=output_rdirs_fieldname)
    call get_command_argument(8,value=cotat_parameters_filename)
    if (num_args == 10) then
      call get_command_argument(9,value=output_cell_numbers_filename)
      call get_command_argument(10,value=output_cell_numbers_fieldname)
      write_cell_numbers = .true.
    else
      write_cell_numbers = .false.
    end if

    write(*,*) "Reading Fine Lat-Lon River Directions"
    call check_return_code(nf90_open(input_fine_rdirs_filename,nf90_nowrite,ncid))
    call check_return_code(nf90_inq_varid(ncid,input_fine_rdirs_fieldname,varid))
    call check_return_code(nf90_inquire_variable(ncid,varid,dimids=dimids))
    call check_return_code(nf90_inquire_dimension(ncid,dimids(1),len=nlon))
    call check_return_code(nf90_inquire_dimension(ncid,dimids(2),len=nlat))
    allocate(input_fine_rdirs_raw(nlon,nlat))
    call check_return_code(nf90_get_var(ncid,varid,input_fine_rdirs_raw))
    allocate(pixel_center_lats(nlat))
    call check_return_code(nf90_inq_varid(ncid,"lat",varid))
    call check_return_code(nf90_get_var(ncid, varid,pixel_center_lats))
    allocate(pixel_center_lons(nlon))
    call check_return_code(nf90_inq_varid(ncid,"lon",varid))
    call check_return_code(nf90_get_var(ncid, varid,pixel_center_lons))
    call check_return_code(nf90_close(ncid))

    write(*,*) "Reading Fine Lat-Lon Cumulative Flow"
    call check_return_code(nf90_open(input_fine_total_cumulative_flow_filename,nf90_nowrite,ncid))
    call check_return_code(nf90_inq_varid(ncid,input_fine_total_cumulative_flow_fieldname,varid))
    allocate(input_fine_total_cumulative_flow_raw(nlon,nlat))
    call check_return_code(nf90_get_var(ncid, varid,input_fine_total_cumulative_flow_raw))
    call check_return_code(nf90_close(ncid))

    allocate(input_fine_rdirs(nlat,nlon))
    allocate(input_fine_total_cumulative_flow(nlat,nlon))
    do j = 1,nlon
      do i = 1,nlat
        input_fine_rdirs(i,j) = input_fine_rdirs_raw(j,i)
        input_fine_total_cumulative_flow(i,j) = input_fine_total_cumulative_flow_raw(j,i)
      end do
    end do
    deallocate(input_fine_rdirs_raw)
    deallocate(input_fine_total_cumulative_flow_raw)
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
    if(write_cell_numbers) then
      call cotat_plus_icon_icosohedral_cell_latlon_pixel(nint(input_fine_rdirs),&
                                                         input_fine_total_cumulative_flow,&
                                                         output_coarse_next_cell_index,&
                                                         pixel_center_lats,&
                                                         pixel_center_lons,&
                                                         cell_vertices_lats, &
                                                         cell_vertices_lons, &
                                                         cell_neighbors, .true., &
                                                         cotat_parameters_filename,&
                                                         output_cell_numbers)
    else
      call cotat_plus_icon_icosohedral_cell_latlon_pixel(nint(input_fine_rdirs),&
                                                         input_fine_total_cumulative_flow,&
                                                         output_coarse_next_cell_index,&
                                                         pixel_center_lats,&
                                                         pixel_center_lons,&
                                                         cell_vertices_lats, &
                                                         cell_vertices_lons, &
                                                         cell_neighbors, .true., &
                                                         cotat_parameters_filename)
    end if
    write(*,*) "Writing Icosohedral River Directions File"
    call check_return_code(nf90_create(output_rdirs_filename,nf90_netcdf4,ncid))
    call check_return_code(nf90_def_dim(ncid,"vertices",3,vertices_dimid))
    call check_return_code(nf90_def_dim(ncid,"ncells",ncells_coarse,ncells_dimid))
    dimids = (/vertices_dimid,ncells_dimid/)
    call check_return_code(nf90_def_var(ncid,"clat", nf90_double,ncells_dimid,clat_varid))
    call check_return_code(nf90_def_var(ncid,"clat_bnds", nf90_double,dimids,clat_vert_varid))
    call check_return_code(nf90_put_att(ncid,clat_varid,"units","radian") )
    call check_return_code(nf90_put_att(ncid,clat_varid,"long_name","center latitude") )
    call check_return_code(nf90_put_att(ncid,clat_varid,"standard_name","latitude") )
    call check_return_code(nf90_put_att(ncid,clat_varid,"bounds","clat_bnds") )
    call check_return_code(nf90_def_var(ncid,"clon_bnds", nf90_double,dimids,clon_vert_varid))
    call check_return_code(nf90_def_var(ncid,"clon", nf90_double,ncells_dimid,clon_varid))
    call check_return_code(nf90_put_att(ncid,clon_varid,"units","radian") )
    call check_return_code(nf90_put_att(ncid,clon_varid,"long_name","center longitude") )
    call check_return_code(nf90_put_att(ncid,clon_varid,"standard_name","longitude") )
    call check_return_code(nf90_put_att(ncid,clon_varid,"bounds","clon_bnds") )
    call check_return_code(nf90_def_var(ncid, output_rdirs_fieldname,nf90_int,ncells_dimid,rdirs_varid))
    call check_return_code(nf90_put_att(ncid, rdirs_varid,"coordinates","clat clon") )
    call check_return_code(nf90_put_att(ncid, rdirs_varid,"grid_type","unstructured") )
    call check_return_code(nf90_put_att(ncid, rdirs_varid,"long_name","next cell index") )
    call check_return_code(nf90_enddef(ncid))
    call check_return_code(nf90_put_var(ncid,rdirs_varid,output_coarse_next_cell_index))
    call check_return_code(nf90_put_var(ncid,clat_vert_varid,cell_vertices_lats_raw,start=(/1,1/), &
                                        count=(/3,ncells_coarse/)))
    call check_return_code(nf90_put_var(ncid,clon_vert_varid,cell_vertices_lons_raw,start=(/1,1/), &
                                        count=(/3,ncells_coarse/)))
    call check_return_code(nf90_put_var(ncid,clat_varid,cell_lats))
    call check_return_code(nf90_put_var(ncid,clon_varid,cell_lons))
    call check_return_code(nf90_close(ncid))
    if (write_cell_numbers) then
      write(*,*) "Writing Cell Numbers File"
      allocate(output_cell_numbers_processed(nlon,nlat))
      do j = 1,nlon
        do i = 1,nlat
          output_cell_numbers_processed(j,i) = output_cell_numbers(i,j)
        end do
      end do
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
    end if
    deallocate(input_fine_rdirs)
    deallocate(input_fine_total_cumulative_flow)
    deallocate(output_coarse_next_cell_index)
    deallocate(pixel_center_lats)
    deallocate(pixel_center_lons)
    deallocate(cell_lats)
    deallocate(cell_lons)
    deallocate(cell_vertices_lats)
    deallocate(cell_vertices_lons)
    deallocate(cell_vertices_lats_raw)
    deallocate(cell_vertices_lons_raw)
    if(write_cell_numbers) deallocate(output_cell_numbers)
contains

  subroutine check_return_code(return_code)
  integer, intent(in) :: return_code
    if(return_code /= nf90_noerr) then
      print *,trim(nf90_strerror(return_code))
      stop
    end if
  end subroutine check_return_code

end program cotat_plus_latlon_to_icon_simple_interface
