program latlon_to_icon_loop_breaker_simple_interface

  use netcdf
  use break_loops_mod
  use check_return_code_netcdf_mod
  implicit none

  integer, parameter :: MAX_NAME_LENGTH = 1000
  real(kind=double_precision), parameter :: ABS_TOL_FOR_DEG = 0.00001
  real(kind=double_precision), parameter :: PI = 4.0*atan(1.0)
  integer :: ncid
  integer :: varid, dimid, ncells_dimid, vertices_dimid
  integer :: rdirs_varid, clat_varid, clon_varid, clat_vert_varid, clon_vert_varid
  real(kind=double_precision), dimension(:,:), allocatable :: input_fine_rdirs
  integer, dimension(:,:), pointer :: input_fine_rdirs_int
  integer, dimension(:,:), pointer :: input_fine_total_cumulative_flow
  real(kind=double_precision), dimension(:,:), allocatable :: input_fine_rdirs_raw
  integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow_raw
  integer, dimension(:,:), pointer :: cell_numbers_data
  integer, dimension(:,:), pointer :: cell_numbers_data_raw
  integer, dimension(:), pointer :: coarse_cumulative_flow
  integer, dimension(:), pointer :: coarse_catchments
  integer, dimension(:), pointer :: coarse_rdirs
  integer, dimension(:), pointer :: loop_nums_list
  real(kind=double_precision), dimension(:), pointer :: pixel_center_lats
  real(kind=double_precision), dimension(:), pointer :: pixel_center_lons
  real(kind=double_precision), dimension(:), allocatable :: cell_lats
  real(kind=double_precision), dimension(:), allocatable :: cell_lons
  real(kind=double_precision), dimension(:,:), pointer :: cell_vertices_lats
  real(kind=double_precision), dimension(:,:), pointer :: cell_vertices_lons
  real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lats_raw
  real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lons_raw
  character(len=MAX_NAME_LENGTH) :: arg
  character(len=MAX_NAME_LENGTH) :: input_fine_rdirs_filename
  character(len=MAX_NAME_LENGTH) :: input_fine_total_cumulative_flow_filename
  character(len=MAX_NAME_LENGTH) :: input_fine_cell_numbers_filename
  character(len=MAX_NAME_LENGTH) :: coarse_grid_params_filename
  character(len=MAX_NAME_LENGTH) :: output_rdirs_filename
  character(len=MAX_NAME_LENGTH) :: input_fine_rdirs_fieldname
  character(len=MAX_NAME_LENGTH) :: input_fine_total_cumulative_flow_fieldname
  character(len=MAX_NAME_LENGTH) :: input_fine_cell_numbers_fieldname
  character(len=MAX_NAME_LENGTH) :: output_rdirs_fieldname
  character(len=MAX_NAME_LENGTH) :: coarse_catchments_filename
  character(len=MAX_NAME_LENGTH) :: coarse_cumulative_filename
  character(len=MAX_NAME_LENGTH) :: coarse_rdirs_filename
  character(len=MAX_NAME_LENGTH) :: coarse_catchments_fieldname
  character(len=MAX_NAME_LENGTH) :: coarse_cumulative_fieldname
  character(len=MAX_NAME_LENGTH) :: coarse_rdirs_fieldname
  character(len=MAX_NAME_LENGTH) :: loop_log_filename
  character(len=MAX_NAME_LENGTH) :: dummy_string
  integer, dimension(:,:), pointer :: cell_neighbors
  integer, dimension(2) :: dimids
  integer :: ncells_coarse
  integer :: num_args
  integer :: nlat,nlon
  integer :: i,j
  integer :: loop_number
  integer :: num_loops
  integer :: readiostat

    num_args = command_argument_count()
    if (num_args == 1) then
      call get_command_argument(1,value=arg)
      if (arg == '-h' .or. arg == '--help') then
        call print_help()
        stop
      end if
    end if
    if (num_args /= 16) then
      write(*,*) "Wrong number of command line arguments given"
      call print_usage()
      stop
    end if
    call get_command_argument(1,value=input_fine_rdirs_filename)
    call get_command_argument(2,value=input_fine_total_cumulative_flow_filename)
    call get_command_argument(3,value=input_fine_cell_numbers_filename)
    call get_command_argument(4,value=coarse_grid_params_filename)
    call get_command_argument(5,value=output_rdirs_filename)
    call get_command_argument(6,value=coarse_catchments_filename)
    call get_command_argument(7,value=coarse_cumulative_filename)
    call get_command_argument(8,value=coarse_rdirs_filename)
    call get_command_argument(9,value=input_fine_rdirs_fieldname)
    call get_command_argument(10,value=input_fine_total_cumulative_flow_fieldname)
    call get_command_argument(11,value=input_fine_cell_numbers_fieldname)
    call get_command_argument(12,value=output_rdirs_fieldname)
    call get_command_argument(13,value=coarse_catchments_fieldname)
    call get_command_argument(14,value=coarse_cumulative_fieldname)
    call get_command_argument(15,value=coarse_rdirs_fieldname)
    call get_command_argument(16,value=loop_log_filename)

    write(*,*) "Reading Loops"
    open(unit=100,file=loop_log_filename,&
         access='sequential',form='formatted',action='read')
    read(100,*) dummy_string
    num_loops=0
    readiostat = 0
    do
      read(100,*,iostat=readiostat) dummy_string
      if (readiostat /= 0) exit
      num_loops = num_loops + 1
    end do
    rewind(100)
    read(100,*) dummy_string
    allocate(loop_nums_list(num_loops))
    do i = 1,num_loops
      read(100,*) loop_number
      loop_nums_list(i) = loop_number
    end do
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
    input_fine_rdirs = transpose(input_fine_rdirs_raw)
    input_fine_total_cumulative_flow = transpose(input_fine_total_cumulative_flow_raw)
    deallocate(input_fine_rdirs_raw)
    deallocate(input_fine_total_cumulative_flow_raw)

    write(*,*) "Reading Coarse to Fine Cell Number"
    call check_return_code(nf90_open(input_fine_cell_numbers_filename,nf90_nowrite,ncid))
    call check_return_code(nf90_inq_varid(ncid,input_fine_cell_numbers_fieldname,varid))
    allocate(cell_numbers_data_raw(nlon,nlat))
    call check_return_code(nf90_get_var(ncid, varid,cell_numbers_data_raw))
    call check_return_code(nf90_close(ncid))

    allocate(cell_numbers_data(nlat,nlon))
    cell_numbers_data = transpose(cell_numbers_data_raw)
    deallocate(cell_numbers_data_raw)

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

    write(*,*) "Reading Coarse River Directions"
    call check_return_code(nf90_open(coarse_rdirs_filename,nf90_nowrite,ncid))

    call check_return_code(nf90_inq_dimid(ncid,'ncells',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=ncells_coarse))

    allocate(coarse_rdirs(ncells_coarse))
    call check_return_code(nf90_inq_varid(ncid,coarse_rdirs_fieldname,varid))
    call check_return_code(nf90_get_var(ncid,varid,coarse_rdirs))

    call check_return_code(nf90_close(ncid))

    write(*,*) "Reading Coarse Catchments"
    call check_return_code(nf90_open(coarse_catchments_filename,nf90_nowrite,ncid))

    call check_return_code(nf90_inq_dimid(ncid,'ncells',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=ncells_coarse))

    allocate(coarse_catchments(ncells_coarse))
    call check_return_code(nf90_inq_varid(ncid,coarse_catchments_fieldname,varid))
    call check_return_code(nf90_get_var(ncid,varid,coarse_catchments))

    call check_return_code(nf90_close(ncid))

    write(*,*) "Reading Coarse Cumulative Flow"
    call check_return_code(nf90_open(coarse_cumulative_filename,nf90_nowrite,ncid))

    call check_return_code(nf90_inq_dimid(ncid,'ncells',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=ncells_coarse))

    allocate(coarse_cumulative_flow(ncells_coarse))
    call check_return_code(nf90_inq_varid(ncid,coarse_cumulative_fieldname,varid))
    call check_return_code(nf90_get_var(ncid,varid,coarse_cumulative_flow))

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
    allocate(input_fine_rdirs_int(size(input_fine_rdirs,1),size(input_fine_rdirs,2)))
    input_fine_rdirs_int = nint(input_fine_rdirs)
    call break_loops_icon_icosohedral_cell_latlon_pixel(coarse_rdirs, &
                                                        coarse_cumulative_flow, &
                                                        coarse_catchments, &
                                                        input_fine_rdirs_int, &
                                                        input_fine_total_cumulative_flow, &
                                                        loop_nums_list, &
                                                        cell_neighbors, &
                                                        cell_vertices_lats, &
                                                        cell_vertices_lons, &
                                                        pixel_center_lats, &
                                                        pixel_center_lons, &
                                                        cell_numbers_data)
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
    call check_return_code(nf90_put_var(ncid,rdirs_varid,coarse_rdirs))
    call check_return_code(nf90_put_var(ncid,clat_vert_varid,cell_vertices_lats_raw,start=(/1,1/), &
                                        count=(/3,ncells_coarse/)))
    call check_return_code(nf90_put_var(ncid,clon_vert_varid,cell_vertices_lons_raw,start=(/1,1/), &
                                        count=(/3,ncells_coarse/)))
    call check_return_code(nf90_put_var(ncid,clat_varid,cell_lats))
    call check_return_code(nf90_put_var(ncid,clon_varid,cell_lons))
    call check_return_code(nf90_close(ncid))
    deallocate(input_fine_rdirs)
    deallocate(input_fine_total_cumulative_flow)
    deallocate(pixel_center_lats)
    deallocate(pixel_center_lons)
    deallocate(cell_lats)
    deallocate(cell_lons)
    deallocate(cell_vertices_lats)
    deallocate(cell_vertices_lons)
    deallocate(cell_vertices_lats_raw)
    deallocate(cell_vertices_lons_raw)

  contains

  subroutine print_usage()
    write(*,*) "Usage: "
    write(*,*) "Lat-Lon to Icon Loop Breaker"
    write(*,*) "./LatLon_To_Icon_Loop_Breaker_Fortran_Exec [input_fine_rdirs_filename]" // &
               "[input_fine_total_cumulative_flow_filename] [input_fine_cell_numbers_filename]" // &
               "[coarse_grid_params_filename] [output_rdirs_filename]" // &
               "[coarse_catchments_filename] [coarse_cumulative_filename]" // &
               "[coarse_rdirs_filename] [input_fine_rdirs_fieldname]" // &
               "[input_fine_total_cumulative_flow_fieldname]" // &
               "[input_fine_cell_numbers_fieldname]" // &
               "[output_rdirs_fieldname] [coarse_catchments_fieldname]" // &
               "[coarse_cumulative_fieldname]"  // &
               "[coarse_rdirs_fieldname] [loop_log_filename]"
  end subroutine print_usage

  subroutine print_help()
    call print_usage()
    write(*,*) "Upscale river directions from the latitude-longitude grid to an ICON"
    write(*,*) "icosehedral grid."
    write(*,*) "Arguments:"
    write(*,*) "input_fine_rdirs_filename - Full path to the file containing fine lat-lon river directions"
    write(*,*) "input_fine_total_cumulative_flow_filename: Full path to the file containing the total "
    write(*,*) " cumulative flow field generated from the fine lat-lon river directions"
    write(*,*) "coarse_grid_params_filename - Full path to the grid description file for the ICON icosohedral"
    write(*,*) " grid"
    write(*,*) "output_rdirs_filename - Full path to the output next cell index file path; the"
    write(*,*) " next cell index values are the ICON equivalent of river directions"
    write(*,*) "coarse catchments filename - Full path to a file containing the prior coarse catchments"
    write(*,*) "coarse cumulative flow filename - Full path to a file containing the prior coarse"
    write(*,*) " cumulative flow"
    write(*,*) "coarse rdirs filename - Full path to a file containing the prior coarse river directions"
    write(*,*) " as a next cell index"
    write(*,*) "input_fine_rdirs_fieldname - Field name of fine lat-lon river directions in specified file"
    write(*,*) "input_fine_total_cumulative_flow_fieldname: Field name of the fine lat-lon cumulative"
    write(*,*) " flow field in the specified file"
    write(*,*) "output_rdirs_fieldname - Field name for the next cell index field in the specified file."
    write(*,*) "coarse catchments fieldname - Field name for the prior coarse catchments in the specified file"
    write(*,*) "coarse cumulative flow fieldname - Field name for the prior coarse cumulative flow in the"
    write(*,*) " specified file"
    write(*,*) "coarse rdirs fieldname - Field name for the prior coarse river directions in the"
    write(*,*) " specified file"
  end subroutine print_help

end program latlon_to_icon_loop_breaker_simple_interface
