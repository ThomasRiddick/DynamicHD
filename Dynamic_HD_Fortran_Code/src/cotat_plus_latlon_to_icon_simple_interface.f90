program cotat_plus_latlon_to_icon_simple_interface

  use netcdf
  use cotat_plus
  implicit none

  integer, parameter :: MAX_NAME_LENGTH = 1000
  integer :: ncid
  integer :: varid, dimid, ncells_dimid, vertices_dimid
  integer :: rdirs_varid, clat_varid, clon_varid, clat_vert_varid, clon_vert_varid
  integer, dimension(:,:), allocatable :: input_fine_rdirs
  integer, dimension(:,:), allocatable :: input_fine_total_cumulative_flow
  integer, dimension(:), allocatable :: output_coarse_next_cell_index
  real(kind=double_precision), dimension(:), allocatable :: pixel_center_lats
  real(kind=double_precision), dimension(:), allocatable :: pixel_center_lons
  real(kind=double_precision), dimension(:), allocatable :: cell_lats
  real(kind=double_precision), dimension(:), allocatable :: cell_lons
  real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lats
  real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lons
  character(len=MAX_NAME_LENGTH) :: cotat_parameters_filename
  character(len=MAX_NAME_LENGTH) :: input_fine_rdirs_filename
  character(len=MAX_NAME_LENGTH) :: input_fine_total_cumulative_flow_filename
  character(len=MAX_NAME_LENGTH) :: coarse_grid_params_filename
  character(len=MAX_NAME_LENGTH) :: output_rdirs_filename
  character(len=MAX_NAME_LENGTH) :: input_fine_rdirs_fieldname
  character(len=MAX_NAME_LENGTH) :: input_fine_total_cumulative_flow_fieldname
  character(len=MAX_NAME_LENGTH) :: output_rdirs_fieldname
  integer, dimension(:,:), pointer :: cell_neighbors
  integer, dimension(2) :: dimids
  integer :: ncells_coarse
  integer :: num_args

    num_args = command_argument_count()
    if (num_args /= 8 ) then
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

    call check_return_code(nf90_open(input_fine_rdirs_filename,nf90_nowrite,ncid))
    call check_return_code(nf90_inq_varid(ncid,input_fine_rdirs_fieldname,varid))
    call check_return_code(nf90_get_var(ncid, varid,input_fine_rdirs))
    call check_return_code(nf90_inq_varid(ncid,"lat",varid))
    call check_return_code(nf90_get_var(ncid, varid,pixel_center_lats))
    call check_return_code(nf90_inq_varid(ncid,"lon",varid))
    call check_return_code(nf90_get_var(ncid, varid,pixel_center_lons))
    call check_return_code(nf90_close(ncid))

    call check_return_code(nf90_open(input_fine_total_cumulative_flow_filename,nf90_nowrite,ncid))
    call check_return_code(nf90_inq_varid(ncid,input_fine_total_cumulative_flow_fieldname,varid))
    call check_return_code(nf90_get_var(ncid, varid,input_fine_total_cumulative_flow))
    call check_return_code(nf90_close(ncid))

    call check_return_code(nf90_open(coarse_grid_params_filename,nf90_nowrite,ncid))

    call check_return_code(nf90_inq_varid(ncid,"clat",varid))
    call check_return_code(nf90_get_var(ncid,varid,cell_lats))

    call check_return_code(nf90_inq_varid(ncid,"clon",varid))
    call check_return_code(nf90_get_var(ncid,varid,cell_lons))

    call check_return_code(nf90_inq_varid(ncid,"clat_vertices",varid))
    call check_return_code(nf90_get_var(ncid,varid,cell_vertices_lats))

    call check_return_code(nf90_inq_varid(ncid,"clon_vertices",varid))
    call check_return_code(nf90_get_var(ncid,varid,cell_vertices_lons))

    call check_return_code(nf90_inq_varid(ncid,"neighbor_cell_index",varid))
    call check_return_code(nf90_get_var(ncid,varid,cell_neighbors))

    call check_return_code(nf90_inq_dimid(ncid,'cell',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=ncells_coarse))

    call check_return_code(nf90_close(ncid))

    call cotat_plus_icon_icosohedral_cell_latlon_pixel(input_fine_rdirs,&
                                                       input_fine_total_cumulative_flow,&
                                                       output_coarse_next_cell_index,&
                                                       pixel_center_lats,&
                                                       pixel_center_lons,&
                                                       cell_vertices_lats, &
                                                       cell_vertices_lons, &
                                                       cell_neighbors, &
                                                       cotat_parameters_filename)

    call check_return_code(nf90_create(output_rdirs_filename,nf90_noclobber,ncid))
    call check_return_code(nf90_def_dim(ncid,"ncells",ncells_coarse,ncells_dimid))
    call check_return_code(nf90_def_dim(ncid,"vertices",3,vertices_dimid))
    call check_return_code(nf90_def_var(ncid,"clat", nf90_real,ncells_dimid,clat_varid))
    call check_return_code(nf90_def_var(ncid,"clat_bnds", nf90_real,dimids,clat_vert_varid))
    call check_return_code(nf90_put_att(ncid,clat_varid,"units","radians") )
    call check_return_code(nf90_put_att(ncid,clat_varid,"long_name","center latitude") )
    call check_return_code(nf90_put_att(ncid,clat_varid,"bounds","clat_bnds") )
    call check_return_code(nf90_def_var(ncid,"clon_bnds", nf90_real,dimids,clon_vert_varid))
    call check_return_code(nf90_def_var(ncid,"clon", nf90_real,ncells_dimid,clon_varid))
    call check_return_code(nf90_put_att(ncid,clon_varid,"units","radians") )
    call check_return_code(nf90_put_att(ncid,clon_varid,"long_name","center longitude") )
    call check_return_code(nf90_put_att(ncid,clon_varid,"bounds","clon_bnds") )
    dimids = (/ncells_dimid,vertices_dimid/)
    call check_return_code(nf90_def_var(ncid,"next_cell_index",nf90_int,ncells_dimid,rdirs_varid))
    call check_return_code(nf90_put_att(ncid, rdirs_varid,"coordinates","clat clon") )
    call check_return_code(nf90_put_att(ncid, rdirs_varid,"grid_type","unstructured") )
    call check_return_code(nf90_put_att(ncid, rdirs_varid,"long_name","next cell index") )
    call check_return_code(nf90_enddef(ncid))
    call check_return_code(nf90_put_var(ncid,rdirs_varid,output_coarse_next_cell_index))
    call check_return_code(nf90_put_var(ncid,clat_vert_varid,cell_vertices_lats))
    call check_return_code(nf90_put_var(ncid,clon_vert_varid,cell_vertices_lons))
    call check_return_code(nf90_put_var(ncid,clat_varid,cell_lats))
    call check_return_code(nf90_put_var(ncid,clon_varid,cell_lons))
    call check_return_code(nf90_close(ncid))

contains

  subroutine check_return_code(return_code)
  integer, intent(in) :: return_code
    if(return_code /= nf90_noerr) then
      print *,trim(nf90_strerror(return_code))
      stop
    end if
  end subroutine check_return_code

end program cotat_plus_latlon_to_icon_simple_interface
