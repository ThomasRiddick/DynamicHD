program accumulate_flow_icon_simple_interface

use netcdf
use accumulate_flow_mod
use check_return_code_netcdf_mod
implicit none

    integer, parameter :: MAX_NAME_LENGTH = 1000
    integer :: num_args
    integer :: ncells
    integer :: clat_varid, clon_varid
    integer :: clat_vert_varid, clon_vert_varid
    integer :: cumulative_flow_varid
    integer :: dimid, ncid, varid, vertices_dimid
    integer :: ncells_dimid
    integer, dimension(2) :: dimids
    integer, dimension(:), pointer  :: rdirs
    integer, dimension(:,:), pointer  ::bifurcated_rdirs
    integer, dimension(:), pointer  :: cumulative_flow
    real(kind=double_precision), dimension(:), allocatable :: cell_lats
    real(kind=double_precision), dimension(:), allocatable :: cell_lons
    real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lats_raw
    real(kind=double_precision), dimension(:,:), allocatable :: cell_vertices_lons_raw
    integer, dimension(:,:), pointer :: cell_neighbors
    character(len=MAX_NAME_LENGTH) :: arg
    character(len=MAX_NAME_LENGTH) :: grid_params_filename
    character(len=MAX_NAME_LENGTH) :: input_rdirs_filename
    character(len=MAX_NAME_LENGTH) :: output_cumulative_flow_filename
    character(len=MAX_NAME_LENGTH) :: input_rdirs_fieldname
    character(len=MAX_NAME_LENGTH) :: output_cumulative_flow_fieldname
    character(len=MAX_NAME_LENGTH) :: input_bifurcated_rdirs_filename
    character(len=MAX_NAME_LENGTH) :: input_bifurcated_rdirs_fieldname

        num_args = command_argument_count()
        if (num_args == 1) then
          call get_command_argument(1,value=arg)
          if (arg == '-h' .or. arg == '--help') then
            call print_help()
            stop
          end if
        end if
        if (num_args /= 5 .and. num_args /=7) then
          write(*,*) "Wrong number of command line arguments given"
          call print_usage()
          stop
        end if
        call get_command_argument(1,value=grid_params_filename)
        call get_command_argument(2,value=input_rdirs_filename)
        call get_command_argument(3,value=output_cumulative_flow_filename)
        call get_command_argument(4,value=input_rdirs_fieldname)
        call get_command_argument(5,value=output_cumulative_flow_fieldname)
        if (num_args == 7) then
          call get_command_argument(6,value=input_bifurcated_rdirs_filename)
          call get_command_argument(7,value=input_bifurcated_rdirs_fieldname)
        end if

        write(*,*) "Reading Icosohedral Grid Parameters"
        call check_return_code(nf90_open(grid_params_filename,nf90_nowrite,ncid))

        call check_return_code(nf90_inq_dimid(ncid,'cell',dimid))
        call check_return_code(nf90_inquire_dimension(ncid,dimid,len=ncells))

        allocate(cell_lats(ncells))
        call check_return_code(nf90_inq_varid(ncid,"clat",varid))
        call check_return_code(nf90_get_var(ncid,varid,cell_lats))

        allocate(cell_lons(ncells))
        call check_return_code(nf90_inq_varid(ncid,"clon",varid))
        call check_return_code(nf90_get_var(ncid,varid,cell_lons))

        allocate(cell_vertices_lats_raw(3,ncells))
        call check_return_code(nf90_inq_varid(ncid,"clat_vertices",varid))
        call check_return_code(nf90_get_var(ncid,varid,cell_vertices_lats_raw))

        allocate(cell_vertices_lons_raw(3,ncells))
        call check_return_code(nf90_inq_varid(ncid,"clon_vertices",varid))
        call check_return_code(nf90_get_var(ncid,varid,cell_vertices_lons_raw))

        allocate(cell_neighbors(ncells,3))
        call check_return_code(nf90_inq_varid(ncid,"neighbor_cell_index",varid))
        call check_return_code(nf90_get_var(ncid,varid,cell_neighbors))

        call check_return_code(nf90_close(ncid))

        write(*,*) "Reading Icon River Directions"
        call check_return_code(nf90_open(input_rdirs_filename,nf90_nowrite,ncid))
        allocate(rdirs(ncells))
        call check_return_code(nf90_inq_varid(ncid,input_rdirs_fieldname,varid))
        call check_return_code(nf90_get_var(ncid,varid,rdirs))
        call check_return_code(nf90_close(ncid))

        if (num_args == 7) then
          write(*,*) "Reading Bifurcated Icon River Directions"
          call check_return_code(nf90_open(input_bifurcated_rdirs_filename,nf90_nowrite,ncid))
          allocate(bifurcated_rdirs(ncells,11))
          call check_return_code(nf90_inq_varid(ncid,input_bifurcated_rdirs_fieldname,varid))
          call check_return_code(nf90_get_var(ncid,varid,rdirs))
          call check_return_code(nf90_close(ncid))
        end if

        write(*,*) "Generating Cumulative Flow"
        allocate(cumulative_flow(ncells))
        if (num_args == 7) then
          call accumulate_flow_icon_single_index(cell_neighbors, &
                                                 rdirs, cumulative_flow, &
                                                 bifurcated_rdirs)
        else
          call accumulate_flow_icon_single_index(cell_neighbors, &
                                                 rdirs, cumulative_flow)
        end if

        write(*,*) "Writing Icosohedral Cumulative Flow File"
        call check_return_code(nf90_create(output_cumulative_flow_filename,nf90_netcdf4,ncid))
        call check_return_code(nf90_def_dim(ncid,"vertices",3,vertices_dimid))
        call check_return_code(nf90_def_dim(ncid,"ncells",ncells,ncells_dimid))
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
        call check_return_code(nf90_def_var(ncid, &
                                            output_cumulative_flow_fieldname, &
                                            nf90_int,ncells_dimid,cumulative_flow_varid))
        call check_return_code(nf90_put_att(ncid, cumulative_flow_varid,"coordinates","clat clon") )
        call check_return_code(nf90_put_att(ncid, cumulative_flow_varid,"grid_type","unstructured") )
        call check_return_code(nf90_put_att(ncid, cumulative_flow_varid,"long_name","next cell index") )
        call check_return_code(nf90_enddef(ncid))
        call check_return_code(nf90_put_var(ncid,cumulative_flow_varid,cumulative_flow))
        call check_return_code(nf90_put_var(ncid,clat_vert_varid,cell_vertices_lats_raw,start=(/1,1/), &
                                            count=(/3,ncells/)))
        call check_return_code(nf90_put_var(ncid,clon_vert_varid,cell_vertices_lons_raw,start=(/1,1/), &
                                            count=(/3,ncells/)))
        call check_return_code(nf90_put_var(ncid,clat_varid,cell_lats))
        call check_return_code(nf90_put_var(ncid,clon_varid,cell_lons))
        call check_return_code(nf90_close(ncid))

  contains

  subroutine print_usage()
    write(*,*) "Usage: "
    write(*,*) "Accumulated Flow Calculation Algorithm"
    write(*,*) "./Accumulate_Flow_Icon_Simple_Interface_Exec [grid_params_filename]" // &
               " [input_rdirs_filename] [output_cumulative_flow_filename]" // &
               " [input_rdirs_fieldname] [output_cumulative_flow_fieldname]"
  end subroutine print_usage

  subroutine print_help()
    call print_usage()
    write(*,*) "Calculate the cumulative flow on an ICON icosehedral grid."
    write(*,*) "Arguments:"
    write(*,*) "grid_params_filename - Full path to the grid description file for the ICON icosohedral"
    write(*,*) " grid"
    write(*,*) "input_rdirs_filename - Full path to the output next cell index file path; the"
    write(*,*) " next cell index values are the ICON equivalent of river directions"
    write(*,*) "cumulative_flow_filename - Full path to the target file to write the output "
    write(*,*) " cumulative flow to"
    write(*,*) "input_rdirs_fieldname - Field name for the next cell index field in the specified file."
    write(*,*) "cumulative_flow_fieldname - Field name for the output cumulative flow field in the"
    write(*,*) " specified file"
  end subroutine print_help

end program accumulate_flow_icon_simple_interface
