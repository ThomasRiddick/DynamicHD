module cotat_parameters_mod
    implicit none
    
    !> The minimum upstream flow path; the minimum distance through the cell that the
    !! highest flow path leading to a given exit pixel must have passed through
    real :: MUFP = 1.5
    !> The miminum additional area a downstream pixel must have gathered before
    !! the cell it is in can be chosen as the downstream cell of a given centre cell
    integer :: area_threshold = 9
    !> Check for sinks within a cell; this can be turned off for the sake of efficiency
    !! when using a set of fine river directions known not to contain sinks
    logical :: run_check_for_sinks = .True.
    !> The maximum number of cells to let a path go for between outlet pixels for the
    !! yamazaki algorithm
    integer :: yamazaki_max_range = 9

contains

    !> Read in and set the parameters from a given file
    subroutine read_cotat_parameters_namelist(filename)
        character(len=*) :: filename
        namelist /cotat_parameters/ MUFP,area_threshold,run_check_for_sinks,yamazaki_max_range
            open(unit=1,file=filename)
            read(unit=1,nml=cotat_parameters)
            close(1)
    end subroutine read_cotat_parameters_namelist

end module cotat_parameters_mod
