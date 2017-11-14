module cotat_parameters_mod
    implicit none
    
    real :: MUFP = 1.5
    integer :: area_threshold = 9
    logical :: run_check_for_sinks = .True.
    integer :: yamazaki_max_range = 9

contains

    subroutine read_cotat_parameters_namelist(filename)
        character(len=*) :: filename
        namelist /cotat_parameters/ MUFP,area_threshold,run_check_for_sinks,yamazaki_max_range
            open(unit=1,file=filename)
            read(unit=1,nml=cotat_parameters)
            close(1)
    end subroutine read_cotat_parameters_namelist

end module cotat_parameters_mod
