module cotat_plus_driver_mod
use cotat_plus
implicit none

contains

    !> Fortran2Python (F2Py) wrapper for the latitude longitude implementation of the COTAT+ algorithm. Input
    !! is the fine cumulative flow and fine river directions along with the file path to the COTAT+ parameters
    !! namelist file. Output is the course river directions. Also takes the bounds of those arrays as arguments.
    subroutine cotat_plus_latlon_f2py_wrapper(input_fine_river_directions,input_fine_total_cumulative_flow,&
                                              output_course_river_directions,cotat_parameters_filepath,&
                                              nlat_fine,nlon_fine,nlat_course,nlon_course)
        integer, intent(in) :: nlat_fine,nlon_fine,nlat_course,nlon_course
        integer, intent(in), dimension(nlat_fine,nlon_fine)     :: input_fine_river_directions
        integer, intent(in), dimension(nlat_fine,nlon_fine)     :: input_fine_total_cumulative_flow
        integer, intent(out), dimension(nlat_course,nlon_course) :: output_course_river_directions
        character(len=*),intent(in) :: cotat_parameters_filepath
            write (*,*) "Running COTAT+ up-scaling algorithm"
            call cotat_plus_latlon(input_fine_river_directions,input_fine_total_cumulative_flow,&
                                   output_course_river_directions,cotat_parameters_filepath)
    end subroutine

end module cotat_plus_driver_mod
