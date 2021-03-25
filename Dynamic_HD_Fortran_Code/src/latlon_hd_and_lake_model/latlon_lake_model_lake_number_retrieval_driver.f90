program latlon_lake_model_number_retrieval_driver

  use latlon_lake_model_retrieve_lake_numbers
  use parameters_mod
  implicit none

  integer :: num_args

  character(len = max_name_length) :: lake_parameters_filename
  character(len = max_name_length) :: output_lake_numbers_filename


    num_args = command_argument_count()
    if (num_args /= 2) then
      write(*,*) "Wrong number of command line arguments given"
      stop
    end if
    call get_command_argument(1,value=lake_parameters_filename)
    call get_command_argument(2,value=output_lake_numbers_filename)
    call retrieve_lake_numbers(lake_parameters_filename,output_lake_numbers_filename)

end program latlon_lake_model_number_retrieval_driver
