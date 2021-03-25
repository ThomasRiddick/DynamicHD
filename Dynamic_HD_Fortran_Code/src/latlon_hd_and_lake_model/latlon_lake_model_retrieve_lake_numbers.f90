module latlon_lake_model_retrieve_lake_numbers

use latlon_lake_model_mod
use latlon_lake_model_io_mod
use parameters_mod

implicit none

type(lakeparameters), pointer ::  global_lake_parameters
type(lakeprognostics), pointer :: global_lake_prognostics
type(lakefields), pointer ::      global_lake_fields

contains

subroutine retrieve_lake_numbers(lake_parameters_filename,output_lake_numbers_filename)
    character(len = max_name_length) :: lake_parameters_filename
    character(len = max_name_length) :: output_lake_numbers_filename
      call set_lake_parameters_filename(lake_parameters_filename)
      global_lake_parameters => read_lake_parameters(.true.)
      global_lake_fields => lakefields(global_lake_parameters)
      global_lake_prognostics => lakeprognostics(global_lake_parameters, &
                                                 global_lake_fields)
      call run_lake_number_retrieval(global_lake_prognostics)
      call write_lake_numbers_field(output_lake_numbers_filename,global_lake_parameters,&
                                    global_lake_fields,-2)
end subroutine retrieve_lake_numbers

end module latlon_lake_model_retrieve_lake_numbers
