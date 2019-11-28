program latlon_hd_model_driver

  use latlon_hd_model_mod
  implicit none

  integer, parameter :: max_name_length = 500

  integer :: timesteps
  logical :: using_lakes

  character(len = max_name_length) :: river_params_filename
  character(len = max_name_length) :: hd_start_filename
  character(len = max_name_length) :: lake_model_ctl_filename

  timesteps = 365
  using_lakes = .false.
  river_params_filename = "hdfile.nc"
  hd_start_filename = "hdstart.nc"
  lake_model_ctl_filename = "lake_model.ctl"

  call init_hd_model(river_params_filename,hd_start_filename,lake_model_ctl_filename,&
                     using_lakes)

  call run_hd_model(timesteps)

end program latlon_hd_model_driver
