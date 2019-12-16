program latlon_hd_model_driver

  use latlon_hd_model_interface_mod
  use parameters_mod
  implicit none

  real   ,dimension(:,:,:), allocatable :: runoffs
  real   ,dimension(:,:,:), allocatable :: drainages

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
  runoffs = 1.0
  drainages = 1.0
  call run_hd_model(timesteps,drainages,runoffs)

end program latlon_hd_model_driver
