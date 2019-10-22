module latlon_hd_model_mod

type RiverParameters
  real, allocatable, dimension(:,:) :: rdirs
  integer, allocatable, dimension(:,:) :: river_reservoir_nums
  integer, allocatable, dimension(:,:) :: overland_reservoir_nums
  integer, allocatable, dimension(:,:) :: base_reservoir_nums
  real, allocatable, dimension(:,:) :: river_retention_coefficients
  real, allocatable, dimension(:,:) :: overland_retention_coefficients
  real, allocatable, dimension(:,:) :: base_retention_coefficients
  logical, allocatable, dimension(:,:) :: landsea_mask
  logical, allocatable, dimension(:,:) :: cascade_flag
  grid::Grid
  function RiverParameters(flow_directions::DirectionIndicators,
                           river_reservoir_nums::Field{Int64},
                           overland_reservoir_nums::Field{Int64},
                           base_reservoir_nums::Field{Int64},
                           river_retention_coefficients::Field{Float64},
                           overland_retention_coefficients::Field{Float64},
                           base_retention_coefficients::Field{Float64},
                           landsea_mask::Field{Bool},
                           grid::Grid)
    cascade_flag::Field{Bool} = invert(landsea_mask)
    return new(flow_directions,river_reservoir_nums,overland_reservoir_nums,
               base_reservoir_nums,river_retention_coefficients,
               overland_retention_coefficients,base_retention_coefficients,
               landsea_mask,cascade_flag,grid)
  end
end type

mutable struct RiverPrognosticFields
  runoff::Field{Float64}
  drainage::Field{Float64}
  river_inflow::Field{Float64}
  base_flow_reservoirs::Array{Field{Float64},1}
  overland_flow_reservoirs::Array{Field{Float64},1}
  river_flow_reservoirs::Array{Field{Float64},1}
  water_to_ocean::Field{Float64}
  function RiverPrognosticFields(river_parameters::RiverParameters)
    runoff = Field{Float64}(river_parameters.grid,0.0)
    drainage = Field{Float64}(river_parameters.grid,0.0)
    river_inflow = Field{Float64}(river_parameters.grid,0.0)
    base_flow_reservoirs =      repeat(Field{Float64}(river_parameters.grid,0.0),
                                       maximum(river_parameters.base_reservoir_nums))
    overland_flow_reservoirs =  repeat(Field{Float64}(river_parameters.grid,0.0),
                                       maximum(river_parameters.overland_reservoir_nums))
    river_flow_reservoirs =     repeat(Field{Float64}(river_parameters.grid,0.0),
                                       maximum(river_parameters.river_reservoir_nums))
    water_to_ocean = Field{Float64}(river_parameters.grid,0.0)
    return new(runoff,drainage,river_inflow,base_flow_reservoirs,
               overland_flow_reservoirs,river_flow_reservoirs,
               water_to_ocean)
  end
end

mutable struct RiverDiagnosticFields
  runoff_to_rivers::Field{Float64}
  drainage_to_rivers::Field{Float64}
  river_outflow::Field{Float64}
  function RiverDiagnosticFields(river_parameters::RiverParameters)
    runoff_to_rivers::Field{Float64} =
      Field{Float64}(river_parameters.grid,0.0)
    drainage_to_rivers::Field{Float64} =
      Field{Float64}(river_parameters.grid,0.0)
    river_outflow::Field{Float64} =
      Field{Float64}(river_parameters.grid,0.0)
    new(runoff_to_rivers,drainage_to_rivers,river_outflow)
  end
end

mutable struct RiverDiagnosticOutputFields
  cumulative_river_flow::Field{Float64}
  function RiverDiagnosticOutputFields(river_parameters::RiverParameters)
    cumulative_river_flow::Field{Float64} =
      Field{Float64}(river_parameters.grid,0.0)
    new(cumulative_river_flow)
  end
end

get_river_parameters(obj::T) where {T <: PrognosticFields} =
  obj.river_parameters::RiverParameters

get_river_fields(obj::T) where {T <: PrognosticFields} =
  obj.river_fields::RiverPrognosticFields

get_river_diagnostic_fields(obj::T) where {T <: PrognosticFields} =
  obj.river_diagnostic_fields::RiverDiagnosticFields

get_river_diagnostic_output_fields(obj::T) where {T <: PrognosticFields} =
  obj.river_diagnostic_output_fields::RiverDiagnosticOutputFields

get_using_lakes(obj::T) where {T <: PrognosticFields} =
  obj.using_lakes::Bool

struct RiverPrognosticFieldsOnly <: PrognosticFields
  river_parameters::RiverParameters
  river_fields::RiverPrognosticFields
  river_diagnostic_fields::RiverDiagnosticFields
  river_diagnostic_output_fields::RiverDiagnosticOutputFields
  using_lakes::Bool
  RiverPrognosticFieldsOnly(river_parameters::RiverParameters,
                            river_fields::RiverPrognosticFields) =
    new(river_parameters,river_fields,RiverDiagnosticFields(river_parameters),
        RiverDiagnosticOutputFields(river_parameters),false)
end

function handle_event(prognostic_fields::PrognosticFields,
                      run_hd::RunHD)
  river_parameters::RiverParameters = get_river_parameters(prognostic_fields)
  river_fields::RiverPrognosticFields = get_river_fields(prognostic_fields)
  river_diagnostic_fields::RiverDiagnosticFields =
    get_river_diagnostic_fields(prognostic_fields)
  using_lakes::Bool =
              get_using_lakes(prognostic_fields)
  if using_lakes
    lake_water_input::Field{Float64} =
      water_from_lakes(prognostic_fields)
    for_all(river_parameters.grid) do coords::Coords
      cell_lake_water_input::Float64 = lake_water_input(coords)
      if river_parameters.cascade_flag(coords)
        set!(river_fields.river_inflow,coords,
             get(river_fields.river_inflow,coords)+cell_lake_water_input)
      else
        set!(river_diagnostic_fields.river_outflow,
             coords,cell_lake_water_input)
      end
    end
  end
  cascade(river_fields.overland_flow_reservoirs,
          river_fields.runoff,
          river_diagnostic_fields.runoff_to_rivers,
          river_parameters.overland_retention_coefficients,
          river_parameters.base_reservoir_nums,
          river_parameters.cascade_flag,river_parameters.grid)
  cascade(river_fields.base_flow_reservoirs,
          river_fields.drainage,
          river_diagnostic_fields.drainage_to_rivers,
          river_parameters.base_retention_coefficients,
          river_parameters.base_reservoir_nums,
          river_parameters.cascade_flag,river_parameters.grid)
  cascade(river_fields.river_flow_reservoirs,
          river_fields.river_inflow,
          river_diagnostic_fields.river_outflow,
          river_parameters.river_retention_coefficients,
          river_parameters.river_reservoir_nums,
          river_parameters.cascade_flag,river_parameters.grid)
  fill!(river_fields.river_inflow,0.0)
  route(river_parameters.flow_directions,
        river_diagnostic_fields.river_outflow+
        river_diagnostic_fields.runoff_to_rivers+
        river_diagnostic_fields.drainage_to_rivers,
        river_fields.river_inflow,
        river_parameters.grid)
  fill!(river_diagnostic_fields.river_outflow,0.0)
  fill!(river_diagnostic_fields.runoff_to_rivers,0.0)
  fill!(river_diagnostic_fields.drainage_to_rivers,0.0)
  for_all(river_parameters.grid) do coords::Coords
            flow_direction::DirectionIndicator =
              get(river_parameters.flow_directions,coords)
            if is_ocean(flow_direction) || is_outflow(flow_direction) ||
               is_truesink(flow_direction)
                set!(river_fields.water_to_ocean,coords,
                      get(river_fields.river_inflow,coords) +
                      get(river_fields.runoff,coords) +
                      get(river_fields.drainage,coords))
                set!(river_fields.river_inflow,coords,0.0)
            elseif using_lakes && is_lake(flow_direction)
                water_to_lakes(prognostic_fields,coords,
                               get(river_fields.river_inflow,coords) +
                               get(river_fields.runoff,coords) +
                               get(river_fields.drainage,coords))
                set!(river_fields.river_inflow,coords,0.0)
            end
          end
  fill!(river_fields.runoff,0.0)
  fill!(river_fields.drainage,0.0)
  return prognostic_fields
end

function water_to_lakes(prognostic_fields::PrognosticFields,coords::Coords,inflow::Float64)
  throw(UserError())
end

function water_from_lakes(prognostic_fields::PrognosticFields)
  throw(UserError())
end

function cascade(reservoirs::Array{Field{Float64},1},
                 inflow::Field{Float64},
                 outflow::Field{Float64},
                 retention_coefficients::Field{Float64},
                 reservoir_nums::Field{Int64},
                 cascade_flag::Field{Bool},grid::Grid)
  for_all(grid) do coords::Coords
    if get(cascade_flag,coords)
      cascade_kernel(coords,
                    reservoirs,
                    inflow,
                    outflow,
                    retention_coefficients,
                    reservoir_nums)
    end
    return
  end
end

function cascade(reservoirs::Array{Array{Field{Float64},1},1},
                 inflow::Array{Field{Float64},1},
                 outflow::Array{Field{Float64},1},
                 retention_coefficients::Array{Field{Float64},1},
                 reservoir_nums::Array{Field{Int64},1},
                 cascade_flag::Field{Bool},grid::Grid,
                 cascade_num::Int64)
  for_all(grid) do coords::Coords
    if get(cascade_flag,coords)
      for i = 1:cascade_num
        reservoirs_i::Array{Field{Float64},1} = reservoirs[i]
        inflow_i::Field{Float64} = inflow[i]
        outflow_i::Field{Float64} = outflow[i]
        retention_coefficients_i::Field{Float64} = retention_coefficients[i]
        reservoir_nums_i::Field{Int64} = reservoir_nums[i]
        cascade_kernel(coords,
                       reservoirs_i,
                       inflow_i,
                       outflow_i,
                       retention_coefficients_i,
                       reservoir_nums_i)
      end
    end
    return
  end
end

function cascade_kernel(coords::Coords,
                        reservoirs::Array{Field{Float64},1},
                        inflow::Field{Float64},
                        outflow::Field{Float64},
                        retention_coefficients::Field{Float64},
                        reservoir_nums::Field{Int64})
  flow::Float64 = get(inflow,coords)
  for i = 1:(get(reservoir_nums,coords))
    reservoir::Field{Float64} = reservoirs[i]
    new_reservoir_value::Float64 = get(reservoir,coords) + flow
    flow = new_reservoir_value/(get(retention_coefficients,coords)+1.0)
    set!(reservoir,coords,new_reservoir_value - flow)
  end
  set!(outflow,coords,flow)
  return
end

function route(flow_directions::DirectionIndicators,
               flow_in::Field{Float64},
               flow_out::Field{Float64},
               grid::Grid)
  for_all(grid) do coords::Coords
    flow_in_local::Float64 = flow_in(coords)
    if flow_in_local != 0.0
      flow_direction::DirectionIndicator =
        get(flow_directions,coords)
      if is_truesink(flow_direction) || is_lake(flow_direction) ||
         is_ocean(flow_direction) || is_outflow(flow_direction)
        flow_in_local += get(flow_out,coords)
        set!(flow_out,coords,flow_in_local)
      else
        new_coords::Coords =
          find_downstream_coords(grid,
                                flow_direction,
                                coords)
        flow_in_local += get(flow_out,new_coords)
        set!(flow_out,new_coords,flow_in_local)
      end
    end
    return
  end
end

struct SetDrainage <: Event
  new_drainage::Field{Float64}
end

struct SetRunoff <: Event
  new_runoff::Field{Float64}
end

function handle_event(prognostic_fields::PrognosticFields,
                      set_drainage::SetDrainage)
  river_fields::RiverPrognosticFields = get_river_fields(prognostic_fields)
  river_fields.drainage = set_drainage.new_drainage
  return prognostic_fields
end

function handle_event(prognostic_fields::PrognosticFields,
                      set_runoff::SetRunoff)
  river_fields::RiverPrognosticFields = get_river_fields(prognostic_fields)
  river_fields.runoff = set_runoff.new_runoff
  return prognostic_fields
end

struct PrintResults <: Event
  timestep::Int64
end

struct PrintSection <: Event end

function handle_event(prognostic_fields::PrognosticFields,
                      print_results::PrintResults)
  println("Timestep: $(print_results.timestep)")
  print_river_results(prognostic_fields)
  return prognostic_fields
end

function print_river_results(prognostic_fields::PrognosticFields)
  river_fields::RiverPrognosticFields = get_river_fields(prognostic_fields)
  println("")
  println("River Flow")
  println(river_fields.river_inflow)
  println("")
  println("Water to Ocean")
  println(river_fields.water_to_ocean)
end

function handle_event(prognostic_fields::PrognosticFields,
                      print_results::PrintSection)
   print_river_results_section(prognostic_fields)
  return prognostic_fields
end

function print_river_results_section(prognostic_fields::PrognosticFields)
  section_coords::LatLonSectionCoords = LatLonSectionCoords(65,75,125,165)
  river_fields::RiverPrognosticFields = get_river_fields(prognostic_fields)
  river_parameters::RiverParameters = get_river_parameters(prognostic_fields)
  for_section_with_line_breaks(river_parameters.grid,section_coords) do coords::Coords
    @printf("%.2f ",river_fields.river_inflow(coords))
    flush(stdout)
  end
  println()
end

struct WriteRiverInitialValues <: Event end

function handle_event(prognostic_fields::PrognosticFields,
                      print_results::WriteRiverInitialValues)
  river_fields::RiverPrognosticFields = get_river_fields(prognostic_fields)
  river_parameters::RiverParameters = get_river_parameters(prognostic_fields)
  hd_start_filepath::AbstractString = "/Users/thomasriddick/Documents/data/temp/transient_sim_1/river_model_out.nc"
  write_river_initial_values(hd_start_filepath,river_parameters,river_fields)
  return prognostic_fields
end

function write_river_initial_values(hd_start_filepath::AbstractString,
                                    river_parameters::RiverParameters,
                                    river_prognostic_fields::RiverPrognosticFields)
  throw(UserError())
end

struct WriteRiverFlow <: Event
  timestep::Int64
end

function handle_event(prognostic_fields::PrognosticFields,
                      write_river_flow::WriteRiverFlow)
  river_fields::RiverPrognosticFields = get_river_fields(prognostic_fields)
  river_parameters::RiverParameters = get_river_parameters(prognostic_fields)
  write_river_flow_field(river_parameters,river_fields.river_inflow,
                         timestep=write_river_flow.timestep)
  return prognostic_fields
end

function write_river_flow_field(river_parameters::RiverParameters,
                                river_flow_field::Field{Float64};
                                timestep::Int64=-1)
  throw(UserError())
end

struct AccumulateRiverFlow <: Event end

function handle_event(prognostic_fields::PrognosticFields,
                      accumulate_river_flow::AccumulateRiverFlow)
  river_fields::RiverPrognosticFields = get_river_fields(prognostic_fields)
  river_diagnostic_output_fields::RiverDiagnosticOutputFields =
    get_river_diagnostic_output_fields(prognostic_fields)
  river_diagnostic_output_fields.cumulative_river_flow += river_fields.river_inflow
  return prognostic_fields
end

struct ResetCumulativeRiverFlow <: Event end

function handle_event(prognostic_fields::PrognosticFields,
                      reset_cumulative_river_flow::ResetCumulativeRiverFlow)
  river_diagnostic_output_fields::RiverDiagnosticOutputFields =
    get_river_diagnostic_output_fields(prognostic_fields)
  fill!(river_diagnostic_output_fields.cumulative_river_flow,0.0)
  return prognostic_fields
end

struct WriteMeanRiverFlow <: Event
  timestep::Int64
  number_of_timesteps::Int64
end

function handle_event(prognostic_fields::PrognosticFields,
                      write_mean_river_flow::WriteMeanRiverFlow)
  river_diagnostic_output_fields::RiverDiagnosticOutputFields =
    get_river_diagnostic_output_fields(prognostic_fields)
  river_parameters::RiverParameters = get_river_parameters(prognostic_fields)
  mean_river_flow::Field{Float64} =
      Field{Float64}(river_parameters.grid,0.0)
  for_all(river_parameters.grid) do coords::Coords
    set!(mean_river_flow,coords,
         river_diagnostic_output_fields.cumulative_river_flow(coords)/
         convert(Float64,write_mean_river_flow.number_of_timesteps))
  end
  write_river_flow_field(river_parameters,mean_river_flow,
                         timestep=write_mean_river_flow.timestep)
  return prognostic_fields
end

end module latlon_hd_model_mod
