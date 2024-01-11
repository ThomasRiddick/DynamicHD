module HDModule

using HierarchicalStateMachineModule: Event,State
using FieldModule: Field,DirectionIndicators,maximum,set!,fill!,+,invert,get
using FieldModule: get_data_vector, repeat_init
using CoordsModule: Coords, DirectionIndicator,LatLonSectionCoords,
      is_ocean, is_outflow, is_truesink, is_lake, get_linear_index
using GridModule: Grid, for_all,for_all_parallel, get_linear_indices
using GridModule: find_downstream_coords,for_section_with_line_breaks
using GridModule: get_number_of_neighbors, for_all_parallel_sum
using OutputModule: write_river_initial_values, write_river_flow_field
using UserExceptionModule: UserError
import HierarchicalStateMachineModule: handle_event
using InteractiveUtils
using SharedArrays
using Printf: @printf

struct RunHD <: Event end

abstract type PrognosticFields <: State end

struct RiverParameters
  flow_directions::DirectionIndicators
  cells_up::Array{Field{Int64}}
  nsplit::Field{Int64}
  river_reservoir_nums::Field{Int64}
  overland_reservoir_nums::Field{Int64}
  base_reservoir_nums::Field{Int64}
  river_retention_coefficients::Field{Float64}
  overland_retention_coefficients::Field{Float64}
  base_retention_coefficients::Field{Float64}
  landsea_mask::Field{Bool}
  cascade_flag::Field{Bool}
  outflow_ocean_truesink_mask::Field{Bool}
  lake_mask::Field{Bool}
  grid::Grid
  step_length::Float64
  function RiverParameters(flow_directions::DirectionIndicators,
                           river_reservoir_nums::Field{Int64},
                           overland_reservoir_nums::Field{Int64},
                           base_reservoir_nums::Field{Int64},
                           river_retention_coefficients::Field{Float64},
                           overland_retention_coefficients::Field{Float64},
                           base_retention_coefficients::Field{Float64},
                           landsea_mask::Field{Bool},
                           grid::Grid,
                           day_length::Float64,
                           step_length::Float64)
    cascade_flag::Field{Bool} = invert(landsea_mask)
    river_retention_coefficients = river_retention_coefficients *
                                   (day_length/step_length)
    overland_retention_coefficients = overland_retention_coefficients *
                                      (day_length/step_length)
    base_retention_coefficients = base_retention_coefficients *
                                  (day_length/step_length)
    outflow_ocean_truesink_mask = Field{Bool}(grid,false)
    lake_mask = Field{Bool}(grid,false)
    linear_indices::LinearIndices = get_linear_indices(grid)
    nsplit::Field{Int64} = Field{Int64}(grid,1)
    cells_up::Array{Field{Int64}} = repeat_init(grid,-1,
                                                get_number_of_neighbors(grid))
    for_all(grid) do coords::Coords
      flow_direction::DirectionIndicator =
        get(flow_directions,coords)
      if is_ocean(flow_direction) || is_outflow(flow_direction) ||
         is_truesink(flow_direction)
         set!(outflow_ocean_truesink_mask,coords,true)
      elseif is_lake(flow_direction)
         set!(lake_mask,coords,true)
      else
        new_coords::Coords =
          find_downstream_coords(grid,
                                 flow_direction,
                                 coords)
        #Explicitly use basic forward stepping index
        for i in 1:length(cells_up)
          if get(cells_up[i],new_coords) == -1
            set!(cells_up[i],new_coords,get_linear_index(coords,
                                                         linear_indices))
            break
          end
        end
      end
    end
    return new(flow_directions,cells_up,nsplit,
               river_reservoir_nums,overland_reservoir_nums,
               base_reservoir_nums,river_retention_coefficients,
               overland_retention_coefficients,base_retention_coefficients,
               landsea_mask,cascade_flag,outflow_ocean_truesink_mask,
               lake_mask,grid,step_length)
  end
  function RiverParameters(flow_directions::DirectionIndicators,
                           cells_up::Array{Field{Int64}},
                           nsplit::Field{Int64},
                           river_reservoir_nums::Field{Int64},
                           overland_reservoir_nums::Field{Int64},
                           base_reservoir_nums::Field{Int64},
                           river_retention_coefficients::Field{Float64},
                           overland_retention_coefficients::Field{Float64},
                           base_retention_coefficients::Field{Float64},
                           landsea_mask::Field{Bool},
                           grid::Grid,
                           day_length::Float64,
                           step_length::Float64)
    cascade_flag::Field{Bool} = invert(landsea_mask)
    river_retention_coefficients = river_retention_coefficients *
                                   (day_length/step_length)
    overland_retention_coefficients = overland_retention_coefficients *
                                      (day_length/step_length)
    base_retention_coefficients = base_retention_coefficients *
                                  (day_length/step_length)
    outflow_ocean_truesink_mask = Field{Bool}(grid,false)
    lake_mask = Field{Bool}(grid,false)
    for_all(grid) do coords::Coords
      flow_direction::DirectionIndicator =
        get(flow_directions,coords)
      if is_ocean(flow_direction) || is_outflow(flow_direction) ||
         is_truesink(flow_direction)
         set!(outflow_ocean_truesink_mask,coords,true)
      elseif is_lake(flow_direction)
         set!(lake_mask,coords,true)
      end
    end
    return new(flow_directions,cells_up,nsplit,
               river_reservoir_nums,overland_reservoir_nums,
               base_reservoir_nums,river_retention_coefficients,
               overland_retention_coefficients,base_retention_coefficients,
               landsea_mask,cascade_flag,outflow_ocean_truesink_mask,
               lake_mask,grid,step_length)
  end
end

RiverParameters(flow_directions::DirectionIndicators,
                river_reservoir_nums::Field{Int64},
                overland_reservoir_nums::Field{Int64},
                base_reservoir_nums::Field{Int64},
                river_retention_coefficients::Field{Float64},
                overland_retention_coefficients::Field{Float64},
                base_retention_coefficients::Field{Float64},
                landsea_mask::Field{Bool},
                grid::Grid) = RiverParameters(flow_directions::DirectionIndicators,
                                              river_reservoir_nums::Field{Int64},
                                              overland_reservoir_nums::Field{Int64},
                                              base_reservoir_nums::Field{Int64},
                                              river_retention_coefficients::Field{Float64},
                                              overland_retention_coefficients::Field{Float64},
                                              base_retention_coefficients::Field{Float64},
                                              landsea_mask::Field{Bool},
                                              grid::Grid,86400.0,86400.0)

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
    base_flow_reservoirs =      repeat_init(river_parameters.grid,0.0,
                                            maximum(river_parameters.base_reservoir_nums))
    overland_flow_reservoirs =  repeat_init(river_parameters.grid,0.0,
                                            maximum(river_parameters.overland_reservoir_nums))
    river_flow_reservoirs =     repeat_init(river_parameters.grid,0.0,
                                            maximum(river_parameters.river_reservoir_nums))
    water_to_ocean = Field{Float64}(river_parameters.grid,0.0)
    return new(runoff,drainage,river_inflow,base_flow_reservoirs,
               overland_flow_reservoirs,river_flow_reservoirs,water_to_ocean)
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
  mean_river_flow::Field{Float64}
  function RiverDiagnosticOutputFields(river_parameters::RiverParameters)
    cumulative_river_flow::Field{Float64} =
      Field{Float64}(river_parameters.grid,0.0)
    mean_river_flow::Field{Float64} =
      Field{Float64}(river_parameters.grid,0.0)
    new(cumulative_river_flow,mean_river_flow)
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
  local lake_water_from_ocean::Field{Float64}
  if using_lakes
    lake_water_input::Field{Float64},lake_water_from_ocean =
      water_from_lakes(prognostic_fields,river_parameters.step_length)
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
          river_parameters.cascade_flag,river_parameters.grid,
          river_parameters.step_length)
  cascade(river_fields.base_flow_reservoirs,
          river_fields.drainage,
          river_diagnostic_fields.drainage_to_rivers,
          river_parameters.base_retention_coefficients,
          river_parameters.base_reservoir_nums,
          river_parameters.cascade_flag,river_parameters.grid,
          river_parameters.step_length)
  cascade(river_fields.river_flow_reservoirs,
          river_fields.river_inflow,
          river_diagnostic_fields.river_outflow,
          river_parameters.river_retention_coefficients,
          river_parameters.river_reservoir_nums,
          river_parameters.cascade_flag,river_parameters.grid,
          river_parameters.step_length)
  fill!(river_fields.river_inflow,0.0)
  data::Dict{Symbol,SharedArray} =
    Dict{Symbol,SharedArray}(
      :river_outflow => river_diagnostic_fields.river_outflow.data,
      :runoff_to_rivers => river_diagnostic_fields.runoff_to_rivers.data,
      :drainage_to_rivers => river_diagnostic_fields.drainage_to_rivers.data)
  for_all_parallel(river_parameters.grid) do coords::CartesianIndex
    data[:river_outflow][coords] =
    data[:river_outflow][coords]+
    data[:runoff_to_rivers][coords]+
    data[:drainage_to_rivers][coords]
  end
  route(river_parameters.cells_up,
        river_parameters.nsplit,
        river_diagnostic_fields.river_outflow,
        river_fields.river_inflow,
        river_parameters.outflow_ocean_truesink_mask,
        river_parameters.lake_mask,
        river_parameters.grid)
  fill!(river_diagnostic_fields.river_outflow,0.0)
  fill!(river_diagnostic_fields.runoff_to_rivers,0.0)
  fill!(river_diagnostic_fields.drainage_to_rivers,0.0)
  local water_to_lakes_local::Field{Float64}
  if using_lakes
    water_to_lakes_local = water_to_lakes(prognostic_fields)
  end
  data =
    Dict{Symbol,SharedArray}(
        :outflow_ocean_truesink_mask =>
        river_parameters.outflow_ocean_truesink_mask.data,
        :lake_mask => river_parameters.lake_mask.data,
        :water_to_ocean => river_fields.water_to_ocean.data,
        :river_inflow => river_fields.river_inflow.data,
        :runoff => river_fields.runoff.data,
        :drainage => river_fields.drainage.data)
  if using_lakes
    water_to_lakes_local_data::SharedArray{Float64} =
      water_to_lakes_local.data
    lake_water_from_ocean_data::SharedArray{Float64} =
      lake_water_from_ocean.data
  end
  for_all_parallel(river_parameters.grid) do coords::CartesianIndex
    if data[:outflow_ocean_truesink_mask][coords]
        data[:water_to_ocean][coords] =
              data[:river_inflow][coords] +
              data[:runoff][coords] +
              data[:drainage][coords]
        data[:river_inflow][coords] = 0.0
    elseif using_lakes && data[:lake_mask][coords]
        water_to_lakes_local_data[coords] =
             data[:river_inflow][coords] +
             data[:runoff][coords] +
             data[:drainage][coords]
        data[:water_to_ocean][coords] =
             -1.0*lake_water_from_ocean_data[coords]
        data[:river_inflow][coords] = 0.0
    end
  end
  if using_lakes
    water_to_lakes_local *= river_parameters.step_length
  end
  return prognostic_fields
end

function water_to_lakes(prognostic_fields::PrognosticFields)
  throw(UserError())
end

function water_from_lakes(prognostic_fields::PrognosticFields,step_length::Float64)
  throw(UserError())
end

function cascade(reservoirs::Array{Field{Float64},1},
                 inflow::Field{Float64},
                 outflow::Field{Float64},
                 retention_coefficients::Field{Float64},
                 reservoir_nums::Field{Int64},
                 cascade_flag::Field{Bool},grid::Grid,
                 step_length::Float64)
  data::Dict{Symbol,SharedArray} =
    Dict{Symbol,SharedArray}(:inflow => inflow.data,
                             :outflow => outflow.data,
                             :retention_coefficients =>
                             retention_coefficients.data,
                             :reservoir_nums =>
                             reservoir_nums.data,
                             :cascade_flag =>
                             cascade_flag.data)
  reservoirs_data::Array{SharedArray{Float64},1} = get_data_vector(reservoirs)
  for_all_parallel(grid) do coords::CartesianIndex
    if data[:cascade_flag][coords]
      cascade_kernel(coords,
                     reservoirs_data,
                     data[:inflow],
                     data[:outflow],
                     data[:retention_coefficients],
                     data[:reservoir_nums],
                     step_length)
    end
    return
  end
end

# function cascade(reservoirs::Array{Array{Field{Float64},1},1},
#                  inflow::Array{Field{Float64},1},
#                  outflow::Array{Field{Float64},1},
#                  retention_coefficients::Array{Field{Float64},1},
#                  reservoir_nums::Array{Field{Int64},1},
#                  cascade_flag::Field{Bool},grid::Grid,
#                  cascade_num::Int64,
#                  step_length::Float64)
#   for_all(grid) do coords::Coords
#     if get(cascade_flag,coords)
#       for i = 1:cascade_num
#         reservoirs_i::Array{Field{Float64},1} = reservoirs[i]
#         inflow_i::Field{Float64} = inflow[i]
#         outflow_i::Field{Float64} = outflow[i]
#         retention_coefficients_i::Field{Float64} = retention_coefficients[i]
#         reservoir_nums_i::Field{Int64} = reservoir_nums[i]
#         cascade_kernel(coords,
#                        reservoirs_i,
#                        inflow_i,
#                        outflow_i,
#                        retention_coefficients_i,
#                        reservoir_nums_i,
#                        step_length)
#       end
#     end
#     return
#   end
# end

function cascade_kernel(coords::CartesianIndex,
                        reservoirs::Array{SharedArray{Float64},1},
                        inflow::SharedArray{Float64},
                        outflow::SharedArray{Float64},
                        retention_coefficients::SharedArray{Float64},
                        reservoir_nums::SharedArray{Int64},
                        step_length::Float64)
  flow::Float64 = inflow[coords]*step_length
  for i = 1:reservoir_nums[coords]
    reservoir::SharedArray{Float64} = reservoirs[i]
    new_reservoir_value::Float64 = reservoir[coords] + flow
    flow = new_reservoir_value/(retention_coefficients[coords]+1.0)
    reservoir[coords] = new_reservoir_value - flow
  end
  flow /= step_length
  outflow[coords] = flow
  return
end

function route(cells_up::Array{Field{Int64}},
               nsplit::Field{Int64},
               flow_in::Field{Float64},
               flow_out::Field{Float64},
               outflow_ocean_truesink_mask::Field{Bool},
               lake_mask::Field{Bool},
               grid::Grid)
  data::Dict{Symbol,SharedArray} =
    Dict{Symbol,SharedArray}(
    :nsplit => nsplit.data,
    :flow_in => flow_in.data,
    :flow_out => flow_out.data,
    :outflow_ocean_truesink_mask => outflow_ocean_truesink_mask.data,
    :lake_mask => lake_mask.data)
  cells_up_data::Array{SharedArray{Int64},1} =
    get_data_vector(cells_up)
  for_all_parallel(grid) do coords::CartesianIndex
    flow_in_local::Float64 = data[:flow_in][coords]
    if flow_in_local != 0.0 &&
          (data[:outflow_ocean_truesink_mask][coords] ||
           data[:lake_mask][coords])
        flow_in_local += data[:flow_out][coords]
        data[:flow_out][coords] = flow_in_local
    end
    flow_in_from_nbrs::Float64 = 0.0
    #Explicitly use basic forward stepping index
    for i in 1:length(cells_up_data)
      linear_index::Int64 = cells_up_data[i][coords]
      if linear_index != -1
        flow_in_from_nbrs += data[:flow_in][linear_index] /
                             data[:nsplit][linear_index]
      else
        break
      end
    end
    if flow_in_from_nbrs != 0.0
      data[:flow_out][coords]=data[:flow_out][coords]+flow_in_from_nbrs
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
  println("")
end

struct WriteRiverInitialValues <: Event
  hd_start_filepath::AbstractString
end

function handle_event(prognostic_fields::PrognosticFields,
                      write_river_initial_values_event::WriteRiverInitialValues)
  river_fields::RiverPrognosticFields = get_river_fields(prognostic_fields)
  river_parameters::RiverParameters = get_river_parameters(prognostic_fields)
  write_river_initial_values(write_river_initial_values_event.hd_start_filepath,
                             river_parameters.grid,river_parameters.step_length,
                             river_fields.river_inflow,
                             river_fields.base_flow_reservoirs
                             river_fields.overland_flow_reservoirs
                             river_fields.river_flow_reservoirs)
  return prognostic_fields
end

struct WriteRiverFlow <: Event
  timestep::Int64
  river_flow_filepath::AbstractString
end

function handle_event(prognostic_fields::PrognosticFields,
                      write_river_flow::WriteRiverFlow)
  river_fields::RiverPrognosticFields = get_river_fields(prognostic_fields)
  river_parameters::RiverParameters = get_river_parameters(prognostic_fields)
  write_river_flow_field(river_parameters.grid,river_fields.river_inflow,
                         write_river_flow.river_flow_filepath)
  return prognostic_fields
end

struct AccumulateRiverFlow <: Event end

function handle_event(prognostic_fields::PrognosticFields,
                      accumulate_river_flow::AccumulateRiverFlow)
  river_fields::RiverPrognosticFields = get_river_fields(prognostic_fields)
  river_parameters::RiverParameters = get_river_parameters(prognostic_fields)
  river_diagnostic_output_fields::RiverDiagnosticOutputFields =
    get_river_diagnostic_output_fields(prognostic_fields)
  data::Dict{Symbol,SharedArray} =
    Dict{Symbol,SharedArray}(
      :cumulative_river_flow => river_diagnostic_output_fields.cumulative_river_flow.data,
      :river_inflow => river_fields.river_inflow.data)
  for_all_parallel(river_parameters.grid) do coords::CartesianIndex
    data[:cumulative_river_flow][coords] += data[:river_inflow][coords]
  end
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
  mean_river_flow_filepath::AbstractString
end

function handle_event(prognostic_fields::PrognosticFields,
                      write_mean_river_flow::WriteMeanRiverFlow)
  river_diagnostic_output_fields::RiverDiagnosticOutputFields =
    get_river_diagnostic_output_fields(prognostic_fields)
  river_parameters::RiverParameters = get_river_parameters(prognostic_fields)
  number_of_timesteps::Float64 = convert(Float64,write_mean_river_flow.number_of_timesteps)
  data::Dict{Symbol,SharedArray} =
    Dict{Symbol,SharedArray}(
      :mean_river_flow => river_diagnostic_output_fields.mean_river_flow.data,
      :cumulative_river_flow => river_diagnostic_output_fields.cumulative_river_flow.data)
  for_all_parallel(river_parameters.grid) do coords::CartesianIndex
    data[:mean_river_flow][coords] =
         data[:cumulative_river_flow][coords]/number_of_timesteps
  end
  write_river_flow_field(river_parameters.grid,
                         river_diagnostic_output_fields.mean_river_flow,
                         write_mean_river_flow.mean_river_flow_filepath)
  return prognostic_fields
end

struct PrintGlobalValues <: Event
end

function handle_event(prognostic_fields::PrognosticFields,
                      print_global_values::PrintGlobalValues)
  river_fields::RiverPrognosticFields = get_river_fields(prognostic_fields)
  river_parameters::RiverParameters = get_river_parameters(prognostic_fields)
  println("Global sums:")
  base_flow_reservoirs_data::Array{SharedArray{Float64},1} =
    get_data_vector(river_fields.base_flow_reservoirs)
  overland_flow_reservoirs_data::Array{SharedArray{Float64},1} =
    get_data_vector(river_fields.overland_flow_reservoirs)
  river_flow_reservoirs_data::Array{SharedArray{Float64},1} =
    get_data_vector(river_fields.river_flow_reservoirs)
  global_reservoir_totals::Array{Float64,1} =
      for_all_parallel_sum(river_parameters.grid) do coords::CartesianIndex
    local_reservoir_totals::Array{Float64,1} = vec(Float64[0.0 0.0 0.0])
    local_reservoir_totals[1] += base_flow_reservoirs_data[1][coords]
    local_reservoir_totals[2] += overland_flow_reservoirs_data[1][coords]
    for i = 1:5
      local_reservoir_totals[3] += river_flow_reservoirs_data[i][coords]
    end
    return local_reservoir_totals
  end
  global_sum_base_flow_res::Float64 = global_reservoir_totals[1]
  global_sum_overland_flow_res::Float64 = global_reservoir_totals[2]
  global_sum_river_flow_res::Float64 = global_reservoir_totals[3]
  println("Base Flow Res Content: $(global_sum_base_flow_res)")
  println("Overland Flow Res Content: $(global_sum_overland_flow_res)")
  println("River Flow Res Content: $(global_sum_river_flow_res)")
  return prognostic_fields
end

end
