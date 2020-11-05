module HDParameterGenerator

using SharedArrays

abstract Type Formula end

abstract Type RiverFlowFormula <: Formula end

struct RiverFlowSausen <: RiverFlowFormula
  riverflow_dx::Float64 = 228000
  minimum_height_threshold::Float64 = 0.0
  default_height_change::Float64 = 0.0
  riverflow_v0::Float64 = 1.0039
  alpha::Float64 = 0.1
  C::Float64 = 2.0
  riverflow_k0 = 0.4112
end

function generate_parameters(landsea_mask_filepath::AbstractString,
                             orography_filepath::AbstractString;
                             river_direction_filepath::AbstractString=nothing,
                             inner_slope_filepath::AbstractString=nothing)
  landsea_mask::SharedArray{Float64} = read_landsea_mask(landsea_mask_filepath)
  grid_dimensions::Tuple{Int64} = size(landsea_mask)
  number_of_riverflow_reservoirs = SharedArray{Float64}(grid_dimensions)
  riverflow_retention_coefficients = SharedArray{Float64}(grid_dimensions)
  number_of_overlandflow_reservoirs = SharedArray{Float64}(grid_dimensions)
  overlandflow_retention_coefficients = SharedArray{Float64}(grid_dimensions)
  number_of_baseflow_reservoirs = SharedArray{Float64}(grid_dimensions)
  baseflow_retention_coefficients = SharedArray{Float64}(grid_dimensions)
  @sync @distributed for i in eachindex(landsea_mask)
    if ( ! landsea_mask[i] || ! glacier_mask[i] )
      number_of_riverflow_reservoirs[i],
      riverflow_retention_coefficients[i] =
        generate_riverflow_parameters(i,riverflow_formula)
      number_of_overlandflow_reservoirs[i],
      overlandflow_retention_coefficients[i] =
        generate_overlandflow_parameters(i,overlandflow_formula)
      number_of_baseflow_reservoirs[i],
      baseflow_retention_coefficients[i]=
        generate_baseflow_parameters(i,baseflow_formula)
    end
  end
end

function calculate_distance(i::CartesianIndices)
  next_cell::CartesianIndices = get_next_cell_coords(i)
  earth_radius::Float64 = 6371000.0
  if size(i) = 2

  else
    dlat::Float64  =
    dlon::Float64 =
  end
  return distance
end

function generate_riverflow_parameters(i::CartesianIndices,formula::RiverFlowSausen)
  if size(i) == 2
    number_of_riverflow_reservoirs = 5.478720
  else
    number_of_riverflow_reservoirs = 5.0
  end
  if height_change < formula.minimum_height_threshold
    height_change = formula.default_height_change
  end
  distance = calculate_distance(i)
  vsau = formula.C*(height_change/distance)^formula.alpha
  riverflow_retention_coefficient = formula.riverflow_k0*distance/
                                    formula.riverflow_dx*formula.riverflow_v0/vsau
  return number_of_riverflow_reservoirs,riverflow_retention_coefficient
end

end
