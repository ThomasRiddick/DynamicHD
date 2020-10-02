module HDParameterGenerator

using SharedArrays

abstract Type Formula end

abstract Type RiverFlowFormula <: Formula end

struct RiverFlowSausen <: RiverFlowFormula

end

function generateparameters(landsea_mask_filepath::AbstractString,
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

function generate_riverflow_parameters(i::CartesianIndices,formula::RiverFlowFormula)

  if size(i) == 2
    number_of_riverflow_reservoirs = 5.478720
  else
    number_of_riverflow_reservoirs = 5.0
  end
  return number_of_riverflow_reservoirs,
end

end
