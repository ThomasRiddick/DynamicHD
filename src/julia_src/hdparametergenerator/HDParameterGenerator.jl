module HDParameterGenerator

using SharedArrays
using Distributed: @distributed

abstract type Grid end

struct LatLonGrid <: Grid
  nlat::Int64
  nlon::Int64
  lats::Array{Float64,1}
  lons::Array{Float64,1}
  dlat::Array{Float64,1}
  dlon::Float64
  grid_dimensions::Tuple{Int64,Int64}
  function LatLonGrid(nlat::Int64,nlon::Int64,
                      lats::Array{Float64,1},lons::Array{Float64,1})
    grid_dimensions::Tuple{Int64,Int64} = (nlat,nlon)
    dlat::Array{Float64,1} = [x-y for (x,y) in zip(lats[2:nlat],lats[1:nlat-1])]
    dlon::Float64 = lons[2] - lons[1]
    new(nlat,nlon,lats,lons,dlat,dlon,grid_dimensions)
  end
end

struct UnstructuredGrid <: Grid
  ncells::Int64
  clat::Array{Float64,1}
  clon::Array{Float64,1}
  clat_bounds::Array{Float64,2}
  clon_bounds::Array{Float64,2}
  grid_dimensions::Tuple{Int64}
  function UnstructuredGrid(ncells,clat,clon,clat_bounds,clon_bounds)
    grid_dimensions::Tuple{Int64} = (ncells)
    new(ncells,clat,clon,clat_bounds,clon_bounds,grid_dimensions)
  end
end

get_grid_dimensions(obj::T) where {T <: Grid} =
  obj.grid_dimensions::Tuple

struct CommonParameters
  minimum_height_threshold::Float64
  default_height_change::Float64
  alpha::Float64
  C::Float64
  function CommonParameters()
    minimum_height_threshold::Float64 = 0.0
    default_height_change::Float64 = 0.0
    alpha::Float64 = 0.1
    C::Float64 = 2.0
    new(minimum_height_threshold,default_height_change,
        alpha,C)
  end
end

abstract type Formula end

abstract type RiverFlowFormula <: Formula end

struct RiverFlowSausen <: RiverFlowFormula
  riverflow_dx::Float64
  riverflow_v0::Float64
  riverflow_k0::Float64
  common_parameters::CommonParameters
  function RiverFlowSausen()
  riverflow_dx::Float64 = 228000.0
  riverflow_v0::Float64 = 1.0039
  riverflow_k0::Float64 = 0.4112
  common_parameters::CommonParameters = CommonParameters()
    new(riverflow_dx,riverflow_v0,
        riverflow_k0,common_parameters)
  end
end

abstract type OverlandFlowFormula <: Formula end

struct OverlandFlowSausen <: OverlandFlowFormula
  overlandflow_dx::Float64
  overlandflow_v0::Float64
  overlandflow_k0::Float64
  overlandflow_torneaelven_k_multiplier::Float64
  common_parameters::CommonParameters
  function OverlandFlowSausen()
    overlandflow_dx::Float64 = 171000.0
    overlandflow_v0::Float64 = 1.0885
    overlandflow_k0::Float64 = 16.8522
    overlandflow_torneaelven_k_multiplier::Float64 = 3.0
    common_parameters::CommonParameters = CommonParameters()
    new(overlandflow_dx,overlandflow_v0,
        overlandflow_k0,overlandflow_torneaelven_k_multiplier,
        common_parameters)
  end
end

abstract type BaseFlowFormula <: Formula end

struct BaseFlowConstant <: BaseFlowFormula
  baseflow_k0::Float64
  function BaseFlowConstant()
    baseflow_k0::Float64 = 300.0
    new(baseflow_k0)
  end
end

struct BaseFlowDistanceAndOrography <: BaseFlowFormula
  baseflow_k0::Float64
  baseflow_d0::Float64
  function BaseFlowDistanceAndOrography()
    baseflow_k0::Float64 = 300.0
    baseflow_d0::Float64 = 50000.0
    new(baseflow_k0,baseflow_d0)
  end
end

abstract type GridSpecificInputData end

struct LatLonGridInputData <: GridSpecificInputData
  dlat::Float64
  dlon::SharedArray{Float64}
  river_directions::SharedArray{Int64}
  function LatLonGridInputData(river_directions::Array{Int64},grid::LatLonGrid)
    #Need formula for thiS!!!!
    dlat::Float64 = abs(grid.lats[2] - grid.lats[1])* 1.0
    dlon::SharedArray{Float64} = SharedArray{Float64}((grid.nlat))
    for i in eachindex(grid.lats)
      #Need formula for thiS!!!!
      dlon[i] = abs(grid.lons[2] - grid.lons[1])* 1.0
    end
    new(dlat,dlon,convert(SharedArray{Int64},river_directions))
  end
end

struct IcosohedralGridInputData <: GridSpecificInputData
  lat::SharedArray{Float64}
  lon::SharedArray{Float64}
  next_cell_index::SharedArray{Int64}
  function IcosohedralGridInputData(next_cell_index::Array{Int64},
                                    grid::UnstructuredGrid)
    lat::SharedArray{Float64} = convert(SharedArray{Float64},grid.clat)
    lon::SharedArray{Float64} = convert(SharedArray{Float64},grid.clon)
    new(lat,lon,convert(SharedArray{Int64},next_cell_index))
  end
end

struct InputData
  landsea_mask::SharedArray{Bool}
  glacier_mask::SharedArray{Bool}
  orography::SharedArray{Float64}
  innerslope::SharedArray{Float64}
  orography_variance::SharedArray{Float64}
  cell_areas::SharedArray{Float64}
  grid_specific_input_data::GridSpecificInputData
  function InputData(landsea_mask::Array{Bool},
                     glacier_mask::Array{Bool},
                     orography::Array{Float64},
                     innerslope::Array{Float64},
                     orography_variance::Array{Float64},
                     river_directions::Array{Int64},
                     cell_areas::Array{Float64},
                     grid::Grid)
    if isa(grid,LatLonGrid)
      grid_specific_input_data = LatLonGridInputData(river_directions,grid)
    else
      grid_specific_input_data = IcosohedralGridInputData(river_directions,grid)
    end
    new(convert(SharedArray{Bool},landsea_mask),convert(SharedArray{Bool},glacier_mask),
        convert(SharedArray{Float64},orography),convert(SharedArray{Float64},innerslope),
        convert(SharedArray{Float64},orography_variance),
        convert(SharedArray{Float64},cell_areas),grid_specific_input_data)
  end
end



struct Configuration
  riverflow_formula::RiverFlowFormula
  overlandflow_formula::OverlandFlowFormula
  baseflow_formula::BaseFlowFormula
end

function load_configuration(input_filepaths::Dict)
  println("Loading: " * input_filepaths["configuration_filepath"])
  formulae::Dict{DataType,Formula} = Dict{DataType,Formula}()
  for line in eachline(input_filepaths["configuration_filepath"])
    formula::Formula = getfield(HDParameterGeneration,Symbol(line))
    formulae[supertype(formula)] = formula()
  end
  return Configuration(formulae[RiverFlowFormula],
                       formulae[OverlandFlowFormula],
                       formulae[BaseFlowFormula])
end



function generate_parameters(configuration::Configuration,
                             input_data::InputData,
                             grid::Grid)
  grid_dimensions::Tuple = get_grid_dimensions(grid)
  number_of_riverflow_reservoirs = SharedArray{Float64}(grid_dimensions)
  riverflow_retention_coefficients = SharedArray{Float64}(grid_dimensions)
  number_of_overlandflow_reservoirs = SharedArray{Float64}(grid_dimensions)
  overlandflow_retention_coefficients = SharedArray{Float64}(grid_dimensions)
  number_of_baseflow_reservoirs = SharedArray{Float64}(grid_dimensions)
  baseflow_retention_coefficients = SharedArray{Float64}(grid_dimensions)
  @sync @distributed for i in CartesianIndices(input_data.landsea_mask)
    if ( ! input_data.landsea_mask[i] || ! input_data.glacier_mask[i] )
      distance::Float64 = calculate_distance(i,input_data,grid)
      height_change::Float64 = calculate_height_change(i,input_data,grid)
      number_of_riverflow_reservoirs[i],
      riverflow_retention_coefficients[i] =
        generate_riverflow_parameters(i,configuration.riverflow_formula,distance,
                                      height_change,grid)
      number_of_overlandflow_reservoirs[i],
      overlandflow_retention_coefficients[i] =
        generate_overlandflow_parameters(i,configuration.overlandflow_formula,distance,
                                         height_change,input_data,grid)
      number_of_baseflow_reservoirs[i],
      baseflow_retention_coefficients[i]=
        generate_baseflow_parameters(i,configuration.baseflow_formula,distance,
                                     height_change,input_data,grid)
    else
      number_of_riverflow_reservoirs[i] = 0
      riverflow_retention_coefficients[i] = 0.0
      number_of_overlandflow_reservoirs[i] = 0
      overlandflow_retention_coefficients[i] = 0.0
      number_of_baseflow_reservoirs[i] = 0
      baseflow_retention_coefficients[i] = 0.0
    end
  end
  return sdata(number_of_riverflow_reservoirs),
         sdata(riverflow_retention_coefficients),
         sdata(number_of_overlandflow_reservoirs),
         sdata(overlandflow_retention_coefficients),
         sdata(number_of_baseflow_reservoirs),
         sdata(baseflow_retention_coefficients)
end

function calculate_height_change(i::CartesianIndex,input_data::InputData,grid::Grid)
  next_cell::CartesianIndex =
    get_next_cell_coords(i,input_data,grid)
  height_change = input_data.orography[i] - input_data.orography[next_cell]
  return height_change
end

function calculate_distance(i::CartesianIndex,input_data::InputData,
                            grid::LatLonGrid)
  earth_radius::Float64 = 6371000.0
  river_direction::Int64 =
    input_data.grid_specific_input_data.river_directions[i]
  local lat_index_change::Int64
  local lon_index_change::Int64
  if river_direction <= 3
    lat_index_change = 1
  elseif river_direction >= 7
    lat_index_change = -1
  else
    lat_index_change = 0
  end
  if river_direction == 7 ||
     river_direction == 4 ||
     river_direction == 1
    lon_index_change = -1
  elseif river_direction == 9 ||
          river_direction == 6 ||
          river_direction == 3
    lon_index_change = 1
  else
    lon_index_change = 0
  end
  distance::Float64 = (((lat_index_change^2)*(grid.dlat[i]^2))+
                       ((lon_index_change^2)*(grid.dlon^2)))
  return distance
end

function calculate_distance(i::CartesianIndex,input_data::InputData,
                            grid::UnstructuredGrid)
  earth_radius::Float64 = 6371000.0
  working_dlat::Float64 = abs(grid.clon(j) - grid.clon(i))
  if working_dlat > 300
    working_dlat = abs(working_dlat - 360)
  end
  pi_factor::Float64 = pi/180.0
  earths_radius::Float64 = 6371000.0
  dlon::Float64 = working_dlat*pi_factor*
                  cos(pi_factor*(grid.clat(j)+grid.clat(i))/2)*earths_radius
  dlat::Float64 = abs(grid.clat(i)+grid.clat(i))*pi_factor*earths_radius
  distance::Float64 = sqrt(dlat^2+dlon^2)
  return distance
end

function get_next_cell_coords(i::CartesianIndex,input_data::InputData,
                              grid::LatLonGrid)
  println("NOT WORKING PROPERLY")
  return i::CartesianIndex
end

function get_next_cell_coords(i::CartesianIndex,input_data::InputData,
                              grid::UnstructuredGrid)
  return CartesianIndex(input_data.grid_specific_input_data.next_cell_index[i])::CartesianIndex
end

function generate_riverflow_parameters(i::CartesianIndex,formula::RiverFlowSausen,
                                       distance::Float64,height_change::Float64,
                                       grid::Grid)
  local number_of_riverflow_reservoirs::Float64
  if isa(grid,LatLonGrid)
    number_of_riverflow_reservoirs = 5.478720
  else
    number_of_riverflow_reservoirs = 5.0
  end
  if height_change < formula.common_parameters.minimum_height_threshold
    height_change = formula.common_parameters.default_height_change
  end
  vsau = formula.common_parameters.C*
           ((height_change/distance)^formula.common_parameters.alpha)
  riverflow_retention_coefficient = (formula.riverflow_k0*distance/
                                    formula.riverflow_dx)*(formula.riverflow_v0/vsau)
  return number_of_riverflow_reservoirs,riverflow_retention_coefficient
end

function generate_overlandflow_parameters(i::CartesianIndex,formula::OverlandFlowSausen,
                                          distance::Float64,height_change::Float64,
                                          input_data::InputData,grid::Grid)
  if isa(grid,LatLonGrid)
    number_of_overlandflow_reservoirs = 1.1107
  else
    number_of_overlandflow_reservoirs = 1.0
  end
  if height_change < formula.common_parameters.minimum_height_threshold
    height_change = formula.common_parameters.default_height_change
  end
  if input_data.innerslope[i] > 0
    if isa(grid,LatLonGrid)
      dx0 = sqrt(grid.dlat[i]^2 + grid.dlon^2)
    else
      dx0 = distance
    end
    vso = formula.common_parameters.C*
          (input_data.innerslope[i]^formula.common_parameters.alpha)
    overlandflow_retention_coefficient = (formula.overlandflow_k0*dx0/
                                          formula.overlandflow_dx)*(formula.overlandflow_v0/vso)
  else
    vsau = formula.common_parameters.C*
           ((height_change/distance)^formula.common_parameters.alpha)
    overlandflow_retention_coefficient = (formula.overlandflow_k0*distance/
                                          formula.overlandflow_dx)*(formula.overlandflow_v0/vsau)
  end
  overlandflow_retention_coefficient *= formula.overlandflow_torneaelven_k_multiplier
  return number_of_overlandflow_reservoirs,overlandflow_retention_coefficient
end

function generate_baseflow_parameters(i::CartesianIndex,formula::BaseFlowConstant,
                                      distance::Float64,height_change::Float64,
                                      input_data::InputData,grid::Grid)
  return 1.0,formula.baseflow_k0
end

function generate_baseflow_parameters(i::CartesianIndex,formula::BaseFlowDistanceAndOrography,
                                      distance::Float64,height_change::Float64,
                                      input_data::InputData,grid::Grid)
  bb = (input_data.orography_variance[i] - 100.0)/(input_data.orography_variance[i] + 1000.0)
  if bb < 0.01
    bb = 0.01
  end
  xib = 1.0 - bb + 0.01
  baseflow_retention_coefficient = (formula.baseflow_k0/xib)*
                                   (distance/formula.baseflow_d0)
  return 1.0,baseflow_retention_coefficient
end

function load_input_data(input_filepaths::Dict)
    throw(UserError())
end

function write_hdpara_file(output_hdpara_filepath::AbstractString,
                           input_data::InputData,
                           number_of_riverflow_reservoirs::Array{Float64},
                           riverflow_retention_coefficients::Array{Float64},
                           number_of_overlandflow_reservoirs::Array{Float64},
                           overlandflow_retention_coefficients::Array{Float64},
                           number_of_baseflow_reservoirs::Array{Float64},
                           baseflow_retention_coefficients::Array{Float64})
    throw(UserError())
end

end
