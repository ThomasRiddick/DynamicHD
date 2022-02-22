module HDParameterGenerator

using SharedArrays

abstract type Grid end

struct LatLonGrid <: Grid
  nlat::Int64
  nlon::Int64
  lats::Array{Float64,1}
  lons::Array{Float64,1}
  grid_dimensions::Tuple{Int64}
  function LatLonGrid(nlat::Int64,nlon::Int64,
                      lats::Array{Float64,1},lons::Array{Float64,1})
    grid_dimensions::Tuple{Int64} = (nlat,nlon)
    new(nlat,nlon,lats,lons,grid_dimensions)
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
  obj.grid_dimensions::Tuple{Int64}

struct CommonParameters
  minimum_height_threshold::Float64 = 0.0
  default_height_change::Float64 = 0.0
  alpha::Float64 = 0.1
  C::Float64 = 2.0
end CommonParameters

abstract Type Formula end

abstract Type RiverFlowFormula <: Formula end

struct RiverFlowSausen <: RiverFlowFormula
  riverflow_dx::Float64 = 228000.0
  riverflow_v0::Float64 = 1.0039
  riverflow_k0::Float64 = 0.4112
  common_parameters::CommonParameters
end

abstract Type OverlandFlowFormula <: Formula end

struct OverlandFlowSausen <: OverlandFlowFormula
  overlandflow_dx::Float64 = 171000.0
  overlandflow_v0::Float64 = 1.0885
  overlandflow_k0::Float64 = 16.8522
  overlandflow_torneaelven_k_multiplier::Float64 = 3.0
  common_parameters::CommonParameters
end

abstract Type BaseFlowFormula <: Formula end

struct BaseFlowConstant <: BaseFlowFormula
  baseflow_k0::Float64 = 300.0
end

struct BaseFlowDistanceAndOrography <: BaseFlowFormula
  baseflow_k0::Float64 = 300.0
  baseflow_d0::Float64 = 50000.0
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
                     cell_areas::Array{Float64}
                     grid::Grid)
    if isa(grid,LatLonGrid)
      grid_specific_input_data = LatLonGridInputData(river_directions,grid)
    else
      grid_specific_input_data = IcosohedralGridInputData(river_directions,grid)
    end
    new(convert(SharedArray{Bool},landsea_mask),convert(SharedArray{Bool},glacier_mask),
        convert(SharedArray{Float64},orography),convert(SharedArray{Float64},innerslope))
        convert(SharedArray{Float64},orography_variance),
        convert(SharedArray{Float64},cell_areas),grid_specific_input_data)
  end
end

abstract Type GridSpecificInputData end

struct LatLonGridInputData <: GridSpecificInputData
  dlat::Float64
  dlon::SharedArray{Float64}
  river_directions::SharedArray{Float64}
  function LatLonGridInputData(river_directions::Array{Int64},grid::LatLonGrid)
    dlat::Float64 = abs(grid.lats[2] - grid.lats[1])* FORMULA FOR THIS IS?
    dlon::SharedArray{Float64} = SharedArray{Float64}((grid.nlat))
    for i in eachindex(grid.lats)
      dlon[i] = abs(grid.lons[2] - grid.lons[1])* FORMULA FOR THIS IS?
    end
    new(dlat,dlon,convert(SharedArray{Int64},river_directions))
  end
end

struct IcosohedralGridInputData <: GridSpecificInputData
  lat::SharedArray{Float64}
  lon::SharedArray{Float64}
  next_cell_index::SharedArray{Float64}
  function IcosohedralGridInputData(next_cell_index::Array{Int64},
                                    grid::UnstructuredGrid)
    lat::SharedArray{Float64} = convert(SharedArray{Float64,grid.clat)
    lon::SharedArray{Float64} = convert(SharedArray{Float64,grid.clon)
    new(lat,lon,convert(SharedArray{Int64},next_cell_index))
  end
end

struct Configuration
  riverflow_formula::RiverFlowFormula
  overlandflow_formula::OverlandFlowFormula
  baseflow_formula::BaseFlowFormula
end Configuration

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

function parameter_generation_driver(input_filepaths::Dict,
                                     output_hdpara_filepath::AbstractString)
  configuration::Configuration = load_configuration(input_filepaths)
  input_data::InputData,grid::Grid = load_input_data(input_filepaths)
  number_of_riverflow_reservoirs::Array,
  riverflow_retention_coefficients::Array,
  number_of_overlandflow_reservoirs::Array,
  overlandflow_retention_coefficients::Array,
  number_of_baseflow_reservoirs::Array,
  baseflow_retention_coefficients::Array = generate_parameters(configuration,input_data,grid)
  write_hdpara_file(output_hdpara_filepath,input_data,
                    number_of_riverflow_reservoirs,
                    riverflow_retention_coefficients,
                    number_of_overlandflow_reservoirs,
                    overlandflow_retention_coefficients,
                    number_of_baseflow_reservoirs,
                    baseflow_retention_coefficients)
end

function generate_parameters(configuration::Configuration,
                             input_data::InputData,
                             grid::Grid)
  grid_dimensions::Tuple{Int64} = get_grid_dimensions(grid)
  number_of_riverflow_reservoirs = SharedArray{Float64}(grid_dimensions)
  riverflow_retention_coefficients = SharedArray{Float64}(grid_dimensions)
  number_of_overlandflow_reservoirs = SharedArray{Float64}(grid_dimensions)
  overlandflow_retention_coefficients = SharedArray{Float64}(grid_dimensions)
  number_of_baseflow_reservoirs = SharedArray{Float64}(grid_dimensions)
  baseflow_retention_coefficients = SharedArray{Float64}(grid_dimensions)
  @sync @distributed for i in eachindex(input_data.landsea_mask)
    if ( ! input_data.landsea_mask[i] || ! input_data.glacier_mask[i] )
      distance::Float64 = calculate_distance(i,input_data)
      height_change::Float64 = calculate_height_change(i,input_data)
      number_of_riverflow_reservoirs[i],
      riverflow_retention_coefficients[i] =
        generate_riverflow_parameters(i,configuration.riverflow_formula,distance,height)
      number_of_overlandflow_reservoirs[i],
      overlandflow_retention_coefficients[i] =
        generate_overlandflow_parameters(i,configuration.overlandflow_formula,distance,height
                                         input_data)
      number_of_baseflow_reservoirs[i],
      baseflow_retention_coefficients[i]=
        generate_baseflow_parameters(i,configuration.baseflow_formula,distance,height,input_data)
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

function calculate_height_change(i::CartesianIndices,input_data::InputData)
  next_cell::CartesianIndices = input_data.get_next_cell_coords(i)
  height_change = orography(i) - orography(next_cell)
  return height_change
end

function calculate_distance(i::CartesianIndices,input_data::InputData)
  next_cell::CartesianIndices = input_data.get_next_cell_coords(i)
  earth_radius::Float64 = 6371000.0
  local distance::Float64
  if size(i) = 2
    river_direction::Int64 = input_data.river_directions(i)
    local lat_index_change::Int64
    local lon_index_change::Int64
    if river_direction <= 3
      lat_index_change = 1
    elseif river_direction >= 7
      lat_index_change = -1
    else
      lat_index_change = 0
    end
    if river_direction == 7 or
       river_direction == 4 or
       river_direction == 1
      lon_index_change = -1
    else if river_direction == 9 or
            river_direction == 6 or
            river_direction == 3
      lon_index_change = 1
    else
      lon_index_change = 0
    end
    distance = (((lat_index_change^2)*(input_data.dlat^2))+
                ((lon_index_change^2)*(input_data.dlon(i)^2))
  else
    working_dlat::Float64 = abs(input_data.lon(j) - input_data.lon(i))
    if working_dlat > 300
      working_dlat = abs(working_dlat - 360)
    end if
    pi_factor::Float64 = pi/180.0
    earths_radius::Float64 = 6371000.0
    dlon::Float64 = working_dlat*pi_factor*
                    cos(pi_factor*(input_data.lat(j)+input_data.lat(i))/2)*earths_radius
    dlat::Float64 = abs(input_data.lat(i)+input_data.lat(i))*pi_factor*earths_radius
    distance = sqrt(dlat^2+dlon^2)
  end
  return distance
end

function generate_riverflow_parameters(i::CartesianIndices,formula::RiverFlowSausen,
                                       distance::Float64,height_change::Float64)
  local number_of_riverflow_reservoirs::Float64
  if size(i) == 2
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

function generate_overlandflow_parameters(i::CartesianIndices,formula::OverlandFlowSausen,
                                          distance::Float64,height_change::Float64,
                                          input_data::InputData)
  if size(i) == 2
    number_of_overlandflow_reservoirs = 1.1107
  else
    number_of_overlandflow_reservoirs = 1.0
  end
  if height_change < formula.common_parameters.minimum_height_threshold
    height_change = formula.common_parameters.default_height_change
  end
  if input_data.innerslope[i] > 0
    if size(i) == 2
      dx0 = sqrt(input_data.dlat^2 + input_data.dlon[i]^2)
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
  end if
  overlandflow_retention_coefficient *= formula.overlandflow_torneaelven_k_multiplier
  return number_of_overlandflow_reservoirs,overlandflow_retention_coefficient
end

function generate_baseflow_parameters(i::CartesianIndices,formula::BaseFlowConstant,
                                      distance::Float64,height_change::Float64,
                                      input_data::InputData)
  return 1.0,formula.baseflow_k0
end

function generate_baseflow_parameters(i::CartesianIndices,formula::BaseFlowDistanceAndOrography,
                                      distance::Float64,height_change::Float64,
                                      input_data::InputData)
  bb = (input_data.orography_variance - 100.0)/(input_data.orography_variance + 1000.0)
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
