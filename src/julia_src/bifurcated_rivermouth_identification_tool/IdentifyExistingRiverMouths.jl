module IdentifyExistingRiverMouths

using IdentifyBifurcatedRiverMouths: Cells

struct Area
  min_lat::Float64
  max_lat::Float64
  min_lon::Float64
  max_lon::Float64
end

function check_if_cell_is_inside_area(cell_lat_min::Float64,
                                      cell_lat_max::Float64,
                                      cell_lon_min::Float64,
                                      cell_lon_max::Float64,
                                      is_wrapped_cell::Bool,
                                      area::Area)
  wrapped_area::Bool = (area.min_lon > area.max_lon)
  is_in_area::Bool = area.max_lat > cell_min_lat &&
                     area.min_lat < cell_max_lat
  if is_wrapped_cell && wrapped_area
    return is_in_area
  elseif is_wrapped_cell || wrapped_area
    is_in_area = is_in_area &&
                   (area.max_lon > cell_min_lon ||
                    area.min_lon < cell_max_lon)
  else
    is_in_area = is_in_area &&
                   area.max_lon > cell_min_lon &&
                   area.min_lon < cell_max_lon
  end
  return is_in_area
end

function identify_existing_river_mouth(cells::Cells,
                                       accumulated_flow::Array{Int64},
                                       area::Area)
  filtered_cell_indices::Array{CartesianIndex} =
    filter(i::CartesianIndex -> check_if_cell_is_inside_area(
                                  cells.cell_extremes.min_lats[i],
                                  cells.cell_extremes.max_lats[i],
                                  cells.cell_extremes.min_lons[i],
                                  cells.cell_extremes.max_lons[i],
                                  cells.is_wrapped_cell[i],
                                  area),
           cells.cell_indices)
  filtered_accumulated_flow::Array{Int64} =
    map(i::CartesianIndex -> accumulated_flow[i],filtered_cell_indices)
  return filtered_cell_indices[argmax(filtered_accumulated_flow)]
end

function identify_existing_river_mouths(cells::Cells,
                                        accumulated_flow::Array{Int64},
                                        search_areas::Dict{String,Area})
  existing_river_mouths::Dict{String,Area} = Dict{String,Area}()
  for (name,search_area) in search_areas
    existing_river_mouths[name] =
      identify_existing_river_mouth(cells,
                                    accumulated_flow,
                                    search_area)
  end
  return existing_river_mouths
end

end
