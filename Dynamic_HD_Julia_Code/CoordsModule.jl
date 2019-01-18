module CoordsModule

using UserExceptionModule: UserError
using SpecialDirectionCodesModule

abstract type Coords end

function is_ocean(coords::Coords)
  throw(UserError())
end

function is_outflow(coords::Coords)
  throw(UserError())
end

function is_truesink(coords::Coords)
  throw(UserError())
end

function is_lake(coords::Coords)
  throw(UserError())
end

struct LatLonCoords <: Coords
  lat::Int64
  lon::Int64
end

function is_ocean(coords::LatLonCoords)
  return coords.lat == coord_base_indicator_ocean &&
         coords.lon == coord_base_indicator_ocean
end

function is_outflow(coords::LatLonCoords)
  return coords.lat == coord_base_indicator_outflow &&
         coords.lon == coord_base_indicator_outflow
end

function is_truesink(coords::LatLonCoords)
  return coords.lat == coord_base_indicator_truesink &&
         coords.lon == coord_base_indicator_truesink
end

function is_lake(coords::LatLonCoords)
  return coords.lat == coord_base_indicator_lake &&
         coords.lon == coord_base_indicator_lake
end

struct DirectionIndicator
  coords::Coords
end

function get_next_cell_coords(flow_direction::DirectionIndicator,
                              coords::Coords)
  return flow_direction.coords
end

function is_ocean(flow_direction::DirectionIndicator)
  return is_ocean(flow_direction.coords)
end

function is_outflow(flow_direction::DirectionIndicator)
  return is_outflow(flow_direction.coords)
end

function is_truesink(flow_direction::DirectionIndicator)
  return is_truesink(flow_direction.coords)
end

function is_lake(flow_direction::DirectionIndicator)
  return is_lake(flow_direction.coords)
end

end
