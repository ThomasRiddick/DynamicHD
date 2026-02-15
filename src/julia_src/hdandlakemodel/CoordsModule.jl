module CoordsModule

using UserExceptionModule: UserError
using SpecialDirectionCodesModule
import Base.==

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

function are_valid(coords::Coords)
  throw(UserError())
end

function get_linear_index(coords::Coords,
                          linear_indices::LinearIndices)
  throw(UserError())
end

function ==(lcoords::Coords,rcoords::Coords)
  throw(UserError())
end

struct LatLonCoords <: Coords
  lat::Int64
  lon::Int64
end

struct Generic1DCoords <: Coords
  index::Int64
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

function ==(lcoords::LatLonCoords,rcoords::LatLonCoords)
  return (lcoords.lat == rcoords.lat &&
          lcoords.lon == rcoords.lon)::Bool
end

function get_linear_index(coords::LatLonCoords,
                          linear_indices::LinearIndices)
  return linear_indices[coords.lat,coords.lon]
end

function is_ocean(coords::Generic1DCoords)
  return coords.index == coord_base_indicator_ocean
end

function is_outflow(coords::Generic1DCoords)
  return coords.index == coord_base_indicator_outflow
end

function is_truesink(coords::Generic1DCoords)
  return coords.index == coord_base_indicator_truesink
end

function is_lake(coords::Generic1DCoords)
  return coords.index == coord_base_indicator_lake
end

function ==(lcoords::Generic1DCoords,rcoords::Generic1DCoords)
  return lcoords.index == rcoords.index
end

function get_linear_index(coords::Generic1DCoords,
                          linear_indices::LinearIndices)
  return coords.index
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

abstract type SectionCoords end

struct LatLonSectionCoords <: SectionCoords
  min_lat::Int64
  max_lat::Int64
  min_lon::Int64
  max_lon::Int64
end

end
