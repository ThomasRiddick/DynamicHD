module GridModule

using UserExceptionModule: UserError
using CoordsModule: Coords,DirectionIndicator,LatLonCoords,get_next_cell_coords
using CoordsModule: LatLonSectionCoords,Generic1DCoords
using InteractiveUtils

abstract type Grid end

function for_all(function_on_point::Function,
                 grid::Grid)
  throw(UserError())
end

function for_all_fine_cells_in_coarse_cell(function_on_point::Function,
                                           fine_grid::Grid,coarse_grid::Grid,
                                           coarse_cell_coords::Coords)
  throw(UserError())
end

function find_downstream_coords(grid::Grid,
                                flow_direction::DirectionIndicator,
                                coords::Coords)
  throw(UserError())
end

function coords_in_grid(grid::Grid,coords::Coords)
  throw(UserError())
end

function wrap_coords!(grid::Grid,coords::Coords)
  throw(UserError())
end

function get_number_of_cells(grid::Grid)
  throw(UserError)
end

get_number_of_dimensions(obj::T) where {T <: Grid} =
  obj.number_of_dimensions::Int64

struct LatLonGrid <: Grid
  nlat::Int64
  nlon::Int64
  wrap_east_west::Bool
  number_of_dimensions::Int64
  function LatLonGrid(nlat::Int64,
                      nlon::Int64,
                      wrap_east_west::Bool)
    return new(nlat,nlon,wrap_east_west,2)
  end
end

struct UnstructuredGrid <: Grid
  ncells::Int64
  number_of_dimensions::Int64
  clat::Array{Float64,1}
  clon::Array{Float64,1}
  clat_bounds::Array{Float64,2}
  clon_bounds::Array{Float64,2}
  function UnstructuredGrid(ncells::Int64,clat::Array{Float64,1},clon::Array{Float64,1},
                            clat_bounds::Array{Float64,2},clon_bounds::Array{Float64,2})
    return new(ncells,1,clat,clon,clat_bounds,clon_bounds)
  end
end

UnstructuredGrid(ncells::Int64) = UnstructuredGrid(ncells,Array{Float64,1}(undef,ncells),Array{Float64,1}(undef,ncells),
                                                   Array{Float64,2}(undef,3,ncells),Array{Float64,2}(undef,3,ncells))

LatLonGridOrUnstructuredGrid = Union{LatLonGrid,UnstructuredGrid}

function for_all(function_on_point::Function,
                 grid::LatLonGrid)
  for j = 1:grid.nlon
    for i = 1:grid.nlat
        function_on_point(LatLonCoords(i,j))
      end
  end
end

function for_all(function_on_point::Function,
                 grid::UnstructuredGrid)
  for i = 1:grid.ncells
    function_on_point(Generic1DCoords(i))
  end
end

function for_all_with_line_breaks(function_on_point::Function,
                 grid::LatLonGrid)
  for i = 1:grid.nlat
    for j = 1:grid.nlon
      function_on_point(LatLonCoords(i,j))
    end
    println("")
  end
end

function for_section_with_line_breaks(function_on_point::Function,
                 grid::LatLonGrid,section_coords::LatLonSectionCoords)
  for i = section_coords.min_lat:section_coords.max_lat
    for j = section_coords.min_lon:section_coords.max_lon
      function_on_point(LatLonCoords(i,j))
    end
    println("")
  end
end

function for_all_fine_cells_in_coarse_cell(function_on_point::Function,
                                           fine_grid::LatLonGrid,
                                           coarse_grid::LatLonGrid,
                                           coarse_cell_coords::Coords)
  nlat_scale_factor = fine_grid.nlat/coarse_grid.nlat
  nlon_scale_factor = fine_grid.nlon/coarse_grid.nlon
  for j = 1+(coarse_cell_coords.lon - 1)*nlon_scale_factor:coarse_cell_coords.lon*nlon_scale_factor
    for i = 1+(coarse_cell_coords.lat - 1)*nlat_scale_factor:coarse_cell_coords.lat*nlat_scale_factor
      function_on_point(LatLonCoords(i,j))
    end
  end
end

function find_downstream_coords(grid::T,
                                flow_direction::DirectionIndicator,
                                coords::Coords) where {T<:LatLonGridOrUnstructuredGrid}
  return get_next_cell_coords(flow_direction,coords)
end

function coords_in_grid(grid::LatLonGrid,coords::LatLonCoords)
  return ((1 <= coords.lat <= grid.nlat) || wrap_east_west) &&
          (1 <= coords.lon <= grid.nlon)
end

function wrap_coords(grid::LatLonGrid,coords::LatLonCoords)
  wrapped_lon::Int64 = coords.lon
  if coords.lon < 1
    wrapped_lon += grid.nlon
  elseif coords.lon > grid.nlon
    wrapped_lon -= grid.nlon
  end
  return LatLonCoords(coords.lat,wrapped_lon)
end

function get_number_of_cells(grid::LatLonGrid)
  return grid.nlat*grid.nlon
end

function get_number_of_cells(grid::UnstructuredGrid)
  return grid.ncells
end

end
