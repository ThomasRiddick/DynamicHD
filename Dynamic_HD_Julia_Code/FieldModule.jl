module FieldModule

using Printf: @printf
using UserExceptionModule: UserError
using CoordsModule: Coords,LatLonCoords,DirectionIndicator
using GridModule: Grid,LatLonGrid, for_all, wrap_coords, coords_in_grid,for_all_with_line_breaks
using InteractiveUtils: subtypes
using SpecialDirectionCodesModule
import Base.maximum
import Base.+
import Base.fill!
import Base.show

abstract type Field{T} end

function set!(field::Field,coords::Coords)
  throw(UserError())
end

function fill!(field::Field)
  throw(UserError())
end

function maximum(field::Field)
  throw(UserError())
end

function +(lfield::Field,rfield::Field)
  throw(UserError())
end

function invert(field::Field)
    throw(UserError())
end

function add_offset(field::Field,offset::T,skip_offset_on_values::Vector{T}) where {T}
    throw(UserError())
end

function add_offset(field::T2,offset::T,skip_offset_on_values::Vector{T}) where {T,T2<:Field{T}}
  for_all(get_grid(field)) do coords::Coords
    if ! (field(coords) in skip_offset_on_values)
      set!(field,coords,field(coords)+offset)
    end
  end
end

get_grid(obj::T) where {T <: Field} =
  obj.grid::Grid

function repeat(field::Field{T}, count::Integer) where {T}
  array::Array{Field{T},1} = Array{Field{T},1}(undef,count)
  for i in 1:count
    array[i] = deepcopy(field)
  end
  return array
end



struct LatLonField{T} <: Field{T}
  data::AbstractArray{T,2}
  grid::LatLonGrid
  function LatLonField{T}(grid::LatLonGrid,value::T) where {T}
    data::Array{T,2} = value == zero(T) ? zeros(T,grid.nlat,grid.nlon) :
                                          ones(T,grid.nlat,grid.nlon)*value
    return new(data,grid)
  end
  function LatLonField{T}(grid::LatLonGrid,values::AbstractArray{T,2}) where {T}
    if size(values,1) != grid.nlat ||
       size(values,2) != grid.nlon
       error("Values provided don't match selected grid")
    end
    return new(values,grid)
  end
end

LatLonField{T}(grid::LatLonGrid) where {T} = LatLonField{T}(grid,zeros(T,grid.nlat,grid.nlon))

Field{T}(grid::LatLonGrid,value::T) where {T} = LatLonField{T}(grid::LatLonGrid,value::T)
Field{T}(grid::LatLonGrid) where {T} = LatLonField{T}(grid::LatLonGrid)

function (latlon_field::LatLonField)(coords::LatLonCoords)
    return latlon_field.data[coords.lat,coords.lon]
end

function set!(latlon_field::LatLonField{T},coords::LatLonCoords,value::T) where {T}
    latlon_field.data[coords.lat,coords.lon] = value
end

function +(lfield::LatLonField{T},rfield::LatLonField{T}) where {T}
  return LatLonField{T}(lfield.grid,lfield.data + rfield.data)
end

function invert(field::LatLonField{T}) where {T}
  return LatLonField{T}(field.grid,.!field.data)
end

function show(io::IO,field::LatLonField{T}) where {T <: Number}
  for_all_with_line_breaks(field.grid) do coords::Coords
    if T <: AbstractFloat
      @printf(io,"%6.2f",field(coords))
    elseif T == Bool
      print(io,field(coords) ? " X " : " - ")
    else
      @printf(io,"%4.0f",field(coords))
    end
    print(io," ")
  end
end

abstract type DirectionIndicators end

function set!(field::DirectionIndicators,coords::Coords)
  throw(UserError())
end

function maximum(field::LatLonField)
  return maximum(field.data)
end

struct LatLonDirectionIndicators <: DirectionIndicators
  next_cell_coords::Array{Field{Int64},1}
  function LatLonDirectionIndicators(dir_based_rdirs::LatLonField{Int64})
    grid = dir_based_rdirs.grid
    nlat = dir_based_rdirs.grid.nlat
    nlon = dir_based_rdirs.grid.nlon
    lat_indices::LatLonField{Int64} =
      LatLonField{Int64}(grid,zeros(Int64,nlat,nlon))
    lon_indices::LatLonField{Int64} =
      LatLonField{Int64}(grid,zeros(Int64,nlat,nlon))
    for_all(grid) do coords::LatLonCoords
      dir_based_rdir::Int64 = dir_based_rdirs(coords)
      if dir_based_rdir == dir_based_indicator_ocean
        set!(lat_indices,coords,coord_base_indicator_ocean)
        set!(lon_indices,coords,coord_base_indicator_ocean)
      elseif dir_based_rdir == dir_based_indicator_outflow
        set!(lat_indices,coords,coord_base_indicator_outflow)
        set!(lon_indices,coords,coord_base_indicator_outflow)
      elseif dir_based_rdir == dir_based_indicator_truesink
        set!(lat_indices,coords,coord_base_indicator_truesink)
        set!(lon_indices,coords,coord_base_indicator_truesink)
      elseif dir_based_rdir == dir_based_indicator_lake
        set!(lat_indices,coords,coord_base_indicator_lake)
        set!(lon_indices,coords,coord_base_indicator_lake)
      elseif 1 <= dir_based_rdir <= 9
        lat_offset::Int64 = dir_based_rdir in [7,8,9] ? -1 :
                           (dir_based_rdir in [1,2,3] ?  1 : 0)
        lon_offset::Int64 = dir_based_rdir in [7,4,1] ? -1 :
                           (dir_based_rdir in [9,6,3] ?  1 : 0)
        destination_coords::LatLonCoords = LatLonCoords(coords.lat + lat_offset,
                                                        coords.lon + lon_offset)
        destination_coords = wrap_coords(grid,destination_coords)
        if ! coords_in_grid(grid,destination_coords)
          error("Direction based direction indicator points out of grid")
        end
        set!(lat_indices,coords,destination_coords.lat)
        set!(lon_indices,coords,destination_coords.lon)
      else
        error("Invalid direction based direction indicator")
      end
    end
    new(Field{Int64}[lat_indices,lon_indices])
  end
end

function (latlon_direction_indicators::LatLonDirectionIndicators)(coords::LatLonCoords)
  return DirectionIndicator(LatLonCoords(latlon_direction_indicators.next_cell_coords[1](coords),
                                         latlon_direction_indicators.next_cell_coords[2](coords)))
end


function set!(latlon_direction_indicators::DirectionIndicators,
               coords::LatLonCoords,direction::DirectionIndicator)
  next_cell_coords =  get_next_cell_coords(direction,coords)
  set!(latlon_direction_indicators.next_cell_coords[1],coords,next_cell_coords.lat)
  set!(latlon_direction_indicators.next_cell_coords[2],coords,next_cell_coords.lon)
end

function fill!(field::LatLonField{T},value::T) where {T}
  return fill!(field.data,value)
end

#Cannot define functors on abstract types in Julia yet. Use this
#workaround - needs to be at end of file as it only acts on subtypes of
#Field/DirectionIndicators that are already defined

for T in subtypes(Field)
 @eval function (field::$T)(coords::Coords) throw(UserError()) end
end

for T in subtypes(DirectionIndicators)
  @eval function (direction_indicators::$T)(coords::Coords) throw(UserError()) end
end

end
