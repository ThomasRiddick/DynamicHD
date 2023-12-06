module FieldModule

using Printf: @printf
using UserExceptionModule: UserError
using CoordsModule: Coords,LatLonCoords,DirectionIndicator,Generic1DCoords
using GridModule: Grid,LatLonGrid, for_all, wrap_coords, coords_in_grid,for_all_with_line_breaks
using GridModule: UnstructuredGrid
using InteractiveUtils: subtypes
using SpecialDirectionCodesModule
import Base.maximum
import Base.*
import Base./
import Base.+
import Base.-
import Base.==
import Base.isapprox
import Base.fill!
import Base.show
import Base.round
import Base.convert
import Base.sum
import Base.count
import Base.Broadcast
using InteractiveUtils

ArrayOrSharedArray = Union{Array,SharedArray}

abstract type Field{T,A<:ArrayOrSharedArray} end

function get(field::Field{T,A},coords::Coords) where {T,A<:ArrayOrSharedArray}
  throw(UserError())
end

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

function -(lfield::Field,rfield::Field)
  throw(UserError())
end

function *(lfield::Field,value::T) where {T<:Number}
  throw(UserError())
end

function /(lfield::Field,value::T) where {T<:Number}
  throw(UserError())
end

function ==(lfield::Field,rfield::Field)
  throw(UserError())
end

function sum(field::Field)
  throw(UserError())
end

function count(field::Field)
  throw(UserError())
end

# This could be done by overloading broadcast but this is overcomplicated
# Assumes result is fractional and hence returns a field of floating point numbers
function elementwise_divide(lfield::Field{T,A},rfield::Field{T,A}) where {T,A<:ArrayOrSharedArray}
  return Field{Float64}(lfield.grid,lfield.data./rfield.data)::Field{Float64}
end

# This could be done by overloading broadcast but this is overcomplicated
function elementwise_multiple(lfield::Field{T,A},rfield::Field{T,A}) where {T,A<:ArrayOrSharedArray}
  return Field{T,A}(lfield.grid,lfield.data.*rfield.data)::Field{T,A}
end

# This could be done by overloading broadcast but this is overcomplicated
function equals(lfield::Field,value::T) where {T}
  return Field{Bool}(lfield.grid,Bool.(lfield.data .== value))::Field{Bool}
end

function isapprox(lfield::Field,rfield::Field;
                  rtol::Real=atol>0 ? 0 : sqrt(eps),
                  atol::Real,nans::Bool)
  throw(UserError())
end

function invert(field::Field)
    throw(UserError())
end

function add_offset(field::Field,offset::T,skip_offset_on_values::Vector{T}) where {T}
    throw(UserError())
end

function round!(type::DataType,field::Field{T,A}) where {T,A<:ArrayOrSharedArray}
  throw(UserError())
end

function add_offset(field::T2,offset::T,skip_offset_on_values::Vector{T}) where
    {T,T2<:Field{T,A<:ArrayOrSharedArray}}
  for_all(get_grid(field)) do coords::Coords
    if ! (field(coords) in skip_offset_on_values)
      set!(field,coords,field(coords)+offset)
    end
  end
end

function divide(field::T2,divisor::T)where
    {T,T2<:Field{T,A<:ArrayOrSharedArray}}
  for_all(get_grid(field)) do coords::Coords
    set!(field,coords,field(coords)/divisor)
  end
end

get_grid(obj::T) where {T <: Field} =
  obj.grid::Grid

get_data(obj::T) where {T <: Field{T2,A<:ArrayOrSharedArray}} =
  obj.data::Array{T2,A}

function repeat(field::Field{T,A}, count::Integer) where {T,A<:ArrayOrSharedArray}
  array::Array{Field{T,A},1} = Array{Field{T,A},1}(undef,count)
  for i in 1:count
    array[i] = deepcopy(field)
  end
  return array
end


struct LatLonField{T,A} <: Field{T,A}
  data::Array{T,2}
  grid::LatLonGrid
  function LatLonField{T,A}(grid::LatLonGrid,value::T) where {T,A<:ArrayOrSharedArray}
    data::A{T,2} = value == zero(T) ? convert(A,zeros(T,grid.nlat,grid.nlon)) :
                                      convert(A,ones(T,grid.nlat,grid.nlon))*value
    return new(data,grid)
  end
  function LatLonField{T,A}(grid::LatLonGrid,values::AbstractArray{T,2}) where
      {T,A<:ArrayOrSharedArray}
    if size(values,1) != grid.nlat ||
       size(values,2) != grid.nlon
       error("Values provided don't match selected grid")
    end
    return new(values,grid)
  end
end

struct UnstructuredField{T,A} <: Field{T,A}
  data::A{T,1}
  grid::UnstructuredGrid
  function UnstructuredField{T,A}(grid::UnstructuredGrid,value::T) where {T,A<:ArrayOrSharedArray}
    data::A{T,1} = value == zero(T) ? convert(A,zeros(T,grid.ncells)) :
                                      convert(A,ones(T,grid.ncells))*value
    return new(data,grid)
  end
  function UnstructuredField{T,A}(grid::UnstructuredGrid,values::AbstractArray{T,1}) where
      {T,A<:ArrayOrSharedArray}
    if size(values,1) != grid.ncells
       error("Values provided don't match selected grid")
    end
    return new(values,grid)
  end
end

LatLonField{T,A}(grid::LatLonGrid) where {T,A<:ArrayOrSharedArray} =
  LatLonField{T,A}(grid,convert(A,zeros(T,grid.nlat,grid.nlon)))

UnstructuredField{T,A}(grid::UnstructuredGrid) where {T,A<:ArrayOrSharedArray} =
  UnstructuredField{T,A}(grid,convert(A,zeros(T,grid.ncells)))

Field{T,A}(grid::LatLonGrid,value::T) where {T,A<:ArrayOrSharedArray} =
  LatLonField{T,A}(grid::LatLonGrid,value::T)
Field{T,A}(grid::LatLonGrid) where {T,A<:ArrayOrSharedArray} =
  LatLonField{T,A}(grid::LatLonGrid)

Field{T,A}(grid::UnstructuredGrid,value::T) where {T,A<:ArrayOrSharedArray} =
  UnstructuredField{T,A}(grid::UnstructuredGrid,value::T)
Field{T,A}(grid::UnstructuredGrid) where {T,A<:ArrayOrSharedArray} =
  UnstructuredField{T,A}(grid::UnstructuredGrid)

Field{T,A}(grid::LatLonGrid,values::AbstractArray{T,2}) where {T,A<:ArrayOrSharedArray} =
  LatLonField{T,A}(grid::LatLonGrid,values::AbstractArray{T,2})

Field{T,A}(grid::UnstructuredGrid,values::AbstractArray{T,1}) where
  {T,A<:ArrayOrSharedArray} = UnstructuredField{T,A}(grid::UnstructuredGrid,
                                                     values::AbstractArray{T,1})

function (latlon_field::LatLonField{T,A})(coords::LatLonCoords) where {T,A<:ArrayOrSharedArray}
  return latlon_field.data[coords.lat,coords.lon]::T
end

function get(latlon_field::LatLonField{T,A},coords::LatLonCoords) where {T,A<:ArrayOrSharedArray}
  return latlon_field(coords)::T
end

function set!(latlon_field::LatLonField{T,A},coords::LatLonCoords,value::T) where
    {T,A<:ArrayOrSharedArray}
  latlon_field.data[coords.lat,coords.lon] = value
end

function +(lfield::LatLonField{T,A},rfield::LatLonField{T,A}) where {T,A<:ArrayOrSharedArray}
  return LatLonField{T,A}(lfield.grid,lfield.data + rfield.data)
end

function -(lfield::LatLonField{T,A},rfield::LatLonField{T,A}) where {T,A<:ArrayOrSharedArray}
  return LatLonField{T,A}(lfield.grid,lfield.data - rfield.data)
end

function *(lfield::Field,value::T) where {T<:Number,A<:ArrayOrSharedArray}
  return LatLonField{T,A}(lfield.grid,lfield.data * value)
end

function /(lfield::Field,value::T) where {T<:Number,A<:ArrayOrSharedArray}
  return LatLonField{T,A}(lfield.grid,lfield.data / value)
end

function ==(lfield::LatLonField{T,A},rfield::LatLonField{T,A}) where {T,A<:ArrayOrSharedArray}
  return (lfield.data == rfield.data)::Bool
end

LatLonFieldOrUnstructuredField = Union{LatLonField{T,A},UnstructuredField{T,A}} where
  {T,A<:ArrayOrSharedArray}

function sum(field::T) where {T<:LatLonFieldOrUnstructuredField}
  return sum(field.data)
end

function count(field::T) where {T<:LatLonFieldOrUnstructuredField}
  return count(field.data)
end

function isapprox(lfield::LatLonField{T,A},rfield::LatLonField{T,A};
                  rtol::Real=atol>0 ? 0 : sqrt(eps),
                  atol::Real=0.0,nans::Bool=false) where {T,A<:ArrayOrSharedArray}
  return isapprox(lfield.data,rfield.data,
                 rtol=rtol,atol=atol,nans=nans)::Bool
end

function invert(field::LatLonField{T,A}) where {T,A<:ArrayOrSharedArray}
  return LatLonField{T,A}(field.grid,.!field.data)
end

function round(type::DataType,field::LatLonField{T,A}) where {T,A<:ArrayOrSharedArray}
  return LatLonField{type}(field.grid,round.(type,field.data))
end

function maximum(field::LatLonField)
  return maximum(field.data)
end

function show(io::IO,field::LatLonField{T,A}) where {T <: Number,A<:ArrayOrSharedArray}
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

function fill!(field::LatLonField{T,A},value::T) where {T,A<:ArrayOrSharedArray}
  return fill!(field.data,value)
end

function (unstructured_field::UnstructuredField{T,A})(coords::Generic1DCoords) where
    {T,A<:ArrayOrSharedArray}
  return unstructured_field.data[coords.index]::T
end

function get(unstructured_field::UnstructuredField{T,A},coords::Generic1DCoords) where
    {T,A<:ArrayOrSharedArray}
  return unstructured_field(coords)::T
end

function set!(unstructured_field::UnstructuredField{T,A},coords::Generic1DCoords,value::T) where
    {T,A<:ArrayOrSharedArray}
  unstructured_field.data[coords.index] = value
end

function +(lfield::UnstructuredField{T,A},rfield::UnstructuredField{T,A}) where
    {T,A<:ArrayOrSharedArray}
  return UnstructuredField{T}(lfield.grid,lfield.data + rfield.data)
end

function -(lfield::UnstructuredField{T,A},rfield::UnstructuredField{T,A}) where
    {T,A<:ArrayOrSharedArray}
  return UnstructuredField{T}(lfield.grid,lfield.data - rfield.data)
end

function *(lfield::UnstructuredField,value::T) where {T<:Number,A<:ArrayOrSharedArray}
  return UnstructuredField{T,A}(lfield.grid,lfield.data * value)
end

function /(lfield::UnstructuredField,value::T) where {T<:Number,A<:ArrayOrSharedArray}
  return UnstructuredField{T,A}(lfield.grid,lfield.data / value)
end

function ==(lfield::UnstructuredField{T,A},rfield::UnstructuredField{T,A}) where
    {T,A<:ArrayOrSharedArray}
  return (lfield.data == rfield.data)::Bool
end

function isapprox(lfield::UnstructuredField{T,A},rfield::UnstructuredField{T,A};
                  rtol::Real=atol>0 ? 0 : sqrt(eps),
                  atol::Real=0.0,nans::Bool=false) where {T,A<:ArrayOrSharedArray}
  return isapprox(lfield.data,rfield.data,
                 rtol=rtol,atol=atol,nans=nans)::Bool
end

function invert(field::UnstructuredField{T,A}) where {T,A<:ArrayOrSharedArray}
  return UnstructuredField{T}(field.grid,.!field.data)
end

function round(type::DataType,field::UnstructuredField{T,A}) where {T,A<:ArrayOrSharedArray}
  return UnstructuredField{type}(field.grid,round.(type,field.data))
end

function maximum(field::UnstructuredField)
  return maximum(field.data)
end

function fill!(field::UnstructuredField{T,A},value::T) where {T,A<:ArrayOrSharedArray}
  return fill!(field.data,value)
end

function show(io::IO,field::UnstructuredField{T,A}) where
    {T <: Number,A<:ArrayOrSharedArray}
  for_all_with_line_breaks(field.grid) do coords::Coords
    if T <: AbstractFloat
      @printf(io,"%6.8f",field(coords))
    elseif T == Bool
      print(io,field(coords) ? " X " : " - ")
    else
      @printf(io,"%4.0f",field(coords))
    end
    print(io," ")
  end
end

abstract type DirectionIndicators{A} end where {A<:ArrayOrSharedArray}

function get(field::DirectionIndicators{A},coords::Coords) where
    {A<:ArrayOrSharedArray}
  throw(UserError())
end

function set!(field::DirectionIndicators{A},coords::Coords) where
    {A<:ArrayOrSharedArray}
  throw(UserError())
end

struct LatLonDirectionIndicators{A} <: DirectionIndicators{A} where
    {A<:ArrayOrSharedArray}
  next_cell_coords::A{LatLonField{Int64},1}
  function LatLonDirectionIndicators(dir_based_rdirs::LatLonField{Int64,A}) where
      {A<:ArrayOrSharedArray}
    grid = dir_based_rdirs.grid
    nlat = dir_based_rdirs.grid.nlat
    nlon = dir_based_rdirs.grid.nlon
    lat_indices::LatLonField{Int64,A} =
      LatLonField{Int64,A}(grid,convert(A,zeros(Int64,nlat,nlon)))
    lon_indices::LatLonField{Int64,A} =
      LatLonField{Int64,A}(grid,convert(A,zeros(Int64,nlat,nlon)))
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
    new(LatLonField{Int64,A}[lat_indices,lon_indices])
  end
end

struct UnstructuredFieldDirectionIndicators{A} <: DirectionIndicators{A} where
    {A<:ArrayOrSharedArray}
  next_cell_coords::UnstructuredField{Int64,A}
end

function (latlon_direction_indicators::LatLonDirectionIndicators)(coords::LatLonCoords)
  return DirectionIndicator(LatLonCoords(latlon_direction_indicators.next_cell_coords[1](coords),
                                         latlon_direction_indicators.next_cell_coords[2](coords)))
end

function get(latlon_direction_indicators::LatLonDirectionIndicators,coords::LatLonCoords)
  return latlon_direction_indicators(coords)
end


function set!(latlon_direction_indicators::DirectionIndicators,
               coords::LatLonCoords,direction::DirectionIndicator)
  next_cell_coords =  get_next_cell_coords(direction,coords)
  set!(latlon_direction_indicators.next_cell_coords[1],coords,next_cell_coords.lat)
  set!(latlon_direction_indicators.next_cell_coords[2],coords,next_cell_coords.lon)
end

function fill!(field::LatLonField{T,A},value::T) where {T,A<:ArrayOrSharedArray}
  return fill!(field.data,value)
end

struct UnstructuredDirectionIndicators{A} <: DirectionIndicators{A} where
    {A<:ArrayOrSharedArray}
  next_cell_coords::UnstructuredField{Int64,A}
end

function (unstructured_direction_indicators::UnstructuredDirectionIndicators)(coords::Generic1DCoords)
  return DirectionIndicator(Generic1DCoords(unstructured_direction_indicators.next_cell_coords(coords)))
end

function get(unstructured_direction_indicators::UnstructuredDirectionIndicators,
             coords::Generic1DCoords)
  return unstructured_direction_indicators(coords)
end


function set!(unstructured_direction_indicators::UnstructuredDirectionIndicators,
              coords::Generic1DCoords,direction::DirectionIndicator)
  next_cell_coords =  get_next_cell_coords(direction,coords)
  set!(unstructured_direction_indicators.next_cell_coords,coords,next_cell_coords.index)
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
