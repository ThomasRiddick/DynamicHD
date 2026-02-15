module FieldModule

using Printf: @printf
using UserExceptionModule: UserError
using CoordsModule: Coords,LatLonCoords,DirectionIndicator,Generic1DCoords
using GridModule: Grid,LatLonGrid, for_all, wrap_coords, coords_in_grid,for_all_with_line_breaks
using GridModule: UnstructuredGrid, for_all_parallel
using InteractiveUtils: subtypes
using SpecialDirectionCodesModule
using SharedArrays
import Base.maximum
import Base.*
import Base./
import Base.+
import Base.-
import Base.>=
import Base.==
import Base.isapprox
import Base.fill!
import Base.show
import Base.round
import Base.convert
import Base.sum
import Base.count
import Base.length
import Base.broadcastable
using InteractiveUtils

array_type = :SharedArray

abstract type Field{T} end

function get(field::Field{T},coords::Coords) where {T}
  throw(UserError())
end

function set!(field::Field{T},coords::Coords,value::T) where{T}
  throw(UserError())
end

function get(field::Field{T},coords::CartesianIndex) where {T}
  throw(UserError())
end

function set!(field::Field,coords::CartesianIndex,value::T) where {T}
  field.data[coords] = value
end

for operator in (:fill!,:maximum,:sum,:count,:invert,:length,:broadcastable)
  @eval function $operator(field::Field) throw(UserError()) end
end

for operator in (:+, :-, Symbol("=="))
  @eval function $operator(lfield::Field,rfield::Field) throw(UserError()) end
end

for operator in (:*, :/, :>=)
  @eval function $operator(lfield::Field,value::T) where {T<:Number} throw(UserError()) end
end

# This could be done by overloading broadcast but this is overcomplicated
# Assumes result is fractional and hence returns a field of floating point numbers
function elementwise_divide(lfield::Field{T},rfield::Field{T}) where {T}
  return Field{Float64}(lfield.grid,
                        Float64.(lfield.data)./Float64.(rfield.data))::Field{Float64}
end

# This could be done by overloading broadcast but this is overcomplicated
function elementwise_multiple(lfield::Field{T},rfield::Field{T}) where {T}
  return Field{T}(lfield.grid,lfield.data.*rfield.data)::Field{T}
end

@eval begin
  # This could be done by overloading broadcast but this is overcomplicated
  function equals(lfield::Field,value::T) where {T}
    return Field{Bool}(lfield.grid,
                       convert($(array_type){Bool},
                               map(x->x==value,lfield.data)))::Field{Bool}
  end
end

function isapprox(lfield::Field,rfield::Field;
                  rtol::Real=atol>0 ? 0 : sqrt(eps),
                  atol::Real,nans::Bool)
  throw(UserError())
end

function add_offset(field::Field,offset::T,skip_offset_on_values::Vector{T}) where {T}
    throw(UserError())
end

function round!(type::DataType,field::Field{T}) where {T}
  throw(UserError())
end

function add_offset(field::T2,offset::T,skip_offset_on_values::Vector{T}) where
    {T,T2<:Field{T}}
  for_all(get_grid(field)) do coords::Coords
    if ! (field(coords) in skip_offset_on_values)
      set!(field,coords,field(coords)+offset)
    end
  end
end

function divide(field::T2,divisor::T)where
    {T,T2<:Field{T}}
  for_all(get_grid(field)) do coords::Coords
    set!(field,coords,field(coords)/divisor)
  end
end

function repeat_init(grid::Grid,value::T,count::Integer) where {T}
  throw(UserError())
end

get_grid(obj::T) where {T <: Field} =
  obj.grid::Grid

@eval begin
  get_data(obj::T2) where {T,T2<:Field{T}} =
    obj.data::$(array_type){T}
end

function repeat(field::Field{T}, count::Integer,
                deep_copy::Bool=true) where {T}
  array::Array{Field{T},1} = Array{Field{T},1}(undef,count)
  for i in 1:count
    if deep_copy
      array[i] = deepcopy(field)
    else
      array[i] = field
    end
  end
  return array
end

@eval begin
  struct LatLonField{T} <: Field{T}
    data::$(array_type){T,2}
    grid::LatLonGrid
    function LatLonField{T}(grid::LatLonGrid,value::T) where {T}
      data::$(array_type){T,2} =
        value == zero(T) ? convert($(array_type),zeros(T,grid.nlat,grid.nlon)) :
                           convert($(array_type),ones(T,grid.nlat,grid.nlon))*value
      return new(data,grid)
    end
    function LatLonField{T}(grid::LatLonGrid,values::AbstractArray{T,2}) where
        {T}
      if size(values,1) != grid.nlat ||
         size(values,2) != grid.nlon
         error("Values provided don't match selected grid")
      end
      return new(convert($(array_type),values),grid)
    end
  end

  struct UnstructuredField{T} <: Field{T}
    data::$(array_type){T,1}
    grid::UnstructuredGrid
    function UnstructuredField{T}(grid::UnstructuredGrid,value::T) where {T}
      data::$(array_type){T,1} = value == zero(T) ? convert($(array_type),zeros(T,grid.ncells)) :
                                                  convert($(array_type),ones(T,grid.ncells))*value
      return new(data,grid)
    end
    function UnstructuredField{T}(grid::UnstructuredGrid,values::AbstractArray{T,1}) where
        {T}
      if size(values,1) != grid.ncells
         error("Values provided don't match selected grid")
      end
      return new(convert($(array_type),values),grid)
    end
  end

  LatLonField{T}(grid::LatLonGrid) where {T} =
    LatLonField{T}(grid,convert($(array_type),zeros(T,grid.nlat,grid.nlon)))

  UnstructuredField{T}(grid::UnstructuredGrid) where {T} =
    UnstructuredField{T}(grid,convert($(array_type),zeros(T,grid.ncells)))
end

for type_prefix in (:LatLon, :Unstructured)
  grid_type = Symbol(type_prefix,"Grid")
  field_type = Symbol(type_prefix,"Field")
  @eval begin
    Field{T}(grid::$(grid_type),value::T) where {T} =
      $(field_type){T}(grid::$(grid_type),value::T)

    Field{T}(grid::$(grid_type)) where {T} =
      $(field_type){T}(grid::$(grid_type))
  end
end

Field{T}(grid::LatLonGrid,values::AbstractArray{T,2}) where {T} =
  LatLonField{T}(grid::LatLonGrid,values::AbstractArray{T,2})

Field{T}(grid::UnstructuredGrid,values::AbstractArray{T,1}) where {T} =
  UnstructuredField{T}(grid::UnstructuredGrid,values::AbstractArray{T,1})

function (latlon_field::LatLonField{T})(coords::LatLonCoords) where {T}
  return latlon_field.data[coords.lat,coords.lon]::T
end

function get(latlon_field::LatLonField{T},coords::LatLonCoords) where {T}
  return latlon_field(coords)::T
end

function set!(latlon_field::LatLonField{T},coords::LatLonCoords,value::T) where
    {T}
  latlon_field.data[coords.lat,coords.lon] = value
end

function (latlon_field::LatLonField{T})(coords::CartesianIndex) where {T}
  return latlon_field.data[coords]::T
end

function get(field::Field{T},coords::CartesianIndex) where {T}
  return latlon_field(coords)::T
end

function repeat_init(grid::LatLonGrid,value::T,count::Integer) where {T}
  array::Array{Field{T},1} = Array{Field{T},1}(undef,count)
  for i in 1:count
    array[i] = LatLonField{T}(grid,value)
  end
  return array
end

for field_type in (:LatLonField,:UnstructuredField)
  @eval begin
    function +(lfield::$(field_type){T},rfield::$(field_type){T}) where {T}
      return $(field_type){T}(lfield.grid,lfield.data + rfield.data)
    end

    function -(lfield::$(field_type){T},rfield::$(field_type){T}) where {T}
      return $(field_type){T}(lfield.grid,lfield.data - rfield.data)
    end

    function *(lfield::$(field_type),value::T) where {T<:Number}
      return $(field_type){T}(lfield.grid,lfield.data * value)
    end

    function >=(lfield::$(field_type),value::T) where {T<:Number}
      return $(field_type){Bool}(lfield.grid,lfield.data .>= value)
    end

    function /(lfield::$(field_type),value::T) where {T<:Number}
      return $(field_type){T}(lfield.grid,lfield.data / value)
    end

    function ==(lfield::$(field_type){T},rfield::$(field_type){T}) where {T}
      return (lfield.data == rfield.data)::Bool
    end

    function isapprox(lfield::$(field_type){T},rfield::$(field_type){T};
                      rtol::Real=atol>0 ? 0 : sqrt(eps),
                      atol::Real=0.0,nans::Bool=false) where {T}
      return isapprox(lfield.data,rfield.data,
                     rtol=rtol,atol=atol,nans=nans)::Bool
    end

    function invert(field::$(field_type){T}) where {T}
      #Avoid converting to bit array by using map
      return $(field_type){T}(field.grid,convert($(array_type),map(!,field.data)))
    end

    function round(type::DataType,field::$(field_type){T}) where {T}
      return $(field_type){type}(field.grid,round.(type,field.data))
    end

    function maximum(field::$(field_type))
      return maximum(field.data)
    end

    function fill!(field::$(field_type){T},value::T) where {T}
      if isa(field.data,SharedArray)
        field_data::SharedArray = field.data
        for_all_parallel(field.grid) do coords::CartesianIndex
          field_data[coords] = value
        end
      else
        fill!(field.data,value)
      end
    end

    function sum(field::$(field_type))
      return sum(field.data)
    end

    function count(field::$(field_type))
      return count(field.data)
    end

    function length(field::$(field_type))
      return length(field.data)
    end

    function broadcastable(field::$(field_type))
      return field.data::$(array_type)
    end
  end
end

function repeat_init(grid::UnstructuredGrid,value::T,count::Integer) where {T}
  array::Array{Field{T},1} = Array{Field{T},1}(undef,count)
  for i in 1:count
    array[i] = UnstructuredField{T}(grid,value)
  end
  return array
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

function (unstructured_field::UnstructuredField{T})(coords::Generic1DCoords) where
    {T}
  return unstructured_field.data[coords.index]::T
end

function get(unstructured_field::UnstructuredField{T},coords::Generic1DCoords) where
    {T}
  return unstructured_field(coords)::T
end

function set!(unstructured_field::UnstructuredField{T},coords::Generic1DCoords,value::T) where
    {T}
  unstructured_field.data[coords.index] = value
end

function show(io::IO,field::UnstructuredField{T}) where
    {T <: Number}
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

function get_data_vector(field_vector::Array{Field{T}}) where {T}
  data_vector::Array{SharedArray{T}} = Array{SharedArray{T}}(undef,0)
  for field in field_vector
    push!(data_vector,field.data)
  end
  return data_vector
end

abstract type DirectionIndicators end

function get(field::DirectionIndicators,coords::Coords)
  throw(UserError())
end

function set!(field::DirectionIndicators,coords::Coords)
  throw(UserError())
end

function fill!(field::DirectionIndicators,value::Int64)
  throw(UserError())
end

@eval begin
  struct LatLonDirectionIndicators <: DirectionIndicators
    next_cell_coords::Array{LatLonField{Int64},1}
    function LatLonDirectionIndicators(dir_based_rdirs::LatLonField{Int64})
      grid = dir_based_rdirs.grid
      nlat = dir_based_rdirs.grid.nlat
      nlon = dir_based_rdirs.grid.nlon
      lat_indices::LatLonField{Int64} =
        LatLonField{Int64}(grid,convert($(array_type),zeros(Int64,nlat,nlon)))
      lon_indices::LatLonField{Int64} =
        LatLonField{Int64}(grid,convert($(array_type),zeros(Int64,nlat,nlon)))
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
      new(LatLonField{Int64}[lat_indices,lon_indices])
    end
  end
end

struct UnstructuredFieldDirectionIndicators <: DirectionIndicators
  next_cell_coords::UnstructuredField{Int64}
end

function (latlon_direction_indicators::LatLonDirectionIndicators)(coords::LatLonCoords)
  return DirectionIndicator(LatLonCoords(latlon_direction_indicators.next_cell_coords[1](coords),
                                         latlon_direction_indicators.next_cell_coords[2](coords)))
end

function get(latlon_direction_indicators::LatLonDirectionIndicators,coords::LatLonCoords)
  return latlon_direction_indicators(coords)
end


function set!(latlon_direction_indicators::LatLonDirectionIndicators,
               coords::LatLonCoords,direction::DirectionIndicator)
  next_cell_coords =  get_next_cell_coords(direction,coords)
  set!(latlon_direction_indicators.next_cell_coords[1],coords,next_cell_coords.lat)
  set!(latlon_direction_indicators.next_cell_coords[2],coords,next_cell_coords.lon)
end

function fill!(latlon_direction_indicators::LatLonDirectionIndicators,value::Int64)
  return fill!(latlon_direction_indicators.next_cell_coords,value)
end

struct UnstructuredDirectionIndicators <: DirectionIndicators
  next_cell_coords::UnstructuredField{Int64}
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

function fill!(unstructured_direction_indicators::UnstructuredDirectionIndicators,value::Int64)
  return fill!(unstructured_direction_indicators.next_cell_coords,value)
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
