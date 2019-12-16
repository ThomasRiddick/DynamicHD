module FieldModule

using Printf: @printf
using UserExceptionModule: UserError
using CoordsModule: Coords,LatLonCoords,DirectionIndicator,Generic1DCoords
using GridModule: Grid,LatLonGrid, for_all, wrap_coords, coords_in_grid,for_all_with_line_breaks
using GridModule: UnstructuredGrid
using InteractiveUtils: subtypes
using SpecialDirectionCodesModule
using MergeTypesModule
import Base.maximum
import Base.+
import Base.==
import Base.isapprox
import Base.fill!
import Base.show
import Base.round
import Base.convert
using InteractiveUtils

abstract type Field{T} end

function get(field::Field{T},coords::Coords) where {T}
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

function ==(lfield::Field,rfield::Field)
  throw(UserError())
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

function round!(type::DataType,field::Field{T}) where {T}
  throw(UserError())
end

function add_offset(field::T2,offset::T,skip_offset_on_values::Vector{T}) where {T,T2<:Field{T}}
  for_all(get_grid(field)) do coords::Coords
    if ! (field(coords) in skip_offset_on_values)
      set!(field,coords,field(coords)+offset)
    end
  end
end

function divide(field::T2,divisor::T) where {T,T2<:Field{T}}
  for_all(get_grid(field)) do coords::Coords
    set!(field,coords,field(coords)/divisor)
  end
end

get_grid(obj::T) where {T <: Field} =
  obj.grid::Grid

get_data(obj::T) where {T <: Field} =
  obj.data::Array

function repeat(field::Field{T}, count::Integer) where {T}
  array::Array{Field{T},1} = Array{Field{T},1}(undef,count)
  for i in 1:count
    array[i] = deepcopy(field)
  end
  return array
end


struct LatLonField{T} <: Field{T}
  data::Array{T,2}
  grid::LatLonGrid
  function LatLonField{T}(grid::LatLonGrid,value::T) where {T}
    data::Array{T,2} = value == zero(T) ? zeros(T,grid.nlat,grid.nlon) :
                                          ones(T,grid.nlat,grid.nlon)*value
    return new(data,grid)
  end
  function LatLonField{MergeTypes}(grid::LatLonGrid,value::MergeTypes)
    data::Array{MergeTypes,2} =  zeros(MergeTypes,grid.nlat,grid.nlon)
    fill!(data,value)
    return new(data,grid)
  end
  function LatLonField{T}(grid::LatLonGrid,values::AbstractArray{T,2}) where {T}
    if size(values,1) != grid.nlat ||
       size(values,2) != grid.nlon
       error("Values provided don't match selected grid")
    end
    return new(values,grid)
  end
  function LatLonField{MergeTypes}(grid::LatLonGrid,
                                   integer_field::LatLonField{T}) where {T<:Signed}
    grid = get_grid(integer_field)
    merge_type_field = LatLonField{MergeTypes}(grid,null_mtype)
    for_all(grid) do coords::Coords
      element::MergeTypes = MergeTypes(get(integer_field,coords))
      set!(merge_type_field,coords,element)
    end
    return new(merge_type_field.data,grid)
  end
end

struct UnstructuredField{T} <: Field{T}
  data::Array{T,1}
  grid::UnstructuredGrid
  function UnstructuredField{T}(grid::UnstructuredGrid,value::T) where {T}
    data::Array{T,1} = value == zero(T) ? zeros(T,grid.ncells) :
                                          ones(T,grid.ncells)*value
    return new(data,grid)
  end
  function UnstructuredField{T}(grid::UnstructuredGrid,values::AbstractArray{T,1}) where {T}
    if size(values,1) != grid.ncells ||
       error("Values provided don't match selected grid")
    end
    return new(values,grid)
  end
end

LatLonField{T}(grid::LatLonGrid) where {T} = LatLonField{T}(grid,zeros(T,grid.nlat,grid.nlon))

UnstructuredField{T}(grid::UnstructuredGrid) where {T} = UnstructuredField{T}(grid,zeros(T,grid.ncells))

Field{T}(grid::LatLonGrid,value::T) where {T} = LatLonField{T}(grid::LatLonGrid,value::T)
Field{T}(grid::LatLonGrid) where {T} = LatLonField{T}(grid::LatLonGrid)

Field{T}(grid::LatLonGrid,values::AbstractArray{T,2}) where {T} =
  LatLonField{T}(grid::LatLonGrid,values::AbstractArray{T,2})

Field{MergeTypes}(grid::LatLonGrid,integer_field::LatLonField{T}) where {T<:Signed} =
  LatLonField{MergeTypes}(grid,integer_field)

convert(::Type{Field{MergeTypes}},x::T2) where {T<:Signed,T2<:Field{T}} = Field{MergeTypes}(get_grid(x),x)


function (latlon_field::LatLonField{T})(coords::LatLonCoords) where {T}
    return latlon_field.data[coords.lat,coords.lon]::T
end

function get(latlon_field::LatLonField{T},coords::LatLonCoords) where {T}
    return latlon_field(coords)::T
end

function set!(latlon_field::LatLonField{T},coords::LatLonCoords,value::T) where {T}
    latlon_field.data[coords.lat,coords.lon] = value
end

function +(lfield::LatLonField{T},rfield::LatLonField{T}) where {T}
  return LatLonField{T}(lfield.grid,lfield.data + rfield.data)
end

function ==(lfield::LatLonField{T},rfield::LatLonField{T}) where {T}
  return (lfield.data == rfield.data)::Bool
end

function isapprox(lfield::LatLonField{T},rfield::LatLonField{T};
                  rtol::Real=atol>0 ? 0 : sqrt(eps),
                  atol::Real=0.0,nans::Bool=false) where {T}
  return isapprox(lfield.data,rfield.data,
                 rtol=rtol,atol=atol,nans=nans)::Bool
end

function invert(field::LatLonField{T}) where {T}
  return LatLonField{T}(field.grid,.!field.data)
end

function round(type::DataType,field::LatLonField{T}) where {T}
  return LatLonField{type}(field.grid,round.(type,field.data))
end

function maximum(field::LatLonField)
  return maximum(field.data)
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

function fill!(field::LatLonField{T},value::T) where {T}
  return fill!(field.data,value)
end

function (unstructured_field::UnstructuredField{T})(coords::Generic1DCoords) where {T}
    return unstructured__field.data[coords.index]::T
end

function get(unstructured_field::UnstructuredField{T},coords::Generic1DCoords) where {T}
    return unstructured_field(coords)::T
end

function set!(unstructured_field::UnstructuredField{T},coords::Generic1DCoords,value::T) where {T}
    unstructured_field.data[coords.index] = value
end

function +(lfield::UnstructuredField{T},rfield::UnstructuredField{T}) where {T}
  return UnstructuredField{T}(lfield.grid,lfield.data + rfield.data)
end

function ==(lfield::UnstructuredField{T},rfield::UnstructuredField{T}) where {T}
  return (lfield.data == rfield.data)::Bool
end

function isapprox(lfield::UnstructuredField{T},rfield::UnstructuredField{T};
                  rtol::Real=atol>0 ? 0 : sqrt(eps),
                  atol::Real=0.0,nans::Bool) where {T}
  return isapprox(lfield.data,rfield.data,
                 rtol=rtol,atol=atol,nans=nans)::Bool
end

function invert(field::UnstructuredField{T}) where {T}
  return UnstructuredField{T}(field.grid,.!field.data)
end

function round(type::DataType,field::UnstructuredField{T}) where {T}
  return UnstructuredField{type}(field.grid,round.(type,field.data))
end

function maximum(field::UnstructuredField)
  return maximum(field.data)
end

function fill!(field::UnstructuredField{T},value::T) where {T}
  return fill!(field.data,value)
end

abstract type DirectionIndicators end

function get(field::DirectionIndicators,coords::Coords)
  throw(UserError())
end

function set!(field::DirectionIndicators,coords::Coords)
  throw(UserError())
end

struct LatLonDirectionIndicators <: DirectionIndicators
  next_cell_coords::Array{LatLonField{Int64},1}
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
    new(LatLonField{Int64}[lat_indices,lon_indices])
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


function set!(latlon_direction_indicators::DirectionIndicators,
               coords::LatLonCoords,direction::DirectionIndicator)
  next_cell_coords =  get_next_cell_coords(direction,coords)
  set!(latlon_direction_indicators.next_cell_coords[1],coords,next_cell_coords.lat)
  set!(latlon_direction_indicators.next_cell_coords[2],coords,next_cell_coords.lon)
end

function fill!(field::LatLonField{T},value::T) where {T}
  return fill!(field.data,value)
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
