module L2LakeInputModule

mutable struct ArrayDecoder
  array::Vector{Float64}
  current_index::Int64
  object_count::Int64
  object_start_index::Int64
  expected_total_objects::Int64
  expected_object_length::Int64
  function ArrayDecoder(array::Vector{Float64})
    expected_total_objects = array[1]
    return new(array,2,0,expected_total_objects)
  end
end

function start_next_object(decoder::ArrayDecoder)
  decoder.expected_object_length = Int64(decoder.array[decoder.current_index])
  decoder.current_index += 1
  decoder.object_count += 1
  #Expected object length excludes the first entry (i.e. the length itself)
  decoder.expected_start_index = decoder.current_index
end

function finish_object(decoder::ArrayDecoder)
  if expected_object_length != decoder.current_index - decoder.object_start_index
    error("Object read incorrectly - length doesn't match expectation")
  end
end

function finish_array(decoder::ArrayDecoder)
  if decoder.object_count != decoder.expected_total_objects
    error("Array read incorrectly - number of object doesn't match expectation")
  end
  if length(decoder.array) != decoder.current_index - 1
    error("Array read incorrectly - length doesn't match expectation")
  end
end

function read_float(decoder::ArrayDecoder)
  val::Float64 = decoder.array[decoder.current_index]
  decoder.current_index += 1
  return val
end

function read_integer(decoder::ArrayDecoder)
  return Int64(read_float(decoder))
end

function read_bool(decoder::ArrayDecoder)
  return Bool(read_float(decoder))
end

function read_coords(decoder::ArrayDecoder;single_index=false)
  if single_index
    return CartesianIndex(read_number(decoder))
  else
    y,x = decoder.array[decoder.current_index:decoder.current_index+1]
    decoder.current_index += 2
    return CartesianIndex(y,x)
  end
end

function read_field(decoder::ArrayDecoder;integer_field=false)
  field_length::Int64 = Int64(decoder.array[decoder.current_index])
  decoder.current_index += 1
  field::Array{Float64} =
    decoder.array[decoder.current_index:decoder.current_index+field_length-1]
  decoder.current_index += field_length
  println("Does this work for length 1 arrays??")
  println(field)
  if integer_field
    return [Int64(x) for x in field]::Array{Int64}
  else
    return field
  end
end

function read_outflow_points_dict(decoder::ArrayDecoder;single_index=false)
  length::Int64 = Int64(decoder.array[decoder.current_index])
  decoder.current_index += 1
  entry_length::Int64 = single_index ? 3 : 4
  offset::Int64 = single_index ? 0 : 1
  outflow_points = Dict{Int64,Redirect}
  for ___ in 1:length
    entry::Array{Float64} =
      decoder.array[decoder.current_index:decoder.current_index+entry_length-1]
    decoder.current_index += entry_length
    lake_number = entry[1]
    is_local::Bool = entry[3+offset]
    if ! is_local
      coords_array::Array{Int64} = single_index ? Int64[Int64(entry[2])] :
                                   [Int64(x) for x in entry[2:3]]
      coords::CartesianIndex = CartesianIndex(coords_array...)
    else
      coords::CartesianIndex = CartesianIndex(-1)
    end
    redirect::Redirect = Redirect(is_local,lake_number,coords)
    outflow_points[lake_number] = redirect
  end
  return outflow_points
end

function read_filling_order(decoder::ArrayDecoder;single_index=false)
  length::Int64 = Int64(decoder.array[decoder.current_index])
  decoder.current_index += 1
  entry_length::Int64 = single_index ? 4 : 5
  offset::Int64 = single_index ? 0 : 1
  filling_order::Vector{Cell} = Cell[]
  for ___ in 1:length
    entry::Array{Float64} =
      decoder.array[decoder.current_index:decoder.current_index+entry_length-1]
    decoder.current_index += entry_length
    coords::Array{Int64} = single_index ?
                           Int64[Int64(entry[1])] :
                           [Int64(x) for x in entry[1:2]]
    height_type_int::Int64 = Int64(entry[2+offset])
    height_type::HeightType = height_type_int == 1 ? connect_height : flood_height
    threshold::Float64 = entry[3+offset]
    height::Float64 = entry[4+offset]
    push!(filling_order,Cell(CartesianIndex(coords),height_type,threshold,height))
  end
  return filling_order
end

function get_lakes_from_array(array::Array{Float64};single_index=false)
  decoder = ArrayDecoder(array)
  lake_parameters::Array{LakeParameters} = LakeParameters[]
  for ___ in expected_total_objects
    start_next_object(decoder)
    lake_number::Int64 = read_integer(decoder)
    primary_lake::Int64 = read_integer(decoder)
    secondary_lakes::Int64 = read_field(decoder)
    center_coords::CartesianIndex = read_coord(decoder;
                                               single_index=single_index)
    filling_order::Vector{Cell} =
      read_filling_order(decoder;
                         single_index=single_index)
    outflow_points::Dict{Int64,Redirect} =
      read_outflow_points_dict(decoder;
                               single_index=single_index)
    finish_object(decoder)
    push!(lakes,LakeParameters(lake_number,
                               primary_lake,
                               secondary_lakes,
                               center_coords,
                               filling_order,
                               outflow_points))
  end
  finish_array(decoder)
  return lakes
end

end
