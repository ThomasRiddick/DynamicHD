module L2LakeInputModule

struct ArrayDecoder
  array::Vector{Float64}
  current_index::Int64
  object_count::Int64
  expected_total_objects::Int64
  function ArrayDecoder(array::Vector{Float64})
    expected_total_objects = array[1]
    return new(array,2,0,expected_total_objects)
  end
end

function read_float(decoder::ArrayDecoder)
  decoder.object_count += 1
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

function read_coord(decoder::ArrayDecoder;single_index=false)
  if single_index
    return CartesianIndex(read_number(decoder))
  else
    decoder.object_count += 1
    y,x = decoder.array[decoder.current_index:decoder.current_index+1]
    decoder.current_index += 2
    return CartesianIndex(y,x)
  end
end

function read_field(decoder::ArrayDecoder;integer_field=false)
  decoder.object_count += 1
  field_length::Int64 = Int64(decoder.array[current_index])
  decoder.current_index += 1
  field::Array{Float64} = decoder.array[current_index:current_index+field_length-1]
  decoder.current_index += field_length
  println("Does this work for length 1 arrays??")
  println(field)
  if integer_field
    return [Int64(x) for x in field]::Array{Int64}
  else
    return field
  end
end

function read_dict(decoder::ArrayDecoder)
  decoder.object_count += 1
  throw(UserError("Needs work"))
end

function read_filling_order(decoder::ArrayDecoder;single_index=false)
  decoder.object_count += 1
  length::Int64 = Int64(decoder.array[current_index]
  current_index += 1
  entry_length::Int64 = single_index ? 4 : 5
  offset:Int64 = single_index ? 0 : 1
  filling_order = Any[]
  for ___ in 1:length
    entry = decoder.array[current_index:current_index+entry_length-1]
    decoder.current_index += entry_length
    coords::Array{Int64} = single_index ?
                           Int64[Int64(entry[1])]
                           [Int64(x) for x in entry[1:2]]
    height_type::Int64 = Int64(entry[2+offset])
    threshold::Float64 = entry[3+offset]
    height::Float64 = entry[4+offset]
    push!(filling_order,Any[coords,height_type,threshold,height])
  end
  return filling_order
end

end
