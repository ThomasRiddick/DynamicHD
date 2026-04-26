module IdentifyBifurcatedRiverMouthsOutput

function write_river_mouth_indices(river_mouth_indices::Dict{String,Array{CartesianIndex}},
                                   existing_river_mouths::Dict{String,CartesianIndex},
                                   output_river_mouths_filepath::String)
  f = open(output_river_mouths_filepath,"w")
  for (name,indices) in river_mouth_indices
    println(f,"#$(name)")
    if haskey(existing_river_mouths,name)
      println(f,"primary mouth: $(Tuple(existing_river_mouths[name])[1])")
    end
    for index in indices
      println(f,"secondary mouth: $(Tuple(index)[1])")
    end
  end
  close(f)
end

end
