module IdentifyBifurcatedRiverMouthsOutput

function write_river_mouth_indices(river_mouth_indices::Dict{Array{CartesianIndex}},
                                   output_river_mouths_filepath::String)
  f = open(output_river_mouths_filepath,"w")
  for (name,indices) in river_mouth_indices
    println(f,name)
    for index in indices
      println(f,index)
    end
  end
  close(f)
end

end
