module L2LakeArrayDecoderTestModule

using Test: @test, @testset
using L2LakeArrayDecoderModule: get_lake_parameters_from_array
using GridModule: LatLonGrid

@testset "Lake Array Decoder tests" begin
  array_in::Vector{Float64} = vec(Float64[1.0 66.0 1.0 -1.0 0.0 4.0 3.0 11.0 4.0 3.0 1.0 0.0 5.0 4.0 4.0 1.0 0.0 5.0 3.0 4.0 1.0 0.0 5.0 3.0 3.0 1.0 0.0 5.0 2.0 5.0 1.0 5.0 6.0 4.0 5.0 1.0 5.0 6.0 3.0 5.0 1.0 12.0 7.0 3.0 2.0 1.0 12.0 7.0 4.0 2.0 1.0 21.0 8.0 2.0 3.0 1.0 21.0 8.0 2.0 4.0 1.0 43.0 10.0 1.0 -1.0 2.0 2.0 0.0])
  println(get_lake_parameters_from_array(array_in,
                                         LatLonGrid(6,6,true),
                                         LatLonGrid(2,2,true)))
end

end
