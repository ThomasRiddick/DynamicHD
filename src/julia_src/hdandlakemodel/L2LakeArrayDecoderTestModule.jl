module L2LakeArrayDecoderTestModule

using Test: @test, @testset
using L2LakeArrayDecoderModule: get_lake_parameters_from_array
using L2LakeModule: Cell, Redirect, flood_height
using GridModule: LatLonGrid

@testset "Lake Array Decoder tests" begin
  #One lake
  array_in::Vector{Float64} = vec(Float64[1.0 66.0 1.0 -1.0 0.0 4.0 3.0 11.0 4.0 3.0 1.0 0.0 5.0 4.0 4.0 1.0 0.0 5.0 3.0 4.0 1.0 0.0 5.0 3.0 3.0 1.0 0.0 5.0 2.0 5.0 1.0 5.0 6.0 4.0 5.0 1.0 5.0 6.0 3.0 5.0 1.0 12.0 7.0 3.0 2.0 1.0 12.0 7.0 4.0 2.0 1.0 21.0 8.0 2.0 3.0 1.0 21.0 8.0 2.0 4.0 1.0 43.0 10.0 1.0 -1.0 2.0 2.0 0.0])
  lake_parameters = get_lake_parameters_from_array(array_in,
                                                   LatLonGrid(6,6,true),
                                                   LatLonGrid(3,3,true))
  @test length(lake_parameters) == 1
  @test lake_parameters[1].center_coords == CartesianIndex(4,3)
  @test lake_parameters[1].center_cell_coarse_coords == CartesianIndex(2,2)
  @test lake_parameters[1].lake_number == 1
  @test lake_parameters[1].is_primary == true
  @test lake_parameters[1].is_leaf == true
  @test lake_parameters[1].primary_lake == -1
  @test lake_parameters[1].secondary_lakes == Int64[]
  @test lake_parameters[1].filling_order == Cell[Cell(CartesianIndex(4, 3), flood_height, 0.0, 5.0), Cell(CartesianIndex(4, 4), flood_height, 0.0, 5.0), Cell(CartesianIndex(3, 4), flood_height, 0.0, 5.0), Cell(CartesianIndex(3, 3), flood_height, 0.0, 5.0), Cell(CartesianIndex(2, 5), flood_height, 5.0, 6.0), Cell(CartesianIndex(4, 5), flood_height, 5.0, 6.0), Cell(CartesianIndex(3, 5), flood_height, 12.0, 7.0), Cell(CartesianIndex(3, 2), flood_height, 12.0, 7.0), Cell(CartesianIndex(4, 2), flood_height, 21.0, 8.0), Cell(CartesianIndex(2, 3), flood_height, 21.0, 8.0), Cell(CartesianIndex(2, 4), flood_height, 43.0, 10.0)]
  @test lake_parameters[1].outflow_points == Dict{Int64,Redirect}(-1 => Redirect(false, -1, CartesianIndex(2, 2)))
  #Two lakes that join
  array_in = Float64[3.0, 21.0, 1.0, 3.0, 0.0, 4.0, 5.0, 2.0, 4.0, 5.0, 1.0, 0.0, 5.0, 3.0, 5.0, 1.0, 6.0, 8.0, 1.0, 2.0, 2.0, 2.0, 0.0, 26.0, 2.0, 3.0, 0.0, 4.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 5.0, 3.0, 2.0, 1.0, 2.0, 6.0, 4.0, 3.0, 1.0, 8.0, 8.0, 1.0, 1.0, -1.0, -1.0, 1.0, 43.0, 3.0, -1.0, 2.0, 1.0, 2.0, 4.0, 5.0, 6.0, 4.0, 5.0, 1.0, 0.0, 8.0, 3.0, 5.0, 1.0, 0.0, 8.0, 4.0, 4.0, 1.0, 0.0, 8.0, 4.0, 3.0, 1.0, 0.0, 8.0, 4.0, 2.0, 1.0, 0.0, 8.0, 3.0, 2.0, 1.0, 12.0, 10.0, 1.0, -1.0, 3.0, 3.0, 0.0]
  lake_parameters = get_lake_parameters_from_array(array_in,
                                         LatLonGrid(6,6,true),
                                         LatLonGrid(3,3,true))
  @test length(lake_parameters) == 3
  @test lake_parameters[1].center_coords == CartesianIndex(4, 5)
  @test lake_parameters[1].center_cell_coarse_coords == CartesianIndex(2, 3)
  @test lake_parameters[1].lake_number == 1
  @test lake_parameters[1].is_primary == false
  @test lake_parameters[1].is_leaf == true
  @test lake_parameters[1].primary_lake == 3
  @test lake_parameters[1].secondary_lakes == Int64[]
  @test lake_parameters[1].filling_order == Cell[Cell(CartesianIndex(4, 5), flood_height, 0.0, 5.0), Cell(CartesianIndex(3, 5), flood_height, 6.0, 8.0)]
  @test lake_parameters[1].outflow_points == Dict{Int64, Redirect}(2 => Redirect(false, 2, CartesianIndex(2, 2)))
  @test lake_parameters[2].center_coords == CartesianIndex(4, 2)
  @test lake_parameters[2].center_cell_coarse_coords == CartesianIndex(2, 1)
  @test lake_parameters[2].lake_number == 2
  @test lake_parameters[2].is_primary == false
  @test lake_parameters[2].is_leaf == true
  @test lake_parameters[2].primary_lake == 3
  @test lake_parameters[2].secondary_lakes == Int64[]
  @test lake_parameters[2].filling_order == Cell[Cell(CartesianIndex(4, 2), flood_height, 0.0, 5.0), Cell(CartesianIndex(3, 2), flood_height, 2.0, 6.0), Cell(CartesianIndex(4, 3), flood_height, 8.0, 8.0)]
  @test lake_parameters[2].outflow_points == Dict{Int64, Redirect}(1 => Redirect(true, 1, CartesianIndex(-1,)))
  @test lake_parameters[3].center_coords == CartesianIndex(4, 5)
  @test lake_parameters[3].center_cell_coarse_coords == CartesianIndex(2, 3)
  @test lake_parameters[3].lake_number == 3
  @test lake_parameters[3].is_primary == true
  @test lake_parameters[3].is_leaf == false
  @test lake_parameters[3].primary_lake == -1
  @test lake_parameters[3].secondary_lakes == [1, 2]
  @test lake_parameters[3].filling_order == Cell[Cell(CartesianIndex(4, 5), flood_height, 0.0, 8.0), Cell(CartesianIndex(3, 5), flood_height, 0.0, 8.0), Cell(CartesianIndex(4, 4), flood_height, 0.0, 8.0), Cell(CartesianIndex(4, 3), flood_height, 0.0, 8.0), Cell(CartesianIndex(4, 2), flood_height, 0.0, 8.0), Cell(CartesianIndex(3, 2), flood_height, 12.0, 10.0)]
  @test lake_parameters[3].outflow_points == Dict{Int64, Redirect}(-1 => Redirect(false, -1, CartesianIndex(3, 3)))
end

end
