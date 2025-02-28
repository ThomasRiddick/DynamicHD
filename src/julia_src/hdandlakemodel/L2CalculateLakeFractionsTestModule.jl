module L2CalculateLakeFractionsTestModule

using GridModule: Grid, LatLonGrid, find_coarse_cell_containing_fine_cell
using Test: @test, @testset
using L2CalculateLakeFractionsModule: LakeInput, calculate_lake_fractions
using FieldModule: Field,LatLonField, set!

          # 0 0 1 1  1 1 0 0  0 0 0 0  0 0 0 0  0 0 0 0
          # 0 0 0 0  1 1 0 0  0 0 0 0  0 0 0 0  0 0 0 0

          # 0 0 1 1  1 1 0 0  0 0 0 0  0 0 0 0  0 0 0 0
          # 0 0 1 1  1 1 0 0  0 0 0 0  0 0 0 0  0 0 0 0
          # 0 0 1 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

@testset "Lake Fraction Calculation Test 1" begin
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  lake_number::Int64 = 1
  lake_pixel_mask::Field{Bool} = LatLonField{Bool}(lake_grid,
    Bool[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
          0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
          0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
          0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

          0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
          0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
          0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
          0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

          0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
          0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
          0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 1 0
          0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 1 0

          0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 1 1
          0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 1  1 1 1 1
          0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 1  1 1 1 1
          0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 0 0

          0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 0  0 1 0 0
          0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 0  0 1 1 0
          0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 1  1 1 0 0
          0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16 ])
  lake_pixel_coords_list::Vector{CartesianIndex} = findall(lake_pixel_mask.data)
  potential_lake_pixel_coords_list::Vector{CartesianIndex} = lake_pixel_coords_list
  cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
  for pixel in potential_lake_pixel_coords_list
    cell_coords::CartesianIndex =
      find_coarse_cell_containing_fine_cell(lake_grid,surface_grid,
                                            pixel)
    set!(cell_mask,cell_coords,true)
  end
  cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
  input = LakeInput(lake_number,
                    lake_pixel_coords_list,
                    potential_lake_pixel_coords_list,
                    cell_coords_list)
  push!(lakes,input)
  calculate_lake_fractions(lakes,
                           cell_pixel_counts,
                           lake_grid,
                           surface_grid)
end

end
