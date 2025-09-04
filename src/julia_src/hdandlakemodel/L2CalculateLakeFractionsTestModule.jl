module L2CalculateLakeFractionsTestModule

using GridModule: Grid, LatLonGrid
using Test: @test, @testset
using L2CalculateLakeFractionsModule: LakeInput, LakeProperties,Pixel
using L2CalculateLakeFractionsModule: calculate_lake_fractions
using L2CalculateLakeFractionsModule: setup_lake_for_fraction_calculation
using L2CalculateLakeFractionsModule: add_pixel_by_coords,remove_pixel_by_coords
using L2CalculateLakeFractionsModule: LakeFractionCalculationPrognostics
using L2LakeModelDefsModule: GridSpecificLakeModelParameters
using L2LakeModelGridSpecificDefsModule: LatLonLakeModelParameters
using L2LakeModelGridSpecificDefsModule: get_corresponding_surface_model_grid_cell
using FieldModule: Field,LatLonField, set!

@testset "Lake Fraction Calculation Test 1" begin
  #Single lake
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  lake_number::Int64 = 1
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
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
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 0 0 0 0
           0 0 0 0 0
           0 0 0 0 0
           0 0 0 0 16
           0 0 0 0 14 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0 0.0 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 1.0
             0.0 0.0 0.0 0.0 0.875 ])
  expected_binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false false false false false
          false false false false false
          false false false false false
          false false false false true
          false false false false true ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == 1,
                                                           lake_pixel_mask.data)
  potential_lake_pixel_coords_list::Vector{CartesianIndex} = lake_pixel_coords_list
  cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
  for pixel in potential_lake_pixel_coords_list
    cell_coords::CartesianIndex =
      get_corresponding_surface_model_grid_cell(pixel,
                                                grid_specific_lake_model_parameters)
    set!(cell_mask,cell_coords,true)
  end
  cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
  input = LakeInput(lake_number,
                    lake_pixel_coords_list,
                    potential_lake_pixel_coords_list,
                    cell_coords_list)
  push!(lakes,input)
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
                             non_lake_mask,
                             grid_specific_lake_model_parameters,
                             lake_grid,
                             surface_grid)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  @test lake_fractions_field == expected_lake_fractions_field
  @test binary_lake_mask == expected_binary_lake_mask
end

@testset "Lake Fraction Calculation Test 2" begin
  #Multiple lakes
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 2 2 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 7 0
           0 2 2 2  2 0 0 0  0 0 3 3  0 0 0 0  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 6  6 6 6 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 6 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 4 4 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 4 4 4  4 4 0 0  0 0 0 0  0 1 1 0
           0 0 0 0  0 4 4 0  0 0 0 0  0 0 0 0  0 1 1 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 1 1
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 1  1 1 1 1
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 1  1 1 1 1
           0 0 0 0  0 5 0 0  0 0 0 0  0 0 0 0  0 1 0 0

           0 0 0 0  0 5 5 0  0 0 0 0  0 0 1 0  0 1 0 0
           0 0 0 0  0 5 0 0  0 0 0 0  0 0 1 0  0 1 1 0
           0 0 0 5  5 5 0 0  0 0 0 0  0 0 1 1  1 1 0 0
           0 0 0 0  0 5 0 0  0 0 0 0  0 0 0 0  0 1 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 5 1 2 0 1
           0 0 0 0 11
           0 9 0 0 0
           0 0 0 0 16
           0 8 0 0 14 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.3125 0.0625 0.125 0.0 0.0625
             0.0 0.0 0.0 0.0 0.6875
             0.0 0.5625 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 1.0
             0.0 0.5 0.0 0.0 0.875 ])
  expected_binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false false false false false
          false false false false  true
          false  true false false false
          false false false false  true
          false  true false false  true ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  for lake_number::Int64=1:7
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} = lake_pixel_coords_list
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel,
                                                  grid_specific_lake_model_parameters)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      lake_pixel_coords_list,
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
                             non_lake_mask,
                             grid_specific_lake_model_parameters,
                             lake_grid,
                             surface_grid)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  @test lake_fractions_field == expected_lake_fractions_field
  @test binary_lake_mask == expected_binary_lake_mask
end

@testset "Lake Fraction Calculation Test 3" begin
  #Multiple lakes
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 1 0  1 1 0 0  0 0 0 0  0 0 0 0  0 0 7 0
           0 0 1 0  1 1 0 1  1 1 1 0  0 0 8 8  0 7 7 0
           0 0 1 0  1 1 1 1  1 1 0 0  0 0 8 8  0 7 7 0

           0 0 1 1  1 1 1 1  0 0 0 0  0 0 0 8  0 7 7 0
           0 0 1 1  1 0 1 1  1 1 1 0  0 0 0 0  0 7 7 0
           0 0 1 1  1 2 1 1  1 1 0 0  0 0 8 8  8 8 0 9
           0 0 1 0  1 2 1 1  0 0 0 0  0 0 8 8  8 8 9 9

           0 0 1 0  2 2 1 1  0 0 0 0  0 0 0 0  0 0 9 9
           0 0 1 1  2 2 1 1  0 0 6 0  0 0 0 9  9 9 9 9
           0 0 0 0  2 2 2 1  0 0 0 0  0 0 0 0  0 9 9 0
           0 0 0 2  2 2 0 1  0 0 0 0  0 0 0 0  0 5 5 0

           0 0 2 2  2 2 0 0  0 0 0 0  0 3 3 5  0 5 0 0
           0 0 2 2  2 2 0 0  0 0 6 0  4 3 3 5  5 5 0 0
           0 0 0 0  2 0 0 0  0 0 0 0  4 4 0 5  5 0 0 0
           0 0 0 0  2 0 0 0  0 0 0 4  4 4 0 5  5 0 0 0

           0 0 0 0  2 2 2 0  0 0 4 4  4 4 0 0  5 5 0 0
           0 0 6 0  0 0 0 0  0 0 0 4  4 0 0 0  0 5 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 16  0  0 9
           16 14 0 13 0
           0 15  1  0 12
           0 15  1 12 14
           1 0  3  1 0 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0    1.0     0.0    0.0    0.5625
             1.0    0.875   0.0    0.8125 0.0
             0.0    0.9375  0.0625 0.0    0.75
             0.0    0.9375  0.0625 0.75   0.875
             0.0625 0.0     0.1875 0.0625 0.0 ])
  expected_binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false  true false false  true
           true  true false  true false
          false  true false false  true
          false  true false  true  true
          false false false false false ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  for lake_number::Int64=1:9
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} = lake_pixel_coords_list
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel,
                                                  grid_specific_lake_model_parameters)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      lake_pixel_coords_list,
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
                             non_lake_mask,
                             grid_specific_lake_model_parameters,
                             lake_grid,
                             surface_grid)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  @test lake_fractions_field == expected_lake_fractions_field
  @test binary_lake_mask == expected_binary_lake_mask
end

@testset "Lake Fraction Calculation Test 4" begin
  #Multiple lakes
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 1  0 1 1 1  1 1 1 0  0 0 0 0  0 0 0 0
           0 0 1 1  0 1 1 1  1 0 1 1  1 0 1 0  0 1 1 0
           0 1 1 1  0 1 1 1  1 1 1 1  1 1 1 1  1 1 0 0

           0 0 0 0  1 0 1 1  1 0 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  1 1 1 1  1 1 0 1  0 1 1 0  0 0 1 1
           0 1 1 1  1 1 1 1  1 1 0 1  1 0 1 1  1 1 1 0
           0 0 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 0

           0 0 0 0  1 1 1 1  1 1 1 1  1 0 1 1  0 0 0 0
           0 2 2 2  2 0 1 1  1 0 1 1  1 1 0 1  1 0 0 0
           0 2 2 0  1 1 1 1  1 1 1 0  1 1 1 1  1 1 0 0
           0 0 2 0  0 1 1 1  0 1 1 0  0 1 1 1  0 0 0 0

           0 0 2 1  0 1 0 1  0 0 1 0  1 1 1 0  0 1 0 0
           1 1 0 1  1 1 1 1  1 0 1 0  1 1 1 0  1 1 1 0
           0 1 1 0  1 1 1 1  1 1 1 0  1 1 1 1  1 1 0 0
           0 1 1 0  1 1 1 1  1 1 0 0  0 1 0 1  0 0 0 0

           0 0 0 0  1 1 0 1  1 1 0 1  1 1 0 0  1 1 0 0
           0 0 1 1  0 1 1 1  1 1 0 1  1 1 0 0  0 1 1 0
           0 0 0 1  1 1 1 1  1 1 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  0 1 1 1  1 0 0 0  1 0 0 0  0 0 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 16 16 0 0
           0 16 16 16 16
           8 15 16 16 0
          11 16  0 16 0
           0 16 16 0 0 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0 1.0 1.0 0.0 0.0
             0.0 1.0 1.0 1.0 1.0
             0.5 0.9375 1.0 1.0 0.0
             0.6875 1.0 0.0 1.0 0.0
             0.0 1.0 1.0 0.0 0.0 ])
  expected_binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false  true  true false false
          false  true  true  true  true
           true  true  true  true false
           true  true  false  true false
          false  true  true false false ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} = lake_pixel_coords_list
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel,
                                                  grid_specific_lake_model_parameters)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      lake_pixel_coords_list,
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
                             non_lake_mask,
                             grid_specific_lake_model_parameters,
                             lake_grid,
                             surface_grid)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  @test lake_fractions_field == expected_lake_fractions_field
  @test binary_lake_mask == expected_binary_lake_mask
end

@testset "Lake Fraction Calculation Test 5" begin
  #Multiple lakes
  lake_grid::Grid = LatLonGrid(15,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 2 2 2
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  2 2 2 2

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 2 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 2 2

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 1 1 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           1 1 1 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           1 1 1 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
            8  8  8  8  8
           16 16 16 16 16
            4  4  4  4  4
           16 16 16 16 16 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 0 0 0 10
           0 0 0 0 0
           0 0 0 0 0
           4 0 0 0 0
           6 0 0 0 0 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0 0.0 0.0 0.0 0.625
             0.0 0.0 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 0.0
             1.0 0.0 0.0 0.0 0.0
             0.375 0.0 0.0 0.0 0.0 ])
  expected_binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false false false false  true
          false false false false false
          false false false false false
           true false false false false
          false  false false false false ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} = lake_pixel_coords_list
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel,
                                                  grid_specific_lake_model_parameters)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      lake_pixel_coords_list,
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
                             non_lake_mask,
                             grid_specific_lake_model_parameters,
                             lake_grid,
                             surface_grid)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  @test lake_fractions_field == expected_lake_fractions_field
  @test binary_lake_mask == expected_binary_lake_mask
end

@testset "Lake Fraction Calculation Test 6" begin
  #Partially filled single lake
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  lake_number::Int64 = 1
  potential_lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
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
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 1 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 0 1
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 1  1 0 0 1
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
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 0 0 0 0
           0 0 0 0 0
           0 0 0 0 0
           0 0 0 0 5
           0 0 0 0 16 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0 0.0 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 0.3125
             0.0 0.0 0.0 0.0 1.0 ])
  expected_binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false false false false false
          false false false false false
          false false false false false
          false false false false false
          false false false false true ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == 1,
                                                           lake_pixel_mask.data)
  potential_lake_pixel_coords_list::Vector{CartesianIndex} =
    findall(x -> x == 1,potential_lake_pixel_mask.data)
  cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
  for pixel in potential_lake_pixel_coords_list
    cell_coords::CartesianIndex =
      get_corresponding_surface_model_grid_cell(pixel,
                                                grid_specific_lake_model_parameters)
    set!(cell_mask,cell_coords,true)
  end
  cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
  input = LakeInput(lake_number,
                    lake_pixel_coords_list,
                    potential_lake_pixel_coords_list,
                    cell_coords_list)
  push!(lakes,input)
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
                             non_lake_mask,
                             grid_specific_lake_model_parameters,
                             lake_grid,
                             surface_grid)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  @test lake_fractions_field == expected_lake_fractions_field
  @test binary_lake_mask == expected_binary_lake_mask
end

@testset "Lake Fraction Calculation Test 7" begin
  #Multiple partially filled lakes
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  potential_lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 2 2 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 7 0
           0 2 2 2  2 0 0 0  0 0 3 3  0 0 0 0  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 6  6 6 6 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 6 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 4 4 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 4 4 4  4 4 0 0  0 0 0 0  0 1 1 0
           0 0 0 0  0 4 4 0  0 0 0 0  0 0 0 0  0 1 1 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 1 1
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 1  1 1 1 1
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 1  1 1 1 1
           0 0 0 0  0 5 0 0  0 0 0 0  0 0 0 0  0 1 0 0

           0 0 0 0  0 5 5 0  0 0 0 0  0 0 1 0  0 1 0 0
           0 0 0 0  0 5 0 0  0 0 0 0  0 0 1 0  0 1 1 0
           0 0 0 5  5 5 0 0  0 0 0 0  0 0 1 1  1 1 0 0
           0 0 0 0  0 5 0 0  0 0 0 0  0 0 0 0  0 1 0 0 ])
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 2 2 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 7 0
           0 2 2 2  2 0 0 0  0 0 3 3  0 0 0 0  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 6  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 4 4 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 4 4 0  4 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 4 4 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 1 1
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  1 1 1 1
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  1 1 1 1
           0 0 0 0  0 5 0 0  0 0 0 0  0 0 0 0  0 1 0 0

           0 0 0 0  0 5 5 0  0 0 0 0  0 0 0 0  0 1 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 1 0
           0 0 0 5  5 0 0 0  0 0 0 0  0 0 0 0  1 1 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 5 1 2 0 1
           0 0 0 0 9
           0 6 1 0 0
           0 1 0 0 16
           1 3 0 0 2 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.3125 0.0625 0.125 0.0 0.0625
             0.0 0.0 0.0 0.0 0.5625
             0.0 0.375 0.0625 0.0 0.0
             0.0 0.0625 0.0 0.0 1.0
             0.0625 0.1875 0.0 0.0 0.125 ])
  expected_binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false false false false false
          false false false false  true
          false false false false false
          false false false false  true
          false false false false false ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  for lake_number::Int64=1:7
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                               potential_lake_pixel_mask.data)
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel,
                                                  grid_specific_lake_model_parameters)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      lake_pixel_coords_list,
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
                             non_lake_mask,
                             grid_specific_lake_model_parameters,
                             lake_grid,
                             surface_grid)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  @test lake_fractions_field == expected_lake_fractions_field
  @test binary_lake_mask == expected_binary_lake_mask
end

@testset "Lake Fraction Calculation Test 8" begin
  #Multiple partially filled lakes
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  potential_lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 1 0  1 1 0 0  0 0 1 0  0 0 0 0  0 0 7 0
           0 0 1 0  1 1 0 1  1 1 1 0  0 0 8 8  0 7 7 0
           0 0 1 0  1 1 1 1  1 1 0 0  0 0 8 8  0 7 7 0

           0 0 1 1  1 1 1 1  0 0 0 0  0 0 0 8  0 7 7 0
           0 0 1 1  1 0 1 1  1 1 1 0  0 0 0 0  0 7 7 0
           0 0 1 1  1 2 1 1  1 1 0 0  0 0 8 8  8 8 0 9
           0 0 1 0  1 2 1 1  0 0 0 0  0 0 8 8  8 8 9 9

           0 0 1 0  2 2 1 1  0 0 0 0  0 0 0 0  0 0 9 9
           0 0 1 1  2 2 1 1  0 0 6 0  0 0 0 9  9 9 9 9
           0 0 0 0  2 2 2 1  0 0 0 0  0 0 0 0  0 9 9 0
           0 0 0 2  2 2 0 1  0 0 0 0  0 0 0 0  0 5 5 0

           0 0 2 2  2 2 0 0  0 0 0 0  0 3 3 5  0 5 0 0
           0 0 2 2  2 2 0 0  0 0 6 0  4 3 3 5  5 5 0 0
           0 0 0 0  2 0 0 0  0 0 0 0  4 4 0 5  5 0 0 0
           0 0 0 0  2 0 0 0  0 0 0 4  4 4 0 5  5 0 0 0

           0 0 0 0  2 2 2 0  0 0 4 4  4 4 0 0  5 5 0 0
           0 0 6 0  0 0 0 0  0 0 0 4  4 0 0 0  0 5 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 ])
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  1 1 0 0  0 0 0 0  0 0 0 0  0 0 7 0
           0 0 0 0  1 1 0 0  0 0 0 0  0 0 8 8  0 0 7 0
           0 0 0 0  1 1 1 1  0 1 0 0  0 0 8 8  0 7 0 0

           0 0 0 0  0 0 1 0  0 0 0 0  0 0 0 8  0 7 7 0
           0 0 0 0  0 0 1 0  0 0 0 0  0 0 0 0  0 7 7 0
           0 0 1 0  0 0 1 1  0 0 0 0  0 0 0 0  8 8 0 9
           0 0 1 0  0 0 1 1  0 0 0 0  0 0 8 0  8 8 9 9

           0 0 0 0  2 2 1 1  0 0 0 0  0 0 0 0  0 0 9 9
           0 0 0 0  2 2 1 1  0 0 6 0  0 0 0 9  9 0 0 0
           0 0 0 0  2 2 2 1  0 0 0 0  0 0 0 0  0 9 9 0
           0 0 0 2  2 2 0 1  0 0 0 0  0 0 0 0  0 0 5 0

           0 0 2 2  0 0 0 0  0 0 0 0  0 3 3 5  0 0 0 0
           0 0 2 2  0 0 0 0  0 0 6 0  4 3 3 5  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  4 4 0 5  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 4  4 4 0 5  0 0 0 0

           0 0 0 0  2 2 2 0  0 0 4 0  4 0 0 0  5 0 0 0
           0 0 6 0  0 0 0 0  0 0 0 0  4 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 16 0 10 3
           0 6  0  0 4
           0 11 1  0 10
           4 0  1 16 0
           1 3  0  1 1 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0 1.0 0.0 0.625 0.1875
             0.0 0.375 0.0 0.0 0.25
             0.0 0.6875 0.0625 0.0 0.625
             0.25 0.0 0.0625 1.0 0.0
             0.0625 0.1875 0.0 0.0625 0.0625 ])
  expected_binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false  true false  true false
          false false false false false
          false  true false false  true
          false false false  true false
          false false false false false ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  for lake_number::Int64=1:9
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                               potential_lake_pixel_mask.data)
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel,
                                                  grid_specific_lake_model_parameters)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      lake_pixel_coords_list,
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
                             non_lake_mask,
                             grid_specific_lake_model_parameters,
                             lake_grid,
                             surface_grid)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  @test lake_fractions_field == expected_lake_fractions_field
  @test binary_lake_mask == expected_binary_lake_mask
end

@testset "Lake Fraction Calculation Test 9" begin
  #Multiple partially filled lakes
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  potential_lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 1  0 1 1 1  1 1 1 0  0 0 0 0  0 0 0 0
           0 0 1 1  0 1 1 1  1 0 1 1  1 0 1 0  0 1 1 0
           0 1 1 1  0 1 1 1  1 1 1 1  1 1 1 1  1 1 0 0

           0 0 0 0  1 0 1 1  1 0 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  1 1 1 1  1 1 0 1  0 1 1 0  0 0 1 1
           0 1 1 1  1 1 1 1  1 1 0 1  1 0 1 1  1 1 1 0
           0 0 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 0

           0 0 0 0  1 1 1 1  1 1 1 1  1 0 1 1  0 0 0 0
           0 2 2 2  2 0 1 1  1 0 1 1  1 1 0 1  1 0 0 0
           0 2 2 0  1 1 1 1  1 1 1 0  1 1 1 1  1 1 0 0
           0 0 0 0  0 1 1 1  0 1 1 0  0 1 1 1  0 0 0 0

           0 0 0 1  0 1 0 1  0 0 1 0  1 1 1 0  0 1 0 0
           1 1 0 1  1 1 1 1  1 0 1 0  1 1 1 0  1 1 1 0
           0 1 1 0  1 1 1 1  1 1 1 0  1 1 1 1  1 1 0 0
           0 1 1 0  1 1 1 1  1 1 0 0  0 1 0 1  0 0 0 0

           0 0 0 0  1 1 0 1  1 1 0 1  1 1 0 0  1 1 0 0
           0 0 1 1  0 1 1 1  1 1 0 1  1 1 0 0  0 1 1 0
           0 0 0 1  1 1 1 1  1 1 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  0 1 1 1  1 0 0 0  1 0 0 0  0 0 0 0 ])
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 1  0 1 1 1  1 1 1 0  0 0 0 0  0 0 0 0
           0 0 1 1  0 1 1 1  1 0 1 1  1 0 1 0  0 1 1 0
           0 1 1 1  0 1 1 1  0 1 0 1  1 1 1 1  1 1 0 0

           0 0 0 0  1 0 1 1  1 0 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  0 0 0 1  0 0 0 1  0 1 1 0  0 0 1 1
           0 1 1 1  0 0 0 0  0 0 0 1  1 0 1 1  1 1 1 0
           0 0 1 1  0 0 1 0  0 0 1 1  1 1 1 1  1 1 1 0

           0 0 0 0  1 1 1 1  1 1 1 1  1 0 1 1  0 0 0 0
           0 2 2 2  0 0 1 1  1 0 1 1  0 1 0 1  1 0 0 0
           0 2 0 0  1 1 1 1  1 1 1 0  1 0 0 1  1 1 0 0
           0 0 0 0  0 1 1 1  0 1 1 0  0 1 1 1  0 0 0 0

           0 0 0 1  0 0 0 1  0 0 1 0  1 1 1 0  0 0 0 0
           1 1 0 1  0 0 0 0  1 0 1 0  1 1 1 0  0 0 1 0
           0 1 1 0  0 0 1 0  1 1 1 0  1 1 1 1  0 0 0 0
           0 1 1 0  1 0 1 1  1 1 0 0  0 1 0 1  0 0 0 0

           0 0 0 0  1 0 0 1  1 1 0 1  1 1 0 0  1 1 0 0
           0 0 1 1  0 1 1 1  1 1 0 1  1 1 0 0  0 1 1 0
           0 0 0 1  1 1 1 1  1 1 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  0 1 1 1  1 0 0 0  1 0 0 0  0 0 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 16  16  0  0
           0 0  0  16 16
           4 15 16 16 0
           16 0  7  16 0
           0 16 16 0  0 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0 1.0 1.0 0.0 0.0
             0.0 0.0 0.0 1.0 1.0
             0.25 0.9375 1.0 1.0 0.0
             1.0 0.0 0.4375 1.0 0.0
             0.0 1.0 1.0 0.0 0.0 ])
  expected_binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false  true  true false false
          false false false  true  true
          false  true true  true  false
           true false false  true false
          false  true  true false false ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                               potential_lake_pixel_mask.data)
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel,
                                                  grid_specific_lake_model_parameters)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      lake_pixel_coords_list,
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
                             non_lake_mask,
                             grid_specific_lake_model_parameters,
                             lake_grid,
                             surface_grid)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  @test lake_fractions_field == expected_lake_fractions_field
  @test binary_lake_mask == expected_binary_lake_mask
end

@testset "Lake Fraction Calculation Test 10" begin
  #Multiple partially filled lakes with uneven cell sizes
  lake_grid::Grid = LatLonGrid(15,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  potential_lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 2 2 2
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  2 2 2 2

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 2 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 2 2

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 1 1 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           1 1 1 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           1 1 1 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 ])
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 2
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  2 0 0 2

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 2 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 2 2

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           1 1 1 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           1 1 1 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
           16 16 16 16  8
           16 16 16 16 16
            4  4  4  4  4
           16 16 16 16 16 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 0 0 0 0
           0 0 0 0 6
           0 0 0 0 0
           0 0 0 0 0
           8 0 0 0 0 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0 0.0 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 0.75
             0.0 0.0 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 0.0
             0.5 0.0 0.0 0.0 0.0 ])
  expected_binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false false false false false
          false false false false  true
          false false false false false
          false false false false false
           true false false false false ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                               potential_lake_pixel_mask.data)
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel,
                                                  grid_specific_lake_model_parameters)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      lake_pixel_coords_list,
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
                             non_lake_mask,
                             grid_specific_lake_model_parameters,
                             lake_grid,
                             surface_grid)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  @test lake_fractions_field == expected_lake_fractions_field
  @test binary_lake_mask == expected_binary_lake_mask
end

@testset "Lake Fraction Calculation Test 11" begin
  #Single lake
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  lake_number::Int64 = 1
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
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
  binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false false false false false
          false false false false false
          false false false false false
          false false false false true
          false false false false true ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 0 0 0 0
           0 0 0 0 0
           0 0 0 0 0
           0 0 0 0 16
           0 0 0 0 14 ])
  expected_immediate_lake_pixel_counts_field::Field{Int64} =
    LatLonField{Int64}(surface_grid,
      Int64[ 0 0 0 0 0
             0 0 0 0 0
             0 0 0 0 0
             0 0 0 0 13
             0 0 0 0 14 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0 0.0 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 1.0
             0.0 0.0 0.0 0.0 0.875 ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == 1,
                                                           lake_pixel_mask.data)
  potential_lake_pixel_coords_list::Vector{CartesianIndex} = lake_pixel_coords_list
  cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
  for pixel in potential_lake_pixel_coords_list
    cell_coords::CartesianIndex =
      get_corresponding_surface_model_grid_cell(pixel,
                                                grid_specific_lake_model_parameters)
    set!(cell_mask,cell_coords,true)
  end
  cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
  input = LakeInput(lake_number,
                    CartesianIndex[],
                    potential_lake_pixel_coords_list,
                    cell_coords_list)
  push!(lakes,input)
  prognostics::LakeFractionCalculationPrognostics =
    setup_lake_for_fraction_calculation(lakes,
                                        cell_pixel_counts,
                                        non_lake_mask,
                                        binary_lake_mask,
                                        lake_pixel_mask,
                                        grid_specific_lake_model_parameters,
                                        lake_grid,
                                        surface_grid)
  lake_pixel_counts_field::Field{Int64} = Field{Int64}(surface_grid,0)
  for coords in lake_pixel_coords_list
    add_pixel_by_coords(coords,
                        lake_pixel_counts_field,prognostics)
  end
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  #@test lake_fractions_field == expected_lake_fractions_field
  remove_pixel_by_coords(CartesianIndex(15,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(11,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,20),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_immediate_lake_pixel_counts_field
  add_pixel_by_coords(CartesianIndex(15,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(11,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,20),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
end

@testset "Lake Fraction Calculation Test 12" begin
  #Multiple lakes
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 2 2 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 7 0
           0 2 2 2  2 0 0 0  0 0 3 3  0 0 0 0  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 6  6 6 6 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 6 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 4 4 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 4 4 4  4 4 0 0  0 0 0 0  0 1 1 0
           0 0 0 0  0 4 4 0  0 0 0 0  0 0 0 0  0 1 1 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 1 1
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 1  1 1 1 1
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 1  1 1 1 1
           0 0 0 0  0 5 0 0  0 0 0 0  0 0 0 0  0 1 0 0

           0 0 0 0  0 5 5 0  0 0 0 0  0 0 1 0  0 1 0 0
           0 0 0 0  0 5 0 0  0 0 0 0  0 0 1 0  0 1 1 0
           0 0 0 5  5 5 0 0  0 0 0 0  0 0 1 1  1 1 0 0
           0 0 0 0  0 5 0 0  0 0 0 0  0 0 0 0  0 1 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false false false false false
          false false false false  true
          false  true false false false
          false false false false  true
          false  true false false  true ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 5 1 2 0 1
           0 0 0 0 11
           0 9 0 0 0
           0 0 0 0 16
           0 8 0 0 14 ])
  expected_immediate_lake_pixel_counts_field::Field{Int64} =
    LatLonField{Int64}(surface_grid,
      Int64[  3 1 0 0 0
              0 0 0 0 0
              0 3 0 0 0
              0 0 0 0 7
              0 6 0 0 1  ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.3125 0.0625 0.125 0.0 0.0625
             0.0 0.0 0.0 0.0 0.6875
             0.0 0.5625 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 1.0
             0.0 0.5 0.0 0.0 0.875 ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  for lake_number::Int64=1:7
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} = lake_pixel_coords_list
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel,
                                                  grid_specific_lake_model_parameters)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      CartesianIndex[],
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  prognostics::LakeFractionCalculationPrognostics =
    setup_lake_for_fraction_calculation(lakes,
                                        cell_pixel_counts,
                                        non_lake_mask,
                                        binary_lake_mask,
                                        lake_pixel_mask,
                                        grid_specific_lake_model_parameters,
                                        lake_grid,
                                        surface_grid)
  lake_pixel_counts_field::Field{Int64} = Field{Int64}(surface_grid,0)
  for lake_number::Int64=1:7
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    for coords in lake_pixel_coords_list
      add_pixel_by_coords(coords,
                          lake_pixel_counts_field,prognostics)
    end
  end
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  #@test lake_fractions_field == expected_lake_fractions_field
  remove_pixel_by_coords(CartesianIndex(2,2),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(16,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(8,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(5,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(6,17),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,17),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(2,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(12,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(12,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(11,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(11,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(12,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(12,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(11,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(14,16),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,16),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,15),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(14,15),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(14,17),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,17),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(20,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,17),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,16),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,15),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,15),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,15),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(13,20),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(14,20),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,20),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(6,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(7,17),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(7,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(5,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(5,17),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(5,16),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_immediate_lake_pixel_counts_field
  add_pixel_by_coords(CartesianIndex(2,2),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(16,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(8,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(5,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(6,17),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,17),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(2,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(12,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(12,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(11,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(11,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(12,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(12,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(11,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(14,16),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,16),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,15),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(14,15),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(14,17),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,17),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(20,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,17),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,16),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,15),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,15),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,15),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(13,20),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(14,20),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,20),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(6,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(7,17),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(7,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(5,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(5,17),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(5,16),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
end

@testset "Lake Fraction Calculation Test 13" begin
  #Multiple lakes
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 1 0  1 1 0 0  0 0 1 0  0 0 0 0  0 0 7 0
           0 0 1 0  1 1 0 1  1 1 1 0  0 0 8 8  0 7 7 0
           0 0 1 0  1 1 1 1  1 1 0 0  0 0 8 8  0 7 7 0

           0 0 1 1  1 1 1 1  0 0 0 0  0 0 0 8  0 7 7 0
           0 0 1 1  1 0 1 1  1 1 1 0  0 0 0 0  0 7 7 0
           0 0 1 1  1 2 1 1  1 1 0 0  0 0 8 8  8 8 0 9
           0 0 1 0  1 2 1 1  0 0 0 0  0 0 8 8  8 8 9 9

           0 0 1 0  2 2 1 1  0 0 0 0  0 0 0 0  0 0 9 9
           0 0 1 1  2 2 1 1  0 0 6 0  0 0 0 9  9 9 9 9
           0 0 0 0  2 2 2 1  0 0 0 0  0 0 0 0  0 9 9 0
           0 0 0 2  2 2 0 1  0 0 0 0  0 0 0 0  0 5 5 0

           0 0 2 2  2 2 0 0  0 0 0 0  0 3 3 5  0 5 0 0
           0 0 2 2  2 2 0 0  0 0 6 0  4 3 3 5  5 5 0 0
           0 0 0 0  2 0 0 0  0 0 0 0  4 4 0 5  5 0 0 0
           0 0 0 0  2 0 0 0  0 0 0 4  4 4 0 5  5 0 0 0

           0 0 0 0  2 2 2 0  0 0 4 4  4 4 0 0  5 5 0 0
           0 0 6 0  0 0 0 0  0 0 0 4  4 0 0 0  0 5 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false  true false false  true
           true  true false  true false
          false  true false false  true
          false  true false  true  true
          false false false false false ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 16  0  0 9
           16 16 0 13 0
           0 16  1  0 14
           0 13  2 16 8
           1 0  0  3 0 ])
  expected_lake_pixel_counts_field_two::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 16  0  0 9
           16 16 0 13 0
           0 16  1  0 14
           0 13  1 16 8
           1 0  1  3 0 ])
  expected_intermediate_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 12  0  0 5
           11 1 0   9 0
           0 16  0  0 8
           0 10  1 10 5
           1 0  0  0 0 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0    1.0     0.0    0.0    0.5625
             1.0    0.875   0.0    0.8125 0.0
             0.0    1.0     0.0625 0.0    0.75
             0.0    0.9375  0.0625 0.75   0.875
             0.0625 0.0     0.1875 0.0625 0.0 ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  for lake_number::Int64=1:9
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} = lake_pixel_coords_list
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel,
                                                  grid_specific_lake_model_parameters)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      CartesianIndex[],
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  prognostics::LakeFractionCalculationPrognostics =
    setup_lake_for_fraction_calculation(lakes,
                                        cell_pixel_counts,
                                        non_lake_mask,
                                        binary_lake_mask,
                                        lake_pixel_mask,
                                        grid_specific_lake_model_parameters,
                                        lake_grid,
                                        surface_grid)
  lake_pixel_counts_field::Field{Int64} = Field{Int64}(surface_grid,0)
  for lake_number::Int64=1:9
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    for coords in lake_pixel_coords_list
      add_pixel_by_coords(coords,
                          lake_pixel_counts_field,prognostics)
    end
  end
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  #@test lake_fractions_field == expected_lake_fractions_field
  remove_pixel_by_coords(CartesianIndex(2,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(2,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(5,5),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(5,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(5,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(6,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(6,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(6,5),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(7,5),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(7,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(7,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(7,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(8,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(8,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(8,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(8,5),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,5),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,14),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(14,14),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(14,15),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(13,15),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(13,14),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(8,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(8,20),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,20),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,16),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,17),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(7,17),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(8,17),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(8,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(7,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(6,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(5,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(5,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(6,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,11),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_intermediate_lake_pixel_counts_field
  add_pixel_by_coords(CartesianIndex(2,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(2,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(5,5),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(5,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(5,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(6,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(6,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(6,5),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(7,5),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(7,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(7,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(7,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(8,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(8,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(8,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(8,5),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,5),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,14),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(14,14),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(14,15),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(13,15),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(13,14),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(8,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(8,20),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,20),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,16),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,17),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(7,17),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(8,17),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(8,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(7,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(6,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(5,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(5,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(6,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,11),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field_two
end

@testset "Lake Fraction Calculation Test 14" begin
  #Multiple lakes
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 1  0 1 1 1  1 1 1 0  0 0 0 0  0 0 0 0
           0 0 1 1  0 1 1 1  1 0 1 1  1 0 1 0  0 1 1 0
           0 1 1 1  0 1 1 1  1 1 1 1  1 1 1 1  1 1 0 0

           0 0 0 0  1 0 1 1  1 0 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  1 1 1 1  1 1 0 1  0 1 1 0  0 0 1 1
           0 1 1 1  1 1 1 1  1 1 0 1  1 0 1 1  1 1 1 0
           0 0 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 0

           0 0 0 0  1 1 1 1  1 1 1 1  1 0 1 1  0 0 0 0
           0 2 2 2  2 0 1 1  1 0 1 1  1 1 0 1  1 0 0 0
           0 2 2 0  1 1 1 1  1 1 1 0  1 1 1 1  1 1 0 0
           0 0 2 0  0 1 1 1  0 1 1 0  0 1 1 1  0 0 0 0

           0 0 2 1  0 1 0 1  0 0 1 0  1 1 1 0  0 1 0 0
           1 1 0 1  1 1 1 1  1 0 1 0  1 1 1 0  1 1 1 0
           0 1 1 0  1 1 1 1  1 1 1 0  1 1 1 1  1 1 0 0
           0 1 1 0  1 1 1 1  1 1 0 0  0 1 0 1  0 0 0 0

           0 0 0 0  1 1 0 1  1 1 0 1  1 1 0 0  1 1 0 0
           0 0 1 1  0 1 1 1  1 1 0 1  1 1 0 0  0 1 1 0
           0 0 0 1  1 1 1 1  1 1 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  0 1 1 1  1 0 0 0  1 0 0 0  0 0 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false  true  true false false
          false  true  true  true  true
           true  true  true  true false
           true  true  false  true false
          false  true  true false false ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 16 16 0 0
           0 16 16 16 12
           6 16 16 16 0
          16 16  0 16 0
           0 16 16 0 0 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0 1.0 1.0 0.0 0.0
             0.0 1.0 1.0 1.0 1.0
             0.5 0.9375 1.0 1.0 0.0
             0.6875 1.0 0.0 1.0 0.0
             0.0 1.0 1.0 0.0 0.0 ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} = lake_pixel_coords_list
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel,
                                                  grid_specific_lake_model_parameters)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      CartesianIndex[],
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  prognostics::LakeFractionCalculationPrognostics =
    setup_lake_for_fraction_calculation(lakes,
                                        cell_pixel_counts,
                                        non_lake_mask,
                                        binary_lake_mask,
                                        lake_pixel_mask,
                                        grid_specific_lake_model_parameters,
                                        lake_grid,
                                        surface_grid)
  lake_pixel_counts_field::Field{Int64} = Field{Int64}(surface_grid,0)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    for coords in lake_pixel_coords_list
      add_pixel_by_coords(coords,
                          lake_pixel_counts_field,prognostics)
    end
  end
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  #@test lake_fractions_field == expected_lake_fractions_field
end

@testset "Lake Fraction Calculation Test 15" begin
  #Multiple lakes
  lake_grid::Grid = LatLonGrid(15,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 2 2 2
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  2 2 2 2

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 2 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 2 2

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 1 1 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           1 1 1 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           1 1 1 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
            8  8  8  8  8
           16 16 16 16 16
            4  4  4  4  4
           16 16 16 16 16 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false false false false  true
          false false false false false
          false false false false false
           true false false false false
          false  false false false false ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 0 0 0 10
           0 0 0 0 0
           0 0 0 0 0
           4 0 0 0 0
           6 0 0 0 0 ])
  expected_intermediate_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 0 0 0 3
           0 0 0 0 0
           0 0 0 0 0
           4 0 0 0 0
           3 0 0 0 0 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0 0.0 0.0 0.0 0.625
             0.0 0.0 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 0.0
             1.0 0.0 0.0 0.0 0.0
             0.375 0.0 0.0 0.0 0.0 ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} = lake_pixel_coords_list
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel,
                                                  grid_specific_lake_model_parameters)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      CartesianIndex[],
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  prognostics::LakeFractionCalculationPrognostics =
    setup_lake_for_fraction_calculation(lakes,
                                        cell_pixel_counts,
                                        non_lake_mask,
                                        binary_lake_mask,
                                        lake_pixel_mask,
                                        grid_specific_lake_model_parameters,
                                        lake_grid,
                                        surface_grid)
  lake_pixel_counts_field::Field{Int64} = Field{Int64}(surface_grid,0)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    for coords in lake_pixel_coords_list
      add_pixel_by_coords(coords,
                          lake_pixel_counts_field,prognostics)
    end
  end
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  #@test lake_fractions_field == expected_lake_fractions_field
  remove_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(11,4),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(11,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,20),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,20),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,17),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_intermediate_lake_pixel_counts_field
  add_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(11,4),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(11,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,20),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,20),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,17),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
end

@testset "Lake Fraction Calculation Test 16" begin
  #Partially filled single lake
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  lake_number::Int64 = 1
  potential_lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
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
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 1 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 0 1
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 1  1 0 0 1
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
    corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false false false false false
          false false false false false
          false false false false false
          false false false false false
          false false false false true ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 0 0 0 0
           0 0 0 0 0
           0 0 0 0 2
           0 0 0 0 2
           0 0 0 1 16 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0 0.0 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 0.3125
             0.0 0.0 0.0 0.0 1.0 ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == 1,
                                                           lake_pixel_mask.data)
  potential_lake_pixel_coords_list::Vector{CartesianIndex} =
    findall(x -> x == 1,potential_lake_pixel_mask.data)
  cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
  for pixel in potential_lake_pixel_coords_list
    cell_coords::CartesianIndex =
      get_corresponding_surface_model_grid_cell(pixel,
                                                grid_specific_lake_model_parameters)
    set!(cell_mask,cell_coords,true)
  end
  cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
  input = LakeInput(lake_number,
                    CartesianIndex[],
                    potential_lake_pixel_coords_list,
                    cell_coords_list)
  push!(lakes,input)
  prognostics::LakeFractionCalculationPrognostics =
    setup_lake_for_fraction_calculation(lakes,
                                        cell_pixel_counts,
                                        non_lake_mask,
                                        binary_lake_mask,
                                        potential_lake_pixel_mask,
                                        grid_specific_lake_model_parameters,
                                        lake_grid,
                                        surface_grid)
  lake_pixel_counts_field::Field{Int64} = Field{Int64}(surface_grid,0)
  for coords in lake_pixel_coords_list
    add_pixel_by_coords(coords,
                        lake_pixel_counts_field,prognostics)
  end
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  #@test lake_fractions_field == expected_lake_fractions_field
end

@testset "Lake Fraction Calculation Test 17" begin
  #Multiple partially filled lakes
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  potential_lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 2 2 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 7 0
           0 2 2 2  2 0 0 0  0 0 3 3  0 0 0 0  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 6  6 6 6 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 6 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 4 4 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 4 4 4  4 4 0 0  0 0 0 0  0 1 1 0
           0 0 0 0  0 4 4 0  0 0 0 0  0 0 0 0  0 1 1 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 1 1
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 1  1 1 1 1
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 1  1 1 1 1
           0 0 0 0  0 5 0 0  0 0 0 0  0 0 0 0  0 1 0 0

           0 0 0 0  0 5 5 0  0 0 0 0  0 0 1 0  0 1 0 0
           0 0 0 0  0 5 0 0  0 0 0 0  0 0 1 0  0 1 1 0
           0 0 0 5  5 5 0 0  0 0 0 0  0 0 1 1  1 1 0 0
           0 0 0 0  0 5 0 0  0 0 0 0  0 0 0 0  0 1 0 0 ])
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 2 2 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 7 0
           0 2 2 2  2 0 0 0  0 0 3 3  0 0 0 0  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 6  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  6 6 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 4 4 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 4 4 0  4 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 4 4 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 1 1
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  1 1 1 1
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  1 1 1 1
           0 0 0 0  0 5 0 0  0 0 0 0  0 0 0 0  0 1 0 0

           0 0 0 0  0 5 5 0  0 0 0 0  0 0 0 0  0 1 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 1 0
           0 0 0 5  5 0 0 0  0 0 0 0  0 0 0 0  1 1 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 1 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16 ])
    corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false false false false false
          false false false false  true
          false false false false false
          false false false false  true
          false false false false false ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 5 1 2 0 1
           0 0 0 0 9
           0 6 1 0 0
           0 1 0 0 16
           1 3 0 0 2 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.3125 0.0625 0.125 0.0 0.0625
             0.0 0.0 0.0 0.0 0.5625
             0.0 0.375 0.0625 0.0 0.0
             0.0 0.0625 0.0 0.0 1.0
             0.0625 0.1875 0.0 0.0 0.125 ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  for lake_number::Int64=1:7
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} =
      findall(x -> x == lake_number,potential_lake_pixel_mask.data)
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel,
                                                  grid_specific_lake_model_parameters)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      CartesianIndex[],
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  prognostics::LakeFractionCalculationPrognostics =
    setup_lake_for_fraction_calculation(lakes,
                                        cell_pixel_counts,
                                        non_lake_mask,
                                        binary_lake_mask,
                                        potential_lake_pixel_mask,
                                        grid_specific_lake_model_parameters,
                                        lake_grid,
                                        surface_grid)
  lake_pixel_counts_field::Field{Int64} = Field{Int64}(surface_grid,0)
  for lake_number::Int64=1:7
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    for coords in lake_pixel_coords_list
      add_pixel_by_coords(coords,
                          lake_pixel_counts_field,prognostics)
    end
  end
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  #@test lake_fractions_field == expected_lake_fractions_field
end

@testset "Lake Fraction Calculation Test 18" begin
  #Multiple partially filled lakes
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  potential_lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 1 0  1 1 0 0  0 0 1 0  0 0 0 0  0 0 7 0
           0 0 1 0  1 1 0 1  1 1 1 0  0 0 8 8  0 7 7 0
           0 0 1 0  1 1 1 1  1 1 0 0  0 0 8 8  0 7 7 0

           0 0 1 1  1 1 1 1  0 0 0 0  0 0 0 8  0 7 7 0
           0 0 1 1  1 0 1 1  1 1 1 0  0 0 0 0  0 7 7 0
           0 0 1 1  1 2 1 1  1 1 0 0  0 0 8 8  8 8 0 9
           0 0 1 0  1 2 1 1  0 0 0 0  0 0 8 8  8 8 9 9

           0 0 1 0  2 2 1 1  0 0 0 0  0 0 0 0  0 0 9 9
           0 0 1 1  2 2 1 1  0 0 6 0  0 0 0 9  9 9 9 9
           0 0 0 0  2 2 2 1  0 0 0 0  0 0 0 0  0 9 9 0
           0 0 0 2  2 2 0 1  0 0 0 0  0 0 0 0  0 5 5 0

           0 0 2 2  2 2 0 0  0 0 0 0  0 3 3 5  0 5 0 0
           0 0 2 2  2 2 0 0  0 0 6 0  4 3 3 5  5 5 0 0
           0 0 0 0  2 0 0 0  0 0 0 0  4 4 0 5  5 0 0 0
           0 0 0 0  2 0 0 0  0 0 0 4  4 4 0 5  5 0 0 0

           0 0 0 0  2 2 2 0  0 0 4 4  4 4 0 0  5 5 0 0
           0 0 6 0  0 0 0 0  0 0 0 4  4 0 0 0  0 5 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 ])
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  1 1 0 0  0 0 0 0  0 0 0 0  0 0 7 0
           0 0 0 0  1 1 0 0  0 0 0 0  0 0 8 8  0 0 7 0
           0 0 0 0  1 1 1 1  0 1 0 0  0 0 8 8  0 7 0 0

           0 0 0 0  0 0 1 0  0 0 0 0  0 0 0 8  0 7 7 0
           0 0 0 0  0 0 1 0  0 0 0 0  0 0 0 0  0 7 7 0
           0 0 1 0  0 0 1 1  0 0 0 0  0 0 0 0  8 8 0 9
           0 0 1 0  0 0 1 1  0 0 0 0  0 0 8 0  8 8 9 9

           0 0 0 0  2 2 1 1  0 0 0 0  0 0 0 0  0 0 9 9
           0 0 0 0  2 2 1 1  0 0 6 0  0 0 0 9  9 0 0 0
           0 0 0 0  2 2 2 1  0 0 0 0  0 0 0 0  0 9 9 0
           0 0 0 2  2 2 0 1  0 0 0 0  0 0 0 0  0 0 5 0

           0 0 2 2  0 0 0 0  0 0 0 0  0 3 3 5  0 0 0 0
           0 0 2 2  0 0 0 0  0 0 6 0  4 3 3 5  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  4 4 0 5  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 4  4 4 0 5  0 0 0 0

           0 0 0 0  2 2 2 0  0 0 4 0  4 0 0 0  5 0 0 0
           0 0 6 0  0 0 0 0  0 0 0 0  4 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16 ])
    corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false  true false  true false
          false false false false false
          false  true false false  true
          false false false  true false
          false false false false false ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 16 0 10 3
           0 0  0  0 4
           1 16 1  0 11
           4 0  2 16 0
           1 3  0  0 0 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0 1.0 0.0 0.625 0.1875
             0.0 0.375 0.0 0.0 0.25
             0.0 0.6875 0.0625 0.0 0.625
             0.25 0.0 0.0625 1.0 0.0
             0.0625 0.1875 0.0 0.0625 0.0625 ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  for lake_number::Int64=1:9
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} =
      findall(x -> x == lake_number,potential_lake_pixel_mask.data)
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel,
                                                  grid_specific_lake_model_parameters)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      CartesianIndex[],
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  prognostics::LakeFractionCalculationPrognostics =
    setup_lake_for_fraction_calculation(lakes,
                                        cell_pixel_counts,
                                        non_lake_mask,
                                        binary_lake_mask,
                                        potential_lake_pixel_mask,
                                        grid_specific_lake_model_parameters,
                                        lake_grid,
                                        surface_grid)
  lake_pixel_counts_field::Field{Int64} = Field{Int64}(surface_grid,0)
  for lake_number::Int64=1:9
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    for coords in lake_pixel_coords_list
      add_pixel_by_coords(coords,
                          lake_pixel_counts_field,prognostics)
    end
  end
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  #@test lake_fractions_field == expected_lake_fractions_field
end

@testset "Lake Fraction Calculation Test 19" begin
  #Multiple partially filled lakes
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  potential_lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 1  0 1 1 1  1 1 1 0  0 0 0 0  0 0 0 0
           0 0 1 1  0 1 1 1  1 0 1 1  1 0 1 0  0 1 1 0
           0 1 1 1  0 1 1 1  1 1 1 1  1 1 1 1  1 1 0 0

           0 0 0 0  1 0 1 1  1 0 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  1 1 1 1  1 1 0 1  0 1 1 0  0 0 1 1
           0 1 1 1  1 1 1 1  1 1 0 1  1 0 1 1  1 1 1 0
           0 0 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 0

           0 0 0 0  1 1 1 1  1 1 1 1  1 0 1 1  0 0 0 0
           0 2 2 2  2 0 1 1  1 0 1 1  1 1 0 1  1 0 0 0
           0 2 2 0  1 1 1 1  1 1 1 0  1 1 1 1  1 1 0 0
           0 0 0 0  0 1 1 1  0 1 1 0  0 1 1 1  0 0 0 0

           0 0 0 1  0 1 0 1  0 0 1 0  1 1 1 0  0 1 0 0
           1 1 0 1  1 1 1 1  1 0 1 0  1 1 1 0  1 1 1 0
           0 1 1 0  1 1 1 1  1 1 1 0  1 1 1 1  1 1 0 0
           0 1 1 0  1 1 1 1  1 1 0 0  0 1 0 1  0 0 0 0

           0 0 0 0  1 1 0 1  1 1 0 1  1 1 0 0  1 1 0 0
           0 0 1 1  0 1 1 1  1 1 0 1  1 1 0 0  0 1 1 0
           0 0 0 1  1 1 1 1  1 1 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  0 1 1 1  1 0 0 0  1 0 0 0  0 0 0 0 ])
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 1  0 1 1 1  1 1 1 0  0 0 0 0  0 0 0 0
           0 0 1 1  0 1 1 1  1 0 1 1  1 0 1 0  0 1 1 0
           0 1 1 1  0 1 1 1  0 1 0 1  1 1 1 1  1 1 0 0

           0 0 0 0  1 0 1 1  1 0 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  0 0 0 1  0 0 0 1  0 1 1 0  0 0 1 1
           0 1 1 1  0 0 0 0  0 0 0 1  1 0 1 1  1 1 1 0
           0 0 1 1  0 0 1 0  0 0 1 1  1 1 1 1  1 1 1 0

           0 0 0 0  1 1 1 1  1 1 1 1  1 0 1 1  0 0 0 0
           0 2 2 2  0 0 1 1  1 0 1 1  0 1 0 1  1 0 0 0
           0 2 0 0  1 1 1 1  1 1 1 0  1 0 0 1  1 1 0 0
           0 0 0 0  0 1 1 1  0 1 1 0  0 1 1 1  0 0 0 0

           0 0 0 1  0 0 0 1  0 0 1 0  1 1 1 0  0 0 0 0
           1 1 0 1  0 0 0 0  1 0 1 0  1 1 1 0  0 0 1 0
           0 1 1 0  0 0 1 0  1 1 1 0  1 1 1 1  0 0 0 0
           0 1 1 0  1 0 1 1  1 1 0 0  0 1 0 1  0 0 0 0

           0 0 0 0  1 0 0 1  1 1 0 1  1 1 0 0  1 1 0 0
           0 0 1 1  0 1 1 1  1 1 0 1  1 1 0 0  0 1 1 0
           0 0 0 1  1 1 1 1  1 1 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  0 1 1 1  1 0 0 0  1 0 0 0  0 0 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16 ])
    corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false  true  true false false
          false false false  true  true
          false  true true  true  false
           true false false  true false
          false  true  true false false ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 16  16  0  0
           0 0  0  16 16
           3 16 16 16 0
           16 0  0  16 1
           0 16 16 0  6 ])
  expected_intermediate_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 14  12  0  0
           0 0  0  14 14
           0 12 11 16 0
           13 0  0  15 0
           0 10 11 0  0 ])
  expected_lake_pixel_counts_field_after_cycle::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 1 16  16  0  0
           0 2  2  16 16
           1 16 16 16 0
           16 0  0  16 0
           1 16 16 1  2 ])
  expected_intermediate_lake_pixel_counts_field_two::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 15  12  0  0
           0 0  0  15 15
           0 10 11 16 0
           13 0  0  15 0
           0 10 10 0  0 ])
  expected_intermediate_lake_pixel_counts_field_three::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 14  13  0  0
           0 0  0  15 16
           0 10 12 16 0
           12 0  0  15 0
           0 7 12 0  0 ])
  expected_lake_pixel_counts_field_after_second_cycle::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 2 16  16  0  0
           0 1  2  16 16
           1 16 16 16 0
           16 0  1  16 1
           1 16 16 0  1 ])
  expected_lake_pixel_counts_field_after_third_cycle::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 16  16  0  0
           0 0  2  16 16
           1 16 16 16 0
           16 0  4  16 0
           0 16 16 1  2 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0 1.0 1.0 0.0 0.0
             0.0 0.0 0.0 1.0 1.0
             0.25 0.9375 1.0 1.0 0.0
             1.0 0.0 0.4375 1.0 0.0
             0.0 1.0 1.0 0.0 0.0 ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} =
      findall(x -> x == lake_number,potential_lake_pixel_mask.data)
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel,
                                                  grid_specific_lake_model_parameters)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      CartesianIndex[],
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  prognostics::LakeFractionCalculationPrognostics =
    setup_lake_for_fraction_calculation(lakes,
                                        cell_pixel_counts,
                                        non_lake_mask,
                                        binary_lake_mask,
                                        potential_lake_pixel_mask,
                                        grid_specific_lake_model_parameters,
                                        lake_grid,
                                        surface_grid)
  lake_pixel_counts_field::Field{Int64} = Field{Int64}(surface_grid,0)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    for coords in lake_pixel_coords_list
      add_pixel_by_coords(coords,
                          lake_pixel_counts_field,prognostics)
    end
  end
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  #@test lake_fractions_field == expected_lake_fractions_field
  remove_pixel_by_coords(CartesianIndex(2,4),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(2,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,4),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,2),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(16,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(14,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(13,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(13,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(14,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(16,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(20,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(8,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(7,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(6,12),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_intermediate_lake_pixel_counts_field
  add_pixel_by_coords(CartesianIndex(2,4),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(2,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,4),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,2),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(16,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(14,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(13,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(13,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(14,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(16,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(20,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(8,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(7,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(6,12),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field_after_cycle
  remove_pixel_by_coords(CartesianIndex(2,4),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(2,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,4),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,2),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(16,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(14,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(13,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(13,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(14,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(16,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(20,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(8,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(7,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(6,12),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_intermediate_lake_pixel_counts_field_two
  add_pixel_by_coords(CartesianIndex(2,4),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(2,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,4),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,2),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(16,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(14,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(13,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(13,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(14,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(16,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(20,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(8,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(7,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(6,12),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field_after_second_cycle
  remove_pixel_by_coords(CartesianIndex(2,4),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(2,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,4),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,2),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(16,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(14,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(13,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(13,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(14,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(16,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(20,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(8,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(7,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(6,12),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_intermediate_lake_pixel_counts_field_three
  add_pixel_by_coords(CartesianIndex(2,4),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(2,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,4),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,2),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(16,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(14,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(13,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(13,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(14,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(16,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(20,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(8,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(7,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(6,12),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field_after_third_cycle
end

@testset "Lake Fraction Calculation Test 20" begin
  #Multiple partially filled lakes with uneven cell sizes
  lake_grid::Grid = LatLonGrid(15,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  potential_lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 2 2 2
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  2 2 2 2

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 2 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 2 2

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 1 1 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           1 1 1 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           1 1 1 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 ])
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 2
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  2 0 0 2

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 2 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 2 2

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 1 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           1 1 1 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           1 1 1 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
            8  8  8  8  8
           16 16 16 16 16
            4  4  4  4  4
           16 16 16 16 16 ])
    corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false false false false false
          false false false false  true
          false false false false false
          false false false false false
           true false false false false ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 0 0 0 0
           0 0 0 0 6
           0 0 0 0 0
           0 0 0 0 0
           8 0 0 0 0 ])
  expected_intermediate_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 0 0 0 0
           0 0 0 0 2
           0 0 0 0 0
           0 0 0 0 0
           2 0 0 0 0 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0 0.0 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 0.75
             0.0 0.0 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 0.0
             0.5 0.0 0.0 0.0 0.0 ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} = lake_pixel_coords_list
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel,
                                                  grid_specific_lake_model_parameters)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      CartesianIndex[],
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  prognostics::LakeFractionCalculationPrognostics =
    setup_lake_for_fraction_calculation(lakes,
                                        cell_pixel_counts,
                                        non_lake_mask,
                                        binary_lake_mask,
                                        potential_lake_pixel_mask,
                                        grid_specific_lake_model_parameters,
                                        lake_grid,
                                        surface_grid)
  lake_pixel_counts_field::Field{Int64} = Field{Int64}(surface_grid,0)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    for coords in lake_pixel_coords_list
      add_pixel_by_coords(coords,
                          lake_pixel_counts_field,prognostics)
    end
  end
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  #@test lake_fractions_field == expected_lake_fractions_field
  remove_pixel_by_coords(CartesianIndex(12,1),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(12,2),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(12,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(13,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(13,2),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(13,1),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(6,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(6,20),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,20),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,20),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_intermediate_lake_pixel_counts_field
  add_pixel_by_coords(CartesianIndex(12,1),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(12,2),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(12,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(13,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(13,2),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(13,1),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(6,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(6,20),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,20),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,20),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
end

@testset "Lake Fraction Calculation Test 21" begin
  #Two lake example taken from lake tests 22
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(3,3,true)
  lakes::Vector{LakeInput} = LakeInput[]
  potential_lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
           0 0 0 1 1 1 0 1 1 0 1 1 1 1 0 0 1 1 1 0
           0 2 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
           0 2 0 1 1 1 0 1 1 1 1 1 1 1 0 0 1 1 1 0
           0 2 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
           0 2 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 1 0 0
           0 2 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 1 0
           0 2 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
           0 2 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
           0 2 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
           0 2 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
           0 2 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
           0 2 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
           0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
           0 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0
           0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
           0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
           0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
           0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
           0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0])
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
           0 0 0 1 1 1 0 1 1 0 1 1 1 1 0 0 1 1 1 0
           0 2 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
           0 2 0 1 1 1 0 1 1 1 1 1 1 1 0 0 1 1 1 0
           0 2 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
           0 2 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 1 0 0
           0 2 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 1 0
           0 2 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
           0 2 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
           0 2 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
           0 2 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
           0 2 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
           0 2 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
           0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
           0 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0
           0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
           0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
           0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
           0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
           0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 36 48 36
           48 64 48
           36 48 36  ])
    corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
           1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
           1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
           1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
           1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
           1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
           2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
           2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
           2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
           2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
           2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
           2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
           2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
           2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
           3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
           3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
           3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
           3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
           3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
           3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
           1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
           1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
           1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
           1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
           1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
           1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
           1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
           1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
           1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
           1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
           1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
           1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
           1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
           1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
           1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
           1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
           1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
           1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
           1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3 ])
  binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false true false
          false false false
          false  true false  ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 1  48 15
           0   6 21
           8  48 0 ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,false)
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} =
      findall(x -> x == lake_number,potential_lake_pixel_mask.data)
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel,
                                                  grid_specific_lake_model_parameters)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      CartesianIndex[],
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  prognostics::LakeFractionCalculationPrognostics =
    setup_lake_for_fraction_calculation(lakes,
                                        cell_pixel_counts,
                                        non_lake_mask,
                                        binary_lake_mask,
                                        potential_lake_pixel_mask,
                                        grid_specific_lake_model_parameters,
                                        lake_grid,
                                        surface_grid)
  lake_pixel_counts_field::Field{Int64} = Field{Int64}(surface_grid,0)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    for coords in lake_pixel_coords_list
      add_pixel_by_coords(coords,
                          lake_pixel_counts_field,prognostics)
    end
  end
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
end

@testset "Lake Fraction Calculation Test 22" begin
  #Multiple partially filled lakes and two non lake points
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  potential_lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 1  0 1 1 1  1 1 1 0  0 0 0 0  0 0 0 0
           0 0 1 1  0 1 1 1  1 0 1 1  1 0 1 0  0 1 1 0
           0 1 1 1  0 1 1 1  1 1 1 1  1 1 1 1  1 1 0 0

           0 0 0 0  1 0 1 1  1 0 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  1 1 1 1  1 1 0 1  0 1 1 0  0 0 1 1
           0 1 1 1  1 1 1 1  1 1 0 1  1 0 1 1  1 1 1 0
           0 0 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 0

           0 0 0 0  1 1 1 1  1 1 1 1  1 0 1 1  0 0 0 0
           0 2 2 2  2 0 1 1  1 0 1 1  1 1 0 1  1 0 0 0
           0 2 2 0  1 1 1 1  1 1 1 0  1 1 1 1  1 1 0 0
           0 0 0 0  0 1 1 1  0 1 1 0  0 1 1 1  0 0 0 0

           0 0 0 1  0 1 0 1  0 0 1 0  1 1 1 0  0 1 0 0
           1 1 0 1  1 1 1 1  1 0 1 0  1 1 1 0  1 1 1 0
           0 1 1 0  1 1 1 1  1 1 1 0  1 1 1 1  1 1 0 0
           0 1 1 0  1 1 1 1  1 1 0 0  0 1 0 1  0 0 0 0

           0 0 0 0  1 1 0 1  1 1 0 1  1 1 0 0  1 1 0 0
           0 0 1 1  0 1 1 1  1 1 0 1  1 1 0 0  0 1 1 0
           0 0 0 1  1 1 1 1  1 1 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  0 1 1 1  1 0 0 0  1 0 0 0  0 0 0 0 ])
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 1  0 1 1 1  1 1 1 0  0 0 0 0  0 0 0 0
           0 0 1 1  0 1 1 1  1 0 1 1  1 0 1 0  0 1 1 0
           0 1 1 1  0 1 1 1  0 1 0 1  1 1 1 1  1 1 0 0

           0 0 0 0  1 0 1 1  1 0 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  0 0 0 1  0 0 0 1  0 1 1 0  0 0 1 1
           0 1 1 1  0 0 0 0  0 0 0 1  1 0 1 1  1 1 1 0
           0 0 1 1  0 0 1 0  0 0 1 1  1 1 1 1  1 1 1 0

           0 0 0 0  1 1 1 1  1 1 1 1  1 0 1 1  0 0 0 0
           0 2 2 2  0 0 1 1  1 0 1 1  0 1 0 1  1 0 0 0
           0 2 0 0  1 1 1 1  1 1 1 0  1 0 0 1  1 1 0 0
           0 0 0 0  0 1 1 1  0 1 1 0  0 1 1 1  0 0 0 0

           0 0 0 1  0 0 0 1  0 0 1 0  1 1 1 0  0 0 0 0
           1 1 0 1  0 0 0 0  1 0 1 0  1 1 1 0  0 0 1 0
           0 1 1 0  0 0 1 0  1 1 1 0  1 1 1 1  0 0 0 0
           0 1 1 0  1 0 1 1  1 1 0 0  0 1 0 1  0 0 0 0

           0 0 0 0  1 0 0 1  1 1 0 1  1 1 0 0  1 1 0 0
           0 0 1 1  0 1 1 1  1 1 0 1  1 1 0 0  0 1 1 0
           0 0 0 1  1 1 1 1  1 1 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  0 1 1 1  1 0 0 0  1 0 0 0  0 0 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 0  16  0  0
           0 0  16  16 16
           4 15 16 16 0
           16 0 16  16 0
           0  0 16 7  0 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0 0.0 1.0 0.0 0.0
             0.0 0.0 1.0 1.0 1.0
             0.25 0.9375 1.0 1.0 0.0
             1.0 0.0 1.0 1.0 0.0
             0.0 0.0 1.0 0.4375 0.0 ])
  expected_binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false false  true false false
          false false  true  true  true
          false  true  true  true  false
           true false  true  true false
          false false  true false false ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false  true false false false
          false false false false false
          false false false false false
          false false false false false
          false  true false false false ])
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                               potential_lake_pixel_mask.data)
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel,
                                                  grid_specific_lake_model_parameters)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      lake_pixel_coords_list,
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
                             non_lake_mask,
                             grid_specific_lake_model_parameters,
                             lake_grid,
                             surface_grid)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  @test lake_fractions_field == expected_lake_fractions_field
  @test binary_lake_mask == expected_binary_lake_mask
end


@testset "Lake Fraction Calculation Test 23" begin
  #Multiple partially filled lakes and two non lake points
  lake_grid::Grid = LatLonGrid(20,20,true)
  surface_grid::Grid = LatLonGrid(5,5,true)
  lakes::Vector{LakeInput} = LakeInput[]
  potential_lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 1  0 1 1 1  1 1 1 0  0 0 0 0  0 0 0 0
           0 0 1 1  0 1 1 1  1 0 1 1  1 0 1 0  0 1 1 0
           0 1 1 1  0 1 1 1  1 1 1 1  1 1 1 1  1 1 0 0

           0 0 0 0  1 0 1 1  1 0 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  1 1 1 1  1 1 0 1  0 1 1 0  0 0 1 1
           0 1 1 1  1 1 1 1  1 1 0 1  1 0 1 1  1 1 1 0
           0 0 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 0

           0 0 0 0  1 1 1 1  1 1 1 1  1 0 1 1  0 0 0 0
           0 2 2 2  2 0 1 1  1 0 1 1  1 1 0 1  1 0 0 0
           0 2 2 0  1 1 1 1  1 1 1 0  1 1 1 1  1 1 0 0
           0 0 0 0  0 1 1 1  0 1 1 0  0 1 1 1  0 0 0 0

           0 0 0 1  0 1 0 1  0 0 1 0  1 1 1 0  0 1 0 0
           1 1 0 1  1 1 1 1  1 0 1 0  1 1 1 0  1 1 1 0
           0 1 1 0  1 1 1 1  1 1 1 0  1 1 1 1  1 1 0 0
           0 1 1 0  1 1 1 1  1 1 0 0  0 1 0 1  0 0 0 0

           0 0 0 0  1 1 0 1  1 1 0 1  1 1 0 0  1 1 0 0
           0 0 1 1  0 1 1 1  1 1 0 1  1 1 0 0  0 1 1 0
           0 0 0 1  1 1 1 1  1 1 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  0 1 1 1  1 0 0 0  1 0 0 0  0 0 0 0 ])
  lake_pixel_mask::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 1  0 1 1 1  1 1 1 0  0 0 0 0  0 0 0 0
           0 0 1 1  0 1 1 1  1 0 1 1  1 0 1 0  0 1 1 0
           0 1 1 1  0 1 1 1  0 1 0 1  1 1 1 1  1 1 0 0

           0 0 0 0  1 0 1 1  1 0 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  0 0 0 1  0 0 0 1  0 1 1 0  0 0 1 1
           0 1 1 1  0 0 0 0  0 0 0 1  1 0 1 1  1 1 1 0
           0 0 1 1  0 0 1 0  0 0 1 1  1 1 1 1  1 1 1 0

           0 0 0 0  1 1 1 1  1 1 1 1  1 0 1 1  0 0 0 0
           0 2 2 2  0 0 1 1  1 0 1 1  0 1 0 1  1 0 0 0
           0 2 0 0  1 1 1 1  1 1 1 0  1 0 0 1  1 1 0 0
           0 0 0 0  0 1 1 1  0 1 1 0  0 1 1 1  0 0 0 0

           0 0 0 1  0 0 0 1  0 0 1 0  1 1 1 0  0 0 0 0
           1 1 0 1  0 0 0 0  1 0 1 0  1 1 1 0  0 0 1 0
           0 1 1 0  0 0 1 0  1 1 1 0  1 1 1 1  0 0 0 0
           0 1 1 0  1 0 1 1  1 1 0 0  0 1 0 1  0 0 0 0

           0 0 0 0  1 0 0 1  1 1 0 1  1 1 0 0  1 1 0 0
           0 0 1 1  0 1 1 1  1 1 0 1  1 1 0 0  0 1 1 0
           0 0 0 1  1 1 1 1  1 1 1 1  1 1 0 0  0 1 1 0
           0 0 0 0  0 1 1 1  1 0 0 0  1 0 0 0  0 0 0 0 ])
  cell_pixel_counts::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16
           16 16 16 16 16 ])
    corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1
           1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1

           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2
           2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2  2 2 2 2

           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3
           3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3  3 3 3 3

           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4
           4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4  4 4 4 4

           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5
           5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5  5 5 5 5 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5

           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5
           1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4  5 5 5 5 ])
  binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false false  true false false
          false false false  true  true
          false  true true  true  false
           true false false  true false
          false  true false false false ])
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 1  16  6  4
           0 0  3  16 16
           3 16 16 16 3
           16 1  7  16 1
           1 16  1 5  6 ])
  expected_intermediate_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 0  15  0  0
           0 0  0  16 16
           0 15 16 16 0
           16 0  0  16 0
           0 16 0 0  0 ])
  expected_lake_pixel_counts_field_after_cycle::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 5  16  4  4
           0 3  2  16 16
           3 16 16 16 0
           16 1  7  16 1
           0 16  5 3  4  ])
  expected_intermediate_lake_pixel_counts_field_two::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 0  15  0  0
           0 0  0  16 16
           0 15 16 16 0
           16 0  0  16 0
           0 16 0 0  0 ])
  expected_intermediate_lake_pixel_counts_field_three::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 0  16  0  0
           0 0  0  16 16
           0 14 16 16 0
           16 0  0  16 0
           0 16 0 0  0 ])
  expected_lake_pixel_counts_field_after_second_cycle::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 5  16  4  4
           0 3  3  16 16
           3 16 16 16 0
           16 1  7  16 1
           2 16  3 2  4  ])
  expected_lake_pixel_counts_field_after_third_cycle::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 1 5  16  5  4
           0 3  3  16 16
           3 16 16 16 0
           16 2  7  16 1
           0 16  4 0  4  ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0 1.0 1.0 0.0 0.0
             0.0 0.0 0.0 1.0 1.0
             0.25 0.9375 1.0 1.0 0.0
             1.0 0.0 0.4375 1.0 0.0
             0.0 1.0 1.0 0.0 0.0 ])
  non_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false  true false false false
          false false false false false
          false false false false false
          false false false false false
          false false  true false false ])
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} =
      findall(x -> x == lake_number,potential_lake_pixel_mask.data)
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(pixel,
                                                  grid_specific_lake_model_parameters)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      CartesianIndex[],
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  prognostics::LakeFractionCalculationPrognostics =
    setup_lake_for_fraction_calculation(lakes,
                                        cell_pixel_counts,
                                        non_lake_mask,
                                        binary_lake_mask,
                                        potential_lake_pixel_mask,
                                        grid_specific_lake_model_parameters,
                                        lake_grid,
                                        surface_grid)
  lake_pixel_counts_field::Field{Int64} = Field{Int64}(surface_grid,0)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    for coords in lake_pixel_coords_list
      add_pixel_by_coords(coords,
                          lake_pixel_counts_field,prognostics)
    end
  end
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  #@test lake_fractions_field == expected_lake_fractions_field
  remove_pixel_by_coords(CartesianIndex(2,4),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(2,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,4),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,2),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(16,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(14,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(13,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(13,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(14,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(16,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(20,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(8,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(7,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(6,12),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_intermediate_lake_pixel_counts_field
  add_pixel_by_coords(CartesianIndex(2,4),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(2,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,4),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,2),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(16,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(14,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(13,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(13,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(14,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(16,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(20,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(8,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(7,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(6,12),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field_after_cycle
  remove_pixel_by_coords(CartesianIndex(2,4),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(2,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,4),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,2),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(16,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(14,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(13,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(13,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(14,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(16,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(20,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(8,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(7,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(6,12),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_intermediate_lake_pixel_counts_field_two
  add_pixel_by_coords(CartesianIndex(2,4),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(2,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,4),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,2),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(16,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(14,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(13,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(13,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(14,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(16,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(20,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(8,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(7,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(6,12),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field_after_second_cycle
  remove_pixel_by_coords(CartesianIndex(2,4),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(2,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(4,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,4),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,2),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(16,3),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(14,19),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(13,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(13,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(14,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,11),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(15,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(16,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(17,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,10),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(20,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(18,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(10,8),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,7),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,6),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(9,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(8,12),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(7,13),lake_pixel_counts_field,prognostics)
  remove_pixel_by_coords(CartesianIndex(6,12),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_intermediate_lake_pixel_counts_field_three
  add_pixel_by_coords(CartesianIndex(2,4),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(2,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(5,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(4,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(3,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,4),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,2),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(11,2),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(16,3),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,17),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,18),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(14,19),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(13,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(13,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(14,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,11),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(15,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(16,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(17,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,10),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(20,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(19,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(18,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(11,9),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(10,8),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,7),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,6),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(9,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(8,12),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(7,13),lake_pixel_counts_field,prognostics)
  add_pixel_by_coords(CartesianIndex(6,12),lake_pixel_counts_field,prognostics)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field_after_third_cycle
end

end
