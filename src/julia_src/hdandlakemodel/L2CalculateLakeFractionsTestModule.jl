module L2CalculateLakeFractionsTestModule

using GridModule: Grid, LatLonGrid, find_coarse_cell_containing_fine_cell
using Test: @test, @testset
using L2CalculateLakeFractionsModule: LakeInput, LakeProperties,Pixel
using L2CalculateLakeFractionsModule: calculate_lake_fractions
using L2CalculateLakeFractionsModule: setup_lake_for_fraction_calculation
using L2CalculateLakeFractionsModule: add_pixel,remove_pixel
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
  lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == 1,
                                                           lake_pixel_mask.data)
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
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
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
  for lake_number::Int64=1:7
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
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
  end
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
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
  expected_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 16  0  0 9
           16 14 0 13 0
           0 16  1  0 12
           0 15  1 12 14
           1 0  3  1 0 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0    1.0     0.0    0.0    0.5625
             1.0    0.875   0.0    0.8125 0.0
             0.0    1.0     0.0625 0.0    0.75
             0.0    0.9375  0.0625 0.75   0.875
             0.0625 0.0     0.1875 0.0625 0.0 ])
  expected_binary_lake_mask::Field{Bool} = LatLonField{Bool}(surface_grid,
    Bool[ false  true false false  true
           true  true false  true false
          false  true false false  true
          false  true false  true  true
          false false false false false ])
  for lake_number::Int64=1:9
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
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
  end
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
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
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
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
  end
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
                             lake_grid,
                             surface_grid)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  @test lake_fractions_field == expected_lake_fractions_field
  @test binary_lake_mask == expected_binary_lake_mask
end

@testset "Lake Fraction Calculation Test 5" begin
  #Multiple lakes
  lake_grid::Grid = LatLonGrid(20,20,true)
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
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 1 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

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
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
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
  end
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
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
  lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == 1,
                                                           lake_pixel_mask.data)
  potential_lake_pixel_coords_list::Vector{CartesianIndex} =
    findall(x -> x == 1,potential_lake_pixel_mask.data)
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
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
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
  for lake_number::Int64=1:7
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                               potential_lake_pixel_mask.data)
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
  end
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
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
  for lake_number::Int64=1:9
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                               potential_lake_pixel_mask.data)
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
  end
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
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
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                               potential_lake_pixel_mask.data)
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
  end
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
                             lake_grid,
                             surface_grid)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  @test lake_fractions_field == expected_lake_fractions_field
  @test binary_lake_mask == expected_binary_lake_mask
end

@testset "Lake Fraction Calculation Test 10" begin
  #Multiple partially filled lakes with uneven cell sizes
  lake_grid::Grid = LatLonGrid(20,20,true)
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
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 1 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

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
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

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
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                               potential_lake_pixel_mask.data)
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
  end
  lake_pixel_counts_field::Field{Int64},
    lake_fractions_field::Field{Float64},
    binary_lake_mask::Field{Bool} =
    calculate_lake_fractions(lakes,
                             cell_pixel_counts,
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
  lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == 1,
                                                           lake_pixel_mask.data)
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
                    CartesianIndex[],
                    potential_lake_pixel_coords_list,
                    cell_coords_list)
  push!(lakes,input)
  lake_properties::Vector{LakeProperties},
    pixel_numbers::Field{Int64},
    pixels::Vector{Pixel},
    lake_pixel_counts_field::Field{Int64} =
      setup_lake_for_fraction_calculation(lakes,
                                          cell_pixel_counts,
                                          binary_lake_mask,
                                          lake_grid,
                                          surface_grid)
  for coords in lake_pixel_coords_list
    lake::LakeProperties = lake_properties[1]
    pixel_number = pixel_numbers(coords)
    pixel::Pixel = pixels[pixel_number]
    add_pixel(lake,pixel,lake_pixel_counts_field)
  end
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  #@test lake_fractions_field == expected_lake_fractions_field
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,19))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  pixel_number = pixel_numbers(CartesianIndex(11,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  pixel_number = pixel_numbers(CartesianIndex(15,20))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  @test lake_pixel_counts_field == expected_immediate_lake_pixel_counts_field
  pixel_number = pixel_numbers(CartesianIndex(15,19))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  pixel_number = pixel_numbers(CartesianIndex(11,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  pixel_number = pixel_numbers(CartesianIndex(15,20))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
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
  for lake_number::Int64=1:7
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
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
                      CartesianIndex[],
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  lake_properties::Vector{LakeProperties},
    pixel_numbers::Field{Int64},
    pixels::Vector{Pixel},
    lake_pixel_counts_field::Field{Int64}  =
      setup_lake_for_fraction_calculation(lakes,
                                          cell_pixel_counts,
                                          binary_lake_mask,
                                          lake_grid,
                                          surface_grid)
  for lake_number::Int64=1:7
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    lake::LakeProperties = lake_properties[lake_number]
    for coords in lake_pixel_coords_list
      pixel_number = pixel_numbers(coords)
      pixel::Pixel = pixels[pixel_number]
      add_pixel(lake,pixel,lake_pixel_counts_field)
    end
  end
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  #@test lake_fractions_field == expected_lake_fractions_field
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(2,2))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(3,3))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(10,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[5]
  pixel_number = pixel_numbers(CartesianIndex(16,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[5]
  pixel_number = pixel_numbers(CartesianIndex(19,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(8,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(5,19))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(6,17))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(3,17))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[7]
  pixel_number = pixel_numbers(CartesianIndex(2,19))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(3,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[3]
  pixel_number = pixel_numbers(CartesianIndex(3,12))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[3]
  pixel_number = pixel_numbers(CartesianIndex(3,11))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(11,9))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(12,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(12,7))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(10,7))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(11,10))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(11,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(12,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(12,19))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(11,19))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,16))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,16))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,15))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,15))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,17))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,17))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,19))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(20,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,17))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,16))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,15))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,15))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,15))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(13,20))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,20))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,20))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(6,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(7,17))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(7,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(5,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(5,17))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(5,16))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  @test lake_pixel_counts_field == expected_immediate_lake_pixel_counts_field
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(2,2))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(3,3))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(10,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[5]
  pixel_number = pixel_numbers(CartesianIndex(16,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[5]
  pixel_number = pixel_numbers(CartesianIndex(19,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(8,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(5,19))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(6,17))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(3,17))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[7]
  pixel_number = pixel_numbers(CartesianIndex(2,19))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(3,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[3]
  pixel_number = pixel_numbers(CartesianIndex(3,12))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[3]
  pixel_number = pixel_numbers(CartesianIndex(3,11))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(11,9))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(12,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(12,7))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(10,7))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(11,10))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(11,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(12,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(12,19))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(11,19))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,16))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,16))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,15))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,15))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,17))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,17))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,19))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(20,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,17))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,16))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,15))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,15))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,15))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(13,20))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,20))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,20))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(6,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(7,17))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(7,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(5,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(5,17))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(5,16))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
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
  expected_intermediate_lake_pixel_counts_field::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 0 12  0  0 5
           11 1 0   9 0
           0 16  0  0 8
           0 10  2 9 5
           1 0  0  0 0 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0    1.0     0.0    0.0    0.5625
             1.0    0.875   0.0    0.8125 0.0
             0.0    1.0     0.0625 0.0    0.75
             0.0    0.9375  0.0625 0.75   0.875
             0.0625 0.0     0.1875 0.0625 0.0 ])
  for lake_number::Int64=1:9
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
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
                      CartesianIndex[],
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  lake_properties::Vector{LakeProperties},
    pixel_numbers::Field{Int64},
    pixels::Vector{Pixel},
    lake_pixel_counts_field::Field{Int64}  =
      setup_lake_for_fraction_calculation(lakes,
                                          cell_pixel_counts,
                                          binary_lake_mask,
                                          lake_grid,
                                          surface_grid)
  for lake_number::Int64=1:9
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    lake::LakeProperties = lake_properties[lake_number]
    for coords in lake_pixel_coords_list
      pixel_number = pixel_numbers(coords)
      pixel::Pixel = pixels[pixel_number]
      add_pixel(lake,pixel,lake_pixel_counts_field)
    end
  end
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  #@test lake_fractions_field == expected_lake_fractions_field
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(2,3))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,3))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(4,3))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,9))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(4,9))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(4,10))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,10))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,11))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(2,11))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(5,5))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(5,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(5,7))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(5,8))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(6,8))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(6,7))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(6,5))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(7,5))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(7,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(7,7))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(7,8))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(8,8))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(8,7))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(8,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(8,5))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(17,5))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(17,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(17,7))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(17,11))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(17,12))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(18,12))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(18,13))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(17,13))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(17,14))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[3]
  pixel_number = pixel_numbers(CartesianIndex(14,14))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[3]
  pixel_number = pixel_numbers(CartesianIndex(14,15))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[3]
  pixel_number = pixel_numbers(CartesianIndex(13,15))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[3]
  pixel_number = pixel_numbers(CartesianIndex(13,14))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[5]
  pixel_number = pixel_numbers(CartesianIndex(17,17))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[5]
  pixel_number = pixel_numbers(CartesianIndex(17,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[5]
  pixel_number = pixel_numbers(CartesianIndex(18,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[9]
  pixel_number = pixel_numbers(CartesianIndex(8,19))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[9]
  pixel_number = pixel_numbers(CartesianIndex(8,20))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[9]
  pixel_number = pixel_numbers(CartesianIndex(9,20))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[9]
  pixel_number = pixel_numbers(CartesianIndex(9,19))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[9]
  pixel_number = pixel_numbers(CartesianIndex(10,16))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[9]
  pixel_number = pixel_numbers(CartesianIndex(10,17))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[8]
  pixel_number = pixel_numbers(CartesianIndex(7,17))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[8]
  pixel_number = pixel_numbers(CartesianIndex(8,17))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[8]
  pixel_number = pixel_numbers(CartesianIndex(8,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[8]
  pixel_number = pixel_numbers(CartesianIndex(7,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[7]
  pixel_number = pixel_numbers(CartesianIndex(6,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[7]
  pixel_number = pixel_numbers(CartesianIndex(5,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[7]
  pixel_number = pixel_numbers(CartesianIndex(5,19))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[7]
  pixel_number = pixel_numbers(CartesianIndex(6,19))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(10,11))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  @test lake_pixel_counts_field == expected_intermediate_lake_pixel_counts_field
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(2,3))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,3))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(4,3))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,9))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(4,9))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(4,10))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,10))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,11))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(2,11))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(5,5))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(5,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(5,7))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(5,8))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(6,8))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(6,7))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(6,5))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(7,5))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(7,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(7,7))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(7,8))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(8,8))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(8,7))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(8,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(8,5))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(17,5))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(17,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(17,7))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(17,11))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(17,12))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(18,12))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(18,13))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(17,13))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[4]
  pixel_number = pixel_numbers(CartesianIndex(17,14))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[3]
  pixel_number = pixel_numbers(CartesianIndex(14,14))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[3]
  pixel_number = pixel_numbers(CartesianIndex(14,15))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[3]
  pixel_number = pixel_numbers(CartesianIndex(13,15))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[3]
  pixel_number = pixel_numbers(CartesianIndex(13,14))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[5]
  pixel_number = pixel_numbers(CartesianIndex(17,17))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[5]
  pixel_number = pixel_numbers(CartesianIndex(17,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[5]
  pixel_number = pixel_numbers(CartesianIndex(18,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[9]
  pixel_number = pixel_numbers(CartesianIndex(8,19))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[9]
  pixel_number = pixel_numbers(CartesianIndex(8,20))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[9]
  pixel_number = pixel_numbers(CartesianIndex(9,20))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[9]
  pixel_number = pixel_numbers(CartesianIndex(9,19))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[9]
  pixel_number = pixel_numbers(CartesianIndex(10,16))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[9]
  pixel_number = pixel_numbers(CartesianIndex(10,17))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[8]
  pixel_number = pixel_numbers(CartesianIndex(7,17))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[8]
  pixel_number = pixel_numbers(CartesianIndex(8,17))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[8]
  pixel_number = pixel_numbers(CartesianIndex(8,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[8]
  pixel_number = pixel_numbers(CartesianIndex(7,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[7]
  pixel_number = pixel_numbers(CartesianIndex(6,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[7]
  pixel_number = pixel_numbers(CartesianIndex(5,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[7]
  pixel_number = pixel_numbers(CartesianIndex(5,19))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[7]
  pixel_number = pixel_numbers(CartesianIndex(6,19))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[6]
  pixel_number = pixel_numbers(CartesianIndex(10,11))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
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
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
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
                      CartesianIndex[],
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  lake_properties::Vector{LakeProperties},
    pixel_numbers::Field{Int64},
    pixels::Vector{Pixel},
    lake_pixel_counts_field::Field{Int64}  =
      setup_lake_for_fraction_calculation(lakes,
                                          cell_pixel_counts,
                                          binary_lake_mask,
                                          lake_grid,
                                          surface_grid)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    lake::LakeProperties = lake_properties[lake_number]
    for coords in lake_pixel_coords_list
      pixel_number = pixel_numbers(coords)
      pixel::Pixel = pixels[pixel_number]
      add_pixel(lake,pixel,lake_pixel_counts_field)
    end
  end
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  #@test lake_fractions_field == expected_lake_fractions_field
end

@testset "Lake Fraction Calculation Test 15" begin
  #Multiple lakes
  lake_grid::Grid = LatLonGrid(20,20,true)
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
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 1 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

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
           1 0 0 0 0
           6 0 0 0 0 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0 0.0 0.0 0.0 0.625
             0.0 0.0 0.0 0.0 0.0
             0.0 0.0 0.0 0.0 0.0
             1.0 0.0 0.0 0.0 0.0
             0.375 0.0 0.0 0.0 0.0 ])
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
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
                      CartesianIndex[],
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  lake_properties::Vector{LakeProperties},
    pixel_numbers::Field{Int64},
    pixels::Vector{Pixel},
    lake_pixel_counts_field::Field{Int64}  =
      setup_lake_for_fraction_calculation(lakes,
                                          cell_pixel_counts,
                                          binary_lake_mask,
                                          lake_grid,
                                          surface_grid)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    lake::LakeProperties = lake_properties[lake_number]
    for coords in lake_pixel_coords_list
      pixel_number = pixel_numbers(coords)
      pixel::Pixel = pixels[pixel_number]
      add_pixel(lake,pixel,lake_pixel_counts_field)
    end
  end
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  #@test lake_fractions_field == expected_lake_fractions_field
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,4))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(16,4))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(16,3))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(4,20))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(4,19))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(4,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(3,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(3,19))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(3,20))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(4,17))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  @test lake_pixel_counts_field == expected_intermediate_lake_pixel_counts_field
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,4))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(16,4))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(16,3))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(4,20))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(4,19))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(4,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(3,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(3,19))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(3,20))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(4,17))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
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
  lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == 1,
                                                           lake_pixel_mask.data)
  potential_lake_pixel_coords_list::Vector{CartesianIndex} =
    findall(x -> x == 1,potential_lake_pixel_mask.data)
  cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
  for pixel in potential_lake_pixel_coords_list
    cell_coords::CartesianIndex =
      find_coarse_cell_containing_fine_cell(lake_grid,surface_grid,
                                            pixel)
    set!(cell_mask,cell_coords,true)
  end
  cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
  input = LakeInput(lake_number,
                    CartesianIndex[],
                    potential_lake_pixel_coords_list,
                    cell_coords_list)
  push!(lakes,input)
  lake_properties::Vector{LakeProperties},
    pixel_numbers::Field{Int64},
    pixels::Vector{Pixel},
    lake_pixel_counts_field::Field{Int64} =
      setup_lake_for_fraction_calculation(lakes,
                                          cell_pixel_counts,
                                          binary_lake_mask,
                                          lake_grid,
                                          surface_grid)
  for coords in lake_pixel_coords_list
    lake::LakeProperties = lake_properties[1]
    pixel_number = pixel_numbers(coords)
    pixel::Pixel = pixels[pixel_number]
    add_pixel(lake,pixel,lake_pixel_counts_field)
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
  for lake_number::Int64=1:7
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} =
      findall(x -> x == lake_number,potential_lake_pixel_mask.data)
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        find_coarse_cell_containing_fine_cell(lake_grid,surface_grid,
                                              pixel)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      CartesianIndex[],
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  lake_properties::Vector{LakeProperties},
    pixel_numbers::Field{Int64},
    pixels::Vector{Pixel},
    lake_pixel_counts_field::Field{Int64}  =
      setup_lake_for_fraction_calculation(lakes,
                                          cell_pixel_counts,
                                          binary_lake_mask,
                                          lake_grid,
                                          surface_grid)
  for lake_number::Int64=1:7
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    lake::LakeProperties = lake_properties[lake_number]
    for coords in lake_pixel_coords_list
      pixel_number = pixel_numbers(coords)
      pixel::Pixel = pixels[pixel_number]
      add_pixel(lake,pixel,lake_pixel_counts_field)
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
  for lake_number::Int64=1:9
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} =
      findall(x -> x == lake_number,potential_lake_pixel_mask.data)
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        find_coarse_cell_containing_fine_cell(lake_grid,surface_grid,
                                              pixel)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      CartesianIndex[],
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  lake_properties::Vector{LakeProperties},
    pixel_numbers::Field{Int64},
    pixels::Vector{Pixel},
    lake_pixel_counts_field::Field{Int64}  =
      setup_lake_for_fraction_calculation(lakes,
                                          cell_pixel_counts,
                                          binary_lake_mask,
                                          lake_grid,
                                          surface_grid)
  for lake_number::Int64=1:9
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    lake::LakeProperties = lake_properties[lake_number]
    for coords in lake_pixel_coords_list
      pixel_number = pixel_numbers(coords)
      pixel::Pixel = pixels[pixel_number]
      add_pixel(lake,pixel,lake_pixel_counts_field)
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
    Int64[ 0 11  12  0  0
           0 0  0  14 14
           0 12 11 16 0
           13 0  0  15 0
           0 10 11 0  3 ])
  expected_lake_pixel_counts_field_after_cycle::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 1 16  16  0  0
           0 1  2  16 16
           1 16 16 16 0
           16 0  1  16 0
           0 16 16 0  4 ])
  expected_intermediate_lake_pixel_counts_field_two::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 1 11  12  0  0
           0 1  0  14 14
           0 10 11 16 0
           13 0  0  14 0
           0 10 12 0  3 ])
  expected_intermediate_lake_pixel_counts_field_three::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 1 11  12  0  0
           0 1  0  14 14
           0 10 12 16 0
           13 0  0  14 0
           0 10 10 1  3 ])
  expected_lake_pixel_counts_field_after_second_cycle::Field{Int64} = LatLonField{Int64}(surface_grid,
    Int64[ 1 16  16  0  0
           0 2  2  16 16
           1 16 16 16 0
           16 0  0  16 0
           0 16 16 1  3 ])
  expected_lake_fractions_field::Field{Float64} = LatLonField{Float64}(surface_grid,
    Float64[ 0.0 1.0 1.0 0.0 0.0
             0.0 0.0 0.0 1.0 1.0
             0.25 0.9375 1.0 1.0 0.0
             1.0 0.0 0.4375 1.0 0.0
             0.0 1.0 1.0 0.0 0.0 ])
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    potential_lake_pixel_coords_list::Vector{CartesianIndex} =
      findall(x -> x == lake_number,potential_lake_pixel_mask.data)
    cell_mask::Field{Bool} = LatLonField{Bool}(lake_grid,false)
    for pixel in potential_lake_pixel_coords_list
      cell_coords::CartesianIndex =
        find_coarse_cell_containing_fine_cell(lake_grid,surface_grid,
                                              pixel)
      set!(cell_mask,cell_coords,true)
    end
    cell_coords_list::Vector{CartesianIndex} = findall(cell_mask.data)
    input = LakeInput(lake_number,
                      CartesianIndex[],
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  lake_properties::Vector{LakeProperties},
    pixel_numbers::Field{Int64},
    pixels::Vector{Pixel},
    lake_pixel_counts_field::Field{Int64}  =
      setup_lake_for_fraction_calculation(lakes,
                                          cell_pixel_counts,
                                          binary_lake_mask,
                                          lake_grid,
                                          surface_grid)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    lake::LakeProperties = lake_properties[lake_number]
    for coords in lake_pixel_coords_list
      pixel_number = pixel_numbers(coords)
      pixel::Pixel = pixels[pixel_number]
      add_pixel(lake,pixel,lake_pixel_counts_field)
    end
  end
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  #@test lake_fractions_field == expected_lake_fractions_field
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(2,4))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(2,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(4,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(5,7))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(4,8))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,11))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,12))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,13))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,19))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(10,4))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(10,3))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(10,2))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(11,2))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,3))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(16,3))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,17))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,19))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(13,13))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(13,11))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,11))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,11))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,10))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(16,10))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,9))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,8))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,8))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,9))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,10))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,8))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(20,7))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(11,9))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(10,8))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(9,7))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(9,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(9,12))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(8,12))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(7,13))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(6,12))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  @test lake_pixel_counts_field == expected_intermediate_lake_pixel_counts_field
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(2,4))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(2,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(4,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(5,7))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(4,8))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,11))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,12))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,13))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,19))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(10,4))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(10,3))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(10,2))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(11,2))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,3))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(16,3))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,17))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,19))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(13,13))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(13,11))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,11))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,11))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,10))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(16,10))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,9))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,8))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,8))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,9))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,10))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,8))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(20,7))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(11,9))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(10,8))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(9,7))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(9,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(9,12))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(8,12))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(7,13))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(6,12))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field_after_cycle
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(2,4))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(2,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(4,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(5,7))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(4,8))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,11))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,12))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,13))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,19))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(10,4))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(10,3))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(10,2))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(11,2))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,3))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(16,3))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,17))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,19))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(13,13))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(13,11))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,11))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,11))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,10))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(16,10))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,9))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,8))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,8))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,9))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,10))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,8))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(20,7))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(11,9))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(10,8))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(9,7))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(9,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(9,12))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(8,12))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(7,13))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(6,12))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  @test lake_pixel_counts_field == expected_intermediate_lake_pixel_counts_field_two
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(2,4))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(2,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(4,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(5,7))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(4,8))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,11))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,12))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,13))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,19))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(10,4))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(10,3))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(10,2))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(11,2))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,3))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(16,3))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,17))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,19))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(13,13))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(13,11))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,11))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,11))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,10))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(16,10))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,9))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,8))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,8))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,9))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,10))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,8))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(20,7))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(11,9))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(10,8))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(9,7))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(9,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(9,12))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(8,12))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(7,13))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(6,12))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field_after_second_cycle
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(2,4))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(2,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(4,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(5,7))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(4,8))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,11))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,12))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,13))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,19))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(10,4))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(10,3))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(10,2))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(11,2))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,3))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(16,3))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,17))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,18))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,19))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(13,13))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(13,11))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,11))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,11))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,10))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(16,10))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,9))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,8))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,8))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,9))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,10))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,8))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(20,7))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(11,9))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(10,8))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(9,7))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(9,6))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(9,12))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(8,12))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(7,13))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(6,12))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  @test lake_pixel_counts_field == expected_intermediate_lake_pixel_counts_field_three
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(2,4))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(2,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(4,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(5,7))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(4,8))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,11))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,12))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,13))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(3,19))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(10,4))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(10,3))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(10,2))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(11,2))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,3))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(16,3))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,17))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,18))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,19))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(13,13))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(13,11))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(14,11))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,11))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(15,10))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(16,10))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,9))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,8))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,8))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,9))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,10))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,8))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(20,7))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(19,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(11,9))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(10,8))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(9,7))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(9,6))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(9,12))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(8,12))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(7,13))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(6,12))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field_after_second_cycle
end

@testset "Lake Fraction Calculation Test 20" begin
  #Multiple partially filled lakes with uneven cell sizes
  lake_grid::Grid = LatLonGrid(20,20,true)
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
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 1 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

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
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 1  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0
           0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0

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
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
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
                      CartesianIndex[],
                      potential_lake_pixel_coords_list,
                      cell_coords_list)
    push!(lakes,input)
  end
  lake_properties::Vector{LakeProperties},
    pixel_numbers::Field{Int64},
    pixels::Vector{Pixel},
    lake_pixel_counts_field::Field{Int64}  =
      setup_lake_for_fraction_calculation(lakes,
                                          cell_pixel_counts,
                                          binary_lake_mask,
                                          lake_grid,
                                          surface_grid)
  for lake_number::Int64=1:2
    lake_pixel_coords_list::Vector{CartesianIndex} = findall(x -> x == lake_number,
                                                             lake_pixel_mask.data)
    lake::LakeProperties = lake_properties[lake_number]
    for coords in lake_pixel_coords_list
      pixel_number = pixel_numbers(coords)
      pixel::Pixel = pixels[pixel_number]
      add_pixel(lake,pixel,lake_pixel_counts_field)
    end
  end
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
  #@test lake_fractions_field == expected_lake_fractions_field
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,1))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,2))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,3))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,3))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,2))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,1))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(6,19))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(6,20))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(4,20))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(3,20))
  pixel = pixels[pixel_number]
  remove_pixel(lake,pixel,lake_pixel_counts_field)
  @test lake_pixel_counts_field == expected_intermediate_lake_pixel_counts_field
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,1))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,2))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(17,3))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,3))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,2))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[1]
  pixel_number = pixel_numbers(CartesianIndex(18,1))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(6,19))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(6,20))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(4,20))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  lake = lake_properties[2]
  pixel_number = pixel_numbers(CartesianIndex(3,20))
  pixel = pixels[pixel_number]
  add_pixel(lake,pixel,lake_pixel_counts_field)
  @test lake_pixel_counts_field == expected_lake_pixel_counts_field
end

end
