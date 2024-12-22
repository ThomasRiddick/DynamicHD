using Profile
using Test: @test, @testset
using L2HDDriverModule: drive_hd_and_lake_model
using HDModule: RiverParameters,RiverPrognosticFields
using GridModule: LatLonGrid
using FieldModule: Field,LatLonField,LatLonDirectionIndicators,set!,repeat
using CoordsModule: LatLonCoords
using L2LakeModule: Lake, OverflowingLake, FillingLake, SubsumedLake
using L2LakeModule: get_lake_volume
using L2LakeModelDefsModule: LatLonLakeModelParameters,LakeModelParameters
using L2LakeModelDefsModule: LakeModelSettings,LakeModelPrognostics
using L2LakeModelDefsModule: LakeModelDiagnostics
using L2LakeModelModule: calculate_lake_fraction_on_surface_grid
using L2LakeModelModule: calculate_diagnostic_lake_volumes_field
using L2LakeModelModule: calculate_effective_lake_height_on_surface_grid

function print_results(river_fields::RiverPrognosticFields,
                       lake_model_parameters::LakeModelParameters,
                       lake_model_prognostics::LakeModelPrognostics,
                       lake_grid::LatLonGrid)
  lake_fractions::Field{Float64} =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes::Array{Float64} = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes::Field{Float64} =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  println(river_fields.river_inflow)
  println(river_fields.water_to_ocean)
  println(lake_model_prognostics.water_to_hd)
  println(lake_model_prognostics.lake_numbers)
  println(lake_types)
  println(lake_volumes)
  println(diagnostic_lake_volumes)
  println(lake_fractions)
  println(lake_model_prognostics.lake_cell_count)
  println(lake_model_parameters.number_fine_grid_cells)
  #println(lake_model_parameters.true_lake_depths)
end

@testset "Lake model tests 1" begin
  #Single lake - fill, overflow, empty
  hd_grid = LatLonGrid(3,3,true)
  surface_model_grid = LatLonGrid(3,3,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
                                                Int64[6  6  2
                                                      6 -2  2
                                                      6  6  0] ))
  river_reservoir_nums = LatLonField{Int64}(hd_grid,5)
  overland_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  base_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  river_retention_coefficients = LatLonField{Float64}(hd_grid,0.7)
  overland_retention_coefficients = LatLonField{Float64}(hd_grid,0.5)
  base_retention_coefficients = LatLonField{Float64}(hd_grid,0.1)
  landsea_mask = LatLonField{Bool}(hd_grid,fill(false,3,3))
  set!(river_reservoir_nums,LatLonCoords(3,3),0)
  set!(overland_reservoir_nums,LatLonCoords(3,3),0)
  set!(base_reservoir_nums,LatLonCoords(3,3),0)
  set!(landsea_mask,LatLonCoords(3,3),true)
  river_parameters = RiverParameters(flow_directions,
                                     river_reservoir_nums,
                                     overland_reservoir_nums,
                                     base_reservoir_nums,
                                     river_retention_coefficients,
                                     overland_retention_coefficients,
                                     base_retention_coefficients,
                                     landsea_mask,
                                     hd_grid,1.0,1.0)
  lake_grid = LatLonGrid(6,6,true)
  cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
    Float64[ 2.5 3.0 2.5
             3.0 4.0 3.0
             2.5 3.0 2.5 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1 1 1
           1 1 1 1 1 1
           2 2 2 2 2 2
           2 2 2 2 2 2
           3 3 3 3 3 3
           3 3 3 3 3 3  ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 2 2 3 3
           1 1 2 2 3 3
           1 1 2 2 3 3
           1 1 2 2 3 3
           1 1 2 2 3 3
           1 1 2 2 3 3 ])
  is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
    Bool[false false false false false false
         false false  true  true  true false
         false  true  true  true  true false
         false  true  true  true  true false
         false false false false false false
         false false false false false false])
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_model_parameters::LakeModelParameters =
    LakeModelParameters(grid_specific_lake_model_parameters,
                        lake_grid,
                        hd_grid,
                        surface_model_grid,
                        cell_areas_on_surface_model_grid,
                        LakeModelSettings(),
                        1,
                        is_lake)
  lake_parameters_as_array::Vector{Float64} = Float64[1.0, 66.0, 1.0, -1.0, 0.0, 4.0, 3.0, 11.0, 4.0, 3.0, 1.0, 0.0, 5.0, 4.0, 4.0, 1.0, 0.0, 5.0, 3.0, 4.0, 1.0, 0.0, 5.0, 3.0, 3.0, 1.0, 0.0, 5.0, 2.0, 5.0, 1.0, 5.0, 6.0, 4.0, 5.0, 1.0, 5.0, 6.0, 3.0, 5.0, 1.0, 12.0, 7.0, 3.0, 2.0, 1.0, 12.0, 7.0, 4.0, 2.0, 1.0, 21.0, 8.0, 2.0, 3.0, 1.0, 21.0, 8.0, 2.0, 4.0, 1.0, 43.0, 10.0, 1.0, -1.0, 3.0, 3.0, 0.0]
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,Float64[ 1.0 1.0 1.0
                                                                                 1.0 1.0 1.0
                                                                                 1.0 1.0 0.0 ])
  drainages::Array{Field{Float64},1} = repeat(drainage,60,false)
  runoffs::Array{Field{Float64},1} = drainages
  evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,30,false)
  additional_evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,5.0)
  additional_evaporations::Array{Field{Float64},1} = repeat(additional_evaporation,30,false)
  append!(evaporations,additional_evaporations)
  expected_river_inflow::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                               Float64[ 0.0 0.0  2.0
                                                                        4.0 0.0 14.0
                                                                        0.0 0.0  0.0 ])
  expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                                 Float64[ 0.0 0.0  0.0
                                                                          0.0 0.0  0.0
                                                                          0.0 0.0 16.0 ])
  expected_water_to_hd::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                              Float64[ 0.0 0.0 10.0
                                                                       0.0 0.0  0.0
                                                                       0.0 0.0  0.0 ])
  # expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
  #     Int64[    0    0    0    1    0    0    0    0    0
  #               0    0    0    1    0    0    0    0    0
  #               0    0    1    1    0    0    0    0    0
  #               0    0    1    1    0    0    0    0    0
  #               0    0    1    1    0    0    0    0    0
  #                ])
  # expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
  #     Int64[    0    0    0    2    0    0    0    0    0
  #               0    0    0    2    0    0    0    0    0
  #               0    0    2    2    0    0    0    0    0
  #               0    0    2    2    0    0    0    0    0
  #               0    0    2    2    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0 ])
  # expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[    0    0    0    80.0    0    0    0    0    0
  #                   0    0    0    80.0   0    0    0    0    0
  #                   0    0    80.0 80.0    0    0    0    0    0
  #                   0    0    80.0 80.0    0    0    0    0    0
  #                   0    0    80.0 80.0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0 ])
  # expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
  #       Float64[ 0.0 1.0/3.0 0.0
  #                0.2     0.2 0.0
  #                0.0     0.0 0.0 ])
  # expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
  #       Int64[ 0 2 0
  #              3 3 0
  #              0 0 0 ])
  # expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
  #       Int64[ 6 6 6
  #             15 15 15
  #              6 6 6 ])
  # expected_lake_volumes::Array{Float64} = Float64[80.0]
  # expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[    0    0    0   7.0    0    0    0    0    0
  #                   0    0    0   7.0   0    0    0    0    0
  #                   0    0    9.0 7.0    0    0    0    0    0
  #                   0    0    8.0 8.0    0    0    0    0    0
  #                   0    0    8.0 8.0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0 ])
  import Profile
  @time river_fields::RiverPrognosticFields,
    lake_model_prognostics::LakeModelPrognostics,_ =
    drive_hd_and_lake_model(river_parameters,lake_model_parameters,
                            lake_parameters_as_array,
                            drainages,runoffs,evaporations,
                            1100,print_timestep_results=false,
                            write_output=false,return_output=true)
  # lake_fractions =
  #   calculate_lake_fraction_on_surface_grid(lake_model_parameters,
  #                                           lake_model_prognostics)
  # lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
  # for i = 1:9
  #   for j = 1:9
  #     coords::LatLonCoords = LatLonCoords(i,j)
  #     lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
  #     if lake_number <= 0 continue end
  #     lake::Lake = lake_model_prognostics.lakes[lake_number]
  #     if isa(lake,FillingLake)
  #       set!(lake_types,coords,1)
  #     elseif isa(lake,OverflowingLake)
  #       set!(lake_types,coords,2)
  #     elseif isa(lake,SubsumedLake)
  #       set!(lake_types,coords,3)
  #     else
  #       set!(lake_types,coords,4)
  #     end
  #   end
  # end
  # lake_volumes::Array{Float64} = Float64[]
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
  # diagnostic_lake_volumes::Field{Float64} =
  #   calculate_diagnostic_lake_volumes_field(lake_model_parameters,
  #                                           lake_model_prognostics)
  # @test expected_river_inflow == river_fields.river_inflow
  # @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  # @test expected_water_to_hd == lake_model_prognostics.water_to_hd
  # @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  # @test expected_lake_types == lake_types
  # @test expected_lake_volumes == lake_volumes
  # @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  # @test expected_lake_fractions == lake_fractions
  # @test expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  # @test expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  # @test expected_true_lake_depths == lake_model_parameters.true_lake_depths
end

@testset "Lake model tests 2" begin
  #Lake with two sub-basins, fill, overflow, empty
  hd_grid = LatLonGrid(3,3,true)
  surface_model_grid = LatLonGrid(3,3,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
                                                Int64[ 2 2  2
                                                      -2 4 -2
                                                       6 6  0 ]))
  river_reservoir_nums = LatLonField{Int64}(hd_grid,5)
  overland_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  base_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  river_retention_coefficients = LatLonField{Float64}(hd_grid,0.7)
  overland_retention_coefficients = LatLonField{Float64}(hd_grid,0.5)
  base_retention_coefficients = LatLonField{Float64}(hd_grid,0.1)
  landsea_mask = LatLonField{Bool}(hd_grid,fill(false,3,3))
  set!(river_reservoir_nums,LatLonCoords(3,3),0)
  set!(overland_reservoir_nums,LatLonCoords(3,3),0)
  set!(base_reservoir_nums,LatLonCoords(3,3),0)
  set!(landsea_mask,LatLonCoords(3,3),true)
  river_parameters = RiverParameters(flow_directions,
                                     river_reservoir_nums,
                                     overland_reservoir_nums,
                                     base_reservoir_nums,
                                     river_retention_coefficients,
                                     overland_retention_coefficients,
                                     base_retention_coefficients,
                                     landsea_mask,
                                     hd_grid,1.0,1.0)
  lake_grid = LatLonGrid(6,6,true)
  cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
    Float64[ 2.5 3.0 2.5
             3.0 4.0 3.0
             2.5 3.0 2.5 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 1 1 1 1
           1 1 1 1 1 1
           2 2 2 2 2 2
           2 2 2 2 2 2
           3 3 3 3 3 3
           3 3 3 3 3 3  ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ 1 1 2 2 3 3
           1 1 2 2 3 3
           1 1 2 2 3 3
           1 1 2 2 3 3
           1 1 2 2 3 3
           1 1 2 2 3 3 ])
  is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
    Bool[false false false false false false
         false false false false false false
         false  true false false  true false
         false  true  true  true  true false
         false false false false false false
         false false false false false false ])
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_model_parameters::LakeModelParameters =
    LakeModelParameters(grid_specific_lake_model_parameters,
                        lake_grid,
                        hd_grid,
                        surface_model_grid,
                        cell_areas_on_surface_model_grid,
                        LakeModelSettings(),
                        3,
                        is_lake)
  lake_parameters_as_array::Vector{Float64} = Float64[3.0, 21.0, 1.0, 3.0, 0.0, 4.0, 5.0, 2.0, 4.0, 5.0, 1.0, 0.0, 5.0, 3.0, 5.0, 1.0, 6.0, 8.0, 1.0, 2.0, 2.0, 2.0, 0.0, 26.0, 2.0, 3.0, 0.0, 4.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 5.0, 3.0, 2.0, 1.0, 2.0, 6.0, 4.0, 3.0, 1.0, 8.0, 8.0, 1.0, 1.0, -1.0, -1.0, 1.0, 43.0, 3.0, -1.0, 2.0, 1.0, 2.0, 4.0, 5.0, 6.0, 4.0, 5.0, 1.0, 0.0, 8.0, 3.0, 5.0, 1.0, 0.0, 8.0, 4.0, 4.0, 1.0, 0.0, 8.0, 4.0, 3.0, 1.0, 0.0, 8.0, 4.0, 2.0, 1.0, 0.0, 8.0, 3.0, 2.0, 1.0, 12.0, 10.0, 1.0, -1.0, 3.0, 3.0, 0.0]
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,Float64[ 1.0 1.0 1.0
                                                                                 1.0 1.0 1.0
                                                                                 1.0 1.0 0.0 ])
  drainages::Array{Field{Float64},1} = repeat(drainage,60,false)
  runoffs::Array{Field{Float64},1} = drainages
  evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,30,false)
  additional_evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,8.0)
  additional_evaporations::Array{Field{Float64},1} = repeat(additional_evaporation,30,false)
  append!(evaporations,additional_evaporations)
  expected_river_inflow::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                               Float64[ 0.0 0.0  2.0
                                                                        4.0 0.0 14.0
                                                                        0.0 0.0  0.0 ])
  expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                                 Float64[ 0.0 0.0  0.0
                                                                          0.0 0.0  0.0
                                                                          0.0 0.0 16.0 ])
  expected_water_to_hd::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                              Float64[ 0.0 0.0 10.0
                                                                       0.0 0.0  0.0
                                                                       0.0 0.0  0.0 ])
  # expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
  #     Int64[    0    0    0    1    0    0    0    0    0
  #               0    0    0    1    0    0    0    0    0
  #               0    0    1    1    0    0    0    0    0
  #               0    0    1    1    0    0    0    0    0
  #               0    0    1    1    0    0    0    0    0
  #                ])
  # expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
  #     Int64[    0    0    0    2    0    0    0    0    0
  #               0    0    0    2    0    0    0    0    0
  #               0    0    2    2    0    0    0    0    0
  #               0    0    2    2    0    0    0    0    0
  #               0    0    2    2    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0 ])
  # expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[    0    0    0    80.0    0    0    0    0    0
  #                   0    0    0    80.0   0    0    0    0    0
  #                   0    0    80.0 80.0    0    0    0    0    0
  #                   0    0    80.0 80.0    0    0    0    0    0
  #                   0    0    80.0 80.0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0 ])
  # expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
  #       Float64[ 0.0 1.0/3.0 0.0
  #                0.2     0.2 0.0
  #                0.0     0.0 0.0 ])
  # expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
  #       Int64[ 0 2 0
  #              3 3 0
  #              0 0 0 ])
  # expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
  #       Int64[ 6 6 6
  #             15 15 15
  #              6 6 6 ])
  # expected_lake_volumes::Array{Float64} = Float64[80.0]
  # expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[    0    0    0   7.0    0    0    0    0    0
  #                   0    0    0   7.0   0    0    0    0    0
  #                   0    0    9.0 7.0    0    0    0    0    0
  #                   0    0    8.0 8.0    0    0    0    0    0
  #                   0    0    8.0 8.0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0 ])
  import Profile
  @time river_fields::RiverPrognosticFields,
    lake_model_prognostics::LakeModelPrognostics,_ =
    drive_hd_and_lake_model(river_parameters,lake_model_parameters,
                            lake_parameters_as_array,
                            drainages,runoffs,evaporations,
                            1100,print_timestep_results=false,
                            write_output=false,return_output=true)
  # lake_fractions =
  #   calculate_lake_fraction_on_surface_grid(lake_model_parameters,
  #                                           lake_model_prognostics)
  # lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
  # for i = 1:9
  #   for j = 1:9
  #     coords::LatLonCoords = LatLonCoords(i,j)
  #     lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
  #     if lake_number <= 0 continue end
  #     lake::Lake = lake_model_prognostics.lakes[lake_number]
  #     if isa(lake,FillingLake)
  #       set!(lake_types,coords,1)
  #     elseif isa(lake,OverflowingLake)
  #       set!(lake_types,coords,2)
  #     elseif isa(lake,SubsumedLake)
  #       set!(lake_types,coords,3)
  #     else
  #       set!(lake_types,coords,4)
  #     end
  #   end
  # end
  # lake_volumes::Array{Float64} = Float64[]
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
#   diagnostic_lake_volumes =
#     calculate_diagnostic_lake_volumes_field(lake_model_parameters,
#                                             lake_model_prognostics)
  # @test expected_river_inflow == river_fields.river_inflow
  # @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  # @test expected_water_to_hd == lake_model_prognostics.water_to_hd
  # @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  # @test expected_lake_types == lake_types
  # @test expected_lake_volumes == lake_volumes
  # @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  # @test expected_lake_fractions == lake_fractions
  # @test expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  # @test expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  # @test expected_true_lake_depths == lake_model_parameters.true_lake_depths
end

@testset "Lake model tests 3" begin
  #Fill a single lake
  hd_grid = LatLonGrid(3,3,true)
  surface_model_grid = LatLonGrid(3,3,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
                                                Int64[-2 6 2
                                                       8 7 2
                                                       8 7 0 ] ))
  river_reservoir_nums = LatLonField{Int64}(hd_grid,5)
  overland_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  base_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  river_retention_coefficients = LatLonField{Float64}(hd_grid,0.7)
  overland_retention_coefficients = LatLonField{Float64}(hd_grid,0.5)
  base_retention_coefficients = LatLonField{Float64}(hd_grid,0.1)
  landsea_mask = LatLonField{Bool}(hd_grid,fill(false,3,3))
  set!(river_reservoir_nums,LatLonCoords(3,3),0)
  set!(overland_reservoir_nums,LatLonCoords(3,3),0)
  set!(base_reservoir_nums,LatLonCoords(3,3),0)
  set!(landsea_mask,LatLonCoords(3,3),true)
  river_parameters = RiverParameters(flow_directions,
                                     river_reservoir_nums,
                                     overland_reservoir_nums,
                                     base_reservoir_nums,
                                     river_retention_coefficients,
                                     overland_retention_coefficients,
                                     base_retention_coefficients,
                                     landsea_mask,
                                     hd_grid,1.0,1.0)
  lake_grid = LatLonGrid(9,9,true)
  cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
    Float64[ 2.5 3.0 2.5
             3.0 4.0 3.0
             2.5 3.0 2.5 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 1 1 1 1 1 1 1 1 1
                                                           1 1 1 1 1 1 1 1 1
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           3 3 3 3 3 3 3 3 3
                                                           3 3 3 3 3 3 3 3 3 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 1 1 1 2 2 2 3 3 3
                                                           1 1 1 2 2 2 3 3 3
                                                           1 1 1 2 2 2 3 3 3
                                                           1 1 1 2 2 2 3 3 3
                                                           1 1 1 2 2 2 3 3 3
                                                           1 1 1 2 2 2 3 3 3
                                                           1 1 1 2 2 2 3 3 3
                                                           1 1 1 2 2 2 3 3 3
                                                           1 1 1 2 2 2 3 3 3 ])
  is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
    Bool[false false false  true false false false false false
         false false false  true false false false false false
         false false  true  true false false false false false
         false false  true  true false false false false false
         false false  true  true false false false false false
         false false false false false false false false false
         false false false false false false false false false
         false false false false false false false false false
         false false false false false false false false false])
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_model_parameters::LakeModelParameters =
    LakeModelParameters(grid_specific_lake_model_parameters,
                        lake_grid,
                        hd_grid,
                        surface_model_grid,
                        cell_areas_on_surface_model_grid,
                        LakeModelSettings(),
                        1,
                        is_lake)
  lake_parameters_as_array::Vector{Float64} = Float64[1.0, 51.0, 1.0, -1.0, 0.0, 3.0, 3.0, 8.0, 3.0, 3.0, 1.0, 1.0, 2.0, 4.0, 4.0, 1.0, 1.0, 2.0, 4.0, 3.0, 1.0, 1.0, 2.0, 5.0, 3.0, 1.0, 1.0, 2.0, 5.0, 4.0, 1.0, 6.0, 3.0, 2.0, 4.0, 1.0, 6.0, 3.0, 3.0, 4.0, 1.0, 6.0, 3.0, 1.0, 4.0, 1.0, 62.0, 10.0, 1.0, -1.0, 3.0, 3.0, 0.0]
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,Float64[ 1.0 1.0 1.0
                                                                                 1.0 1.0 1.0
                                                                                 1.0 1.0 0.0 ])
  drainages::Array{Field{Float64},1} = repeat(drainage,1000,false)
  runoffs::Array{Field{Float64},1} = drainages
  evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,1000,false)
  expected_river_inflow::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                               Float64[ 0.0 0.0  2.0
                                                                        4.0 0.0  4.0
                                                                        0.0 0.0  0.0 ])
  expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                                 Float64[ 0.0 0.0  0.0
                                                                          0.0 0.0  0.0
                                                                          0.0 0.0 16.0 ])
  expected_water_to_hd::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                              Float64[ 0.0 0.0  0.0
                                                                       0.0 0.0  0.0
                                                                       0.0 0.0 10.0 ])
  expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    2    0    0    0    0    0
                0    0    0    2    0    0    0    0    0
                0    0    2    2    0    0    0    0    0
                0    0    2    2    0    0    0    0    0
                0    0    2    2    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[    0    0    0    62.0    0    0    0    0    0
                    0    0    0    62.0   0    0    0    0    0
                    0    0    62.0 62.0    0    0    0    0    0
                    0    0    62.0 62.0    0    0    0    0    0
                    0    0    62.0 62.0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0 ])
  expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.0 1.0/3.0 0.0
                 0.2     0.2 0.0
                 0.0     0.0 0.0 ])
  expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 0 2 0
               3 3 0
               0 0 0 ])
  expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 6 6 6
              15 15 15
               6 6 6 ])
  expected_lake_volumes::Array{Float64} = Float64[62.0]
  # expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[    0    0    0   7.0    0    0    0    0    0
  #                   0    0    0   7.0   0    0    0    0    0
  #                   0    0    9.0 7.0    0    0    0    0    0
  #                   0    0    8.0 8.0    0    0    0    0    0
  #                   0    0    8.0 8.0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0 ])
  import Profile
  @time river_fields::RiverPrognosticFields,
    lake_model_prognostics::LakeModelPrognostics,_ =
    drive_hd_and_lake_model(river_parameters,lake_model_parameters,
                            lake_parameters_as_array,
                            drainages,runoffs,evaporations,
                            1000,print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_fractions::Field{Float64} =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes::Array{Float64} = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes::Field{Float64} =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test expected_river_inflow == river_fields.river_inflow
  @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test expected_lake_types == lake_types
  @test expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  @test expected_lake_fractions == lake_fractions
  @test expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  # @test expected_true_lake_depths == lake_model_parameters.true_lake_depths
end

@testset "Lake model tests 4" begin
  #Three lakes, one of which has several subbasin - fill only
  hd_grid = LatLonGrid(4,4,true)
  surface_model_grid = LatLonGrid(3,3,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
                                                Int64[-2  4  2  2
                                                       8 -2 -2  2
                                                       6  2  3 -2
                                                      -2  0  6  0 ]))
  river_reservoir_nums = LatLonField{Int64}(hd_grid,5)
  overland_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  base_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  river_retention_coefficients = LatLonField{Float64}(hd_grid,0.7)
  overland_retention_coefficients = LatLonField{Float64}(hd_grid,0.5)
  base_retention_coefficients = LatLonField{Float64}(hd_grid,0.1)
  landsea_mask = LatLonField{Bool}(hd_grid,fill(false,4,4))
  set!(river_reservoir_nums,LatLonCoords(4,4),0)
  set!(overland_reservoir_nums,LatLonCoords(4,4),0)
  set!(base_reservoir_nums,LatLonCoords(4,4),0)
  set!(landsea_mask,LatLonCoords(4,4),true)
  set!(river_reservoir_nums,LatLonCoords(4,2),0)
  set!(overland_reservoir_nums,LatLonCoords(4,2),0)
  set!(base_reservoir_nums,LatLonCoords(4,2),0)
  set!(landsea_mask,LatLonCoords(4,2),true)
  river_parameters = RiverParameters(flow_directions,
                                     river_reservoir_nums,
                                     overland_reservoir_nums,
                                     base_reservoir_nums,
                                     river_retention_coefficients,
                                     overland_retention_coefficients,
                                     base_retention_coefficients,
                                     landsea_mask,
                                     hd_grid,1.0,1.0)
  lake_grid = LatLonGrid(20,20,true)
  cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
    Float64[ 2.5 3.0 2.5
             3.0 2.0 3.0
             2.5 3.0 2.5 ])
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
                                                    Int64[ 1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3
                                                           1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3
                                                           1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3
                                                           1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3
                                                           1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3
                                                           1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3
                                                           1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3
                                                           1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3
                                                           1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3
                                                           1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3
                                                           1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3
                                                           1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3
                                                           1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3
                                                           1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3
                                                           1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3
                                                           1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3
                                                           1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3
                                                           1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3
                                                           1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3
                                                           1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 3 ])
  is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
    Bool[false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false  true
          true  true  true false false false false false false false false false  true  true  true  true false false false  true
         false  true  true false false  true  true false false false false  true  true  true  true  true false false false false
         false  true  true false false  true  true  true  true false false false  true  true  true  true  true false false false
         false  true  true  true  true  true false  true  true  true  true  true  true  true  true  true  true  true false false
          true  true  true false  true  true  true  true  true false  true  true  true  true  true  true  true  true false false
         false  true  true false false  true false  true false false false  true  true  true  true  true  true false false false
         false false false false false false false false false false false  true  true  true  true  true  true false false false
         false false false false false false false false false false false false  true  true false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false  true  true  true  true false
         false false false false false false false false false false false false false false false  true  true  true false false
         false false false  true false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false])
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_model_parameters::LakeModelParameters =
    LakeModelParameters(grid_specific_lake_model_parameters,
                        lake_grid,
                        hd_grid,
                        surface_model_grid,
                        cell_areas_on_surface_model_grid,
                        LakeModelSettings(),
                        9,
                        is_lake)
  lake_parameters_as_array::Vector{Float64} = Float64[9.0, 16.0, 1.0, -1.0, 0.0, 16.0, 4.0, 1.0, 16.0, 4.0, 1.0, 1.0, 3.0, 1.0, -1.0, 4.0, 2.0, 0.0, 46.0, 2.0, -1.0, 0.0, 14.0, 16.0, 7.0, 14.0, 16.0, 1.0, 0.0, 2.0, 14.0, 17.0, 1.0, 0.0, 2.0, 15.0, 16.0, 1.0, 3.0, 3.0, 15.0, 17.0, 1.0, 3.0, 3.0, 15.0, 18.0, 1.0, 3.0, 3.0, 14.0, 18.0, 1.0, 3.0, 3.0, 14.0, 19.0, 1.0, 10.0, 4.0, 1.0, -1.0, 4.0, 4.0, 0.0, 16.0, 3.0, 7.0, 0.0, 8.0, 15.0, 1.0, 8.0, 15.0, 1.0, 1.0, 3.0, 1.0, 5.0, -1.0, -1.0, 1.0, 91.0, 4.0, 8.0, 0.0, 7.0, 6.0, 16.0, 7.0, 6.0, 1.0, 0.0, 1.0, 6.0, 6.0, 1.0, 2.0, 2.0, 7.0, 5.0, 1.0, 2.0, 2.0, 7.0, 7.0, 1.0, 2.0, 2.0, 8.0, 6.0, 1.0, 2.0, 2.0, 7.0, 8.0, 1.0, 2.0, 2.0, 6.0, 8.0, 1.0, 2.0, 2.0, 6.0, 5.0, 1.0, 10.0, 3.0, 5.0, 6.0, 1.0, 10.0, 3.0, 5.0, 7.0, 1.0, 10.0, 3.0, 7.0, 9.0, 1.0, 10.0, 3.0, 6.0, 9.0, 1.0, 10.0, 3.0, 4.0, 7.0, 2.0, 10.0, 3.0, 5.0, 9.0, 1.0, 23.0, 4.0, 8.0, 8.0, 1.0, 23.0, 4.0, 5.0, 8.0, 1.0, 38.0, 5.0, 1.0, 7.0, -1.0, -1.0, 1.0, 41.0, 5.0, 7.0, 0.0, 6.0, 14.0, 6.0, 6.0, 14.0, 1.0, 0.0, 2.0, 7.0, 13.0, 1.0, 0.0, 2.0, 5.0, 13.0, 1.0, 0.0, 2.0, 6.0, 12.0, 1.0, 0.0, 2.0, 8.0, 13.0, 1.0, 0.0, 2.0, 7.0, 12.0, 1.0, 6.0, 3.0, 1.0, 3.0, -1.0, -1.0, 1.0, 86.0, 6.0, 9.0, 0.0, 3.0, 1.0, 15.0, 3.0, 1.0, 1.0, 0.0, 1.0, 3.0, 20.0, 1.0, 2.0, 2.0, 2.0, 20.0, 1.0, 5.0, 3.0, 4.0, 2.0, 1.0, 5.0, 3.0, 5.0, 3.0, 1.0, 5.0, 3.0, 4.0, 3.0, 1.0, 5.0, 3.0, 3.0, 3.0, 1.0, 5.0, 3.0, 5.0, 2.0, 1.0, 5.0, 3.0, 6.0, 3.0, 1.0, 5.0, 3.0, 6.0, 2.0, 1.0, 5.0, 3.0, 7.0, 3.0, 1.0, 16.0, 4.0, 7.0, 2.0, 1.0, 16.0, 4.0, 8.0, 3.0, 1.0, 16.0, 4.0, 8.0, 2.0, 1.0, 16.0, 4.0, 7.0, 1.0, 1.0, 46.0, 6.0, 1.0, 8.0, 2.0, 2.0, 0.0, 163.0, 7.0, 8.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 8.0, 15.0, 1.0, 0.0, 3.0, 8.0, 16.0, 1.0, 0.0, 3.0, 7.0, 16.0, 1.0, 0.0, 3.0, 8.0, 14.0, 1.0, 0.0, 3.0, 9.0, 14.0, 1.0, 0.0, 3.0, 9.0, 15.0, 1.0, 0.0, 3.0, 7.0, 13.0, 1.0, 0.0, 3.0, 6.0, 15.0, 1.0, 0.0, 3.0, 8.0, 13.0, 1.0, 0.0, 3.0, 9.0, 13.0, 1.0, 0.0, 3.0, 7.0, 15.0, 1.0, 0.0, 3.0, 6.0, 16.0, 1.0, 0.0, 3.0, 5.0, 14.0, 1.0, 0.0, 3.0, 6.0, 12.0, 1.0, 0.0, 3.0, 4.0, 13.0, 1.0, 0.0, 3.0, 7.0, 11.0, 1.0, 0.0, 3.0, 6.0, 11.0, 1.0, 0.0, 3.0, 5.0, 15.0, 1.0, 0.0, 3.0, 4.0, 14.0, 1.0, 0.0, 3.0, 5.0, 13.0, 1.0, 0.0, 3.0, 7.0, 14.0, 1.0, 0.0, 3.0, 8.0, 12.0, 1.0, 0.0, 3.0, 7.0, 12.0, 1.0, 0.0, 3.0, 6.0, 14.0, 1.0, 0.0, 3.0, 6.0, 13.0, 1.0, 25.0, 4.0, 6.0, 17.0, 1.0, 25.0, 4.0, 7.0, 17.0, 1.0, 25.0, 4.0, 5.0, 16.0, 1.0, 25.0, 4.0, 4.0, 15.0, 1.0, 25.0, 4.0, 9.0, 16.0, 1.0, 55.0, 5.0, 1.0, 4.0, -1.0, -1.0, 1.0, 298.0, 8.0, 9.0, 2.0, 4.0, 7.0, 7.0, 6.0, 57.0, 7.0, 6.0, 1.0, 0.0, 5.0, 7.0, 7.0, 1.0, 0.0, 5.0, 7.0, 5.0, 1.0, 0.0, 5.0, 8.0, 6.0, 1.0, 0.0, 5.0, 6.0, 8.0, 1.0, 0.0, 5.0, 7.0, 8.0, 1.0, 0.0, 5.0, 7.0, 9.0, 1.0, 0.0, 5.0, 6.0, 9.0, 1.0, 0.0, 5.0, 5.0, 7.0, 1.0, 0.0, 5.0, 6.0, 6.0, 1.0, 0.0, 5.0, 6.0, 10.0, 2.0, 0.0, 5.0, 4.0, 7.0, 2.0, 0.0, 5.0, 6.0, 11.0, 1.0, 0.0, 5.0, 5.0, 6.0, 1.0, 0.0, 5.0, 7.0, 12.0, 1.0, 0.0, 5.0, 6.0, 12.0, 1.0, 0.0, 5.0, 8.0, 13.0, 1.0, 0.0, 5.0, 7.0, 11.0, 1.0, 0.0, 5.0, 8.0, 8.0, 1.0, 0.0, 5.0, 6.0, 5.0, 1.0, 0.0, 5.0, 5.0, 13.0, 1.0, 0.0, 5.0, 9.0, 13.0, 1.0, 0.0, 5.0, 9.0, 12.0, 1.0, 0.0, 5.0, 5.0, 14.0, 1.0, 0.0, 5.0, 4.0, 14.0, 1.0, 0.0, 5.0, 4.0, 15.0, 1.0, 0.0, 5.0, 3.0, 15.0, 1.0, 0.0, 5.0, 5.0, 16.0, 1.0, 0.0, 5.0, 4.0, 16.0, 1.0, 0.0, 5.0, 6.0, 17.0, 1.0, 0.0, 5.0, 6.0, 16.0, 1.0, 0.0, 5.0, 5.0, 17.0, 1.0, 0.0, 5.0, 7.0, 18.0, 1.0, 0.0, 5.0, 7.0, 17.0, 1.0, 0.0, 5.0, 7.0, 16.0, 1.0, 0.0, 5.0, 7.0, 15.0, 1.0, 0.0, 5.0, 6.0, 18.0, 1.0, 0.0, 5.0, 8.0, 16.0, 1.0, 0.0, 5.0, 8.0, 15.0, 1.0, 0.0, 5.0, 10.0, 13.0, 1.0, 0.0, 5.0, 4.0, 13.0, 1.0, 0.0, 5.0, 6.0, 15.0, 1.0, 0.0, 5.0, 5.0, 15.0, 1.0, 0.0, 5.0, 9.0, 15.0, 1.0, 0.0, 5.0, 9.0, 16.0, 1.0, 0.0, 5.0, 10.0, 14.0, 1.0, 0.0, 5.0, 8.0, 14.0, 1.0, 0.0, 5.0, 7.0, 14.0, 1.0, 0.0, 5.0, 6.0, 14.0, 1.0, 0.0, 5.0, 5.0, 9.0, 1.0, 0.0, 5.0, 5.0, 8.0, 1.0, 0.0, 5.0, 9.0, 14.0, 1.0, 0.0, 5.0, 8.0, 12.0, 1.0, 0.0, 5.0, 9.0, 17.0, 1.0, 0.0, 5.0, 7.0, 13.0, 1.0, 0.0, 5.0, 6.0, 13.0, 1.0, 0.0, 5.0, 8.0, 17.0, 1.0, 55.0, 6.0, 1.0, 6.0, 2.0, 1.0, 0.0, 418.0, 9.0, -1.0, 2.0, 6.0, 8.0, 3.0, 1.0, 81.0, 3.0, 1.0, 1.0, 0.0, 6.0, 4.0, 2.0, 1.0, 0.0, 6.0, 3.0, 20.0, 1.0, 0.0, 6.0, 2.0, 20.0, 1.0, 0.0, 6.0, 3.0, 3.0, 1.0, 0.0, 6.0, 5.0, 3.0, 1.0, 0.0, 6.0, 4.0, 3.0, 1.0, 0.0, 6.0, 5.0, 2.0, 1.0, 0.0, 6.0, 6.0, 3.0, 1.0, 0.0, 6.0, 6.0, 2.0, 1.0, 0.0, 6.0, 6.0, 4.0, 1.0, 0.0, 6.0, 7.0, 3.0, 1.0, 0.0, 6.0, 6.0, 5.0, 1.0, 0.0, 6.0, 8.0, 3.0, 1.0, 0.0, 6.0, 8.0, 2.0, 1.0, 0.0, 6.0, 6.0, 6.0, 1.0, 0.0, 6.0, 5.0, 6.0, 1.0, 0.0, 6.0, 7.0, 6.0, 1.0, 0.0, 6.0, 7.0, 2.0, 1.0, 0.0, 6.0, 7.0, 1.0, 1.0, 0.0, 6.0, 7.0, 5.0, 1.0, 0.0, 6.0, 7.0, 7.0, 1.0, 0.0, 6.0, 8.0, 6.0, 1.0, 0.0, 6.0, 5.0, 7.0, 1.0, 0.0, 6.0, 6.0, 8.0, 1.0, 0.0, 6.0, 4.0, 7.0, 2.0, 0.0, 6.0, 5.0, 8.0, 1.0, 0.0, 6.0, 7.0, 9.0, 1.0, 0.0, 6.0, 6.0, 9.0, 1.0, 0.0, 6.0, 6.0, 10.0, 2.0, 0.0, 6.0, 5.0, 9.0, 1.0, 0.0, 6.0, 7.0, 11.0, 1.0, 0.0, 6.0, 6.0, 11.0, 1.0, 0.0, 6.0, 8.0, 12.0, 1.0, 0.0, 6.0, 6.0, 12.0, 1.0, 0.0, 6.0, 7.0, 12.0, 1.0, 0.0, 6.0, 9.0, 13.0, 1.0, 0.0, 6.0, 9.0, 12.0, 1.0, 0.0, 6.0, 6.0, 13.0, 1.0, 0.0, 6.0, 5.0, 13.0, 1.0, 0.0, 6.0, 10.0, 14.0, 1.0, 0.0, 6.0, 9.0, 14.0, 1.0, 0.0, 6.0, 8.0, 14.0, 1.0, 0.0, 6.0, 5.0, 14.0, 1.0, 0.0, 6.0, 8.0, 13.0, 1.0, 0.0, 6.0, 7.0, 13.0, 1.0, 0.0, 6.0, 4.0, 15.0, 1.0, 0.0, 6.0, 8.0, 8.0, 1.0, 0.0, 6.0, 7.0, 8.0, 1.0, 0.0, 6.0, 8.0, 15.0, 1.0, 0.0, 6.0, 7.0, 15.0, 1.0, 0.0, 6.0, 3.0, 15.0, 1.0, 0.0, 6.0, 3.0, 14.0, 1.0, 0.0, 6.0, 5.0, 15.0, 1.0, 0.0, 6.0, 9.0, 16.0, 1.0, 0.0, 6.0, 3.0, 13.0, 1.0, 0.0, 6.0, 8.0, 16.0, 1.0, 0.0, 6.0, 9.0, 17.0, 1.0, 0.0, 6.0, 8.0, 17.0, 1.0, 0.0, 6.0, 7.0, 17.0, 1.0, 0.0, 6.0, 7.0, 16.0, 1.0, 0.0, 6.0, 7.0, 18.0, 1.0, 0.0, 6.0, 6.0, 18.0, 1.0, 0.0, 6.0, 6.0, 17.0, 1.0, 0.0, 6.0, 6.0, 16.0, 1.0, 0.0, 6.0, 5.0, 17.0, 1.0, 0.0, 6.0, 4.0, 16.0, 1.0, 0.0, 6.0, 3.0, 16.0, 1.0, 0.0, 6.0, 6.0, 15.0, 1.0, 0.0, 6.0, 5.0, 16.0, 1.0, 0.0, 6.0, 10.0, 13.0, 1.0, 0.0, 6.0, 4.0, 13.0, 1.0, 0.0, 6.0, 9.0, 15.0, 1.0, 0.0, 6.0, 4.0, 12.0, 1.0, 0.0, 6.0, 7.0, 14.0, 1.0, 0.0, 6.0, 6.0, 14.0, 1.0, 0.0, 6.0, 4.0, 14.0, 1.0, 75.0, 7.0, 4.0, 7.0, 1.0, 75.0, 7.0, 4.0, 6.0, 2.0, 151.0, 8.0, 3.0, 2.0, 1.0, 151.0, 8.0, 4.0, 6.0, 1.0, 229.0, 9.0, 1.0, -1.0, 2.0, 4.0, 0.0]
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,Float64[ 1.0 1.0 1.0 1.0
                                                                                 1.0 1.0 1.0 1.0
                                                                                 1.0 1.0 1.0 1.0
                                                                                 1.0 0.0 1.0 0.0 ])
  drainages::Array{Field{Float64},1} = repeat(drainage,10000,false)
  runoffs::Array{Field{Float64},1} = drainages
  evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,10000,false)
  expected_river_inflow::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                               Float64[ 0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 2.0
                                                                        0.0 2.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0 ])
  expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                                 Float64[ 0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0
                                                                          0.0 6.0 0.0 22.0 ])
  expected_water_to_hd::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                              Float64[ 0.0 0.0 0.0  0.0
                                                                       0.0 0.0 0.0 12.0
                                                                       0.0 0.0 0.0  0.0
                                                                       0.0 2.0 0.0 18.0 ])
  expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
             6 9 6 0 0 0 0 0 0 0 0 0 9 9 8 9 0 0 0 6
             0 6 6 0 0 9 9 0 0 0 0 9 7 7 7 8 0 0 0 0
             0 6 6 0 0 4 4 4 4 0 0 0 5 7 7 7 8 0 0 0
             0 6 6 9 4 4 0 4 4 0 7 5 7 5 7 7 7 8 0 0
             6 6 6 0 4 4 4 4 4 0 7 5 5 7 7 7 7 8 0 0
             0 6 6 0 0 4 0 4 0 0 0 7 5 7 3 7 8 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 8 7 7 7 7 8 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 8 8 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3
             3 2 3 0 0 0 0 0 0 0 0 0 2 2 3 2 0 0 0 3
             0 3 3 0 0 2 2 0 0 0 0 2 3 3 3 3 0 0 0 0
             0 3 3 0 0 3 3 3 3 0 0 0 3 3 3 3 3 0 0 0
             0 3 3 2 3 3 0 3 3 0 3 3 3 3 3 3 3 3 0 0
             3 3 3 0 3 3 3 3 3 0 3 3 3 3 3 3 3 3 0 0
             0 3 3 0 0 3 0 3 0 0 0 3 3 3 3 3 3 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 3 3 3 3 3 3 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 3 3 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 0 0
             0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     430.0
                   430.0 430.0 430.0 0     0     0     0     0     0     0     #=
                =# 0     0     430.0 430.0 430.0 430.0 0     0     0     430.0
                   0     430.0 430.0 0     0     430.0 430.0 0     0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 0     0     0     0
                   0     430.0 430.0 0     0     430.0 430.0 430.0 430.0 0     #=
                =# 0     0     430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     430.0 430.0 430.0 430.0 430.0 0     430.0 430.0 0     #=
                =# 430.0 430.0 430.0 430.0 430.0 430.0 430.0 430.0 0     0
                   430.0 430.0 430.0 0     430.0 430.0 430.0 430.0 430.0 0     #=
                =# 430.0 430.0 430.0 430.0 430.0 430.0 430.0 430.0 0     0
                   0     430.0 430.0 0     0     430.0 0     430.0 0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     430.0 430.0 0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     10.0  10.0  10.0  10.0  0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     10.0  10.0  10.0  0     0
                   0     0     0     1.0   0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0 ])
  expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[0.380952 0.305556 0.404762
                0.160714 0.229167 0.321429
                0.0238095 0.0 0.0714286])
  expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 16 11 17
                9 11 18
                1  0  3 ])
  expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[42 36 42
              56 48 56
              42 36 42 ])
  expected_lake_volumes::Array{Float64} = Float64[1.0, 10.0, 1.0, 38.0, 6.0, 46.0, 55.0, 55.0, 229.0]
  # expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[ 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 7.0
  #                8.0 1.0 6.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 4.0 3.0 0.0 0.0 0.0 8.0
  #                0.0 6.0 6.0 0.0 0.0 1.0 2.0 0.0 0.0 0.0 0.0 3.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0 0.0
  #                0.0 6.0 6.0 0.0 0.0 6.0 6.0 5.0 6.0 0.0 0.0 0.0 7.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0
  #                0.0 6.0 6.0 3.0 7.0 8.0 0.0 7.0 6.0 0.0 6.0 7.0 6.0 7.0 6.0 6.0 5.0 4.0 0.0 0.0
  #                5.0 5.0 6.0 0.0 7.0 8.0 7.0 7.0 6.0 0.0 6.0 7.0 7.0 6.0 6.0 6.0 5.0 4.0 0.0 0.0
  #                0.0 5.0 5.0 0.0 0.0 7.0 0.0 5.0 0.0 0.0 0.0 6.0 7.0 6.0 7.0 6.0 4.0 0.0 0.0 0.0
  #                0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0 6.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0
  #                0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0 4.0 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 2.0 1.0 1.0 0.0
  #                0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 1.0 1.0 0.0 0.0
  #                0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ])
  import Profile
  @time river_fields::RiverPrognosticFields,
    lake_model_prognostics::LakeModelPrognostics,_ =
    drive_hd_and_lake_model(river_parameters,lake_model_parameters,
                            lake_parameters_as_array,
                            drainages,runoffs,evaporations,
                            10000,print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_fractions::Field{Float64} =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes::Array{Float64} = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes::Field{Float64} =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test expected_river_inflow == river_fields.river_inflow
  @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test expected_lake_types == lake_types
  @test expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  @test isapprox(expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.000001)
  @test expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  # @test expected_true_lake_depths == lake_model_parameters.true_lake_depths
end

@testset "Lake model tests 5" begin
  #Three lakes, one of which has several subbasin - fill and drain
  hd_grid = LatLonGrid(4,4,true)
  surface_model_grid = LatLonGrid(3,3,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
                                                Int64[-2  4  2  2
                                                       8 -2 -2  2
                                                       6  2  3 -2
                                                      -2  0  6  0 ]))
  river_reservoir_nums = LatLonField{Int64}(hd_grid,5)
  overland_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  base_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  river_retention_coefficients = LatLonField{Float64}(hd_grid,0.7)
  overland_retention_coefficients = LatLonField{Float64}(hd_grid,0.5)
  base_retention_coefficients = LatLonField{Float64}(hd_grid,0.1)
  landsea_mask = LatLonField{Bool}(hd_grid,fill(false,4,4))
  set!(river_reservoir_nums,LatLonCoords(4,4),0)
  set!(overland_reservoir_nums,LatLonCoords(4,4),0)
  set!(base_reservoir_nums,LatLonCoords(4,4),0)
  set!(landsea_mask,LatLonCoords(4,4),true)
  set!(river_reservoir_nums,LatLonCoords(4,2),0)
  set!(overland_reservoir_nums,LatLonCoords(4,2),0)
  set!(base_reservoir_nums,LatLonCoords(4,2),0)
  set!(landsea_mask,LatLonCoords(4,2),true)
  river_parameters = RiverParameters(flow_directions,
                                     river_reservoir_nums,
                                     overland_reservoir_nums,
                                     base_reservoir_nums,
                                     river_retention_coefficients,
                                     overland_retention_coefficients,
                                     base_retention_coefficients,
                                     landsea_mask,
                                     hd_grid,1.0,1.0)
  lake_grid = LatLonGrid(20,20,true)
  cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
    Float64[ 2.5 3.0 2.5
             3.0 4.0 3.0
             2.5 3.0 2.5 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 2 1 1 1 1 1 1 1 1 1 1 2 2 2 2 1 1 1 1
                                                             1 1 1 1 1 2 2 1 1 1 1 2 2 2 2 2 1 1 1 1
                                                             1 1 1 1 1 3 3 3 3 1 1 1 1 2 2 2 2 1 1 1
                                                             1 1 1 2 3 3 1 3 3 2 2 1 2 1 2 2 2 2 1 1
                                                             1 1 1 1 3 3 3 3 3 1 2 1 1 2 2 2 2 2 1 1
                                                             1 1 1 1 1 3 1 3 1 1 1 2 1 2 2 2 2 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 2 2 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 1 1
                                                             1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3 ])

  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                      Int64[ 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1
                                                             1 2 1 3 3 3 3 3 3 3 3 3 2 2 2 2 3 3 3 1
                                                             3 1 1 3 3 2 2 3 3 3 3 2 2 2 2 2 3 3 3 3
                                                             3 1 1 3 3 1 1 1 1 3 3 3 2 2 2 2 2 3 3 3
                                                             3 1 1 2 1 1 3 1 1 2 2 2 2 2 2 2 2 2 3 3
                                                             1 1 1 3 1 1 1 1 1 3 2 2 2 2 2 2 2 2 3 3
                                                             3 1 1 3 3 1 3 1 3 3 3 2 2 2 2 2 2 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 2 2 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 3 3
                                                             3 3 3 1 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3  ])
  is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
    Bool[false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false  true
          true  true  true false false false false false false false false false  true  true  true  true false false false  true
         false  true  true false false  true  true false false false false  true  true  true  true  true false false false false
         false  true  true false false  true  true  true  true false false false  true  true  true  true  true false false false
         false  true  true  true  true  true false  true  true  true  true  true  true  true  true  true  true  true false false
          true  true  true false  true  true  true  true  true false  true  true  true  true  true  true  true  true false false
         false  true  true false false  true false  true false false false  true  true  true  true  true  true false false false
         false false false false false false false false false false false  true  true  true  true  true  true false false false
         false false false false false false false false false false false false  true  true false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false  true  true  true  true false
         false false false false false false false false false false false false false false false  true  true  true false false
         false false false  true false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false])
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_model_parameters::LakeModelParameters =
    LakeModelParameters(grid_specific_lake_model_parameters,
                        lake_grid,
                        hd_grid,
                        surface_model_grid,
                        cell_areas_on_surface_model_grid,
                        LakeModelSettings(),
                        9,
                        is_lake)
  lake_parameters_as_array::Vector{Float64} = Float64[9.0, 16.0, 1.0, -1.0, 0.0, 16.0, 4.0, 1.0, 16.0, 4.0, 1.0, 1.0, 3.0, 1.0, -1.0, 4.0, 2.0, 0.0, 46.0, 2.0, -1.0, 0.0, 14.0, 16.0, 7.0, 14.0, 16.0, 1.0, 0.0, 2.0, 14.0, 17.0, 1.0, 0.0, 2.0, 15.0, 16.0, 1.0, 3.0, 3.0, 15.0, 17.0, 1.0, 3.0, 3.0, 15.0, 18.0, 1.0, 3.0, 3.0, 14.0, 18.0, 1.0, 3.0, 3.0, 14.0, 19.0, 1.0, 10.0, 4.0, 1.0, -1.0, 4.0, 4.0, 0.0, 16.0, 3.0, 7.0, 0.0, 8.0, 15.0, 1.0, 8.0, 15.0, 1.0, 1.0, 3.0, 1.0, 5.0, -1.0, -1.0, 1.0, 91.0, 4.0, 8.0, 0.0, 7.0, 6.0, 16.0, 7.0, 6.0, 1.0, 0.0, 1.0, 6.0, 6.0, 1.0, 2.0, 2.0, 7.0, 5.0, 1.0, 2.0, 2.0, 7.0, 7.0, 1.0, 2.0, 2.0, 8.0, 6.0, 1.0, 2.0, 2.0, 7.0, 8.0, 1.0, 2.0, 2.0, 6.0, 8.0, 1.0, 2.0, 2.0, 6.0, 5.0, 1.0, 10.0, 3.0, 5.0, 6.0, 1.0, 10.0, 3.0, 5.0, 7.0, 1.0, 10.0, 3.0, 7.0, 9.0, 1.0, 10.0, 3.0, 6.0, 9.0, 1.0, 10.0, 3.0, 4.0, 7.0, 2.0, 10.0, 3.0, 5.0, 9.0, 1.0, 23.0, 4.0, 8.0, 8.0, 1.0, 23.0, 4.0, 5.0, 8.0, 1.0, 38.0, 5.0, 1.0, 7.0, -1.0, -1.0, 1.0, 41.0, 5.0, 7.0, 0.0, 6.0, 14.0, 6.0, 6.0, 14.0, 1.0, 0.0, 2.0, 7.0, 13.0, 1.0, 0.0, 2.0, 5.0, 13.0, 1.0, 0.0, 2.0, 6.0, 12.0, 1.0, 0.0, 2.0, 8.0, 13.0, 1.0, 0.0, 2.0, 7.0, 12.0, 1.0, 6.0, 3.0, 1.0, 3.0, -1.0, -1.0, 1.0, 86.0, 6.0, 9.0, 0.0, 3.0, 1.0, 15.0, 3.0, 1.0, 1.0, 0.0, 1.0, 3.0, 20.0, 1.0, 2.0, 2.0, 2.0, 20.0, 1.0, 5.0, 3.0, 4.0, 2.0, 1.0, 5.0, 3.0, 5.0, 3.0, 1.0, 5.0, 3.0, 4.0, 3.0, 1.0, 5.0, 3.0, 3.0, 3.0, 1.0, 5.0, 3.0, 5.0, 2.0, 1.0, 5.0, 3.0, 6.0, 3.0, 1.0, 5.0, 3.0, 6.0, 2.0, 1.0, 5.0, 3.0, 7.0, 3.0, 1.0, 16.0, 4.0, 7.0, 2.0, 1.0, 16.0, 4.0, 8.0, 3.0, 1.0, 16.0, 4.0, 8.0, 2.0, 1.0, 16.0, 4.0, 7.0, 1.0, 1.0, 46.0, 6.0, 1.0, 8.0, 2.0, 2.0, 0.0, 163.0, 7.0, 8.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 8.0, 15.0, 1.0, 0.0, 3.0, 8.0, 16.0, 1.0, 0.0, 3.0, 7.0, 16.0, 1.0, 0.0, 3.0, 8.0, 14.0, 1.0, 0.0, 3.0, 9.0, 14.0, 1.0, 0.0, 3.0, 9.0, 15.0, 1.0, 0.0, 3.0, 7.0, 13.0, 1.0, 0.0, 3.0, 6.0, 15.0, 1.0, 0.0, 3.0, 8.0, 13.0, 1.0, 0.0, 3.0, 9.0, 13.0, 1.0, 0.0, 3.0, 7.0, 15.0, 1.0, 0.0, 3.0, 6.0, 16.0, 1.0, 0.0, 3.0, 5.0, 14.0, 1.0, 0.0, 3.0, 6.0, 12.0, 1.0, 0.0, 3.0, 4.0, 13.0, 1.0, 0.0, 3.0, 7.0, 11.0, 1.0, 0.0, 3.0, 6.0, 11.0, 1.0, 0.0, 3.0, 5.0, 15.0, 1.0, 0.0, 3.0, 4.0, 14.0, 1.0, 0.0, 3.0, 5.0, 13.0, 1.0, 0.0, 3.0, 7.0, 14.0, 1.0, 0.0, 3.0, 8.0, 12.0, 1.0, 0.0, 3.0, 7.0, 12.0, 1.0, 0.0, 3.0, 6.0, 14.0, 1.0, 0.0, 3.0, 6.0, 13.0, 1.0, 25.0, 4.0, 6.0, 17.0, 1.0, 25.0, 4.0, 7.0, 17.0, 1.0, 25.0, 4.0, 5.0, 16.0, 1.0, 25.0, 4.0, 4.0, 15.0, 1.0, 25.0, 4.0, 9.0, 16.0, 1.0, 55.0, 5.0, 1.0, 4.0, -1.0, -1.0, 1.0, 298.0, 8.0, 9.0, 2.0, 4.0, 7.0, 7.0, 6.0, 57.0, 7.0, 6.0, 1.0, 0.0, 5.0, 7.0, 7.0, 1.0, 0.0, 5.0, 7.0, 5.0, 1.0, 0.0, 5.0, 8.0, 6.0, 1.0, 0.0, 5.0, 6.0, 8.0, 1.0, 0.0, 5.0, 7.0, 8.0, 1.0, 0.0, 5.0, 7.0, 9.0, 1.0, 0.0, 5.0, 6.0, 9.0, 1.0, 0.0, 5.0, 5.0, 7.0, 1.0, 0.0, 5.0, 6.0, 6.0, 1.0, 0.0, 5.0, 6.0, 10.0, 2.0, 0.0, 5.0, 4.0, 7.0, 2.0, 0.0, 5.0, 6.0, 11.0, 1.0, 0.0, 5.0, 5.0, 6.0, 1.0, 0.0, 5.0, 7.0, 12.0, 1.0, 0.0, 5.0, 6.0, 12.0, 1.0, 0.0, 5.0, 8.0, 13.0, 1.0, 0.0, 5.0, 7.0, 11.0, 1.0, 0.0, 5.0, 8.0, 8.0, 1.0, 0.0, 5.0, 6.0, 5.0, 1.0, 0.0, 5.0, 5.0, 13.0, 1.0, 0.0, 5.0, 9.0, 13.0, 1.0, 0.0, 5.0, 9.0, 12.0, 1.0, 0.0, 5.0, 5.0, 14.0, 1.0, 0.0, 5.0, 4.0, 14.0, 1.0, 0.0, 5.0, 4.0, 15.0, 1.0, 0.0, 5.0, 3.0, 15.0, 1.0, 0.0, 5.0, 5.0, 16.0, 1.0, 0.0, 5.0, 4.0, 16.0, 1.0, 0.0, 5.0, 6.0, 17.0, 1.0, 0.0, 5.0, 6.0, 16.0, 1.0, 0.0, 5.0, 5.0, 17.0, 1.0, 0.0, 5.0, 7.0, 18.0, 1.0, 0.0, 5.0, 7.0, 17.0, 1.0, 0.0, 5.0, 7.0, 16.0, 1.0, 0.0, 5.0, 7.0, 15.0, 1.0, 0.0, 5.0, 6.0, 18.0, 1.0, 0.0, 5.0, 8.0, 16.0, 1.0, 0.0, 5.0, 8.0, 15.0, 1.0, 0.0, 5.0, 10.0, 13.0, 1.0, 0.0, 5.0, 4.0, 13.0, 1.0, 0.0, 5.0, 6.0, 15.0, 1.0, 0.0, 5.0, 5.0, 15.0, 1.0, 0.0, 5.0, 9.0, 15.0, 1.0, 0.0, 5.0, 9.0, 16.0, 1.0, 0.0, 5.0, 10.0, 14.0, 1.0, 0.0, 5.0, 8.0, 14.0, 1.0, 0.0, 5.0, 7.0, 14.0, 1.0, 0.0, 5.0, 6.0, 14.0, 1.0, 0.0, 5.0, 5.0, 9.0, 1.0, 0.0, 5.0, 5.0, 8.0, 1.0, 0.0, 5.0, 9.0, 14.0, 1.0, 0.0, 5.0, 8.0, 12.0, 1.0, 0.0, 5.0, 9.0, 17.0, 1.0, 0.0, 5.0, 7.0, 13.0, 1.0, 0.0, 5.0, 6.0, 13.0, 1.0, 0.0, 5.0, 8.0, 17.0, 1.0, 55.0, 6.0, 1.0, 6.0, 2.0, 1.0, 0.0, 418.0, 9.0, -1.0, 2.0, 6.0, 8.0, 3.0, 1.0, 81.0, 3.0, 1.0, 1.0, 0.0, 6.0, 4.0, 2.0, 1.0, 0.0, 6.0, 3.0, 20.0, 1.0, 0.0, 6.0, 2.0, 20.0, 1.0, 0.0, 6.0, 3.0, 3.0, 1.0, 0.0, 6.0, 5.0, 3.0, 1.0, 0.0, 6.0, 4.0, 3.0, 1.0, 0.0, 6.0, 5.0, 2.0, 1.0, 0.0, 6.0, 6.0, 3.0, 1.0, 0.0, 6.0, 6.0, 2.0, 1.0, 0.0, 6.0, 6.0, 4.0, 1.0, 0.0, 6.0, 7.0, 3.0, 1.0, 0.0, 6.0, 6.0, 5.0, 1.0, 0.0, 6.0, 8.0, 3.0, 1.0, 0.0, 6.0, 8.0, 2.0, 1.0, 0.0, 6.0, 6.0, 6.0, 1.0, 0.0, 6.0, 5.0, 6.0, 1.0, 0.0, 6.0, 7.0, 6.0, 1.0, 0.0, 6.0, 7.0, 2.0, 1.0, 0.0, 6.0, 7.0, 1.0, 1.0, 0.0, 6.0, 7.0, 5.0, 1.0, 0.0, 6.0, 7.0, 7.0, 1.0, 0.0, 6.0, 8.0, 6.0, 1.0, 0.0, 6.0, 5.0, 7.0, 1.0, 0.0, 6.0, 6.0, 8.0, 1.0, 0.0, 6.0, 4.0, 7.0, 2.0, 0.0, 6.0, 5.0, 8.0, 1.0, 0.0, 6.0, 7.0, 9.0, 1.0, 0.0, 6.0, 6.0, 9.0, 1.0, 0.0, 6.0, 6.0, 10.0, 2.0, 0.0, 6.0, 5.0, 9.0, 1.0, 0.0, 6.0, 7.0, 11.0, 1.0, 0.0, 6.0, 6.0, 11.0, 1.0, 0.0, 6.0, 8.0, 12.0, 1.0, 0.0, 6.0, 6.0, 12.0, 1.0, 0.0, 6.0, 7.0, 12.0, 1.0, 0.0, 6.0, 9.0, 13.0, 1.0, 0.0, 6.0, 9.0, 12.0, 1.0, 0.0, 6.0, 6.0, 13.0, 1.0, 0.0, 6.0, 5.0, 13.0, 1.0, 0.0, 6.0, 10.0, 14.0, 1.0, 0.0, 6.0, 9.0, 14.0, 1.0, 0.0, 6.0, 8.0, 14.0, 1.0, 0.0, 6.0, 5.0, 14.0, 1.0, 0.0, 6.0, 8.0, 13.0, 1.0, 0.0, 6.0, 7.0, 13.0, 1.0, 0.0, 6.0, 4.0, 15.0, 1.0, 0.0, 6.0, 8.0, 8.0, 1.0, 0.0, 6.0, 7.0, 8.0, 1.0, 0.0, 6.0, 8.0, 15.0, 1.0, 0.0, 6.0, 7.0, 15.0, 1.0, 0.0, 6.0, 3.0, 15.0, 1.0, 0.0, 6.0, 3.0, 14.0, 1.0, 0.0, 6.0, 5.0, 15.0, 1.0, 0.0, 6.0, 9.0, 16.0, 1.0, 0.0, 6.0, 3.0, 13.0, 1.0, 0.0, 6.0, 8.0, 16.0, 1.0, 0.0, 6.0, 9.0, 17.0, 1.0, 0.0, 6.0, 8.0, 17.0, 1.0, 0.0, 6.0, 7.0, 17.0, 1.0, 0.0, 6.0, 7.0, 16.0, 1.0, 0.0, 6.0, 7.0, 18.0, 1.0, 0.0, 6.0, 6.0, 18.0, 1.0, 0.0, 6.0, 6.0, 17.0, 1.0, 0.0, 6.0, 6.0, 16.0, 1.0, 0.0, 6.0, 5.0, 17.0, 1.0, 0.0, 6.0, 4.0, 16.0, 1.0, 0.0, 6.0, 3.0, 16.0, 1.0, 0.0, 6.0, 6.0, 15.0, 1.0, 0.0, 6.0, 5.0, 16.0, 1.0, 0.0, 6.0, 10.0, 13.0, 1.0, 0.0, 6.0, 4.0, 13.0, 1.0, 0.0, 6.0, 9.0, 15.0, 1.0, 0.0, 6.0, 4.0, 12.0, 1.0, 0.0, 6.0, 7.0, 14.0, 1.0, 0.0, 6.0, 6.0, 14.0, 1.0, 0.0, 6.0, 4.0, 14.0, 1.0, 75.0, 7.0, 4.0, 7.0, 1.0, 75.0, 7.0, 4.0, 6.0, 2.0, 151.0, 8.0, 3.0, 2.0, 1.0, 151.0, 8.0, 4.0, 6.0, 1.0, 229.0, 9.0, 1.0, -1.0, 2.0, 4.0, 0.0]
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,Float64[ 1.0 1.0 1.0 1.0
                                                                                 1.0 1.0 1.0 1.0
                                                                                 1.0 1.0 1.0 1.0
                                                                                 1.0 0.0 1.0 0.0 ])
  drainages::Array{Field{Float64},1} = repeat(drainage,10000,false)
  runoffs::Array{Field{Float64},1} = drainages
  evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,180,false)
  evaporation = LatLonField{Float64}(surface_model_grid,100.0)
  additional_evaporations::Array{Field{Float64},1} = repeat(evaporation,200,false)
  append!(evaporations,additional_evaporations)
  expected_river_inflow::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                               Float64[ 0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 2.0
                                                                        0.0 2.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0 ])
  expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                                 Float64[ -94.0   0.0   0.0   0.0
                                                                            0.0 -98.0 -196.0   0.0
                                                                            0.0   0.0   0.0 -94.0
                                                                          -98.0   4.0   0.0   4.0 ])
  expected_water_to_hd::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                              Float64[ 0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0 ])
  expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 5 0 0 0 0 0 0
             0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
             0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.06666666 0.16666667 0.0
                 1.0 0.02325581395 0.0
                 0.06666666 0.1428571429 0.0 ])
  expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 1 1 0
               1 1 0
               1 1 0 ])
  expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 6  311
                1 43 1
               15 7  1 ])
  expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  expected_intermediate_river_inflow::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                         Float64[ 0.0 0.0 0.0 0.0
                                                                  0.0 0.0 0.0 2.0
                                                                  0.0 2.0 0.0 0.0
                                                                  0.0 0.0 0.0 0.0 ])
  expected_intermediate_water_to_ocean::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                           Float64[ 0.0 0.0 0.0 0.0
                                                                    0.0 0.0 0.0 0.0
                                                                    0.0 0.0 0.0 0.0
                                                                    0.0 6.0 0.0 22.0 ])
  expected_intermediate_water_to_hd::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                        Float64[ 0.0 0.0 0.0  0.0
                                                                 0.0 0.0 0.0 12.0
                                                                 0.0 0.0 0.0  0.0
                                                                 0.0 2.0 0.0 18.0 ])
  expected_intermediate_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
             6 9 6 0 0 0 0 0 0 0 0 0 9 9 8 9 0 0 0 6
             0 6 6 0 0 9 9 0 0 0 0 9 7 7 7 8 0 0 0 0
             0 6 6 0 0 4 4 4 4 0 0 0 5 7 7 7 8 0 0 0
             0 6 6 9 4 4 0 4 4 0 7 5 7 5 7 7 7 8 0 0
             6 6 6 0 4 4 4 4 4 0 7 5 5 7 7 7 7 8 0 0
             0 6 6 0 0 4 0 4 0 0 0 7 5 7 3 7 8 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 8 7 7 7 7 8 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 8 8 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
  expected_intermediate_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3
             3 2 3 0 0 0 0 0 0 0 0 0 2 2 3 2 0 0 0 3
             0 3 3 0 0 2 2 0 0 0 0 2 3 3 3 3 0 0 0 0
             0 3 3 0 0 3 3 3 3 0 0 0 3 3 3 3 3 0 0 0
             0 3 3 2 3 3 0 3 3 0 3 3 3 3 3 3 3 3 0 0
             3 3 3 0 3 3 3 3 3 0 3 3 3 3 3 3 3 3 0 0
             0 3 3 0 0 3 0 3 0 0 0 3 3 3 3 3 3 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 3 3 3 3 3 3 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 3 3 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 0 0
             0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
    expected_intermediate_diagnostic_lake_volumes::Field{Float64} =
      LatLonField{Float64}(lake_grid,
        Float64[   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     430.0
                   430.0 430.0 430.0 0     0     0     0     0     0     0     #=
                =# 0     0     430.0 430.0 430.0 430.0 0     0     0     430.0
                   0     430.0 430.0 0     0     430.0 430.0 0     0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 0     0     0     0
                   0     430.0 430.0 0     0     430.0 430.0 430.0 430.0 0     #=
                =# 0     0     430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     430.0 430.0 430.0 430.0 430.0 0     430.0 430.0 0     #=
                =# 430.0 430.0 430.0 430.0 430.0 430.0 430.0 430.0 0     0
                   430.0 430.0 430.0 0     430.0 430.0 430.0 430.0 430.0 0     #=
                =# 430.0 430.0 430.0 430.0 430.0 430.0 430.0 430.0 0     0
                   0     430.0 430.0 0     0     430.0 0     430.0 0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     430.0 430.0 0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     10.0  10.0  10.0  10.0  0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     10.0  10.0  10.0  0     0
                   0     0     0     1.0   0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0 ])
    expected_diagnostic_lake_volumes::Field{Float64} =
      LatLonField{Float64}(lake_grid,
        Float64[   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0 ])
    # expected_intermediate_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
    #     Float64[ 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 7.0
    #              8.0 1.0 6.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 4.0 3.0 0.0 0.0 0.0 8.0
    #              0.0 6.0 6.0 0.0 0.0 1.0 2.0 0.0 0.0 0.0 0.0 3.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0 0.0
    #              0.0 6.0 6.0 0.0 0.0 6.0 6.0 5.0 6.0 0.0 0.0 0.0 7.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0
    #              0.0 6.0 6.0 3.0 7.0 8.0 0.0 7.0 6.0 0.0 6.0 7.0 6.0 7.0 6.0 6.0 5.0 4.0 0.0 0.0
    #              5.0 5.0 6.0 0.0 7.0 8.0 7.0 7.0 6.0 0.0 6.0 7.0 7.0 6.0 6.0 6.0 5.0 4.0 0.0 0.0
    #              0.0 5.0 5.0 0.0 0.0 7.0 0.0 5.0 0.0 0.0 0.0 6.0 7.0 6.0 7.0 6.0 4.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0 6.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0 4.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 2.0 1.0 1.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 1.0 1.0 0.0 0.0
    #              0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ])
    # expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
    #     Float64[ 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #              0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ])
  expected_intermediate_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 1.0 0.0
                 1.0 0.976744186 0.0
                 1.0 1.0 0.0 ])
  expected_intermediate_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 6  0
                1 42 0
               15 7  0 ])
  expected_intermediate_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 6  311
                1 43 1
               15 7  1  ])
  expected_intermediate_lake_volumes::Array{Float64} = Float64[1.0, 10.0, 1.0, 38.0, 6.0, 46.0, 55.0, 55.0, 229.0]
  evaporations_copy::Array{Field{Float64},1} = evaporations
  river_fields::RiverPrognosticFields,
    lake_model_prognostics::LakeModelPrognostics,
    lake_model_diagnostics::LakeModelDiagnostics =
    drive_hd_and_lake_model(river_parameters,lake_model_parameters,
                            lake_parameters_as_array,
                            drainages,runoffs,evaporations,
                            5000,print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_fractions::Field{Float64} =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes::Array{Float64} = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes::Field{Float64} =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test expected_intermediate_river_inflow == river_fields.river_inflow
  @test isapprox(expected_intermediate_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_intermediate_water_to_hd == lake_model_prognostics.water_to_hd
  @test expected_intermediate_lake_numbers == lake_model_prognostics.lake_numbers
  @test expected_intermediate_lake_types == lake_types
  @test isapprox(expected_intermediate_lake_volumes,lake_volumes,atol=0.00001)
  @test expected_intermediate_diagnostic_lake_volumes == diagnostic_lake_volumes
  @test isapprox(expected_intermediate_lake_fractions,lake_fractions,rtol=0.0,atol=0.000001)
  @test expected_intermediate_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test expected_intermediate_number_fine_grid_cells ==
          lake_model_parameters.number_fine_grid_cells
  #@test expected_intermediate_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,_ =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            10000,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=5000)
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test expected_river_inflow == river_fields.river_inflow
  @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test expected_lake_types == lake_types
  @test isapprox(expected_lake_volumes,lake_volumes,atol=0.00001)
  @test expected_diagnostic_lake_volumes == diagnostic_lake_volumes
  @test isapprox(expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.000001)
  @test expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test expected_true_lake_depths == lake_model_parameters.true_lake_depths
end

@testset "Lake model tests 6" begin
  #Single lake, non unit timestep
  hd_grid = LatLonGrid(3,3,true)
  surface_model_grid = LatLonGrid(2,2,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
                                                Int64[-2 6 2
                                                       8 7 2
                                                       8 7 0 ] ))
  river_reservoir_nums = LatLonField{Int64}(hd_grid,5)
  overland_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  base_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  river_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  overland_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  base_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  landsea_mask = LatLonField{Bool}(hd_grid,fill(false,3,3))
  set!(river_reservoir_nums,LatLonCoords(3,3),0)
  set!(overland_reservoir_nums,LatLonCoords(3,3),0)
  set!(base_reservoir_nums,LatLonCoords(3,3),0)
  set!(landsea_mask,LatLonCoords(3,3),true)
  river_parameters = RiverParameters(flow_directions,
                                     river_reservoir_nums,
                                     overland_reservoir_nums,
                                     base_reservoir_nums,
                                     river_retention_coefficients,
                                     overland_retention_coefficients,
                                     base_retention_coefficients,
                                     landsea_mask,
                                     hd_grid,86400.0,86400.0)
  lake_grid = LatLonGrid(9,9,true)
  cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
    Float64[ 2.5 3.0
             3.5 4.0 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 2 2 2 1 2 2 2 2 2
                                                           2 2 2 1 2 2 2 2 2
                                                           2 2 1 1 2 2 2 2 2
                                                           2 2 1 1 2 2 2 2 2
                                                           2 2 1 1 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 1 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 2 2 2 1 2 2 2 2 2
                                                           2 2 2 1 2 2 2 2 2
                                                           2 2 1 1 2 2 2 2 2
                                                           2 2 1 1 2 2 2 2 2
                                                           2 2 1 1 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 1
                                                           2 2 2 2 2 2 2 2 2 ])
  is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
    Bool[false false false  true false false false false false
         false false false  true false false false false false
         false false  true  true false false false false false
         false false  true  true false false false false false
         false false  true  true false false false false false
         false false false false false false false false false
         false false false false false false false false false
         false false false false false false false false false
         false false false false false false false false false])
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_model_parameters::LakeModelParameters =
    LakeModelParameters(grid_specific_lake_model_parameters,
                        lake_grid,
                        hd_grid,
                        surface_model_grid,
                        cell_areas_on_surface_model_grid,
                        LakeModelSettings(),
                        1,
                        is_lake)
  lake_parameters_as_array::Vector{Float64} = Float64[1.0, 51.0, 1.0, -1.0, 0.0, 3.0, 3.0, 8.0, 3.0, 3.0, 1.0, 0.0, 2.0, 4.0, 4.0, 1.0, 0.0, 2.0, 3.0, 4.0, 1.0, 0.0, 2.0, 5.0, 3.0, 1.0, 0.0, 2.0, 4.0, 3.0, 1.0, 0.0, 2.0, 5.0, 4.0, 1.0, 172800.0, 5.0, 2.0, 4.0, 1.0, 172800.0, 5.0, 1.0, 4.0, 1.0, 432000.0, 8.375, 1.0, -1.0, 1.0, 2.0, 0.0]
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,Float64[ 1.0 0.0 0.0
                                                                                 0.0 0.0 0.0
                                                                                 0.0 0.0 0.0 ])
  drainages::Array{Field{Float64},1} = repeat(drainage,1000,false)
  runoff::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
  runoffs::Array{Field{Float64},1} = repeat(runoff,1000,false)
  evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,1000,false)
  expected_river_inflow::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                               Float64[ 0.0 0.0  1.0
                                                                        0.0 0.0  1.0
                                                                        0.0 0.0  0.0 ])
  expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                                 Float64[ 0.0 0.0  0.0
                                                                          0.0 0.0  0.0
                                                                          0.0 0.0  1.0 ])
  expected_water_to_hd::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                              Float64[ 0.0 86400.0 0.0
                                                                       0.0 0.0  0.0
                                                                       0.0 0.0  0.0 ])
  expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    2    0    0    0    0    0
                0    0    0    2    0    0    0    0    0
                0    0    2    2    0    0    0    0    0
                0    0    2    2    0    0    0    0    0
                0    0    2    2    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[    0    0    0    432000.0    0    0    0    0    0
                    0    0    0    432000.0   0    0    0    0    0
                    0    0    432000.0 432000.0    0    0    0    0    0
                    0    0    432000.0 432000.0    0    0    0    0    0
                    0    0    432000.0 432000.0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0 ])
  expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0     0.0
                 0.0     0.0  ])
  expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 0
               0 0 ])
  expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 1
               1 71 ])
  expected_lake_volumes::Array{Float64} = Float64[432000.0]
  # expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[ 0    0    0    5.0    0    0    0    0    0
  #                   0    0    0    5.0   0    0    0    0    0
  #                   0    0    8.0 8.0    0    0    0    0    0
  #                   0    0    8.0 8.0    0    0    0    0    0
  #                   0    0    8.0 8.0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0 ])

  first_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  first_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  first_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  first_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  first_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  first_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[    0    0    0    0   0    0    0    0    0
                    0    0    0    0   0    0    0    0    0
                    0    0 172800.0 172800.0 0 0 0    0    0
                    0    0 172800.0 172800.0 0 0 0    0    0
                    0    0 172800.0 172800.0 0 0 0    0    0
                    0    0    0    0   0    0    0    0    0
                    0    0    0    0   0    0    0    0    0
                    0    0    0    0   0    0    0    0    0
                    0    0    0    0   0    0    0    0    0 ])
  first_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.75 0.0
                 0.0  0.0 ])
  first_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 6 0
               0 0 ])
  first_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 1
               1 71 ])
  first_intermediate_expected_lake_volumes::Array{Float64} = Float64[172800.0]
  # first_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    0    0    0    0    0    0
  #                   0    0    0    0   0    0    0    0    0
  #                   0    0    172800.0 0    0    0    0    0    0
  #                   0    0    0 0    0    0    0    0    0
  #                   0    0    0 0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0])

  second_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  second_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  second_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  second_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  second_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  second_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[    0    0    0    259200.0    0    0    0    0    0
                    0    0    0    259200.0   0    0    0    0    0
                    0    0    259200.0 259200.0    0    0    0    0    0
                    0    0    259200.0 259200.0    0    0    0    0    0
                    0    0    259200.0 259200.0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0 ])
  second_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 0.0
                 0.0 0.0 ])
  second_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 0
               0 0 ])
  second_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 1
               1 71 ])
  second_intermediate_expected_lake_volumes::Array{Float64} = Float64[259200.0]
  # second_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    0.0    0    0    0    0    0
  #                   0    0    0    0.0   0    0    0    0    0
  #                   0    0    14400.0 14400.0    0    0    0    0    0
  #                   0    0    14400.0 14400.0    0    0    0    0    0
  #                   0    0    14400.0 14400.0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0])

  third_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  third_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  third_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  third_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  third_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  third_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[    0    0    0    432000.0   0    0    0    0    0
                    0    0    0    432000.0   0    0    0    0    0
                    0    0    432000.0 432000.0    0    0    0    0    0
                    0    0    432000.0 432000.0    0    0    0    0    0
                    0    0    432000.0 432000.0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0 ])
  third_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 0.0
                 0.0 0.0  ])
  third_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 0
               0 0 ])
  third_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 1
               1 71 ])
  third_intermediate_expected_lake_volumes::Array{Float64} = Float64[432000.0]
  # third_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[ 0    0    0    0   0    0    0    0    0
  #                   0    0    0    0   0    0    0    0    0
  #                   0    0    43200.0 43200.0    0    0    0    0    0
  #                   0    0    43200.0 43200.0    0    0    0    0    0
  #                   0    0    43200.0 43200.0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0 ])

  fourth_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0
                                  0.0 0.0 0.0
                                  0.0 0.0 0.0 ])
  fourth_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0
                                  0.0 0.0 0.0
                                  0.0 0.0 0.0 ])
  fourth_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 86400.0 0.0
                                  0.0 0.0 0.0
                                  0.0 0.0 0.0 ])
  fourth_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 0.0
                 0.0 0.0 ])
  fourth_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 0
               0 0 ])
  fourth_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 1
               1 71 ])
  # fourth_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    5.0    0    0    0    0    0
  #                   0    0    0    5.0   0    0    0    0    0
  #                   0    0    8.0 8.0    0    0    0    0    0
  #                   0    0    8.0 8.0    0    0    0    0    0
  #                   0    0    8.0 8.0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0])

  fifth_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 1.0
                                  0.0 0.0 0.0
                                  0.0 0.0 0.0 ])
  fifth_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0
                                  0.0 0.0 0.0
                                  0.0 0.0 0.0 ])
  fifth_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 86400.0 0.0
                                  0.0 0.0 0.0
                                  0.0 0.0 0.0 ])
  fifth_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 0.0
                 0.0 0.0 ])
  fifth_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 0
               0 0 ])
  fifth_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 1
               1 71 ])
  # fifth_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    5.0    0    0    0    0    0
  #                   0    0    0    5.0   0    0    0    0    0
  #                   0    0    8.0 8.0    0    0    0    0    0
  #                   0    0    8.0 8.0    0    0    0    0    0
  #                   0    0    8.0 8.0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0])
  river_fields::RiverPrognosticFields,
    lake_model_prognostics::LakeModelPrognostics,
    lake_model_diagnostics::LakeModelDiagnostics =
    drive_hd_and_lake_model(river_parameters,lake_model_parameters,
                            lake_parameters_as_array,
                            drainages,runoffs,evaporations,
                            2,print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_fractions::Field{Float64} =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes::Array{Float64} = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes::Field{Float64} =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test first_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(first_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test first_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test first_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test first_intermediate_expected_lake_types == lake_types
  @test first_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == first_intermediate_expected_diagnostic_lake_volumes
  @test first_intermediate_expected_lake_fractions == lake_fractions
  @test first_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test first_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test first_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            3,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=2)
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test second_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(second_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test second_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test second_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test second_intermediate_expected_lake_types == lake_types
  @test second_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == second_intermediate_expected_diagnostic_lake_volumes
  @test isapprox(second_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.000001)
  @test second_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test second_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test second_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            5,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=3)
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test third_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(third_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test third_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test third_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test third_intermediate_expected_lake_types == lake_types
  @test third_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == third_intermediate_expected_diagnostic_lake_volumes
  @test third_intermediate_expected_lake_fractions == lake_fractions
  @test third_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test third_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test third_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            6,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=5)
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test fourth_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(fourth_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test fourth_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test expected_lake_types == lake_types
  @test expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  @test fourth_intermediate_expected_lake_fractions == lake_fractions
  @test fourth_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test fourth_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test fourth_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            7,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=6)
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test fifth_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(fifth_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test fifth_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test expected_lake_types == lake_types
  @test expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  @test fifth_intermediate_expected_lake_fractions == lake_fractions
  @test fifth_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test fifth_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test fifth_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,_ =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            9,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=7)
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test expected_river_inflow == river_fields.river_inflow
  @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test expected_lake_types == lake_types
  @test expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  @test expected_lake_fractions == lake_fractions
  @test expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test expected_true_lake_depths == lake_model_parameters.true_lake_depths
end

@testset "Lake model tests 7" begin
  #Detailed filling of single lake
  hd_grid = LatLonGrid(3,3,true)
  surface_model_grid = LatLonGrid(2,2,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
                                                Int64[-2 6 2
                                                       8 7 2
                                                       8 7 0 ] ))
  river_reservoir_nums = LatLonField{Int64}(hd_grid,5)
  overland_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  base_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  river_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  overland_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  base_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  landsea_mask = LatLonField{Bool}(hd_grid,fill(false,3,3))
  set!(river_reservoir_nums,LatLonCoords(3,3),0)
  set!(overland_reservoir_nums,LatLonCoords(3,3),0)
  set!(base_reservoir_nums,LatLonCoords(3,3),0)
  set!(landsea_mask,LatLonCoords(3,3),true)
  river_parameters = RiverParameters(flow_directions,
                                     river_reservoir_nums,
                                     overland_reservoir_nums,
                                     base_reservoir_nums,
                                     river_retention_coefficients,
                                     overland_retention_coefficients,
                                     base_retention_coefficients,
                                     landsea_mask,
                                     hd_grid,86400.0,86400.0)
  lake_grid = LatLonGrid(9,9,true)
  cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
    Float64[ 2.5 5.0
             3.0 4.0 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 2 2 2 1 2 2 2 2 2
                                                           2 2 2 1 2 2 2 2 2
                                                           2 2 1 1 2 2 2 2 2
                                                           2 2 1 1 2 2 2 2 2
                                                           2 2 1 1 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 1 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 2 2 2 1 2 2 2 2 2
                                                           2 2 2 1 2 2 2 2 2
                                                           2 2 1 1 2 2 2 2 2
                                                           2 2 1 1 2 2 2 2 2
                                                           2 2 1 1 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 1
                                                           2 2 2 2 2 2 2 2 2 ])
  is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
    Bool[false false false  true false false false false false
         false false false  true false false false false false
         false false  true  true false false false false false
         false false  true  true false false false false false
         false false  true  true false false false false false
         false false false false false false false false false
         false false false false false false false false false
         false false false false false false false false false
         false false false false false false false false false])
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_model_parameters::LakeModelParameters =
    LakeModelParameters(grid_specific_lake_model_parameters,
                        lake_grid,
                        hd_grid,
                        surface_model_grid,
                        cell_areas_on_surface_model_grid,
                        LakeModelSettings(),
                        1,
                        is_lake)
  lake_parameters_as_array::Vector{Float64} = Float64[1.0, 51.0, 1.0, -1.0, 0.0, 3.0, 3.0, 8.0, 3.0, 3.0, 1.0, 0.0, 2.0, 4.0, 4.0, 1.0, 0.0, 2.0, 3.0, 4.0, 1.0, 0.0, 2.0, 5.0, 3.0, 1.0, 0.0, 2.0, 4.0, 3.0, 1.0, 0.0, 2.0, 5.0, 4.0, 1.0, 172800.0, 5.0, 2.0, 4.0, 1.0, 172800.0, 5.0, 1.0, 4.0, 1.0, 432000.0, 8.375, 1.0, -1.0, 1.0, 2.0, 0.0]
  runoff::Field{Float64} = LatLonField{Float64}(river_parameters.grid,Float64[ 1.0 0.0 0.0
                                                                                 0.0 0.0 0.0
                                                                                 0.0 0.0 0.0 ])
  runoffs::Array{Field{Float64},1} = repeat(runoff,1000,false)
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
  drainages::Array{Field{Float64},1} = repeat(drainage,1000,false)
  evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,1000,false)
  expected_river_inflow::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                               Float64[ 0.0 0.0  1.0
                                                                        0.0 0.0  1.0
                                                                        0.0 0.0  0.0 ])
  expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                                 Float64[ 0.0 0.0  0.0
                                                                          0.0 0.0  0.0
                                                                          0.0 0.0  0.0 ])
  expected_water_to_hd::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                              Float64[ 0.0 86400.0 0.0
                                                                       0.0 0.0  0.0
                                                                       0.0 0.0  0.0 ])
  expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    2    0    0    0    0    0
                0    0    0    2    0    0    0    0    0
                0    0    2    2    0    0    0    0    0
                0    0    2    2    0    0    0    0    0
                0    0    2    2    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[    0    0    0    432000.0    0    0    0    0    0
                    0    0    0    432000.0   0    0    0    0    0
                    0    0    432000.0 432000.0    0    0    0    0    0
                    0    0    432000.0 432000.0    0    0    0    0    0
                    0    0    432000.0 432000.0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0 ])
  expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 0.0
                 0.0 0.0 ])
  expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 0
               0 0 ])
  expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 1
               1 71 ])
  expected_lake_volumes::Array{Float64} = Float64[432000.0]
  # expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    5.0    0    0    0    0    0
  #               0    0    0    5.0   0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0])

  first_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  first_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  first_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  first_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  first_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  first_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[    0    0    0    0   0    0    0    0    0
                    0    0    0    0   0    0    0    0    0
                    0    0    172800.0 172800.0 0 0 0 0    0
                    0    0    172800.0 172800.0 0 0 0 0    0
                    0    0    172800.0 172800.0 0 0 0 0    0
                    0    0    0    0   0    0    0    0    0
                    0    0    0    0   0    0    0    0    0
                    0    0    0    0   0    0    0    0    0
                    0    0    0    0   0    0    0    0    0 ])
  first_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.75 0.0
                 0.0 0.0 ])
  first_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 6 0
               0 0 ])
  first_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 1
               1 71 ])
  first_intermediate_expected_lake_volumes::Array{Float64} = Float64[172800.0]
  # first_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    0.0    0    0    0    0    0
  #               0    0    0    0.0   0    0    0    0    0
  #               0    0    172800.0 0.0    0    0    0    0    0
  #               0    0    0.0 0.0    0    0    0    0    0
  #               0    0    0.0 0.0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0])

  second_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  second_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  second_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  second_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  second_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  second_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[    0    0    0    259200.0 0   0    0    0    0
                    0    0    0    259200.0 0   0    0    0    0
                    0    0    259200.0 259200.0    0    0    0    0    0
                    0    0    259200.0 259200.0    0    0    0    0    0
                    0    0    259200.0 259200.0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0 ])
  second_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 0.0
                 0.0 0.0 ])
  second_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 0
               0 0 ])
  second_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 1
               1 71 ])
  second_intermediate_expected_lake_volumes::Array{Float64} = Float64[259200.0]
  # second_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[ 0    0    0    0.0    0    0    0    0    0
  #                   0    0    0    0.0   0    0    0    0    0
  #                   0    0    14400.0 14400.0    0    0    0    0    0
  #                   0    0    14400.0 14400.0    0    0    0    0    0
  #                   0    0    14400.0 14400.0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0])

  third_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  third_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  third_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  third_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  third_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  third_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[    0    0    0    432000.0   0    0    0    0    0
                    0    0    0    432000.0   0    0    0    0    0
                    0    0    432000.0 432000.0    0    0    0    0    0
                    0    0    432000.0 432000.0    0    0    0    0    0
                    0    0    432000.0 432000.0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0 ])
  third_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[  1.0 0.0
                  0.0  0.0 ])
  third_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 0
               0 0 ])
  third_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 1
               1 71  ])
  third_intermediate_expected_lake_volumes::Array{Float64} = Float64[432000.0]
  # third_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    0.0    0    0    0    0    0
  #               0    0    0    0.0   0    0    0    0    0
  #               0    0    43200.0 43200.0    0    0    0    0    0
  #               0    0    43200.0 43200.0    0    0    0    0    0
  #               0    0    43200.0 43200.0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0])

  fourth_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0
                                  0.0 0.0 0.0
                                  0.0 0.0 0.0 ])
  fourth_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0
                                  0.0 0.0 0.0
                                  0.0 0.0 0.0 ])
  fourth_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 86400.0 0.0
                                  0.0 0.0 0.0
                                  0.0 0.0 0.0 ])
  fourth_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0  0.0
                 0.0  0.0 ])
  fourth_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 0
               0 0 ])
  fourth_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 1
               1 71  ])
  # fourth_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    5.0    0    0    0    0    0
  #               0    0    0    5.0   0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0])

  fifth_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 1.0
                                  0.0 0.0 0.0
                                  0.0 0.0 0.0 ])
  fifth_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0
                                  0.0 0.0 0.0
                                  0.0 0.0 0.0 ])
  fifth_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 86400.0 0.0
                                  0.0 0.0 0.0
                                  0.0 0.0 0.0 ])
  fifth_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0  0.0
                 0.0  0.0 ])
  fifth_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 0
               0 0 ])
  fifth_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 1
               1 71  ])
  # fifth_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    5.0    0    0    0    0    0
  #               0    0    0    5.0   0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0])
  river_fields::RiverPrognosticFields,
    lake_model_prognostics::LakeModelPrognostics,
    lake_model_diagnostics::LakeModelDiagnostics =
    drive_hd_and_lake_model(river_parameters,lake_model_parameters,
                            lake_parameters_as_array,
                            drainages,runoffs,evaporations,
                            2,print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_fractions::Field{Float64} =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes::Array{Float64} = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes::Field{Float64}  =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test first_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(first_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test first_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test first_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test first_intermediate_expected_lake_types == lake_types
  @test first_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == first_intermediate_expected_diagnostic_lake_volumes
  @test first_intermediate_expected_lake_fractions == lake_fractions
  @test first_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test first_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test first_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            3,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=2)
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test second_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(second_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test second_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test second_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test second_intermediate_expected_lake_types == lake_types
  @test second_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == second_intermediate_expected_diagnostic_lake_volumes
  @test second_intermediate_expected_lake_fractions == lake_fractions
  @test second_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test second_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test second_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            5,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=3)
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test third_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(third_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test third_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test third_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test third_intermediate_expected_lake_types == lake_types
  @test third_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == third_intermediate_expected_diagnostic_lake_volumes
  @test third_intermediate_expected_lake_fractions == lake_fractions
  @test third_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test third_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test third_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            6,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=5)
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test fourth_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(fourth_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test fourth_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test expected_lake_types == lake_types
  @test expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  @test fourth_intermediate_expected_lake_fractions == lake_fractions
  @test fourth_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test fourth_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test fourth_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            7,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=6)
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test fifth_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(fifth_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test fifth_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test expected_lake_types == lake_types
  @test expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  @test fifth_intermediate_expected_lake_fractions == lake_fractions
  @test fifth_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test fifth_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test fifth_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            8,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=7)
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test expected_river_inflow == river_fields.river_inflow
  @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test expected_lake_types == lake_types
  @test expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  @test expected_lake_fractions == lake_fractions
  @test expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test expected_true_lake_depths == lake_model_parameters.true_lake_depths
end

@testset "Lake model tests 8" begin
  #Detailed filling and draining of a single lake
  hd_grid = LatLonGrid(3,3,true)
  surface_model_grid = LatLonGrid(2,2,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
                                                Int64[-2 6 2
                                                       8 7 2
                                                       8 7 0 ] ))
  river_reservoir_nums = LatLonField{Int64}(hd_grid,5)
  overland_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  base_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  river_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  overland_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  base_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  landsea_mask = LatLonField{Bool}(hd_grid,fill(false,3,3))
  set!(river_reservoir_nums,LatLonCoords(3,3),0)
  set!(overland_reservoir_nums,LatLonCoords(3,3),0)
  set!(base_reservoir_nums,LatLonCoords(3,3),0)
  set!(landsea_mask,LatLonCoords(3,3),true)
  river_parameters = RiverParameters(flow_directions,
                                     river_reservoir_nums,
                                     overland_reservoir_nums,
                                     base_reservoir_nums,
                                     river_retention_coefficients,
                                     overland_retention_coefficients,
                                     base_retention_coefficients,
                                     landsea_mask,
                                     hd_grid,86400.0,86400.0)
  lake_grid = LatLonGrid(9,9,true)
  cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
    Float64[ 2.5 3.5
             3.0 4.0 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 2 2 2 1 2 2 2 2 2
                                                           2 2 2 1 2 2 2 2 2
                                                           2 2 1 1 2 2 2 2 2
                                                           2 2 1 1 2 2 2 2 2
                                                           2 2 1 1 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 1 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 2 2 2 1 2 2 2 2 2
                                                           2 2 2 1 2 2 2 2 2
                                                           2 2 1 1 2 2 2 2 2
                                                           2 2 1 1 2 2 2 2 2
                                                           2 2 1 1 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 1
                                                           2 2 2 2 2 2 2 2 2 ])
  is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
    Bool[false false false  true false false false false false
         false false false  true false false false false false
         false false  true  true false false false false false
         false false  true  true false false false false false
         false false  true  true false false false false false
         false false false false false false false false false
         false false false false false false false false false
         false false false false false false false false false
         false false false false false false false false false])
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_model_parameters::LakeModelParameters =
    LakeModelParameters(grid_specific_lake_model_parameters,
                        lake_grid,
                        hd_grid,
                        surface_model_grid,
                        cell_areas_on_surface_model_grid,
                        LakeModelSettings(),
                        1,
                        is_lake)
  lake_parameters_as_array::Vector{Float64} = Float64[1.0, 51.0, 1.0, -1.0, 0.0, 3.0, 3.0, 8.0, 3.0, 3.0, 1.0, 0.0, 2.0, 4.0, 4.0, 1.0, 0.0, 2.0, 3.0, 4.0, 1.0, 0.0, 2.0, 5.0, 3.0, 1.0, 0.0, 2.0, 4.0, 3.0, 1.0, 0.0, 2.0, 5.0, 4.0, 1.0, 172800.0, 5.0, 2.0, 4.0, 1.0, 172800.0, 5.0, 1.0, 4.0, 1.0, 432000.0, 8.375, 1.0, -1.0, 1.0, 2.0, 0.0]
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,Float64[ 2.0 0.0 0.0
                                                                                 0.0 0.0 0.0
                                                                                 0.0 0.0 0.0 ])
  drainages::Array{Field{Float64},1} = repeat(drainage,1,false)
  drainage = LatLonField{Float64}(river_parameters.grid,Float64[ 0.0 0.0 0.0
                                                                 0.0 0.0 0.0
                                                                 0.0 0.0 0.0 ])
  drainages_two::Array{Field{Float64},1} = repeat(drainage,999,false)
  drainages = vcat(drainages,drainages_two)
  runoff::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
  runoffs::Array{Field{Float64},1} = repeat(runoff,1000,false)
  evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,Float64[ 86400.0 0.0
                                                                                     0.0 0.0 ])
  evaporations::Array{Field{Float64},1} = repeat(evaporation,1000,false)
  expected_river_inflow::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                               Float64[ 0.0 0.0  0.0
                                                                        0.0 0.0  0.0
                                                                        0.0 0.0  0.0 ])
  expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                                 Float64[ -1.0 0.0  0.0
                                                                          0.0 0.0  0.0
                                                                          0.0 0.0  0.0 ])
  expected_water_to_hd::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                              Float64[ 0.0 0.0  0.0
                                                                       0.0 0.0  0.0
                                                                       0.0 0.0  0.0 ])
  expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    1    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[      0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    1    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0 ])
  expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0 ])
  expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.125 0.0
                 0.0 0.0 ])
  expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 1 0
               0 0 ])
  expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8  1
               1 71 ])
  expected_lake_volumes::Array{Float64} = Float64[0.0]
  # expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    0   0    0    0    0    0
  #               0    0    0    0   0    0    0    0    0
  #               0    0    0    0   0    0    0    0    0
  #               0    0    0    0   0    0    0    0    0
  #               0    0    0    0   0    0    0    0    0
  #               0    0    0    0   0    0    0    0    0
  #               0    0    0    0   0    0    0    0    0
  #               0    0    0    0   0    0    0    0    0
  #               0    0    0    0   0    0    0    0    0])

  first_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  first_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  first_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  first_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  first_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  first_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[    0    0    0    0   0    0    0    0    0
                    0    0    0    0   0    0    0    0    0
                    0    0    172800.0 172800.0 0 0 0 0    0
                    0    0    172800.0 172800.0 0 0 0 0    0
                    0    0    172800.0 172800.0 0 0 0 0    0
                    0    0    0    0   0    0    0    0    0
                    0    0    0    0   0    0    0    0    0
                    0    0    0    0   0    0    0    0    0
                    0    0    0    0   0    0    0    0    0 ])
  first_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.75 0.0
                 0.0 0.0 ])
  first_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 6 0
               0 0 ])
  first_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8  1
               1 71])
  first_intermediate_expected_lake_volumes::Array{Float64} = Float64[172800.0]
  # first_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    0   0    0    0    0    0
  #               0    0    0    0   0    0    0    0    0
  #               0    0    172800.0 172800.0 0 0 0 0    0
  #               0    0    172800.0 172800.0 0 0 0 0    0
  #               0    0    172800.0 172800.0 0 0 0 0    0
  #               0    0    0    0   0    0    0    0    0
  #               0    0    0    0   0    0    0    0    0
  #               0    0    0    0   0    0    0    0    0
  #               0    0    0    0   0    0    0    0    0])

  second_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  second_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  second_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  second_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  second_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  second_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[    0    0    0    259200.0    0    0    0    0    0
                    0    0    0    259200.0   0    0    0    0    0
                    0    0    259200.0 259200.0    0    0    0    0    0
                    0    0    259200.0 259200.0    0    0    0    0    0
                    0    0    259200.0 259200.0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0 ])
  second_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 0.0
                 0.0  0.0 ])
  second_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 0
               0 0 ])
  second_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8  1
               1 71 ])
  second_intermediate_expected_lake_volumes::Array{Float64} = Float64[259200.0]
  # second_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    0    0    0    0    0    0
  #               0    0    0    0   0    0    0    0    0
  #               0    0    14400.0 14400.0    0    0    0    0    0
  #               0    0    14400.0 14400.0    0    0    0    0    0
  #               0    0    14400.0 14400.0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0])

  third_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  third_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  third_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  third_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  third_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  third_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[    0    0    0    432000.0   0    0    0    0    0
                    0    0    0    432000.0   0    0    0    0    0
                    0    0    432000.0 432000.0    0    0    0    0    0
                    0    0    432000.0 432000.0    0    0    0    0    0
                    0    0    432000.0 432000.0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0 ])
  third_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 0.0
                 0.0  0.0 ])
  third_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 0
               0 0 ])
  third_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8  1
               1 71 ])
  third_intermediate_expected_lake_volumes::Array{Float64} = Float64[432000.0]
  # third_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[ 0    0    0    0   0    0    0    0    0
  #                   0    0    0    0   0    0    0    0    0
  #                   0    0    43200.0 43200.0    0    0    0    0    0
  #                   0    0    43200.0 43200.0    0    0    0    0    0
  #                   0    0    43200.0 43200.0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0  ])

  fourth_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0
                                  0.0 0.0 0.0
                                  0.0 0.0 0.0 ])
  fourth_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0
                                  0.0 0.0 0.0
                                  0.0 0.0 0.0 ])
  fourth_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 86400.0 0.0
                                  0.0 0.0 0.0
                                  0.0 0.0 0.0 ])
  fourth_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 0.0
                 0.0 0.0 ])
  fourth_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 0
               0 0 ])
  fourth_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8  1
               1 71 ])
  # fourth_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[ 0    0    0    5.0    0    0    0    0    0
  #               0    0    0    5.0   0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0 ])

  fifth_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 1.0
                                  0.0 0.0 0.0
                                  0.0 0.0 0.0 ])
  fifth_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0
                                  0.0 0.0 0.0
                                  0.0 0.0 0.0 ])
  fifth_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 86400.0 0.0
                                  0.0 0.0 0.0
                                  0.0 0.0 0.0 ])
  fifth_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 0.0
                 0.0 0.0 ])
  fifth_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 0
               0 0 ])
  fifth_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8  1
               1 71 ])
  # fifth_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    5.0    0    0    0    0    0
  #               0    0    0    5.0   0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0])

  sixth_intermediate_expected_river_inflow = LatLonField{Float64}(hd_grid,
                                                                  Float64[ 0.0 0.0 1.0
                                                                           0.0 0.0 1.0
                                                                           0.0 0.0 0.0 ])
  sixth_intermediate_expected_water_to_ocean = LatLonField{Float64}(hd_grid,
                                                                   Float64[ 0.0 0.0 0.0
                                                                            0.0 0.0 0.0
                                                                            0.0 0.0 0.0 ])
  sixth_intermediate_expected_water_to_ocean_variant_two = LatLonField{Float64}(hd_grid,
                                                                   Float64[ 0.0 0.0 0.0
                                                                            0.0 0.0 0.0
                                                                            0.0 0.0 1.0 ])
  sixth_intermediate_expected_water_to_hd = LatLonField{Float64}(hd_grid,
                                                                 Float64[ 0.0 86400.0 0.0
                                                                          0.0 0.0 0.0
                                                                          0.0 0.0 0.0 ])
  sixth_intermediate_expected_lake_numbers = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  sixth_intermediate_expected_lake_types = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    2    0    0    0    0    0
                  0    0    0    2    0    0    0    0    0
                  0    0    2    2    0    0    0    0    0
                  0    0    2    2    0    0    0    0    0
                  0    0    2    2    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0 ])
  sixth_intermediate_expected_diagnostic_lake_volumes =
  LatLonField{Float64}(lake_grid,
        Float64[  0    0    0    432000.0    0    0    0    0    0
                  0    0    0    432000.0   0    0    0    0    0
                  0    0    432000.0 432000.0    0    0    0    0    0
                  0    0    432000.0 432000.0    0    0    0    0    0
                  0    0    432000.0 432000.0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0 ])
  sixth_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0  0.0
                 0.0  0.0 ])
  sixth_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 0
               0 0 ])
  sixth_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8  1
               1 71 ])
  sixth_intermediate_expected_lake_volumes = Float64[432000.0]
  # sixth_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    5.0    0    0    0    0    0
  #               0    0    0    5.0   0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0])

  seven_intermediate_expected_river_inflow = LatLonField{Float64}(hd_grid,
                                                                  Float64[ 0.0 0.0 1.0
                                                                           0.0 0.0 1.0
                                                                           0.0 0.0 0.0 ])
  seven_intermediate_expected_water_to_ocean = LatLonField{Float64}(hd_grid,
                                                                   Float64[ 0.0 0.0 0.0
                                                                            0.0 0.0 0.0
                                                                            0.0 0.0 1.0 ])
  seven_intermediate_expected_water_to_hd = LatLonField{Float64}(hd_grid,
                                                                 Float64[ 0.0 0.0 0.0
                                                                          0.0 0.0 0.0
                                                                          0.0 0.0 0.0 ])
  seven_intermediate_expected_lake_numbers = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  seven_intermediate_expected_lake_types = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  seven_intermediate_expected_diagnostic_lake_volumes =
  LatLonField{Float64}(lake_grid,
        Float64[  0    0    0    345600.0    0    0    0    0    0
                  0    0    0    345600.0    0    0    0    0    0
                  0    0    345600.0 345600.0    0    0    0    0    0
                  0    0    345600.0 345600.0    0    0    0    0    0
                  0    0    345600.0 345600.0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0 ])
  seven_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0  0.0
                 0.0   0.0 ])
  seven_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 0
               0 0 ])
  seven_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8  1
               1 71 ])
  seven_intermediate_expected_lake_volumes = Float64[345600.0]
  # seven_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    0   0    0    0    0    0
  #                   0    0    0    0   0    0    0    0    0
  #                   0    0    28800.0  28800.0    0    0    0    0    0
  #                   0    0    28800.0  28800.0    0    0    0    0    0
  #                   0    0    28800.0  28800.0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0])

  eight_intermediate_expected_river_inflow = LatLonField{Float64}(hd_grid,
                                                                  Float64[ 0.0 0.0 0.0
                                                                           0.0 0.0 1.0
                                                                           0.0 0.0 0.0 ])
  eight_intermediate_expected_water_to_ocean = LatLonField{Float64}(hd_grid,
                                                                   Float64[ 0.0 0.0 0.0
                                                                            0.0 0.0 0.0
                                                                            0.0 0.0 1.0 ])
  eight_intermediate_expected_water_to_hd = LatLonField{Float64}(hd_grid,
                                                                 Float64[ 0.0 0.0 0.0
                                                                          0.0 0.0 0.0
                                                                          0.0 0.0 0.0 ])
  eight_intermediate_expected_lake_numbers = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  eight_intermediate_expected_lake_types = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  eight_intermediate_expected_diagnostic_lake_volumes =
  LatLonField{Float64}(lake_grid,
        Float64[  0    0    0    259200.0   0    0    0    0    0
                  0    0    0    259200.0   0    0    0    0    0
                  0    0    259200.0 259200.0    0    0    0    0    0
                  0    0    259200.0 259200.0    0    0    0    0    0
                  0    0    259200.0 259200.0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0 ])
  eight_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 0.0
                 0.0  0.0 ])
  eight_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 0
               0 0 ])
  eight_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8  1
               1 71 ])
  eight_intermediate_expected_lake_volumes = Float64[259200.0]
  # eight_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    0   0    0    0    0    0
  #                   0    0    0    0   0    0    0    0    0
  #                   0    0    14400.0 14400.0    0    0    0    0    0
  #                   0    0    14400.0 14400.0    0    0    0    0    0
  #                   0    0    14400.0 14400.0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0])

  nine_intermediate_expected_river_inflow = LatLonField{Float64}(hd_grid,
                                                                 Float64[ 0.0 0.0 0.0
                                                                          0.0 0.0 0.0
                                                                          0.0 0.0 0.0 ])
  nine_intermediate_expected_water_to_ocean = LatLonField{Float64}(hd_grid,
                                                                   Float64[ 0.0 0.0 0.0
                                                                            0.0 0.0 0.0
                                                                            0.0 0.0 1.0 ])
  nine_intermediate_expected_water_to_hd = LatLonField{Float64}(hd_grid,
                                                                Float64[ 0.0 0.0 0.0
                                                                         0.0 0.0 0.0
                                                                         0.0 0.0 0.0 ])
  nine_intermediate_expected_lake_numbers = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  nine_intermediate_expected_lake_types = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  nine_intermediate_expected_diagnostic_lake_volumes =
  LatLonField{Float64}(lake_grid,
        Float64[  0    0    0    172800.0   0    0    0    0    0
                  0    0    0    172800.0   0    0    0    0    0
                  0    0    172800.0 172800.0    0    0    0    0    0
                  0    0    172800.0 172800.0    0    0    0    0    0
                  0    0    172800.0 172800.0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0 ])
  nine_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 0.0
                 0.0  0.0 ])
  nine_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8 0
               0 0 ])
  nine_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8  1
               1 71 ])
  nine_intermediate_expected_lake_volumes = Float64[172800.0]
  # nine_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    0   0    0    0    0    0
  #                   0    0    0    0   0    0    0    0    0
  #                   0    0    0.0 0.0    0    0    0    0    0
  #                   0    0    0.0 0.0    0    0    0    0    0
  #                   0    0    0.0 0.0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0])

  ten_intermediate_expected_river_inflow = LatLonField{Float64}(hd_grid,
                                                                Float64[ 0.0 0.0 0.0
                                                                         0.0 0.0 0.0
                                                                         0.0 0.0 0.0 ])
  ten_intermediate_expected_water_to_ocean = LatLonField{Float64}(hd_grid,
                                                                  Float64[ 0.0 0.0 0.0
                                                                           0.0 0.0 0.0
                                                                           0.0 0.0 0.0 ])
  ten_intermediate_expected_water_to_hd = LatLonField{Float64}(hd_grid,
                                                               Float64[ 0.0 0.0 0.0
                                                                        0.0 0.0 0.0
                                                                        0.0 0.0 0.0 ])
  ten_intermediate_expected_lake_numbers = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  ten_intermediate_expected_lake_types = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  ten_intermediate_expected_diagnostic_lake_volumes =
  LatLonField{Float64}(lake_grid,
        Float64[  0    0    0    0   0    0    0    0    0
                  0    0    0    0   0    0    0    0    0
                  0    0    86400.0 86400.0 0 0 0   0    0
                  0    0    86400.0 86400.0 0 0 0   0    0
                  0    0    86400.0 86400.0 0 0 0   0    0
                  0    0    0    0    0    0    0   0    0
                  0    0    0    0    0    0    0   0    0
                  0    0    0    0    0    0    0   0    0
                  0    0    0    0    0    0    0   0    0 ])
  ten_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.75 0.0
                 0.0   0.0 ])
  ten_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 6 0
               0 0 ])
  ten_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8  1
               1 71 ])
  ten_intermediate_expected_lake_volumes = Float64[86400.0]
  # ten_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    0   0    0    0    0    0
  #                 0    0    0    0   0    0    0    0    0
  #                 0    0    86400.0  0    0    0    0    0    0
  #                 0    0    0    0    0    0    0    0    0
  #                 0    0    0    0    0    0    0    0    0
  #                 0    0    0    0    0    0    0    0    0
  #                 0    0    0    0    0    0    0    0    0
  #                 0    0    0    0    0    0    0    0    0
  #                 0    0    0    0    0    0    0    0    0])

  eleven_intermediate_expected_river_inflow = LatLonField{Float64}(hd_grid,
                                                                Float64[ 0.0 0.0 0.0
                                                                         0.0 0.0 0.0
                                                                         0.0 0.0 0.0 ])
  eleven_intermediate_expected_water_to_ocean = LatLonField{Float64}(hd_grid,
                                                                  Float64[ 0.0 0.0 0.0
                                                                           0.0 0.0 0.0
                                                                           0.0 0.0 0.0 ])
  eleven_intermediate_expected_water_to_hd = LatLonField{Float64}(hd_grid,
                                                               Float64[ 0.0 0.0 0.0
                                                                        0.0 0.0 0.0
                                                                        0.0 0.0 0.0 ])
  eleven_intermediate_expected_lake_numbers = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    1    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  eleven_intermediate_expected_lake_types = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    1    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  eleven_intermediate_expected_diagnostic_lake_volumes =
  LatLonField{Float64}(lake_grid,
        Float64[  0    0    0    0   0    0    0    0    0
                  0    0    0    0   0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0 ])
  eleven_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.125 0.0
                 0.0   0.0 ])
  eleven_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 1 0
               0 0 ])
  eleven_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8  1
               1 71 ])
  eleven_intermediate_expected_lake_volumes = Float64[0.0]
  # eleven_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    0   0    0    0    0    0
  #                   0    0    0    0   0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0])

  twelve_intermediate_expected_river_inflow = LatLonField{Float64}(hd_grid,
                                                                Float64[ 0.0 0.0 0.0
                                                                         0.0 0.0 0.0
                                                                         0.0 0.0 0.0 ])
  twelve_intermediate_expected_water_to_ocean = LatLonField{Float64}(hd_grid,
                                                                  Float64[ 0.0 0.0 0.0
                                                                           0.0 0.0 0.0
                                                                           0.0 0.0 0.0 ])
  twelve_intermediate_expected_water_to_hd = LatLonField{Float64}(hd_grid,
                                                               Float64[ 0.0 0.0 0.0
                                                                        0.0 0.0 0.0
                                                                        0.0 0.0 0.0 ])
  twelve_intermediate_expected_lake_numbers = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    1    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  twelve_intermediate_expected_lake_types = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    1    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  twelve_intermediate_expected_diagnostic_lake_volumes =
  LatLonField{Float64}(lake_grid,
        Float64[  0    0    0    0   0    0    0    0    0
                  0    0    0    0   0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0 ])
  twelve_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.125 0.0
                 0.0   0.0 ])
  twelve_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 1 0
               0 0 ])
  twelve_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 8  1
               1 71 ])
  twelve_intermediate_expected_lake_volumes = Float64[0.0]
  # twelve_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    0   0    0    0    0    0
  #                   0    0    0    0   0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0
  #                   0    0    0    0    0    0    0    0    0])
  river_fields::RiverPrognosticFields,
    lake_model_prognostics::LakeModelPrognostics,
    lake_model_diagnostics::LakeModelDiagnostics =
    drive_hd_and_lake_model(river_parameters,lake_model_parameters,
                            lake_parameters_as_array,
                            drainages,runoffs,evaporations,
                            2,print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes::Array{Float64} = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes::Field{Float64} =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test first_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(first_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test first_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test first_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test first_intermediate_expected_lake_types == lake_types
  @test first_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == first_intermediate_expected_diagnostic_lake_volumes
  @test first_intermediate_expected_lake_fractions == lake_fractions
  @test first_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test first_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test first_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            3,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=2)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test second_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(second_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test second_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test second_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test second_intermediate_expected_lake_types == lake_types
  @test second_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == second_intermediate_expected_diagnostic_lake_volumes
  @test second_intermediate_expected_lake_fractions == lake_fractions
  @test second_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test second_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test second_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            5,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=3)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test third_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(third_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test third_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test third_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test third_intermediate_expected_lake_types == lake_types
  @test third_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == third_intermediate_expected_diagnostic_lake_volumes
  @test third_intermediate_expected_lake_fractions == lake_fractions
  @test third_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test third_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test third_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            6,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=5)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test fourth_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(fourth_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test fourth_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test sixth_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test sixth_intermediate_expected_lake_types == lake_types
  @test sixth_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes ==
        sixth_intermediate_expected_diagnostic_lake_volumes
  @test fourth_intermediate_expected_lake_fractions == lake_fractions
  @test fourth_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test fourth_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test fourth_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            7,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=6)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test fifth_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(fifth_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test fifth_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test sixth_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test sixth_intermediate_expected_lake_types == lake_types
  @test sixth_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == sixth_intermediate_expected_diagnostic_lake_volumes
  @test fifth_intermediate_expected_lake_fractions == lake_fractions
  @test fifth_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test fifth_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test fifth_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            8,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=7)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test sixth_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(sixth_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test sixth_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test sixth_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test sixth_intermediate_expected_lake_types == lake_types
  @test sixth_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes ==
        sixth_intermediate_expected_diagnostic_lake_volumes
  @test sixth_intermediate_expected_lake_fractions == lake_fractions
  @test sixth_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test sixth_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test sixth_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            30,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=8)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test sixth_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(sixth_intermediate_expected_water_to_ocean_variant_two ,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test sixth_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test sixth_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test sixth_intermediate_expected_lake_types == lake_types
  @test sixth_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes ==
        sixth_intermediate_expected_diagnostic_lake_volumes
  @test sixth_intermediate_expected_lake_fractions == lake_fractions
  @test sixth_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test sixth_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test sixth_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            31,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=30)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test seven_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(seven_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test seven_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test seven_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test seven_intermediate_expected_lake_types == lake_types
  @test seven_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes ==
        seven_intermediate_expected_diagnostic_lake_volumes
  @test seven_intermediate_expected_lake_fractions == lake_fractions
  @test seven_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test seven_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test seven_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            32,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=31)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test eight_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(eight_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test eight_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test eight_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test eight_intermediate_expected_lake_types == lake_types
  @test eight_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes ==
        eight_intermediate_expected_diagnostic_lake_volumes
  @test eight_intermediate_expected_lake_fractions == lake_fractions
  @test eight_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test eight_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test eight_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            33,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=32)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test nine_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(nine_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test nine_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test nine_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test nine_intermediate_expected_lake_types == lake_types
  @test nine_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes ==
        nine_intermediate_expected_diagnostic_lake_volumes
  @test nine_intermediate_expected_lake_fractions == lake_fractions
  @test nine_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test nine_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test nine_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            34,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=33)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test ten_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(ten_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test ten_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test ten_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test ten_intermediate_expected_lake_types == lake_types
  @test ten_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes ==
        ten_intermediate_expected_diagnostic_lake_volumes
  @test ten_intermediate_expected_lake_fractions == lake_fractions
  @test ten_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test ten_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test ten_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            35,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=34)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test eleven_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(eleven_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test eleven_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test eleven_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test eleven_intermediate_expected_lake_types == lake_types
  @test eleven_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes ==
        eleven_intermediate_expected_diagnostic_lake_volumes
  @test eleven_intermediate_expected_lake_fractions == lake_fractions
  @test eleven_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test eleven_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test eleven_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            36,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=35)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test twelve_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(twelve_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test twelve_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test twelve_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test twelve_intermediate_expected_lake_types == lake_types
  @test twelve_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes ==
        twelve_intermediate_expected_diagnostic_lake_volumes
  @test twelve_intermediate_expected_lake_fractions == lake_fractions
  @test twelve_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test twelve_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test twelve_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,_ =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            37,print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=36)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test expected_river_inflow == river_fields.river_inflow
  @test isapprox(expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test expected_lake_types == lake_types
  @test expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes ==
        expected_diagnostic_lake_volumes
  @test expected_lake_fractions == lake_fractions
  @test expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test expected_true_lake_depths == lake_model_parameters.true_lake_depths
end

@testset "Lake model tests 9" begin
  #Initial water to a single lake
  hd_grid = LatLonGrid(3,3,true)
  surface_model_grid = LatLonGrid(3,3,true)
  seconds_per_day::Float64 = 86400.0
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
                                                Int64[-2 6 2
                                                       8 7 2
                                                       8 7 0 ] ))
  river_reservoir_nums = LatLonField{Int64}(hd_grid,5)
  overland_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  base_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  river_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  overland_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  base_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  landsea_mask = LatLonField{Bool}(hd_grid,fill(false,3,3))
  set!(river_reservoir_nums,LatLonCoords(3,3),0)
  set!(overland_reservoir_nums,LatLonCoords(3,3),0)
  set!(base_reservoir_nums,LatLonCoords(3,3),0)
  set!(landsea_mask,LatLonCoords(3,3),true)
  river_parameters = RiverParameters(flow_directions,
                                     river_reservoir_nums,
                                     overland_reservoir_nums,
                                     base_reservoir_nums,
                                     river_retention_coefficients,
                                     overland_retention_coefficients,
                                     base_retention_coefficients,
                                     landsea_mask,
                                     hd_grid,86400.0,86400.0)
  lake_grid = LatLonGrid(9,9,true)
  cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
    Float64[ 2.5 3.0 2.5
             3.0 4.0 3.0
             2.5 3.0 2.5 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 1 1 1 1 1 1 1 1 1
                                                           1 1 1 1 1 1 1 1 1
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           2 2 2 2 2 2 2 2 2
                                                           3 3 3 3 3 3 3 3 3
                                                           3 3 3 3 3 3 3 3 3
                                                           3 3 3 3 3 3 3 3 3 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 1 1 1 2 2 2 3 3 3
                                                           1 1 1 2 2 2 3 3 3
                                                           1 1 1 2 2 2 3 3 3
                                                           1 1 1 2 2 2 3 3 3
                                                           1 1 1 2 2 2 3 3 3
                                                           1 1 1 2 2 2 3 3 3
                                                           1 1 1 2 2 2 3 3 3
                                                           1 1 1 2 2 2 3 3 3
                                                           1 1 1 2 2 2 3 3 3 ])
  is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
    Bool[false false false  true false false false false false
         false false false  true false false false false false
         false false  true  true false false false false false
         false false  true  true false false false false false
         false false  true  true false false false false false
         false false false false false false false false false
         false false false false false false false false false
         false false false false false false false false false
         false false false false false false false false false])
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_model_parameters::LakeModelParameters =
    LakeModelParameters(grid_specific_lake_model_parameters,
                        lake_grid,
                        hd_grid,
                        surface_model_grid,
                        cell_areas_on_surface_model_grid,
                        LakeModelSettings(),
                        1,
                        is_lake)
  lake_parameters_as_array::Vector{Float64} = Float64[1.0, 51.0, 1.0, -1.0, 0.0, 3.0, 3.0, 8.0, 3.0, 3.0, 1.0, 0.0, 2.0, 4.0, 4.0, 1.0, 0.0, 2.0, 3.0, 4.0, 1.0, 0.0, 2.0, 5.0, 3.0, 1.0, 0.0, 2.0, 4.0, 3.0, 1.0, 0.0, 2.0, 5.0, 4.0, 1.0, 172800.0, 5.0, 2.0, 4.0, 1.0, 172800.0, 5.0, 1.0, 4.0, 1.0, 432000.0, 8.375, 1.0, -1.0, 1.0, 2.0, 0.0]
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
  drainages::Array{Field{Float64},1} = repeat(drainage,1000,false)
  runoff::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
  runoffs::Array{Field{Float64},1} = repeat(runoff,1000,false)
  evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,1000,false)
  high_lake_volume::Float64 = 5.0*seconds_per_day
  initial_water_to_lake_centers = LatLonField{Float64}(lake_grid,
    Float64[ 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0
             0 0 4*high_lake_volume 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 ])
  initial_spillover_to_rivers = LatLonField{Float64}(hd_grid,
                                                     Float64[ 0.0 0.0  86400.0
                                                              0.0 0.0  0.0
                                                              0.0 0.0  0.0 ])
  expected_river_inflow::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                               Float64[ 0.0 0.0  0.0
                                                                        0.0 0.0  15.0
                                                                        0.0 0.0  0.0 ])
  expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                                 Float64[ 0.0 0.0  0.0
                                                                          0.0 0.0  0.0
                                                                          0.0 0.0  1.0 ])
  expected_water_to_hd::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                              Float64[ 0.0 0.0  0.0
                                                                       0.0 0.0  0.0
                                                                       0.0 0.0  0.0 ])
  expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    2    0    0    0    0    0
                  0    0    0    2    0    0    0    0    0
                  0    0    2    2    0    0    0    0    0
                  0    0    2    2    0    0    0    0    0
                  0    0    2    2    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0 ])
  expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[  0    0    0    432000.0    0    0    0    0    0
                  0    0    0    432000.0   0    0    0    0    0
                  0    0    432000.0 432000.0    0    0    0    0    0
                  0    0    432000.0 432000.0    0    0    0    0    0
                  0    0    432000.0 432000.0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0
                  0    0    0    0    0    0    0    0    0 ])
  expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.0 1.0/3.0 0.0
                 0.25     0.25 0.0
                 0.0     0.0 0.0 ])
  expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 0 2 0
               3 3 0
               0 0 0 ])
  expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 6 6 6
              12 12 12
               9 9 9 ])
  expected_lake_volumes::Array{Float64} = Float64[432000.0]
  # expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    5.0    0    0    0    0    0
  #               0    0    0    5.0   0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0])
  first_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  first_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  first_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 1296000.0 86400.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  first_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  first_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    2    0    0    0    0    0
                0    0    0    2    0    0    0    0    0
                0    0    2    2    0    0    0    0    0
                0    0    2    2    0    0    0    0    0
                0    0    2    2    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  first_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[    0    0    0    432000.0   0    0    0    0    0
                    0    0    0    432000.0   0    0    0    0    0
                    0    0    432000.0 432000.0 0  0    0    0    0
                    0    0    432000.0 432000.0 0    0    0    0    0
                    0    0    432000.0 432000.0  0    0    0    0    0
                    0    0    0    0   0    0    0    0    0
                    0    0    0    0   0    0    0    0    0
                    0    0    0    0   0    0    0    0    0
                    0    0    0    0   0    0    0    0    0 ])
  first_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.0 1.0/3.0 0.0
                 0.25     0.25 0.0
                 0.0     0.0 0.0 ])
  first_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 0 2 0
               3 3 0
               0 0 0 ])
  first_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 6 6 6
              12 12 12
               9 9 9 ])
  first_intermediate_expected_lake_volumes::Array{Float64} = Float64[432000.0]
  # first_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    5.0    0    0    0    0    0
  #               0    0    0    5.0   0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0])
  second_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  15.0
                                  0.0 0.0  1.0
                                  0.0 0.0  0.0 ])
  second_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  second_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0  0.0
                                  0.0 0.0  0.0
                                  0.0 0.0  0.0 ])
  second_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    1    0    0    0    0    0
                0    0    0    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    1    1    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  second_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0    0    0    2    0    0    0    0    0
                0    0    0    2    0    0    0    0    0
                0    0    2    2    0    0    0    0    0
                0    0    2    2    0    0    0    0    0
                0    0    2    2    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0
                0    0    0    0    0    0    0    0    0 ])
  second_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[    0    0    0    432000.0    0    0    0    0    0
                    0    0    0    432000.0   0    0    0    0    0
                    0    0    432000.0 432000.0    0    0    0    0    0
                    0    0    432000.0 432000.0    0    0    0    0    0
                    0    0    432000.0 432000.0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0
                    0    0    0    0    0    0    0    0    0 ])
  second_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.0 1.0/3.0 0.0
                 0.25     0.25 0.0
                 0.0     0.0 0.0 ])
  second_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 0 2 0
               3 3 0
               0 0 0 ])
  second_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 6 6 6
              12 12 12
               9 9 9 ])
  second_intermediate_expected_lake_volumes::Array{Float64} = Float64[432000.0]
  # second_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0    0    0    5.0    0    0    0    0    0
  #               0    0    0    5.0   0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    8.0 8.0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0
  #               0    0    0    0    0    0    0    0    0])
  river_fields::RiverPrognosticFields,
    lake_model_prognostics::LakeModelPrognostics,
    lake_model_diagnostics::LakeModelDiagnostics =
    drive_hd_and_lake_model(river_parameters,lake_model_parameters,
                            lake_parameters_as_array,
                            drainages,runoffs,evaporations,
                            0,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes::Array{Float64} = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes::Field{Float64} =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test first_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(first_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test first_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test first_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test first_intermediate_expected_lake_types == lake_types
  @test first_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == first_intermediate_expected_diagnostic_lake_volumes
  @test first_intermediate_expected_lake_fractions == lake_fractions
  @test first_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test first_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test first_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            1,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            #-1 means skip no timesteps but do skip setup
                            forcing_timesteps_to_skip=-1)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test second_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(second_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test second_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test second_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test second_intermediate_expected_lake_types == lake_types
  @test second_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == second_intermediate_expected_diagnostic_lake_volumes
  @test second_intermediate_expected_lake_fractions == lake_fractions
  @test second_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test second_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test second_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,_ =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            2,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=1)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test expected_river_inflow == river_fields.river_inflow
  @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test expected_lake_types == lake_types
  @test expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  @test expected_lake_fractions == lake_fractions
  @test expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test expected_true_lake_depths == lake_model_parameters.true_lake_depths
end

@testset "Lake model tests 10" begin
  #Ensemble of lakes with initial spill over to rivers, no runoff, drainage over evap
  hd_grid = LatLonGrid(4,4,true)
  surface_model_grid = LatLonGrid(3,3,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
                                                Int64[-2  4  2  2
                                                       6 -2 -2  2
                                                       6  2  3 -2
                                                      -2  0  6  0 ]))
  river_reservoir_nums = LatLonField{Int64}(hd_grid,5)
  overland_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  base_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  river_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  overland_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  base_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  landsea_mask = LatLonField{Bool}(hd_grid,fill(false,4,4))
  set!(river_reservoir_nums,LatLonCoords(4,4),0)
  set!(overland_reservoir_nums,LatLonCoords(4,4),0)
  set!(base_reservoir_nums,LatLonCoords(4,4),0)
  set!(landsea_mask,LatLonCoords(4,4),true)
  set!(river_reservoir_nums,LatLonCoords(4,2),0)
  set!(overland_reservoir_nums,LatLonCoords(4,2),0)
  set!(base_reservoir_nums,LatLonCoords(4,2),0)
  set!(landsea_mask,LatLonCoords(4,2),true)
  river_parameters = RiverParameters(flow_directions,
                                     river_reservoir_nums,
                                     overland_reservoir_nums,
                                     base_reservoir_nums,
                                     river_retention_coefficients,
                                     overland_retention_coefficients,
                                     base_retention_coefficients,
                                     landsea_mask,
                                     hd_grid,86400.0,86400.0)
  lake_grid = LatLonGrid(20,20,true)
  cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
    Float64[ 2.5 3.0 2.5
             3.0 4.0 3.0
             2.5 3.0 2.5 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 2 1 1 1 1 1 1 1 1 1 1 2 2 2 2 1 1 1 1
                                                             1 1 1 1 1 2 2 1 1 1 1 2 2 2 2 2 1 1 1 1
                                                             1 1 1 1 1 3 3 3 3 1 1 1 1 2 2 2 2 1 1 1
                                                             1 1 1 2 3 3 1 3 3 2 2 1 2 1 2 2 2 2 1 1
                                                             1 1 1 1 3 3 3 3 3 1 2 1 1 2 2 2 2 2 1 1
                                                             1 1 1 1 1 3 1 3 1 1 1 2 1 2 2 2 2 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 2 2 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 1 1
                                                             1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                      Int64[ 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1
                                                             1 2 1 3 3 3 3 3 3 3 3 3 2 2 2 2 3 3 3 1
                                                             3 1 1 3 3 2 2 3 3 3 3 2 2 2 2 2 3 3 3 3
                                                             3 1 1 3 3 1 1 1 1 3 3 3 2 2 2 2 2 3 3 3
                                                             3 1 1 2 1 1 3 1 1 2 2 2 2 2 2 2 2 2 3 3
                                                             1 1 1 3 1 1 1 1 1 3 2 2 2 2 2 2 2 2 3 3
                                                             3 1 1 3 3 1 3 1 3 3 3 2 2 2 2 2 2 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 2 2 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 3 3
                                                             3 3 3 1 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3  ])
  is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
    Bool[false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false  true
          true  true  true false false false false false false false false false  true  true  true  true false false false  true
         false  true  true false false  true  true false false false false  true  true  true  true  true false false false false
         false  true  true false false  true  true  true  true false false false  true  true  true  true  true false false false
         false  true  true  true  true  true false  true  true  true  true  true  true  true  true  true  true  true false false
          true  true  true false  true  true  true  true  true false  true  true  true  true  true  true  true  true false false
         false  true  true false false  true false  true false false false  true  true  true  true  true  true false false false
         false false false false false false false false false false false  true  true  true  true  true  true false false false
         false false false false false false false false false false false false  true  true false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false  true  true  true  true false
         false false false false false false false false false false false false false false false  true  true  true false false
         false false false  true false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false])
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_model_parameters::LakeModelParameters =
    LakeModelParameters(grid_specific_lake_model_parameters,
                        lake_grid,
                        hd_grid,
                        surface_model_grid,
                        cell_areas_on_surface_model_grid,
                        LakeModelSettings(),
                        9,
                        is_lake)

  lake_parameters_as_array::Vector{Float64} = Float64[9.0, 16.0, 1.0, -1.0, 0.0, 16.0, 4.0, 1.0, 16.0, 4.0, 1.0, 1.0, 3.0, 1.0, -1.0, 4.0, 2.0, 0.0, 46.0, 2.0, -1.0, 0.0, 14.0, 16.0, 7.0, 14.0, 16.0, 1.0, 0.0, 2.0, 14.0, 17.0, 1.0, 0.0, 2.0, 15.0, 16.0, 1.0, 3.0, 3.0, 15.0, 17.0, 1.0, 3.0, 3.0, 15.0, 18.0, 1.0, 3.0, 3.0, 14.0, 18.0, 1.0, 3.0, 3.0, 14.0, 19.0, 1.0, 10.0, 4.0, 1.0, -1.0, 4.0, 4.0, 0.0, 16.0, 3.0, 7.0, 0.0, 8.0, 15.0, 1.0, 8.0, 15.0, 1.0, 1.0, 3.0, 1.0, 5.0, -1.0, -1.0, 1.0, 91.0, 4.0, 8.0, 0.0, 7.0, 6.0, 16.0, 7.0, 6.0, 1.0, 0.0, 1.0, 6.0, 6.0, 1.0, 2.0, 2.0, 7.0, 5.0, 1.0, 2.0, 2.0, 7.0, 7.0, 1.0, 2.0, 2.0, 8.0, 6.0, 1.0, 2.0, 2.0, 7.0, 8.0, 1.0, 2.0, 2.0, 6.0, 8.0, 1.0, 2.0, 2.0, 6.0, 5.0, 1.0, 10.0, 3.0, 5.0, 6.0, 1.0, 10.0, 3.0, 5.0, 7.0, 1.0, 10.0, 3.0, 7.0, 9.0, 1.0, 10.0, 3.0, 6.0, 9.0, 1.0, 10.0, 3.0, 4.0, 7.0, 2.0, 10.0, 3.0, 5.0, 9.0, 1.0, 23.0, 4.0, 8.0, 8.0, 1.0, 23.0, 4.0, 5.0, 8.0, 1.0, 38.0, 5.0, 1.0, 7.0, -1.0, -1.0, 1.0, 41.0, 5.0, 7.0, 0.0, 6.0, 14.0, 6.0, 6.0, 14.0, 1.0, 0.0, 2.0, 7.0, 13.0, 1.0, 0.0, 2.0, 5.0, 13.0, 1.0, 0.0, 2.0, 6.0, 12.0, 1.0, 0.0, 2.0, 8.0, 13.0, 1.0, 0.0, 2.0, 7.0, 12.0, 1.0, 6.0, 3.0, 1.0, 3.0, -1.0, -1.0, 1.0, 86.0, 6.0, 9.0, 0.0, 3.0, 1.0, 15.0, 3.0, 1.0, 1.0, 0.0, 1.0, 3.0, 20.0, 1.0, 2.0, 2.0, 2.0, 20.0, 1.0, 5.0, 3.0, 4.0, 2.0, 1.0, 5.0, 3.0, 5.0, 3.0, 1.0, 5.0, 3.0, 4.0, 3.0, 1.0, 5.0, 3.0, 3.0, 3.0, 1.0, 5.0, 3.0, 5.0, 2.0, 1.0, 5.0, 3.0, 6.0, 3.0, 1.0, 5.0, 3.0, 6.0, 2.0, 1.0, 5.0, 3.0, 7.0, 3.0, 1.0, 16.0, 4.0, 7.0, 2.0, 1.0, 16.0, 4.0, 8.0, 3.0, 1.0, 16.0, 4.0, 8.0, 2.0, 1.0, 16.0, 4.0, 7.0, 1.0, 1.0, 46.0, 6.0, 1.0, 8.0, 2.0, 2.0, 0.0, 163.0, 7.0, 8.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 8.0, 15.0, 1.0, 0.0, 3.0, 8.0, 16.0, 1.0, 0.0, 3.0, 7.0, 16.0, 1.0, 0.0, 3.0, 8.0, 14.0, 1.0, 0.0, 3.0, 9.0, 14.0, 1.0, 0.0, 3.0, 9.0, 15.0, 1.0, 0.0, 3.0, 7.0, 13.0, 1.0, 0.0, 3.0, 6.0, 15.0, 1.0, 0.0, 3.0, 8.0, 13.0, 1.0, 0.0, 3.0, 9.0, 13.0, 1.0, 0.0, 3.0, 7.0, 15.0, 1.0, 0.0, 3.0, 6.0, 16.0, 1.0, 0.0, 3.0, 5.0, 14.0, 1.0, 0.0, 3.0, 6.0, 12.0, 1.0, 0.0, 3.0, 4.0, 13.0, 1.0, 0.0, 3.0, 7.0, 11.0, 1.0, 0.0, 3.0, 6.0, 11.0, 1.0, 0.0, 3.0, 5.0, 15.0, 1.0, 0.0, 3.0, 4.0, 14.0, 1.0, 0.0, 3.0, 5.0, 13.0, 1.0, 0.0, 3.0, 7.0, 14.0, 1.0, 0.0, 3.0, 8.0, 12.0, 1.0, 0.0, 3.0, 7.0, 12.0, 1.0, 0.0, 3.0, 6.0, 14.0, 1.0, 0.0, 3.0, 6.0, 13.0, 1.0, 25.0, 4.0, 6.0, 17.0, 1.0, 25.0, 4.0, 7.0, 17.0, 1.0, 25.0, 4.0, 5.0, 16.0, 1.0, 25.0, 4.0, 4.0, 15.0, 1.0, 25.0, 4.0, 9.0, 16.0, 1.0, 55.0, 5.0, 1.0, 4.0, -1.0, -1.0, 1.0, 298.0, 8.0, 9.0, 2.0, 4.0, 7.0, 7.0, 6.0, 57.0, 7.0, 6.0, 1.0, 0.0, 5.0, 7.0, 7.0, 1.0, 0.0, 5.0, 7.0, 5.0, 1.0, 0.0, 5.0, 8.0, 6.0, 1.0, 0.0, 5.0, 6.0, 8.0, 1.0, 0.0, 5.0, 7.0, 8.0, 1.0, 0.0, 5.0, 7.0, 9.0, 1.0, 0.0, 5.0, 6.0, 9.0, 1.0, 0.0, 5.0, 5.0, 7.0, 1.0, 0.0, 5.0, 6.0, 6.0, 1.0, 0.0, 5.0, 6.0, 10.0, 2.0, 0.0, 5.0, 4.0, 7.0, 2.0, 0.0, 5.0, 6.0, 11.0, 1.0, 0.0, 5.0, 5.0, 6.0, 1.0, 0.0, 5.0, 7.0, 12.0, 1.0, 0.0, 5.0, 6.0, 12.0, 1.0, 0.0, 5.0, 8.0, 13.0, 1.0, 0.0, 5.0, 7.0, 11.0, 1.0, 0.0, 5.0, 8.0, 8.0, 1.0, 0.0, 5.0, 6.0, 5.0, 1.0, 0.0, 5.0, 5.0, 13.0, 1.0, 0.0, 5.0, 9.0, 13.0, 1.0, 0.0, 5.0, 9.0, 12.0, 1.0, 0.0, 5.0, 5.0, 14.0, 1.0, 0.0, 5.0, 4.0, 14.0, 1.0, 0.0, 5.0, 4.0, 15.0, 1.0, 0.0, 5.0, 3.0, 15.0, 1.0, 0.0, 5.0, 5.0, 16.0, 1.0, 0.0, 5.0, 4.0, 16.0, 1.0, 0.0, 5.0, 6.0, 17.0, 1.0, 0.0, 5.0, 6.0, 16.0, 1.0, 0.0, 5.0, 5.0, 17.0, 1.0, 0.0, 5.0, 7.0, 18.0, 1.0, 0.0, 5.0, 7.0, 17.0, 1.0, 0.0, 5.0, 7.0, 16.0, 1.0, 0.0, 5.0, 7.0, 15.0, 1.0, 0.0, 5.0, 6.0, 18.0, 1.0, 0.0, 5.0, 8.0, 16.0, 1.0, 0.0, 5.0, 8.0, 15.0, 1.0, 0.0, 5.0, 10.0, 13.0, 1.0, 0.0, 5.0, 4.0, 13.0, 1.0, 0.0, 5.0, 6.0, 15.0, 1.0, 0.0, 5.0, 5.0, 15.0, 1.0, 0.0, 5.0, 9.0, 15.0, 1.0, 0.0, 5.0, 9.0, 16.0, 1.0, 0.0, 5.0, 10.0, 14.0, 1.0, 0.0, 5.0, 8.0, 14.0, 1.0, 0.0, 5.0, 7.0, 14.0, 1.0, 0.0, 5.0, 6.0, 14.0, 1.0, 0.0, 5.0, 5.0, 9.0, 1.0, 0.0, 5.0, 5.0, 8.0, 1.0, 0.0, 5.0, 9.0, 14.0, 1.0, 0.0, 5.0, 8.0, 12.0, 1.0, 0.0, 5.0, 9.0, 17.0, 1.0, 0.0, 5.0, 7.0, 13.0, 1.0, 0.0, 5.0, 6.0, 13.0, 1.0, 0.0, 5.0, 8.0, 17.0, 1.0, 55.0, 6.0, 1.0, 6.0, 2.0, 1.0, 0.0, 418.0, 9.0, -1.0, 2.0, 6.0, 8.0, 3.0, 1.0, 81.0, 3.0, 1.0, 1.0, 0.0, 6.0, 4.0, 2.0, 1.0, 0.0, 6.0, 3.0, 20.0, 1.0, 0.0, 6.0, 2.0, 20.0, 1.0, 0.0, 6.0, 3.0, 3.0, 1.0, 0.0, 6.0, 5.0, 3.0, 1.0, 0.0, 6.0, 4.0, 3.0, 1.0, 0.0, 6.0, 5.0, 2.0, 1.0, 0.0, 6.0, 6.0, 3.0, 1.0, 0.0, 6.0, 6.0, 2.0, 1.0, 0.0, 6.0, 6.0, 4.0, 1.0, 0.0, 6.0, 7.0, 3.0, 1.0, 0.0, 6.0, 6.0, 5.0, 1.0, 0.0, 6.0, 8.0, 3.0, 1.0, 0.0, 6.0, 8.0, 2.0, 1.0, 0.0, 6.0, 6.0, 6.0, 1.0, 0.0, 6.0, 5.0, 6.0, 1.0, 0.0, 6.0, 7.0, 6.0, 1.0, 0.0, 6.0, 7.0, 2.0, 1.0, 0.0, 6.0, 7.0, 1.0, 1.0, 0.0, 6.0, 7.0, 5.0, 1.0, 0.0, 6.0, 7.0, 7.0, 1.0, 0.0, 6.0, 8.0, 6.0, 1.0, 0.0, 6.0, 5.0, 7.0, 1.0, 0.0, 6.0, 6.0, 8.0, 1.0, 0.0, 6.0, 4.0, 7.0, 2.0, 0.0, 6.0, 5.0, 8.0, 1.0, 0.0, 6.0, 7.0, 9.0, 1.0, 0.0, 6.0, 6.0, 9.0, 1.0, 0.0, 6.0, 6.0, 10.0, 2.0, 0.0, 6.0, 5.0, 9.0, 1.0, 0.0, 6.0, 7.0, 11.0, 1.0, 0.0, 6.0, 6.0, 11.0, 1.0, 0.0, 6.0, 8.0, 12.0, 1.0, 0.0, 6.0, 6.0, 12.0, 1.0, 0.0, 6.0, 7.0, 12.0, 1.0, 0.0, 6.0, 9.0, 13.0, 1.0, 0.0, 6.0, 9.0, 12.0, 1.0, 0.0, 6.0, 6.0, 13.0, 1.0, 0.0, 6.0, 5.0, 13.0, 1.0, 0.0, 6.0, 10.0, 14.0, 1.0, 0.0, 6.0, 9.0, 14.0, 1.0, 0.0, 6.0, 8.0, 14.0, 1.0, 0.0, 6.0, 5.0, 14.0, 1.0, 0.0, 6.0, 8.0, 13.0, 1.0, 0.0, 6.0, 7.0, 13.0, 1.0, 0.0, 6.0, 4.0, 15.0, 1.0, 0.0, 6.0, 8.0, 8.0, 1.0, 0.0, 6.0, 7.0, 8.0, 1.0, 0.0, 6.0, 8.0, 15.0, 1.0, 0.0, 6.0, 7.0, 15.0, 1.0, 0.0, 6.0, 3.0, 15.0, 1.0, 0.0, 6.0, 3.0, 14.0, 1.0, 0.0, 6.0, 5.0, 15.0, 1.0, 0.0, 6.0, 9.0, 16.0, 1.0, 0.0, 6.0, 3.0, 13.0, 1.0, 0.0, 6.0, 8.0, 16.0, 1.0, 0.0, 6.0, 9.0, 17.0, 1.0, 0.0, 6.0, 8.0, 17.0, 1.0, 0.0, 6.0, 7.0, 17.0, 1.0, 0.0, 6.0, 7.0, 16.0, 1.0, 0.0, 6.0, 7.0, 18.0, 1.0, 0.0, 6.0, 6.0, 18.0, 1.0, 0.0, 6.0, 6.0, 17.0, 1.0, 0.0, 6.0, 6.0, 16.0, 1.0, 0.0, 6.0, 5.0, 17.0, 1.0, 0.0, 6.0, 4.0, 16.0, 1.0, 0.0, 6.0, 3.0, 16.0, 1.0, 0.0, 6.0, 6.0, 15.0, 1.0, 0.0, 6.0, 5.0, 16.0, 1.0, 0.0, 6.0, 10.0, 13.0, 1.0, 0.0, 6.0, 4.0, 13.0, 1.0, 0.0, 6.0, 9.0, 15.0, 1.0, 0.0, 6.0, 4.0, 12.0, 1.0, 0.0, 6.0, 7.0, 14.0, 1.0, 0.0, 6.0, 6.0, 14.0, 1.0, 0.0, 6.0, 4.0, 14.0, 1.0, 75.0, 7.0, 4.0, 7.0, 1.0, 75.0, 7.0, 4.0, 6.0, 2.0, 151.0, 8.0, 3.0, 2.0, 1.0, 151.0, 8.0, 4.0, 6.0, 1.0, 229.0, 9.0, 1.0, -1.0, 2.0, 4.0, 0.0]
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
  drainages::Array{Field{Float64},1} = repeat(drainage,10000,false)
  runoffs::Array{Field{Float64},1} = drainages
  evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,180,false)
  initial_water_to_lake_centers = LatLonField{Float64}(lake_grid,
    Float64[  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0 ])
  initial_spillover_to_rivers = LatLonField{Float64}(hd_grid,
                                                     Float64[ 430.0 0.0 0.0 0.0
                                                              0.0 0.0 0.0 0.0
                                                              0.0 2.0*86400.0 0.0 0.0
                                                              0.0 0.0 1.0*86400.0 0.0 ])
  expected_river_inflow::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                               Float64[ 0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0 ])
  expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                                 Float64[ 0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0 ])
  expected_water_to_hd::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                              Float64[ 0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0 ])
  expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
              6 9 6 0 0 0 0 0 0 0 0 0 9 9 8 9 0 0 0 6
              0 6 6 0 0 9 9 0 0 0 0 9 7 7 7 8 0 0 0 0
              0 6 6 0 0 4 4 4 4 0 0 0 5 7 7 7 8 0 0 0
              0 6 6 9 4 4 0 4 4 0 7 5 7 5 7 7 7 8 0 0
              6 6 6 0 4 4 4 4 4 0 7 5 5 7 7 7 7 8 0 0
              0 6 6 0 0 4 0 4 0 0 0 7 5 7 3 7 8 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 8 7 7 7 7 8 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 8 8 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3
             3 1 3 0 0 0 0 0 0 0 0 0 1 1 3 1 0 0 0 3
             0 3 3 0 0 1 1 0 0 0 0 1 3 3 3 3 0 0 0 0
             0 3 3 0 0 3 3 3 3 0 0 0 3 3 3 3 3 0 0 0
             0 3 3 1 3 3 0 3 3 0 3 3 3 3 3 3 3 3 0 0
             3 3 3 0 3 3 3 3 3 0 3 3 3 3 3 3 3 3 0 0
             0 3 3 0 0 3 0 3 0 0 0 3 3 3 3 3 3 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 3 3 3 3 3 3 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 3 3 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
  expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     430.0
                   430.0 430.0 430.0 0     0     0     0     0     0     0     #=
                =# 0     0     430.0 430.0 430.0 430.0 0     0     0     430.0
                   0     430.0 430.0 0     0     430.0 430.0 0     0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 0     0     0     0
                   0     430.0 430.0 0     0     430.0 430.0 430.0 430.0 0     #=
                =# 0     0     430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     430.0 430.0 430.0 430.0 430.0 0     430.0 430.0 0.0 #=
                =# 430.0 430.0 430.0 430.0 430.0 430.0 430.0 430.0 0     0
                   430.0 430.0 430.0 0     430.0 430.0 430.0 430.0 430.0 0     #=
                =# 430.0 430.0 430.0 430.0 430.0 430.0 430.0 430.0 0     0
                   0     430.0 430.0 0     0     430.0 0     430.0 0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     430.0 430.0 0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0 ])
  expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 1.0 0.0
                 1.0 0.976744186 0.0
                 1.0 0.1428571429 0.0])
  expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15  6  0
                1 42  0
               15  1  0 ])
  expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[  15 6  311
                1  43 1
                15 7  1 ])
  expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 1.0, 38.0, 6.0, 46.0, 55.0, 55.0, 229.0]
  # expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 7.0
  #               8.0 1.0 6.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 4.0 3.0 0.0 0.0 0.0 8.0
  #               0.0 6.0 6.0 0.0 0.0 1.0 2.0 0.0 0.0 0.0 0.0 3.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0 0.0
  #               0.0 6.0 6.0 0.0 0.0 6.0 6.0 5.0 6.0 0.0 0.0 0.0 7.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0
  #               0.0 6.0 6.0 3.0 7.0 8.0 0.0 7.0 6.0 0.0 6.0 7.0 6.0 7.0 6.0 6.0 5.0 4.0 0.0 0.0
  #               5.0 5.0 6.0 0.0 7.0 8.0 7.0 7.0 6.0 0.0 6.0 7.0 7.0 6.0 6.0 6.0 5.0 4.0 0.0 0.0
  #               0.0 5.0 5.0 0.0 0.0 7.0 0.0 5.0 0.0 0.0 0.0 6.0 7.0 6.0 7.0 6.0 4.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0 6.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0 4.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])

  first_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  first_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  first_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 430.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 2.0*86400.0 0.0 0.0
                                  0.0 0.0 1.0*86400.0 0.0 ])
  first_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 5 0 0 0 0 0 0
             0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
  first_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
             0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
  first_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
      Float64[     0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0
                   0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0
                   0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0
                   0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0
                   0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0
                   0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0
                   0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0
                   0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0
                   0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0
                   0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0
                   0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0
                   0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0
                   0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0
                   0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0
                   0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0
                   0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0
                   0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0
                   0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0
                   0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0
                   0 0 0 0 0 0 0 0 0 0 #=
                =# 0 0 0 0 0 0 0 0 0 0 ])
  first_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.06666666667 0.1666666667 0.0
                 1.0 0.02325581395 0.0
                 0.06666666667 0.1428571429 0.0])
  first_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[  1  1  0
                1  1  0
                1  1  0 ])
  first_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 6  311
                1 43 1
               15 7  1  ])
  first_intermediate_expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  # first_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])

  second_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  second_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 2.0 0.0 1.0 ])
  second_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 384.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  second_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
              6 0 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
              0 6 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 6 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 6 6 0 0 0 0 0 0 0 0 0 0 5 0 0 0 0 0 0
              6 6 6 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 6 6 0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  second_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2
             2 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2
             0 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 2 2 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
             2 2 2 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 2 2 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
  second_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
      Float64[  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
             =# 0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
             =# 0.0   0.0   0.0   0.0   0.0   0.0  46.0
               46.0   0.0  46.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
             =# 0.0   0.0   0.0   0.0   0.0   0.0  46.0
                0.0  46.0  46.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
             =# 0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0  46.0  46.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0  46.0  46.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
               46.0  46.0  46.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0  46.0  46.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0  ])
  second_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 0.1666666667 0.0
                 1.0 0.02325581395 0.0
                 0.06666666667 0.1428571429 0.0])
  second_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 1 0
                1 1 0
                1 1 0 ])
  second_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 6  311
                1 43 1
               15 7  1  ])
  second_intermediate_expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 0.0, 0.0, 0.0, 46.0, 0.0, 0.0, 0.0]
  # second_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0
  #               5.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 5.0
  #               0.0 3.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 3.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 3.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               2.0 2.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])

  third_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  third_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  third_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  third_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
              6 0 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
              0 6 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 6 6 0 0 4 4 4 4 0 0 0 0 0 0 0 0 0 0 0
              0 6 6 0 4 4 0 4 4 0 0 0 0 5 0 0 0 0 0 0
              6 6 6 0 4 4 4 4 4 0 0 0 0 0 0 0 0 0 0 0
              0 6 6 0 0 4 0 4 0 0 0 0 0 0 3 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  third_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2
             2 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2
             0 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 2 2 0 0 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0
             0 2 2 0 2 2 0 2 2 0 0 0 0 1 0 0 0 0 0 0
             2 2 2 0 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0
             0 2 2 0 0 2 0 2 0 0 0 0 0 0 2 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  third_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
      Float64[   0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0  46.0
                46.0   0   46.0 0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0  46.0
                 0    46.0 46.0 0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0    46.0 46.0 0   0   38.0 38.0 38.0 38.0  0   0   0   0   0   0   0   0   0   0   0
                 0    46.0 46.0 0  38.0 38.0  0   38.0 38.0  0   0   0   0   0   0   0   0   0   0   0
                46.0  46.0 46.0 0  38.0 38.0 38.0 38.0 38.0  0   0   0   0   0   0   0   0   0   0   0
                 0    46.0 46.0 0   0   38.0  0   38.0  0    0   0   0   0   0   346.0 0 0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0  ])
  third_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 1.0/6.0 0.0
                 1.0 1.0/43.0 0.0
                 1.0 0.1428571429 0.0 ])
  third_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 1 0
                1 1 0
               15 1 0] )
  third_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 6  311
                1  43 1
                15 7  1  ])
  third_intermediate_expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 346.0, 38.0, 0.0, 46.0, 0.0, 0.0, 0.0]
  # third_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0
  #               5.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 5.0
  #               0.0 3.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 3.0 3.0 0.0 0.0 2.0 2.0 1.0 2.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 3.0 3.0 0.0 3.0 4.0 0.0 3.0 2.0 0.0 0.0 1.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               2.0 2.0 3.0 0.0 3.0 4.0 3.0 3.0 2.0 0.0 0.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 2.0 2.0 0.0 0.0 3.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])
  river_fields::RiverPrognosticFields,
    lake_model_prognostics::LakeModelPrognostics,
    lake_model_diagnostics::LakeModelDiagnostics =
    drive_hd_and_lake_model(river_parameters,lake_model_parameters,
                            lake_parameters_as_array,
                            drainages,runoffs,evaporations,
                            0,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes::Array{Float64} = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes::Field{Float64} =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test first_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(first_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test first_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test first_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test first_intermediate_expected_lake_types == lake_types
  @test first_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == first_intermediate_expected_diagnostic_lake_volumes
  @test isapprox(first_intermediate_expected_lake_fractions,lake_fractions,
                 rtol=0.0,atol=0.00001)
  @test first_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test first_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test first_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            1,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=-1)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test second_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(second_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test second_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test second_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test second_intermediate_expected_lake_types == lake_types
  @test second_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == second_intermediate_expected_diagnostic_lake_volumes
  @test isapprox(second_intermediate_expected_lake_fractions,lake_fractions,
                 rtol=0.0,atol=0.00001)
  @test second_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test second_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test second_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            2,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=1)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test third_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(third_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test third_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test third_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test third_intermediate_expected_lake_types == lake_types
  @test third_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == third_intermediate_expected_diagnostic_lake_volumes
  @test isapprox(third_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.000001)
  @test third_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test third_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test third_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,_ =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            3,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=2)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test expected_river_inflow == river_fields.river_inflow
  @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test expected_lake_types == lake_types
  @test expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  @test isapprox(expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
  @test expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test expected_true_lake_depths == lake_model_parameters.true_lake_depths
end

@testset "Lake model tests 11" begin
  #Initial water spilled into HD on ensemble lakes - this
  #time with enough water to also fill downstream lake
  hd_grid = LatLonGrid(4,4,true)
  surface_model_grid = LatLonGrid(3,3,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
                                                Int64[-2  4  2  2
                                                       6 -2 -2  2
                                                       6  2  3 -2
                                                      -2  0  6  0 ]))
  river_reservoir_nums = LatLonField{Int64}(hd_grid,5)
  overland_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  base_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  river_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  overland_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  base_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  landsea_mask = LatLonField{Bool}(hd_grid,fill(false,4,4))
  set!(river_reservoir_nums,LatLonCoords(4,4),0)
  set!(overland_reservoir_nums,LatLonCoords(4,4),0)
  set!(base_reservoir_nums,LatLonCoords(4,4),0)
  set!(landsea_mask,LatLonCoords(4,4),true)
  set!(river_reservoir_nums,LatLonCoords(4,2),0)
  set!(overland_reservoir_nums,LatLonCoords(4,2),0)
  set!(base_reservoir_nums,LatLonCoords(4,2),0)
  set!(landsea_mask,LatLonCoords(4,2),true)
  river_parameters = RiverParameters(flow_directions,
                                     river_reservoir_nums,
                                     overland_reservoir_nums,
                                     base_reservoir_nums,
                                     river_retention_coefficients,
                                     overland_retention_coefficients,
                                     base_retention_coefficients,
                                     landsea_mask,
                                     hd_grid,86400.0,86400.0)
  lake_grid = LatLonGrid(20,20,true)
  cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
    Float64[ 2.5 3.0 2.5
             3.0 4.0 3.0
             2.5 3.0 2.5 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 2 1 1 1 1 1 1 1 1 1 1 2 2 2 2 1 1 1 1
                                                             1 1 1 1 1 2 2 1 1 1 1 2 2 2 2 2 1 1 1 1
                                                             1 1 1 1 1 3 3 3 3 1 1 1 1 2 2 2 2 1 1 1
                                                             1 1 1 2 3 3 1 3 3 2 2 1 2 1 2 2 2 2 1 1
                                                             1 1 1 1 3 3 3 3 3 1 2 1 1 2 2 2 2 2 1 1
                                                             1 1 1 1 1 3 1 3 1 1 1 2 1 2 2 2 2 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 2 2 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 1 1
                                                             1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                      Int64[ 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1
                                                             1 2 1 3 3 3 3 3 3 3 3 3 2 2 2 2 3 3 3 1
                                                             3 1 1 3 3 2 2 3 3 3 3 2 2 2 2 2 3 3 3 3
                                                             3 1 1 3 3 1 1 1 1 3 3 3 2 2 2 2 2 3 3 3
                                                             3 1 1 2 1 1 3 1 1 2 2 2 2 2 2 2 2 2 3 3
                                                             1 1 1 3 1 1 1 1 1 3 2 2 2 2 2 2 2 2 3 3
                                                             3 1 1 3 3 1 3 1 3 3 3 2 2 2 2 2 2 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 2 2 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 3 3
                                                             3 3 3 1 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3  ])
  is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
    Bool[false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false  true
          true  true  true false false false false false false false false false  true  true  true  true false false false  true
         false  true  true false false  true  true false false false false  true  true  true  true  true false false false false
         false  true  true false false  true  true  true  true false false false  true  true  true  true  true false false false
         false  true  true  true  true  true false  true  true  true  true  true  true  true  true  true  true  true false false
          true  true  true false  true  true  true  true  true false  true  true  true  true  true  true  true  true false false
         false  true  true false false  true false  true false false false  true  true  true  true  true  true false false false
         false false false false false false false false false false false  true  true  true  true  true  true false false false
         false false false false false false false false false false false false  true  true false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false  true  true  true  true false
         false false false false false false false false false false false false false false false  true  true  true false false
         false false false  true false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false])
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_model_parameters::LakeModelParameters =
    LakeModelParameters(grid_specific_lake_model_parameters,
                        lake_grid,
                        hd_grid,
                        surface_model_grid,
                        cell_areas_on_surface_model_grid,
                        LakeModelSettings(),
                        9,
                        is_lake)
  lake_parameters_as_array::Vector{Float64} = Float64[9.0, 16.0, 1.0, -1.0, 0.0, 16.0, 4.0, 1.0, 16.0, 4.0, 1.0, 1.0, 3.0, 1.0, -1.0, 4.0, 2.0, 0.0, 46.0, 2.0, -1.0, 0.0, 14.0, 16.0, 7.0, 14.0, 16.0, 1.0, 0.0, 2.0, 14.0, 17.0, 1.0, 0.0, 2.0, 15.0, 16.0, 1.0, 3.0, 3.0, 15.0, 17.0, 1.0, 3.0, 3.0, 15.0, 18.0, 1.0, 3.0, 3.0, 14.0, 18.0, 1.0, 3.0, 3.0, 14.0, 19.0, 1.0, 10.0, 4.0, 1.0, -1.0, 4.0, 4.0, 0.0, 16.0, 3.0, 7.0, 0.0, 8.0, 15.0, 1.0, 8.0, 15.0, 1.0, 1.0, 3.0, 1.0, 5.0, -1.0, -1.0, 1.0, 91.0, 4.0, 8.0, 0.0, 7.0, 6.0, 16.0, 7.0, 6.0, 1.0, 0.0, 1.0, 6.0, 6.0, 1.0, 2.0, 2.0, 7.0, 5.0, 1.0, 2.0, 2.0, 7.0, 7.0, 1.0, 2.0, 2.0, 8.0, 6.0, 1.0, 2.0, 2.0, 7.0, 8.0, 1.0, 2.0, 2.0, 6.0, 8.0, 1.0, 2.0, 2.0, 6.0, 5.0, 1.0, 10.0, 3.0, 5.0, 6.0, 1.0, 10.0, 3.0, 5.0, 7.0, 1.0, 10.0, 3.0, 7.0, 9.0, 1.0, 10.0, 3.0, 6.0, 9.0, 1.0, 10.0, 3.0, 4.0, 7.0, 2.0, 10.0, 3.0, 5.0, 9.0, 1.0, 23.0, 4.0, 8.0, 8.0, 1.0, 23.0, 4.0, 5.0, 8.0, 1.0, 38.0, 5.0, 1.0, 7.0, -1.0, -1.0, 1.0, 41.0, 5.0, 7.0, 0.0, 6.0, 14.0, 6.0, 6.0, 14.0, 1.0, 0.0, 2.0, 7.0, 13.0, 1.0, 0.0, 2.0, 5.0, 13.0, 1.0, 0.0, 2.0, 6.0, 12.0, 1.0, 0.0, 2.0, 8.0, 13.0, 1.0, 0.0, 2.0, 7.0, 12.0, 1.0, 6.0, 3.0, 1.0, 3.0, -1.0, -1.0, 1.0, 86.0, 6.0, 9.0, 0.0, 3.0, 1.0, 15.0, 3.0, 1.0, 1.0, 0.0, 1.0, 3.0, 20.0, 1.0, 2.0, 2.0, 2.0, 20.0, 1.0, 5.0, 3.0, 4.0, 2.0, 1.0, 5.0, 3.0, 5.0, 3.0, 1.0, 5.0, 3.0, 4.0, 3.0, 1.0, 5.0, 3.0, 3.0, 3.0, 1.0, 5.0, 3.0, 5.0, 2.0, 1.0, 5.0, 3.0, 6.0, 3.0, 1.0, 5.0, 3.0, 6.0, 2.0, 1.0, 5.0, 3.0, 7.0, 3.0, 1.0, 16.0, 4.0, 7.0, 2.0, 1.0, 16.0, 4.0, 8.0, 3.0, 1.0, 16.0, 4.0, 8.0, 2.0, 1.0, 16.0, 4.0, 7.0, 1.0, 1.0, 46.0, 6.0, 1.0, 8.0, 2.0, 2.0, 0.0, 163.0, 7.0, 8.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 8.0, 15.0, 1.0, 0.0, 3.0, 8.0, 16.0, 1.0, 0.0, 3.0, 7.0, 16.0, 1.0, 0.0, 3.0, 8.0, 14.0, 1.0, 0.0, 3.0, 9.0, 14.0, 1.0, 0.0, 3.0, 9.0, 15.0, 1.0, 0.0, 3.0, 7.0, 13.0, 1.0, 0.0, 3.0, 6.0, 15.0, 1.0, 0.0, 3.0, 8.0, 13.0, 1.0, 0.0, 3.0, 9.0, 13.0, 1.0, 0.0, 3.0, 7.0, 15.0, 1.0, 0.0, 3.0, 6.0, 16.0, 1.0, 0.0, 3.0, 5.0, 14.0, 1.0, 0.0, 3.0, 6.0, 12.0, 1.0, 0.0, 3.0, 4.0, 13.0, 1.0, 0.0, 3.0, 7.0, 11.0, 1.0, 0.0, 3.0, 6.0, 11.0, 1.0, 0.0, 3.0, 5.0, 15.0, 1.0, 0.0, 3.0, 4.0, 14.0, 1.0, 0.0, 3.0, 5.0, 13.0, 1.0, 0.0, 3.0, 7.0, 14.0, 1.0, 0.0, 3.0, 8.0, 12.0, 1.0, 0.0, 3.0, 7.0, 12.0, 1.0, 0.0, 3.0, 6.0, 14.0, 1.0, 0.0, 3.0, 6.0, 13.0, 1.0, 25.0, 4.0, 6.0, 17.0, 1.0, 25.0, 4.0, 7.0, 17.0, 1.0, 25.0, 4.0, 5.0, 16.0, 1.0, 25.0, 4.0, 4.0, 15.0, 1.0, 25.0, 4.0, 9.0, 16.0, 1.0, 55.0, 5.0, 1.0, 4.0, -1.0, -1.0, 1.0, 298.0, 8.0, 9.0, 2.0, 4.0, 7.0, 7.0, 6.0, 57.0, 7.0, 6.0, 1.0, 0.0, 5.0, 7.0, 7.0, 1.0, 0.0, 5.0, 7.0, 5.0, 1.0, 0.0, 5.0, 8.0, 6.0, 1.0, 0.0, 5.0, 6.0, 8.0, 1.0, 0.0, 5.0, 7.0, 8.0, 1.0, 0.0, 5.0, 7.0, 9.0, 1.0, 0.0, 5.0, 6.0, 9.0, 1.0, 0.0, 5.0, 5.0, 7.0, 1.0, 0.0, 5.0, 6.0, 6.0, 1.0, 0.0, 5.0, 6.0, 10.0, 2.0, 0.0, 5.0, 4.0, 7.0, 2.0, 0.0, 5.0, 6.0, 11.0, 1.0, 0.0, 5.0, 5.0, 6.0, 1.0, 0.0, 5.0, 7.0, 12.0, 1.0, 0.0, 5.0, 6.0, 12.0, 1.0, 0.0, 5.0, 8.0, 13.0, 1.0, 0.0, 5.0, 7.0, 11.0, 1.0, 0.0, 5.0, 8.0, 8.0, 1.0, 0.0, 5.0, 6.0, 5.0, 1.0, 0.0, 5.0, 5.0, 13.0, 1.0, 0.0, 5.0, 9.0, 13.0, 1.0, 0.0, 5.0, 9.0, 12.0, 1.0, 0.0, 5.0, 5.0, 14.0, 1.0, 0.0, 5.0, 4.0, 14.0, 1.0, 0.0, 5.0, 4.0, 15.0, 1.0, 0.0, 5.0, 3.0, 15.0, 1.0, 0.0, 5.0, 5.0, 16.0, 1.0, 0.0, 5.0, 4.0, 16.0, 1.0, 0.0, 5.0, 6.0, 17.0, 1.0, 0.0, 5.0, 6.0, 16.0, 1.0, 0.0, 5.0, 5.0, 17.0, 1.0, 0.0, 5.0, 7.0, 18.0, 1.0, 0.0, 5.0, 7.0, 17.0, 1.0, 0.0, 5.0, 7.0, 16.0, 1.0, 0.0, 5.0, 7.0, 15.0, 1.0, 0.0, 5.0, 6.0, 18.0, 1.0, 0.0, 5.0, 8.0, 16.0, 1.0, 0.0, 5.0, 8.0, 15.0, 1.0, 0.0, 5.0, 10.0, 13.0, 1.0, 0.0, 5.0, 4.0, 13.0, 1.0, 0.0, 5.0, 6.0, 15.0, 1.0, 0.0, 5.0, 5.0, 15.0, 1.0, 0.0, 5.0, 9.0, 15.0, 1.0, 0.0, 5.0, 9.0, 16.0, 1.0, 0.0, 5.0, 10.0, 14.0, 1.0, 0.0, 5.0, 8.0, 14.0, 1.0, 0.0, 5.0, 7.0, 14.0, 1.0, 0.0, 5.0, 6.0, 14.0, 1.0, 0.0, 5.0, 5.0, 9.0, 1.0, 0.0, 5.0, 5.0, 8.0, 1.0, 0.0, 5.0, 9.0, 14.0, 1.0, 0.0, 5.0, 8.0, 12.0, 1.0, 0.0, 5.0, 9.0, 17.0, 1.0, 0.0, 5.0, 7.0, 13.0, 1.0, 0.0, 5.0, 6.0, 13.0, 1.0, 0.0, 5.0, 8.0, 17.0, 1.0, 55.0, 6.0, 1.0, 6.0, 2.0, 1.0, 0.0, 418.0, 9.0, -1.0, 2.0, 6.0, 8.0, 3.0, 1.0, 81.0, 3.0, 1.0, 1.0, 0.0, 6.0, 4.0, 2.0, 1.0, 0.0, 6.0, 3.0, 20.0, 1.0, 0.0, 6.0, 2.0, 20.0, 1.0, 0.0, 6.0, 3.0, 3.0, 1.0, 0.0, 6.0, 5.0, 3.0, 1.0, 0.0, 6.0, 4.0, 3.0, 1.0, 0.0, 6.0, 5.0, 2.0, 1.0, 0.0, 6.0, 6.0, 3.0, 1.0, 0.0, 6.0, 6.0, 2.0, 1.0, 0.0, 6.0, 6.0, 4.0, 1.0, 0.0, 6.0, 7.0, 3.0, 1.0, 0.0, 6.0, 6.0, 5.0, 1.0, 0.0, 6.0, 8.0, 3.0, 1.0, 0.0, 6.0, 8.0, 2.0, 1.0, 0.0, 6.0, 6.0, 6.0, 1.0, 0.0, 6.0, 5.0, 6.0, 1.0, 0.0, 6.0, 7.0, 6.0, 1.0, 0.0, 6.0, 7.0, 2.0, 1.0, 0.0, 6.0, 7.0, 1.0, 1.0, 0.0, 6.0, 7.0, 5.0, 1.0, 0.0, 6.0, 7.0, 7.0, 1.0, 0.0, 6.0, 8.0, 6.0, 1.0, 0.0, 6.0, 5.0, 7.0, 1.0, 0.0, 6.0, 6.0, 8.0, 1.0, 0.0, 6.0, 4.0, 7.0, 2.0, 0.0, 6.0, 5.0, 8.0, 1.0, 0.0, 6.0, 7.0, 9.0, 1.0, 0.0, 6.0, 6.0, 9.0, 1.0, 0.0, 6.0, 6.0, 10.0, 2.0, 0.0, 6.0, 5.0, 9.0, 1.0, 0.0, 6.0, 7.0, 11.0, 1.0, 0.0, 6.0, 6.0, 11.0, 1.0, 0.0, 6.0, 8.0, 12.0, 1.0, 0.0, 6.0, 6.0, 12.0, 1.0, 0.0, 6.0, 7.0, 12.0, 1.0, 0.0, 6.0, 9.0, 13.0, 1.0, 0.0, 6.0, 9.0, 12.0, 1.0, 0.0, 6.0, 6.0, 13.0, 1.0, 0.0, 6.0, 5.0, 13.0, 1.0, 0.0, 6.0, 10.0, 14.0, 1.0, 0.0, 6.0, 9.0, 14.0, 1.0, 0.0, 6.0, 8.0, 14.0, 1.0, 0.0, 6.0, 5.0, 14.0, 1.0, 0.0, 6.0, 8.0, 13.0, 1.0, 0.0, 6.0, 7.0, 13.0, 1.0, 0.0, 6.0, 4.0, 15.0, 1.0, 0.0, 6.0, 8.0, 8.0, 1.0, 0.0, 6.0, 7.0, 8.0, 1.0, 0.0, 6.0, 8.0, 15.0, 1.0, 0.0, 6.0, 7.0, 15.0, 1.0, 0.0, 6.0, 3.0, 15.0, 1.0, 0.0, 6.0, 3.0, 14.0, 1.0, 0.0, 6.0, 5.0, 15.0, 1.0, 0.0, 6.0, 9.0, 16.0, 1.0, 0.0, 6.0, 3.0, 13.0, 1.0, 0.0, 6.0, 8.0, 16.0, 1.0, 0.0, 6.0, 9.0, 17.0, 1.0, 0.0, 6.0, 8.0, 17.0, 1.0, 0.0, 6.0, 7.0, 17.0, 1.0, 0.0, 6.0, 7.0, 16.0, 1.0, 0.0, 6.0, 7.0, 18.0, 1.0, 0.0, 6.0, 6.0, 18.0, 1.0, 0.0, 6.0, 6.0, 17.0, 1.0, 0.0, 6.0, 6.0, 16.0, 1.0, 0.0, 6.0, 5.0, 17.0, 1.0, 0.0, 6.0, 4.0, 16.0, 1.0, 0.0, 6.0, 3.0, 16.0, 1.0, 0.0, 6.0, 6.0, 15.0, 1.0, 0.0, 6.0, 5.0, 16.0, 1.0, 0.0, 6.0, 10.0, 13.0, 1.0, 0.0, 6.0, 4.0, 13.0, 1.0, 0.0, 6.0, 9.0, 15.0, 1.0, 0.0, 6.0, 4.0, 12.0, 1.0, 0.0, 6.0, 7.0, 14.0, 1.0, 0.0, 6.0, 6.0, 14.0, 1.0, 0.0, 6.0, 4.0, 14.0, 1.0, 75.0, 7.0, 4.0, 7.0, 1.0, 75.0, 7.0, 4.0, 6.0, 2.0, 151.0, 8.0, 3.0, 2.0, 1.0, 151.0, 8.0, 4.0, 6.0, 1.0, 229.0, 9.0, 1.0, -1.0, 2.0, 4.0, 0.0]
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
  drainages::Array{Field{Float64},1} = repeat(drainage,10000,false)
  runoffs::Array{Field{Float64},1} = drainages
  evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,180,false)
  initial_water_to_lake_centers = LatLonField{Float64}(lake_grid,
    Float64[  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0 ])
  initial_spillover_to_rivers = LatLonField{Float64}(hd_grid,
                                                     Float64[ 440.0 0.0 0.0 0.0
                                                              0.0 0.0 0.0 0.0
                                                              0.0 2.0*86400.0 0.0 0.0
                                                              0.0 0.0 1.0*86400.0 0.0 ])
  expected_river_inflow::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                               Float64[ 0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0 ])
  expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                                 Float64[ 0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0 ])
  expected_water_to_hd::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                              Float64[ 0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0 ])
  expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
              6 9 6 0 0 0 0 0 0 0 0 0 9 9 8 9 0 0 0 6
              0 6 6 0 0 9 9 0 0 0 0 9 7 7 7 8 0 0 0 0
              0 6 6 0 0 4 4 4 4 0 0 0 5 7 7 7 8 0 0 0
              0 6 6 9 4 4 0 4 4 0 7 5 7 5 7 7 7 8 0 0
              6 6 6 0 4 4 4 4 4 0 7 5 5 7 7 7 7 8 0 0
              0 6 6 0 0 4 0 4 0 0 0 7 5 7 3 7 8 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 8 7 7 7 7 8 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 8 8 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 0 0
              0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
  expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3
             3 2 3 0 0 0 0 0 0 0 0 0 2 2 3 2 0 0 0 3
             0 3 3 0 0 2 2 0 0 0 0 2 3 3 3 3 0 0 0 0
             0 3 3 0 0 3 3 3 3 0 0 0 3 3 3 3 3 0 0 0
             0 3 3 2 3 3 0 3 3 0 3 3 3 3 3 3 3 3 0 0
             3 3 3 0 3 3 3 3 3 0 3 3 3 3 3 3 3 3 0 0
             0 3 3 0 0 3 0 3 0 0 0 3 3 3 3 3 3 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 3 3 3 3 3 3 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 3 3 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     430.0
                   430.0 430.0 430.0 0     0     0     0     0     0     0     #=
                =# 0     0     430.0 430.0 430.0 430.0 0     0     0     430.0
                   0     430.0 430.0 0     0     430.0 430.0 0     0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 0     0     0     0
                   0     430.0 430.0 0     0     430.0 430.0 430.0 430.0 0     #=
                =# 0     0     430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     430.0 430.0 430.0 430.0 430.0 0     430.0 430.0 0     #=
                =# 430.0 430.0 430.0 430.0 430.0 430.0 430.0 430.0 0     0
                   430.0 430.0 430.0 0     430.0 430.0 430.0 430.0 430.0 0     #=
                =# 430.0 430.0 430.0 430.0 430.0 430.0 430.0 430.0 0     0
                   0     430.0 430.0 0     0     430.0 0     430.0 0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     430.0 430.0 0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0  10.0  10.0  10.0  10.0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0  10.0  10.0  10.0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0 ])
  expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 1.0 0.0
                 1.0 0.976744186 0.0
                 1.0 1.0 0.0])
  expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15  6  0
                1 42  0
               15  7  0  ])
  expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 6  311
                1  43 1
                15 7  1 ])
  expected_lake_volumes::Array{Float64} = Float64[0.0, 10.0, 1.0, 38.0, 6.0, 46.0, 55.0, 55.0, 229.0]
  # expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 7.0
  #               8.0 1.0 6.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 4.0 3.0 0.0 0.0 0.0 8.0
  #               0.0 6.0 6.0 0.0 0.0 1.0 2.0 0.0 0.0 0.0 0.0 3.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0 0.0
  #               0.0 6.0 6.0 0.0 0.0 6.0 6.0 5.0 6.0 0.0 0.0 0.0 7.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0
  #               0.0 6.0 6.0 3.0 7.0 8.0 0.0 7.0 6.0 0.0 6.0 7.0 6.0 7.0 6.0 6.0 5.0 4.0 0.0 0.0
  #               5.0 5.0 6.0 0.0 7.0 8.0 7.0 7.0 6.0 0.0 6.0 7.0 7.0 6.0 6.0 6.0 5.0 4.0 0.0 0.0
  #               0.0 5.0 5.0 0.0 0.0 7.0 0.0 5.0 0.0 0.0 0.0 6.0 7.0 6.0 7.0 6.0 4.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0 6.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0 4.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 2.0 1.0 1.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 1.0 1.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])
  river_fields::RiverPrognosticFields,
    lake_model_prognostics::LakeModelPrognostics,_ =
    drive_hd_and_lake_model(river_parameters,lake_model_parameters,
                            lake_parameters_as_array,
                            drainages,runoffs,evaporations,
                            5,true,initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes::Field{Float64} =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test expected_river_inflow == river_fields.river_inflow
  @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test expected_lake_types == lake_types
  @test expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  @test isapprox(expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
  @test expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test expected_true_lake_depths == lake_model_parameters.true_lake_depths
end

@testset "Lake model tests 12" begin
  #More initial spill over to river - spill over more spread out
  hd_grid = LatLonGrid(4,4,true)
  surface_model_grid = LatLonGrid(3,3,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
                                                Int64[-2  4  2  2
                                                       6 -2 -2  2
                                                       6  2  3 -2
                                                      -2  0  6  0 ]))
  river_reservoir_nums = LatLonField{Int64}(hd_grid,5)
  overland_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  base_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  river_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  overland_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  base_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  landsea_mask = LatLonField{Bool}(hd_grid,fill(false,4,4))
  set!(river_reservoir_nums,LatLonCoords(4,4),0)
  set!(overland_reservoir_nums,LatLonCoords(4,4),0)
  set!(base_reservoir_nums,LatLonCoords(4,4),0)
  set!(landsea_mask,LatLonCoords(4,4),true)
  set!(river_reservoir_nums,LatLonCoords(4,2),0)
  set!(overland_reservoir_nums,LatLonCoords(4,2),0)
  set!(base_reservoir_nums,LatLonCoords(4,2),0)
  set!(landsea_mask,LatLonCoords(4,2),true)
  river_parameters = RiverParameters(flow_directions,
                                     river_reservoir_nums,
                                     overland_reservoir_nums,
                                     base_reservoir_nums,
                                     river_retention_coefficients,
                                     overland_retention_coefficients,
                                     base_retention_coefficients,
                                     landsea_mask,
                                     hd_grid,86400.0,86400.0)
  lake_grid = LatLonGrid(20,20,true)
  cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
    Float64[ 2.5 3.0 2.5
             3.0 4.0 3.0
             2.5 3.0 2.5 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 2 1 1 1 1 1 1 1 1 1 1 2 2 2 2 1 1 1 1
                                                             1 1 1 1 1 2 2 1 1 1 1 2 2 2 2 2 1 1 1 1
                                                             1 1 1 1 1 3 3 3 3 1 1 1 1 2 2 2 2 1 1 1
                                                             1 1 1 2 3 3 1 3 3 2 2 1 2 1 2 2 2 2 1 1
                                                             1 1 1 1 3 3 3 3 3 1 2 1 1 2 2 2 2 2 1 1
                                                             1 1 1 1 1 3 1 3 1 1 1 2 1 2 2 2 2 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 2 2 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 1 1
                                                             1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                      Int64[ 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1
                                                             1 2 1 3 3 3 3 3 3 3 3 3 2 2 2 2 3 3 3 1
                                                             3 1 1 3 3 2 2 3 3 3 3 2 2 2 2 2 3 3 3 3
                                                             3 1 1 3 3 1 1 1 1 3 3 3 2 2 2 2 2 3 3 3
                                                             3 1 1 2 1 1 3 1 1 2 2 2 2 2 2 2 2 2 3 3
                                                             1 1 1 3 1 1 1 1 1 3 2 2 2 2 2 2 2 2 3 3
                                                             3 1 1 3 3 1 3 1 3 3 3 2 2 2 2 2 2 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 2 2 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 3 3
                                                             3 3 3 1 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3  ])
  is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
    Bool[false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false  true
          true  true  true false false false false false false false false false  true  true  true  true false false false  true
         false  true  true false false  true  true false false false false  true  true  true  true  true false false false false
         false  true  true false false  true  true  true  true false false false  true  true  true  true  true false false false
         false  true  true  true  true  true false  true  true  true  true  true  true  true  true  true  true  true false false
          true  true  true false  true  true  true  true  true false  true  true  true  true  true  true  true  true false false
         false  true  true false false  true false  true false false false  true  true  true  true  true  true false false false
         false false false false false false false false false false false  true  true  true  true  true  true false false false
         false false false false false false false false false false false false  true  true false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false  true  true  true  true false
         false false false false false false false false false false false false false false false  true  true  true false false
         false false false  true false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false])
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_model_parameters::LakeModelParameters =
    LakeModelParameters(grid_specific_lake_model_parameters,
                        lake_grid,
                        hd_grid,
                        surface_model_grid,
                        cell_areas_on_surface_model_grid,
                        LakeModelSettings(),
                        9,
                        is_lake)
  lake_parameters_as_array::Vector{Float64} = Float64[9.0, 16.0, 1.0, -1.0, 0.0, 16.0, 4.0, 1.0, 16.0, 4.0, 1.0, 1.0, 3.0, 1.0, -1.0, 4.0, 2.0, 0.0, 46.0, 2.0, -1.0, 0.0, 14.0, 16.0, 7.0, 14.0, 16.0, 1.0, 0.0, 2.0, 14.0, 17.0, 1.0, 0.0, 2.0, 15.0, 16.0, 1.0, 3.0, 3.0, 15.0, 17.0, 1.0, 3.0, 3.0, 15.0, 18.0, 1.0, 3.0, 3.0, 14.0, 18.0, 1.0, 3.0, 3.0, 14.0, 19.0, 1.0, 10.0, 4.0, 1.0, -1.0, 4.0, 4.0, 0.0, 16.0, 3.0, 7.0, 0.0, 8.0, 15.0, 1.0, 8.0, 15.0, 1.0, 1.0, 3.0, 1.0, 5.0, -1.0, -1.0, 1.0, 91.0, 4.0, 8.0, 0.0, 7.0, 6.0, 16.0, 7.0, 6.0, 1.0, 0.0, 1.0, 6.0, 6.0, 1.0, 2.0, 2.0, 7.0, 5.0, 1.0, 2.0, 2.0, 7.0, 7.0, 1.0, 2.0, 2.0, 8.0, 6.0, 1.0, 2.0, 2.0, 7.0, 8.0, 1.0, 2.0, 2.0, 6.0, 8.0, 1.0, 2.0, 2.0, 6.0, 5.0, 1.0, 10.0, 3.0, 5.0, 6.0, 1.0, 10.0, 3.0, 5.0, 7.0, 1.0, 10.0, 3.0, 7.0, 9.0, 1.0, 10.0, 3.0, 6.0, 9.0, 1.0, 10.0, 3.0, 4.0, 7.0, 2.0, 10.0, 3.0, 5.0, 9.0, 1.0, 23.0, 4.0, 8.0, 8.0, 1.0, 23.0, 4.0, 5.0, 8.0, 1.0, 38.0, 5.0, 1.0, 7.0, -1.0, -1.0, 1.0, 41.0, 5.0, 7.0, 0.0, 6.0, 14.0, 6.0, 6.0, 14.0, 1.0, 0.0, 2.0, 7.0, 13.0, 1.0, 0.0, 2.0, 5.0, 13.0, 1.0, 0.0, 2.0, 6.0, 12.0, 1.0, 0.0, 2.0, 8.0, 13.0, 1.0, 0.0, 2.0, 7.0, 12.0, 1.0, 6.0, 3.0, 1.0, 3.0, -1.0, -1.0, 1.0, 86.0, 6.0, 9.0, 0.0, 3.0, 1.0, 15.0, 3.0, 1.0, 1.0, 0.0, 1.0, 3.0, 20.0, 1.0, 2.0, 2.0, 2.0, 20.0, 1.0, 5.0, 3.0, 4.0, 2.0, 1.0, 5.0, 3.0, 5.0, 3.0, 1.0, 5.0, 3.0, 4.0, 3.0, 1.0, 5.0, 3.0, 3.0, 3.0, 1.0, 5.0, 3.0, 5.0, 2.0, 1.0, 5.0, 3.0, 6.0, 3.0, 1.0, 5.0, 3.0, 6.0, 2.0, 1.0, 5.0, 3.0, 7.0, 3.0, 1.0, 16.0, 4.0, 7.0, 2.0, 1.0, 16.0, 4.0, 8.0, 3.0, 1.0, 16.0, 4.0, 8.0, 2.0, 1.0, 16.0, 4.0, 7.0, 1.0, 1.0, 46.0, 6.0, 1.0, 8.0, 2.0, 1.0, 0.0, 163.0, 7.0, 8.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 8.0, 15.0, 1.0, 0.0, 3.0, 8.0, 16.0, 1.0, 0.0, 3.0, 7.0, 16.0, 1.0, 0.0, 3.0, 8.0, 14.0, 1.0, 0.0, 3.0, 9.0, 14.0, 1.0, 0.0, 3.0, 9.0, 15.0, 1.0, 0.0, 3.0, 7.0, 13.0, 1.0, 0.0, 3.0, 6.0, 15.0, 1.0, 0.0, 3.0, 8.0, 13.0, 1.0, 0.0, 3.0, 9.0, 13.0, 1.0, 0.0, 3.0, 7.0, 15.0, 1.0, 0.0, 3.0, 6.0, 16.0, 1.0, 0.0, 3.0, 5.0, 14.0, 1.0, 0.0, 3.0, 6.0, 12.0, 1.0, 0.0, 3.0, 4.0, 13.0, 1.0, 0.0, 3.0, 7.0, 11.0, 1.0, 0.0, 3.0, 6.0, 11.0, 1.0, 0.0, 3.0, 5.0, 15.0, 1.0, 0.0, 3.0, 4.0, 14.0, 1.0, 0.0, 3.0, 5.0, 13.0, 1.0, 0.0, 3.0, 7.0, 14.0, 1.0, 0.0, 3.0, 8.0, 12.0, 1.0, 0.0, 3.0, 7.0, 12.0, 1.0, 0.0, 3.0, 6.0, 14.0, 1.0, 0.0, 3.0, 6.0, 13.0, 1.0, 25.0, 4.0, 6.0, 17.0, 1.0, 25.0, 4.0, 7.0, 17.0, 1.0, 25.0, 4.0, 5.0, 16.0, 1.0, 25.0, 4.0, 4.0, 15.0, 1.0, 25.0, 4.0, 9.0, 16.0, 1.0, 55.0, 5.0, 1.0, 4.0, -1.0, -1.0, 1.0, 298.0, 8.0, 9.0, 2.0, 4.0, 7.0, 7.0, 6.0, 57.0, 7.0, 6.0, 1.0, 0.0, 5.0, 7.0, 7.0, 1.0, 0.0, 5.0, 7.0, 5.0, 1.0, 0.0, 5.0, 8.0, 6.0, 1.0, 0.0, 5.0, 6.0, 8.0, 1.0, 0.0, 5.0, 7.0, 8.0, 1.0, 0.0, 5.0, 7.0, 9.0, 1.0, 0.0, 5.0, 6.0, 9.0, 1.0, 0.0, 5.0, 5.0, 7.0, 1.0, 0.0, 5.0, 6.0, 6.0, 1.0, 0.0, 5.0, 6.0, 10.0, 2.0, 0.0, 5.0, 4.0, 7.0, 2.0, 0.0, 5.0, 6.0, 11.0, 1.0, 0.0, 5.0, 5.0, 6.0, 1.0, 0.0, 5.0, 7.0, 12.0, 1.0, 0.0, 5.0, 6.0, 12.0, 1.0, 0.0, 5.0, 8.0, 13.0, 1.0, 0.0, 5.0, 7.0, 11.0, 1.0, 0.0, 5.0, 8.0, 8.0, 1.0, 0.0, 5.0, 6.0, 5.0, 1.0, 0.0, 5.0, 5.0, 13.0, 1.0, 0.0, 5.0, 9.0, 13.0, 1.0, 0.0, 5.0, 9.0, 12.0, 1.0, 0.0, 5.0, 5.0, 14.0, 1.0, 0.0, 5.0, 4.0, 14.0, 1.0, 0.0, 5.0, 4.0, 15.0, 1.0, 0.0, 5.0, 3.0, 15.0, 1.0, 0.0, 5.0, 5.0, 16.0, 1.0, 0.0, 5.0, 4.0, 16.0, 1.0, 0.0, 5.0, 6.0, 17.0, 1.0, 0.0, 5.0, 6.0, 16.0, 1.0, 0.0, 5.0, 5.0, 17.0, 1.0, 0.0, 5.0, 7.0, 18.0, 1.0, 0.0, 5.0, 7.0, 17.0, 1.0, 0.0, 5.0, 7.0, 16.0, 1.0, 0.0, 5.0, 7.0, 15.0, 1.0, 0.0, 5.0, 6.0, 18.0, 1.0, 0.0, 5.0, 8.0, 16.0, 1.0, 0.0, 5.0, 8.0, 15.0, 1.0, 0.0, 5.0, 10.0, 13.0, 1.0, 0.0, 5.0, 4.0, 13.0, 1.0, 0.0, 5.0, 6.0, 15.0, 1.0, 0.0, 5.0, 5.0, 15.0, 1.0, 0.0, 5.0, 9.0, 15.0, 1.0, 0.0, 5.0, 9.0, 16.0, 1.0, 0.0, 5.0, 10.0, 14.0, 1.0, 0.0, 5.0, 8.0, 14.0, 1.0, 0.0, 5.0, 7.0, 14.0, 1.0, 0.0, 5.0, 6.0, 14.0, 1.0, 0.0, 5.0, 5.0, 9.0, 1.0, 0.0, 5.0, 5.0, 8.0, 1.0, 0.0, 5.0, 9.0, 14.0, 1.0, 0.0, 5.0, 8.0, 12.0, 1.0, 0.0, 5.0, 9.0, 17.0, 1.0, 0.0, 5.0, 7.0, 13.0, 1.0, 0.0, 5.0, 6.0, 13.0, 1.0, 0.0, 5.0, 8.0, 17.0, 1.0, 55.0, 6.0, 1.0, 6.0, 2.0, 1.0, 0.0, 418.0, 9.0, -1.0, 2.0, 6.0, 8.0, 3.0, 1.0, 81.0, 3.0, 1.0, 1.0, 0.0, 6.0, 4.0, 2.0, 1.0, 0.0, 6.0, 3.0, 20.0, 1.0, 0.0, 6.0, 2.0, 20.0, 1.0, 0.0, 6.0, 3.0, 3.0, 1.0, 0.0, 6.0, 5.0, 3.0, 1.0, 0.0, 6.0, 4.0, 3.0, 1.0, 0.0, 6.0, 5.0, 2.0, 1.0, 0.0, 6.0, 6.0, 3.0, 1.0, 0.0, 6.0, 6.0, 2.0, 1.0, 0.0, 6.0, 6.0, 4.0, 1.0, 0.0, 6.0, 7.0, 3.0, 1.0, 0.0, 6.0, 6.0, 5.0, 1.0, 0.0, 6.0, 8.0, 3.0, 1.0, 0.0, 6.0, 8.0, 2.0, 1.0, 0.0, 6.0, 6.0, 6.0, 1.0, 0.0, 6.0, 5.0, 6.0, 1.0, 0.0, 6.0, 7.0, 6.0, 1.0, 0.0, 6.0, 7.0, 2.0, 1.0, 0.0, 6.0, 7.0, 1.0, 1.0, 0.0, 6.0, 7.0, 5.0, 1.0, 0.0, 6.0, 7.0, 7.0, 1.0, 0.0, 6.0, 8.0, 6.0, 1.0, 0.0, 6.0, 5.0, 7.0, 1.0, 0.0, 6.0, 6.0, 8.0, 1.0, 0.0, 6.0, 4.0, 7.0, 2.0, 0.0, 6.0, 5.0, 8.0, 1.0, 0.0, 6.0, 7.0, 9.0, 1.0, 0.0, 6.0, 6.0, 9.0, 1.0, 0.0, 6.0, 6.0, 10.0, 2.0, 0.0, 6.0, 5.0, 9.0, 1.0, 0.0, 6.0, 7.0, 11.0, 1.0, 0.0, 6.0, 6.0, 11.0, 1.0, 0.0, 6.0, 8.0, 12.0, 1.0, 0.0, 6.0, 6.0, 12.0, 1.0, 0.0, 6.0, 7.0, 12.0, 1.0, 0.0, 6.0, 9.0, 13.0, 1.0, 0.0, 6.0, 9.0, 12.0, 1.0, 0.0, 6.0, 6.0, 13.0, 1.0, 0.0, 6.0, 5.0, 13.0, 1.0, 0.0, 6.0, 10.0, 14.0, 1.0, 0.0, 6.0, 9.0, 14.0, 1.0, 0.0, 6.0, 8.0, 14.0, 1.0, 0.0, 6.0, 5.0, 14.0, 1.0, 0.0, 6.0, 8.0, 13.0, 1.0, 0.0, 6.0, 7.0, 13.0, 1.0, 0.0, 6.0, 4.0, 15.0, 1.0, 0.0, 6.0, 8.0, 8.0, 1.0, 0.0, 6.0, 7.0, 8.0, 1.0, 0.0, 6.0, 8.0, 15.0, 1.0, 0.0, 6.0, 7.0, 15.0, 1.0, 0.0, 6.0, 3.0, 15.0, 1.0, 0.0, 6.0, 3.0, 14.0, 1.0, 0.0, 6.0, 5.0, 15.0, 1.0, 0.0, 6.0, 9.0, 16.0, 1.0, 0.0, 6.0, 3.0, 13.0, 1.0, 0.0, 6.0, 8.0, 16.0, 1.0, 0.0, 6.0, 9.0, 17.0, 1.0, 0.0, 6.0, 8.0, 17.0, 1.0, 0.0, 6.0, 7.0, 17.0, 1.0, 0.0, 6.0, 7.0, 16.0, 1.0, 0.0, 6.0, 7.0, 18.0, 1.0, 0.0, 6.0, 6.0, 18.0, 1.0, 0.0, 6.0, 6.0, 17.0, 1.0, 0.0, 6.0, 6.0, 16.0, 1.0, 0.0, 6.0, 5.0, 17.0, 1.0, 0.0, 6.0, 4.0, 16.0, 1.0, 0.0, 6.0, 3.0, 16.0, 1.0, 0.0, 6.0, 6.0, 15.0, 1.0, 0.0, 6.0, 5.0, 16.0, 1.0, 0.0, 6.0, 10.0, 13.0, 1.0, 0.0, 6.0, 4.0, 13.0, 1.0, 0.0, 6.0, 9.0, 15.0, 1.0, 0.0, 6.0, 4.0, 12.0, 1.0, 0.0, 6.0, 7.0, 14.0, 1.0, 0.0, 6.0, 6.0, 14.0, 1.0, 0.0, 6.0, 4.0, 14.0, 1.0, 75.0, 7.0, 4.0, 7.0, 1.0, 75.0, 7.0, 4.0, 6.0, 2.0, 151.0, 8.0, 3.0, 2.0, 1.0, 151.0, 8.0, 4.0, 6.0, 1.0, 229.0, 9.0, 1.0, -1.0, 2.0, 4.0, 0.0]
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
  drainages::Array{Field{Float64},1} = repeat(drainage,10000,false)
  runoffs::Array{Field{Float64},1} = drainages
  evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,180,false)
  initial_water_to_lake_centers = LatLonField{Float64}(lake_grid,
    Float64[  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0 ])
  initial_spillover_to_rivers = LatLonField{Float64}(hd_grid,
                                                     Float64[ 120.0 0.0 0.0 0.0
                                                              0.0 20.0 0.0 0.0
                                                              0.0 2.0*86400.0 0.0 0.0
                                                              0.0 0.0 1.0*86400.0 0.0 ])
  expected_river_inflow::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                               Float64[ 0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0 ])
  expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                                 Float64[ 0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0 ])
  expected_water_to_hd::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                              Float64[ 0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0 ])
  expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
              6 0 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
              0 6 6 0 0 0 0 0 0 0 0 0 7 7 7 0 0 0 0 0
              0 6 6 0 0 4 4 4 4 0 0 0 5 7 7 7 0 0 0 0
              0 6 6 0 4 4 0 4 4 0 7 5 7 5 7 7 7 0 0 0
              6 6 6 0 4 4 4 4 4 0 7 5 5 7 7 7 7 0 0 0
              0 6 6 0 0 4 0 4 0 0 0 7 5 7 3 7 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 7 7 7 7 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2
             2 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2
             0 2 2 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0
             0 2 2 0 0 2 2 2 2 0 0 0 3 1 1 1 0 0 0 0
             0 2 2 0 2 2 0 2 2 0 1 3 1 3 1 1 1 0 0 0
             2 2 2 0 2 2 2 2 2 0 1 3 3 1 1 1 1 0 0 0
             0 2 2 0 0 2 0 2 0 0 0 1 3 1 3 1 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 46.0
                 46.0 0 46.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 46.0
                 0 46.0 46.0 0 0 0 0 0 0 0 0 0 56.0 56.0 56.0 0 0 0 0 0
                 0 46.0 46.0 0 0 38.0 38.0 38.0 38.0 0 0 0 56.0 56.0 56.0 56.0 0 0 0 0
                 0 46.0 46.0 0 38.0 38.0 0 38.0 38.0 0 56.0 56.0 56.0 56.0 56.0 56.0 56.0 0 0 0
                 46.0 46.0 46.0 0 38.0 38.0 38.0 38.0 38.0 0 56.0 56.0 56.0 56.0 56.0 56.0 56.0 0 0 0
                 0 46.0 46.0 0 0 38.0 0 38.0 0 0 0 56.0 56.0 56.0 56.0 56.0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 56.0 56.0 56.0 56.0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 1.0 0.0
                 1.0 0.5581395349 0.0
                 1.0 0.1428571429 0.0 ])
  expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 6  0
                1 24 0
               15 1  0 ])
  expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 6  311
                1  43 1
                15 7  1 ])
  expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 1.0, 38.0, 6.0, 46.0, 49.0, 0.0, 0.0]
  # expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0
  #               5.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 5.0
  #               0.0 3.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.8 1.8 0.8 0.0 0.0 0.0 0.0 0.0
  #               0.0 3.0 3.0 0.0 0.0 2.0 2.0 1.0 2.0 0.0 0.0 0.0 2.8 1.8 1.8 0.8 0.0 0.0 0.0 0.0
  #               0.0 3.0 3.0 0.0 3.0 4.0 0.0 3.0 2.0 0.0 1.8 2.8 1.8 2.8 1.8 1.8 0.8 0.0 0.0 0.0
  #               2.0 2.0 3.0 0.0 3.0 4.0 3.0 3.0 2.0 0.0 1.8 2.8 2.8 1.8 1.8 1.8 0.8 0.0 0.0 0.0
  #               0.0 2.0 2.0 0.0 0.0 3.0 0.0 1.0 0.0 0.0 0.0 1.8 2.8 1.8 2.8 1.8 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.8 1.8 1.8 0.8 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])
  river_fields::RiverPrognosticFields,
    lake_model_prognostics::LakeModelPrognostics,_ =
    drive_hd_and_lake_model(river_parameters,lake_model_parameters,
                            lake_parameters_as_array,
                            drainages,runoffs,evaporations,
                            5,true,initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes::Field{Float64} =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test expected_river_inflow == river_fields.river_inflow
  @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test expected_lake_types == lake_types
  @test expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  @test isapprox(expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.000001)
  @test expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test  isapprox(expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.000001)
end

@testset "Lake model tests 13" begin
  #Initial water to rivers
  #No drainage,runoff and evaporation
  hd_grid = LatLonGrid(4,4,true)
  surface_model_grid = LatLonGrid(3,3,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
                                                Int64[-2  4  2  2
                                                       8 -2 -2  2
                                                       6  2  3 -2
                                                      -2  0  6  0 ]))
  river_reservoir_nums = LatLonField{Int64}(hd_grid,5)
  overland_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  base_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  river_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  overland_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  base_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  landsea_mask = LatLonField{Bool}(hd_grid,fill(false,4,4))
  set!(river_reservoir_nums,LatLonCoords(4,4),0)
  set!(overland_reservoir_nums,LatLonCoords(4,4),0)
  set!(base_reservoir_nums,LatLonCoords(4,4),0)
  set!(landsea_mask,LatLonCoords(4,4),true)
  set!(river_reservoir_nums,LatLonCoords(4,2),0)
  set!(overland_reservoir_nums,LatLonCoords(4,2),0)
  set!(base_reservoir_nums,LatLonCoords(4,2),0)
  set!(landsea_mask,LatLonCoords(4,2),true)
  river_parameters = RiverParameters(flow_directions,
                                     river_reservoir_nums,
                                     overland_reservoir_nums,
                                     base_reservoir_nums,
                                     river_retention_coefficients,
                                     overland_retention_coefficients,
                                     base_retention_coefficients,
                                     landsea_mask,
                                     hd_grid,86400.0,86400.0)
  lake_grid = LatLonGrid(20,20,true)
  cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
    Float64[ 2.5 3.0 2.5
             3.0 4.0 3.0
             2.5 3.0 2.5 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 2 1 1 1 1 1 1 1 1 1 1 2 2 2 2 1 1 1 1
                                                             1 1 1 1 1 2 2 1 1 1 1 2 2 2 2 2 1 1 1 1
                                                             1 1 1 1 1 3 3 3 3 1 1 1 1 2 2 2 2 1 1 1
                                                             1 1 1 2 3 3 1 3 3 2 2 1 2 1 2 2 2 2 1 1
                                                             1 1 1 1 3 3 3 3 3 1 2 1 1 2 2 2 2 2 1 1
                                                             1 1 1 1 1 3 1 3 1 1 1 2 1 2 2 2 2 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 2 2 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 1 1
                                                             1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3 ])

  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                      Int64[ 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1
                                                             1 2 1 3 3 3 3 3 3 3 3 3 2 2 2 2 3 3 3 1
                                                             3 1 1 3 3 2 2 3 3 3 3 2 2 2 2 2 3 3 3 3
                                                             3 1 1 3 3 1 1 1 1 3 3 3 2 2 2 2 2 3 3 3
                                                             3 1 1 2 1 1 3 1 1 2 2 2 2 2 2 2 2 2 3 3
                                                             1 1 1 3 1 1 1 1 1 3 2 2 2 2 2 2 2 2 3 3
                                                             3 1 1 3 3 1 3 1 3 3 3 2 2 2 2 2 2 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 2 2 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 3 3
                                                             3 3 3 1 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3  ])
  is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
    Bool[false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false  true
          true  true  true false false false false false false false false false  true  true  true  true false false false  true
         false  true  true false false  true  true false false false false  true  true  true  true  true false false false false
         false  true  true false false  true  true  true  true false false false  true  true  true  true  true false false false
         false  true  true  true  true  true false  true  true  true  true  true  true  true  true  true  true  true false false
          true  true  true false  true  true  true  true  true false  true  true  true  true  true  true  true  true false false
         false  true  true false false  true false  true false false false  true  true  true  true  true  true false false false
         false false false false false false false false false false false  true  true  true  true  true  true false false false
         false false false false false false false false false false false false  true  true false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false  true  true  true  true false
         false false false false false false false false false false false false false false false  true  true  true false false
         false false false  true false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false])
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_model_parameters::LakeModelParameters =
    LakeModelParameters(grid_specific_lake_model_parameters,
                        lake_grid,
                        hd_grid,
                        surface_model_grid,
                        cell_areas_on_surface_model_grid,
                        LakeModelSettings(),
                        9,
                        is_lake)
  lake_parameters_as_array::Vector{Float64} = Float64[9.0, 16.0, 1.0, -1.0, 0.0, 16.0, 4.0, 1.0, 16.0, 4.0, 1.0, 1.0, 3.0, 1.0, -1.0, 4.0, 2.0, 0.0, 46.0, 2.0, -1.0, 0.0, 14.0, 16.0, 7.0, 14.0, 16.0, 1.0, 0.0, 2.0, 14.0, 17.0, 1.0, 0.0, 2.0, 15.0, 16.0, 1.0, 3.0, 3.0, 15.0, 17.0, 1.0, 3.0, 3.0, 15.0, 18.0, 1.0, 3.0, 3.0, 14.0, 18.0, 1.0, 3.0, 3.0, 14.0, 19.0, 1.0, 10.0, 4.0, 1.0, -1.0, 4.0, 4.0, 0.0, 16.0, 3.0, 7.0, 0.0, 8.0, 15.0, 1.0, 8.0, 15.0, 1.0, 1.0, 3.0, 1.0, 5.0, -1.0, -1.0, 1.0, 91.0, 4.0, 8.0, 0.0, 7.0, 6.0, 16.0, 7.0, 6.0, 1.0, 0.0, 1.0, 6.0, 6.0, 1.0, 2.0, 2.0, 7.0, 5.0, 1.0, 2.0, 2.0, 7.0, 7.0, 1.0, 2.0, 2.0, 8.0, 6.0, 1.0, 2.0, 2.0, 7.0, 8.0, 1.0, 2.0, 2.0, 6.0, 8.0, 1.0, 2.0, 2.0, 6.0, 5.0, 1.0, 10.0, 3.0, 5.0, 6.0, 1.0, 10.0, 3.0, 5.0, 7.0, 1.0, 10.0, 3.0, 7.0, 9.0, 1.0, 10.0, 3.0, 6.0, 9.0, 1.0, 10.0, 3.0, 4.0, 7.0, 2.0, 10.0, 3.0, 5.0, 9.0, 1.0, 23.0, 4.0, 8.0, 8.0, 1.0, 23.0, 4.0, 5.0, 8.0, 1.0, 38.0, 5.0, 1.0, 7.0, -1.0, -1.0, 1.0, 41.0, 5.0, 7.0, 0.0, 6.0, 14.0, 6.0, 6.0, 14.0, 1.0, 0.0, 2.0, 7.0, 13.0, 1.0, 0.0, 2.0, 5.0, 13.0, 1.0, 0.0, 2.0, 6.0, 12.0, 1.0, 0.0, 2.0, 8.0, 13.0, 1.0, 0.0, 2.0, 7.0, 12.0, 1.0, 6.0, 3.0, 1.0, 3.0, -1.0, -1.0, 1.0, 86.0, 6.0, 9.0, 0.0, 3.0, 1.0, 15.0, 3.0, 1.0, 1.0, 0.0, 1.0, 3.0, 20.0, 1.0, 2.0, 2.0, 2.0, 20.0, 1.0, 5.0, 3.0, 4.0, 2.0, 1.0, 5.0, 3.0, 5.0, 3.0, 1.0, 5.0, 3.0, 4.0, 3.0, 1.0, 5.0, 3.0, 3.0, 3.0, 1.0, 5.0, 3.0, 5.0, 2.0, 1.0, 5.0, 3.0, 6.0, 3.0, 1.0, 5.0, 3.0, 6.0, 2.0, 1.0, 5.0, 3.0, 7.0, 3.0, 1.0, 16.0, 4.0, 7.0, 2.0, 1.0, 16.0, 4.0, 8.0, 3.0, 1.0, 16.0, 4.0, 8.0, 2.0, 1.0, 16.0, 4.0, 7.0, 1.0, 1.0, 46.0, 6.0, 1.0, 8.0, 2.0, 1.0, 0.0, 163.0, 7.0, 8.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 8.0, 15.0, 1.0, 0.0, 3.0, 8.0, 16.0, 1.0, 0.0, 3.0, 7.0, 16.0, 1.0, 0.0, 3.0, 8.0, 14.0, 1.0, 0.0, 3.0, 9.0, 14.0, 1.0, 0.0, 3.0, 9.0, 15.0, 1.0, 0.0, 3.0, 7.0, 13.0, 1.0, 0.0, 3.0, 6.0, 15.0, 1.0, 0.0, 3.0, 8.0, 13.0, 1.0, 0.0, 3.0, 9.0, 13.0, 1.0, 0.0, 3.0, 7.0, 15.0, 1.0, 0.0, 3.0, 6.0, 16.0, 1.0, 0.0, 3.0, 5.0, 14.0, 1.0, 0.0, 3.0, 6.0, 12.0, 1.0, 0.0, 3.0, 4.0, 13.0, 1.0, 0.0, 3.0, 7.0, 11.0, 1.0, 0.0, 3.0, 6.0, 11.0, 1.0, 0.0, 3.0, 5.0, 15.0, 1.0, 0.0, 3.0, 4.0, 14.0, 1.0, 0.0, 3.0, 5.0, 13.0, 1.0, 0.0, 3.0, 7.0, 14.0, 1.0, 0.0, 3.0, 8.0, 12.0, 1.0, 0.0, 3.0, 7.0, 12.0, 1.0, 0.0, 3.0, 6.0, 14.0, 1.0, 0.0, 3.0, 6.0, 13.0, 1.0, 25.0, 4.0, 6.0, 17.0, 1.0, 25.0, 4.0, 7.0, 17.0, 1.0, 25.0, 4.0, 5.0, 16.0, 1.0, 25.0, 4.0, 4.0, 15.0, 1.0, 25.0, 4.0, 9.0, 16.0, 1.0, 55.0, 5.0, 1.0, 4.0, -1.0, -1.0, 1.0, 298.0, 8.0, 9.0, 2.0, 4.0, 7.0, 7.0, 6.0, 57.0, 7.0, 6.0, 1.0, 0.0, 5.0, 7.0, 7.0, 1.0, 0.0, 5.0, 7.0, 5.0, 1.0, 0.0, 5.0, 8.0, 6.0, 1.0, 0.0, 5.0, 6.0, 8.0, 1.0, 0.0, 5.0, 7.0, 8.0, 1.0, 0.0, 5.0, 7.0, 9.0, 1.0, 0.0, 5.0, 6.0, 9.0, 1.0, 0.0, 5.0, 5.0, 7.0, 1.0, 0.0, 5.0, 6.0, 6.0, 1.0, 0.0, 5.0, 6.0, 10.0, 2.0, 0.0, 5.0, 4.0, 7.0, 2.0, 0.0, 5.0, 6.0, 11.0, 1.0, 0.0, 5.0, 5.0, 6.0, 1.0, 0.0, 5.0, 7.0, 12.0, 1.0, 0.0, 5.0, 6.0, 12.0, 1.0, 0.0, 5.0, 8.0, 13.0, 1.0, 0.0, 5.0, 7.0, 11.0, 1.0, 0.0, 5.0, 8.0, 8.0, 1.0, 0.0, 5.0, 6.0, 5.0, 1.0, 0.0, 5.0, 5.0, 13.0, 1.0, 0.0, 5.0, 9.0, 13.0, 1.0, 0.0, 5.0, 9.0, 12.0, 1.0, 0.0, 5.0, 5.0, 14.0, 1.0, 0.0, 5.0, 4.0, 14.0, 1.0, 0.0, 5.0, 4.0, 15.0, 1.0, 0.0, 5.0, 3.0, 15.0, 1.0, 0.0, 5.0, 5.0, 16.0, 1.0, 0.0, 5.0, 4.0, 16.0, 1.0, 0.0, 5.0, 6.0, 17.0, 1.0, 0.0, 5.0, 6.0, 16.0, 1.0, 0.0, 5.0, 5.0, 17.0, 1.0, 0.0, 5.0, 7.0, 18.0, 1.0, 0.0, 5.0, 7.0, 17.0, 1.0, 0.0, 5.0, 7.0, 16.0, 1.0, 0.0, 5.0, 7.0, 15.0, 1.0, 0.0, 5.0, 6.0, 18.0, 1.0, 0.0, 5.0, 8.0, 16.0, 1.0, 0.0, 5.0, 8.0, 15.0, 1.0, 0.0, 5.0, 10.0, 13.0, 1.0, 0.0, 5.0, 4.0, 13.0, 1.0, 0.0, 5.0, 6.0, 15.0, 1.0, 0.0, 5.0, 5.0, 15.0, 1.0, 0.0, 5.0, 9.0, 15.0, 1.0, 0.0, 5.0, 9.0, 16.0, 1.0, 0.0, 5.0, 10.0, 14.0, 1.0, 0.0, 5.0, 8.0, 14.0, 1.0, 0.0, 5.0, 7.0, 14.0, 1.0, 0.0, 5.0, 6.0, 14.0, 1.0, 0.0, 5.0, 5.0, 9.0, 1.0, 0.0, 5.0, 5.0, 8.0, 1.0, 0.0, 5.0, 9.0, 14.0, 1.0, 0.0, 5.0, 8.0, 12.0, 1.0, 0.0, 5.0, 9.0, 17.0, 1.0, 0.0, 5.0, 7.0, 13.0, 1.0, 0.0, 5.0, 6.0, 13.0, 1.0, 0.0, 5.0, 8.0, 17.0, 1.0, 55.0, 6.0, 1.0, 6.0, 2.0, 1.0, 0.0, 418.0, 9.0, -1.0, 2.0, 6.0, 8.0, 3.0, 1.0, 81.0, 3.0, 1.0, 1.0, 0.0, 6.0, 4.0, 2.0, 1.0, 0.0, 6.0, 3.0, 20.0, 1.0, 0.0, 6.0, 2.0, 20.0, 1.0, 0.0, 6.0, 3.0, 3.0, 1.0, 0.0, 6.0, 5.0, 3.0, 1.0, 0.0, 6.0, 4.0, 3.0, 1.0, 0.0, 6.0, 5.0, 2.0, 1.0, 0.0, 6.0, 6.0, 3.0, 1.0, 0.0, 6.0, 6.0, 2.0, 1.0, 0.0, 6.0, 6.0, 4.0, 1.0, 0.0, 6.0, 7.0, 3.0, 1.0, 0.0, 6.0, 6.0, 5.0, 1.0, 0.0, 6.0, 8.0, 3.0, 1.0, 0.0, 6.0, 8.0, 2.0, 1.0, 0.0, 6.0, 6.0, 6.0, 1.0, 0.0, 6.0, 5.0, 6.0, 1.0, 0.0, 6.0, 7.0, 6.0, 1.0, 0.0, 6.0, 7.0, 2.0, 1.0, 0.0, 6.0, 7.0, 1.0, 1.0, 0.0, 6.0, 7.0, 5.0, 1.0, 0.0, 6.0, 7.0, 7.0, 1.0, 0.0, 6.0, 8.0, 6.0, 1.0, 0.0, 6.0, 5.0, 7.0, 1.0, 0.0, 6.0, 6.0, 8.0, 1.0, 0.0, 6.0, 4.0, 7.0, 2.0, 0.0, 6.0, 5.0, 8.0, 1.0, 0.0, 6.0, 7.0, 9.0, 1.0, 0.0, 6.0, 6.0, 9.0, 1.0, 0.0, 6.0, 6.0, 10.0, 2.0, 0.0, 6.0, 5.0, 9.0, 1.0, 0.0, 6.0, 7.0, 11.0, 1.0, 0.0, 6.0, 6.0, 11.0, 1.0, 0.0, 6.0, 8.0, 12.0, 1.0, 0.0, 6.0, 6.0, 12.0, 1.0, 0.0, 6.0, 7.0, 12.0, 1.0, 0.0, 6.0, 9.0, 13.0, 1.0, 0.0, 6.0, 9.0, 12.0, 1.0, 0.0, 6.0, 6.0, 13.0, 1.0, 0.0, 6.0, 5.0, 13.0, 1.0, 0.0, 6.0, 10.0, 14.0, 1.0, 0.0, 6.0, 9.0, 14.0, 1.0, 0.0, 6.0, 8.0, 14.0, 1.0, 0.0, 6.0, 5.0, 14.0, 1.0, 0.0, 6.0, 8.0, 13.0, 1.0, 0.0, 6.0, 7.0, 13.0, 1.0, 0.0, 6.0, 4.0, 15.0, 1.0, 0.0, 6.0, 8.0, 8.0, 1.0, 0.0, 6.0, 7.0, 8.0, 1.0, 0.0, 6.0, 8.0, 15.0, 1.0, 0.0, 6.0, 7.0, 15.0, 1.0, 0.0, 6.0, 3.0, 15.0, 1.0, 0.0, 6.0, 3.0, 14.0, 1.0, 0.0, 6.0, 5.0, 15.0, 1.0, 0.0, 6.0, 9.0, 16.0, 1.0, 0.0, 6.0, 3.0, 13.0, 1.0, 0.0, 6.0, 8.0, 16.0, 1.0, 0.0, 6.0, 9.0, 17.0, 1.0, 0.0, 6.0, 8.0, 17.0, 1.0, 0.0, 6.0, 7.0, 17.0, 1.0, 0.0, 6.0, 7.0, 16.0, 1.0, 0.0, 6.0, 7.0, 18.0, 1.0, 0.0, 6.0, 6.0, 18.0, 1.0, 0.0, 6.0, 6.0, 17.0, 1.0, 0.0, 6.0, 6.0, 16.0, 1.0, 0.0, 6.0, 5.0, 17.0, 1.0, 0.0, 6.0, 4.0, 16.0, 1.0, 0.0, 6.0, 3.0, 16.0, 1.0, 0.0, 6.0, 6.0, 15.0, 1.0, 0.0, 6.0, 5.0, 16.0, 1.0, 0.0, 6.0, 10.0, 13.0, 1.0, 0.0, 6.0, 4.0, 13.0, 1.0, 0.0, 6.0, 9.0, 15.0, 1.0, 0.0, 6.0, 4.0, 12.0, 1.0, 0.0, 6.0, 7.0, 14.0, 1.0, 0.0, 6.0, 6.0, 14.0, 1.0, 0.0, 6.0, 4.0, 14.0, 1.0, 75.0, 7.0, 4.0, 7.0, 1.0, 75.0, 7.0, 4.0, 6.0, 2.0, 151.0, 8.0, 3.0, 2.0, 1.0, 151.0, 8.0, 4.0, 6.0, 1.0, 229.0, 9.0, 1.0, -1.0, 2.0, 4.0, 0.0]
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
  drainages::Array{Field{Float64},1} = repeat(drainage,10000,false)
  runoffs::Array{Field{Float64},1} = drainages
  evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,180,false)
  initial_water_to_lake_centers = LatLonField{Float64}(lake_grid,
    Float64[  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 430.0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0 ])
  initial_spillover_to_rivers = LatLonField{Float64}(hd_grid,
                                                     Float64[ 0.0 0.0 0.0 0.0
                                                              0.0 0.0 0.0 0.0
                                                              0.0 0.0 0.0 0.0
                                                              0.0 0.0 0.0 0.0 ])
  expected_river_inflow::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                               Float64[ 0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0 ])
  expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                                 Float64[ 0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0 ])
  expected_water_to_hd::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                              Float64[ 0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0 ])
  expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
              6 9 6 0 0 0 0 0 0 0 0 0 9 9 8 9 0 0 0 6
              0 6 6 0 0 9 9 0 0 0 0 9 7 7 7 8 0 0 0 0
              0 6 6 0 0 4 4 4 4 0 0 0 5 7 7 7 8 0 0 0
              0 6 6 9 4 4 0 4 4 0 7 5 7 5 7 7 7 8 0 0
              6 6 6 0 4 4 4 4 4 0 7 5 5 7 7 7 7 8 0 0
              0 6 6 0 0 4 0 4 0 0 0 7 5 7 3 7 8 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 8 7 7 7 7 8 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 8 8 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3
             3 1 3 0 0 0 0 0 0 0 0 0 1 1 3 1 0 0 0 3
             0 3 3 0 0 1 1 0 0 0 0 1 3 3 3 3 0 0 0 0
             0 3 3 0 0 3 3 3 3 0 0 0 3 3 3 3 3 0 0 0
             0 3 3 1 3 3 0 3 3 0 3 3 3 3 3 3 3 3 0 0
             3 3 3 0 3 3 3 3 3 0 3 3 3 3 3 3 3 3 0 0
             0 3 3 0 0 3 0 3 0 0 0 3 3 3 3 3 3 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 3 3 3 3 3 3 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 3 3 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
  expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     430.0
                   430.0 430.0 430.0 0     0     0     0     0     0     0     #=
                =# 0     0     430.0 430.0 430.0 430.0 0     0     0     430.0
                   0     430.0 430.0 0     0     430.0 430.0 0     0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 0     0     0     0
                   0     430.0 430.0 0     0     430.0 430.0 430.0 430.0 0     #=
                =# 0     0     430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     430.0 430.0 430.0 430.0 430.0 0     430.0 430.0 0     #=
                =# 430.0 430.0 430.0 430.0 430.0 430.0 430.0 430.0 0     0
                   430.0 430.0 430.0 0     430.0 430.0 430.0 430.0 430.0 0     #=
                =# 430.0 430.0 430.0 430.0 430.0 430.0 430.0 430.0 0     0
                   0     430.0 430.0 0     0     430.0 0     430.0 0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     430.0 430.0 0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0 ])
  expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 1.0 0.0
                 1.0 0.976744186 0.0
                 1.0 0.1428571429 0.0])
  expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 6  0
                1 42 0
               15  1 0 ])
  expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 6  311
                1  43 1
                15 7  1  ])
  expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 1.0, 38.0, 6.0, 46.0, 55.0, 55.0, 229.0]
  # expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 7.0
  #               8.0 1.0 6.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 4.0 3.0 0.0 0.0 0.0 8.0
  #               0.0 6.0 6.0 0.0 0.0 1.0 2.0 0.0 0.0 0.0 0.0 3.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0 0.0
  #               0.0 6.0 6.0 0.0 0.0 6.0 6.0 5.0 6.0 0.0 0.0 0.0 7.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0
  #               0.0 6.0 6.0 3.0 7.0 8.0 0.0 7.0 6.0 0.0 6.0 7.0 6.0 7.0 6.0 6.0 5.0 4.0 0.0 0.0
  #               5.0 5.0 6.0 0.0 7.0 8.0 7.0 7.0 6.0 0.0 6.0 7.0 7.0 6.0 6.0 6.0 5.0 4.0 0.0 0.0
  #               0.0 5.0 5.0 0.0 0.0 7.0 0.0 5.0 0.0 0.0 0.0 6.0 7.0 6.0 7.0 6.0 4.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0 6.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0 4.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])

  first_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  first_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  first_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[   0.0 0.0 0.0 0.0
                                  275.0 0.0 0.0 0.0
                                    0.0 0.0 0.0 0.0
                                    0.0 0.0 0.0 0.0 ])
  first_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             6 0 0 0 0 0 0 0 0 0 0 0 0 0 8 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 7 7 7 8 0 0 0 0
             0 0 0 0 0 4 4 4 4 0 0 0 5 7 7 7 8 0 0 0
             0 0 0 0 4 4 0 4 4 0 7 5 7 5 7 7 7 8 0 0
             0 0 0 0 4 4 4 4 4 0 7 5 5 7 7 7 7 8 0 0
             0 0 0 0 0 4 0 4 0 0 0 7 5 7 3 7 8 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 8 7 7 7 7 8 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 8 8 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
  first_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             1 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 3 3 3 2 0 0 0 0
             0 0 0 0 0 3 3 3 3 0 0 0 3 3 3 3 2 0 0 0
             0 0 0 0 3 3 0 3 3 0 3 3 3 3 3 3 3 2 0 0
             0 0 0 0 3 3 3 3 3 0 3 3 3 3 3 3 3 2 0 0
             0 0 0 0 0 3 0 3 0 0 0 3 3 3 3 3 2 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 2 3 3 3 3 2 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 2 2 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  first_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
      Float64[ 0.0 0.0 0.0 0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0 #=
               =#   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0 0.0
               0.0 0.0 0.0 0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0 #=
               =#   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0 0.0
               0.0 0.0 0.0 0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0 #=
               =#   0.0   0.0   0.0 155.0   0.0   0.0   0.0 0.0 0.0
               0.0 0.0 0.0 0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0 #=
               =#   0.0 155.0 155.0 155.0 155.0   0.0   0.0 0.0 0.0
               0.0 0.0 0.0 0.0   0.0 155.0 155.0 155.0 155.0 0.0   0.0 #=
               =#   0.0 155.0 155.0 155.0 155.0 155.0   0.0 0.0 0.0
               0.0 0.0 0.0 0.0 155.0 155.0   0.0 155.0 155.0 0.0 155.0 #=
               =# 155.0 155.0 155.0 155.0 155.0 155.0 155.0 0.0 0.0
               0.0 0.0 0.0 0.0 155.0 155.0 155.0 155.0 155.0 0.0 155.0 #=
               =# 155.0 155.0 155.0 155.0 155.0 155.0 155.0 0.0 0.0
               0.0 0.0 0.0 0.0   0.0 155.0   0.0 155.0   0.0 0.0   0.0 #=
               =# 155.0 155.0 155.0 155.0 155.0 155.0   0.0 0.0 0.0
               0.0 0.0 0.0 0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0 #=
               =# 155.0 155.0 155.0 155.0 155.0 155.0   0.0 0.0 0.0
               0.0 0.0 0.0 0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0 #=
               =#   0.0 155.0 155.0   0.0   0.0   0.0   0.0 0.0 0.0
               0.0 0.0 0.0 0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0 #=
               =#   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0 0.0
               0.0 0.0 0.0 0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0 #=
               =#   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0 0.0
               0.0 0.0 0.0 0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0 #=
               =#   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0 0.0
               0.0 0.0 0.0 0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0 #=
               =#   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0 0.0
               0.0 0.0 0.0 0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0 #=
               =#   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0 0.0
               0.0 0.0 0.0 0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0 #=
               =#   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0 0.0
               0.0 0.0 0.0 0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0 #=
               =#   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0 0.0
               0.0 0.0 0.0 0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0 #=
               =#   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0 0.0
               0.0 0.0 0.0 0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0 #=
               =#   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0 0.0
               0.0 0.0 0.0 0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0 #=
               =#   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0 0.0   ])
  first_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0/15.0 1.0 0.0
                 1.0 34.0/43.0 0.0
                 1.0 1.0/7.0 0.0 ])
  first_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 1 6  0
               1 34 0
              15 1  0])
  first_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 6  311
                1  43 1
                15 7  1  ])
  first_intermediate_expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 1.0, 38.0, 6.0, 0.0, 55.0, 55.0, 0.0]
  # first_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 2.0 1.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 2.0 2.0 1.0 2.0 0.0 0.0 0.0 3.0 2.0 2.0 1.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 3.0 4.0 0.0 3.0 2.0 0.0 2.0 3.0 2.0 3.0 2.0 2.0 1.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 3.0 4.0 3.0 3.0 2.0 0.0 2.0 3.0 3.0 2.0 2.0 2.0 1.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 3.0 0.0 1.0 0.0 0.0 0.0 2.0 3.0 2.0 3.0 2.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 2.0 2.0 1.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])

  second_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  second_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  second_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  second_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
              6 9 6 0 0 0 0 0 0 0 0 0 9 9 8 9 0 0 0 6
              0 6 6 0 0 9 9 0 0 0 0 9 7 7 7 8 0 0 0 0
              0 6 6 0 0 4 4 4 4 0 0 0 5 7 7 7 8 0 0 0
              0 6 6 9 4 4 0 4 4 0 7 5 7 5 7 7 7 8 0 0
              6 6 6 0 4 4 4 4 4 0 7 5 5 7 7 7 7 8 0 0
              0 6 6 0 0 4 0 4 0 0 0 7 5 7 3 7 8 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 8 7 7 7 7 8 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 8 8 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  second_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3
             3 1 3 0 0 0 0 0 0 0 0 0 1 1 3 1 0 0 0 3
             0 3 3 0 0 1 1 0 0 0 0 1 3 3 3 3 0 0 0 0
             0 3 3 0 0 3 3 3 3 0 0 0 3 3 3 3 3 0 0 0
             0 3 3 1 3 3 0 3 3 0 3 3 3 3 3 3 3 3 0 0
             3 3 3 0 3 3 3 3 3 0 3 3 3 3 3 3 3 3 0 0
             0 3 3 0 0 3 0 3 0 0 0 3 3 3 3 3 3 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 3 3 3 3 3 3 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 3 3 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  second_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
      Float64[  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0 430.0
              430.0 430.0 430.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0   0.0 430.0 430.0 430.0 430.0   0.0   0.0 0.0 430.0
                0.0 430.0 430.0   0.0   0.0 430.0 430.0   0.0   0.0 0.0   0.0 430.0 430.0 430.0 430.0 430.0   0.0   0.0 0.0   0.0
                0.0 430.0 430.0   0.0   0.0 430.0 430.0 430.0 430.0 0.0   0.0   0.0 430.0 430.0 430.0 430.0 430.0   0.0 0.0   0.0
                0.0 430.0 430.0 430.0 430.0 430.0   0.0 430.0 430.0 0.0 430.0 430.0 430.0 430.0 430.0 430.0 430.0 430.0 0.0   0.0
              430.0 430.0 430.0   0.0 430.0 430.0 430.0 430.0 430.0 0.0 430.0 430.0 430.0 430.0 430.0 430.0 430.0 430.0 0.0   0.0
                0.0 430.0 430.0   0.0   0.0 430.0   0.0 430.0   0.0 0.0   0.0 430.0 430.0 430.0 430.0 430.0 430.0   0.0 0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0 430.0 430.0 430.0 430.0 430.0 430.0   0.0 0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0   0.0 430.0 430.0   0.0   0.0   0.0   0.0 0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 0.0   0.0 ])
  second_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 1.0 0.0
                 1.0 42.0/43.0 0.0
                 1.0 1.0/7.0 0.0 ])
  second_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15  6 0
                1 42 0
               15  1 0 ])
  second_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 6  311
                1  43 1
                15 7  1 ])
  second_intermediate_expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 1.0, 38.0, 6.0, 46.0, 55.0, 55.0, 229.0]
  # second_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 2.0 1.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 3.0 3.0 2.0 3.0 0.0 0.0 0.0 4.0 3.0 3.0 2.0 1.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 4.0 5.0 0.0 4.0 3.0 0.0 3.0 4.0 3.0 4.0 3.0 3.0 2.0 1.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 4.0 5.0 4.0 4.0 3.0 0.0 3.0 4.0 4.0 3.0 3.0 3.0 2.0 1.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 4.0 0.0 2.0 0.0 0.0 0.0 3.0 4.0 3.0 4.0 3.0 1.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 3.0 3.0 3.0 2.0 1.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])
  river_fields::RiverPrognosticFields,
    lake_model_prognostics::LakeModelPrognostics,
    lake_model_diagnostics::LakeModelDiagnostics =
    drive_hd_and_lake_model(river_parameters,lake_model_parameters,
                            lake_parameters_as_array,
                            drainages,runoffs,evaporations,
                            0,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes::Array{Float64} = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes::Field{Float64} =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test first_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(first_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test first_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test first_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test first_intermediate_expected_lake_types == lake_types
  @test first_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == first_intermediate_expected_diagnostic_lake_volumes
  @test isapprox(first_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.000001)
  @test first_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test first_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test first_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            1,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=-1)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test second_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(second_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test second_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test second_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test second_intermediate_expected_lake_types == lake_types
  @test second_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == second_intermediate_expected_diagnostic_lake_volumes
  @test isapprox(second_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.000001)
  @test second_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test second_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test second_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,_ =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            2,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=1)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test expected_river_inflow == river_fields.river_inflow
  @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test expected_lake_types == lake_types
  @test expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  @test isapprox(expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
  @test expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test expected_true_lake_depths == lake_model_parameters.true_lake_depths
end

@testset "Lake model tests 14" begin
  #Initial water to lakes with overflow to downstream lake for an
  #ensemble of lakes - no evaporation, runoff or drainage
  hd_grid = LatLonGrid(4,4,true)
  surface_model_grid = LatLonGrid(3,3,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
                                                Int64[-2  4  2  2
                                                       8 -2 -2  2
                                                       6  2  3 -2
                                                      -2  0  6  0 ]))
  river_reservoir_nums = LatLonField{Int64}(hd_grid,5)
  overland_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  base_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  river_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  overland_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  base_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  landsea_mask = LatLonField{Bool}(hd_grid,fill(false,4,4))
  set!(river_reservoir_nums,LatLonCoords(4,4),0)
  set!(overland_reservoir_nums,LatLonCoords(4,4),0)
  set!(base_reservoir_nums,LatLonCoords(4,4),0)
  set!(landsea_mask,LatLonCoords(4,4),true)
  set!(river_reservoir_nums,LatLonCoords(4,2),0)
  set!(overland_reservoir_nums,LatLonCoords(4,2),0)
  set!(base_reservoir_nums,LatLonCoords(4,2),0)
  set!(landsea_mask,LatLonCoords(4,2),true)
  river_parameters = RiverParameters(flow_directions,
                                     river_reservoir_nums,
                                     overland_reservoir_nums,
                                     base_reservoir_nums,
                                     river_retention_coefficients,
                                     overland_retention_coefficients,
                                     base_retention_coefficients,
                                     landsea_mask,
                                     hd_grid,86400.0,86400.0)
  lake_grid = LatLonGrid(20,20,true)
  cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
    Float64[ 2.5 3.0 2.5
             3.0 4.0 3.0
             2.5 3.0 2.5 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 2 1 1 1 1 1 1 1 1 1 1 2 2 2 2 1 1 1 1
                                                             1 1 1 1 1 2 2 1 1 1 1 2 2 2 2 2 1 1 1 1
                                                             1 1 1 1 1 3 3 3 3 1 1 1 1 2 2 2 2 1 1 1
                                                             1 1 1 2 3 3 1 3 3 2 2 1 2 1 2 2 2 2 1 1
                                                             1 1 1 1 3 3 3 3 3 1 2 1 1 2 2 2 2 2 1 1
                                                             1 1 1 1 1 3 1 3 1 1 1 2 1 2 2 2 2 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 2 2 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 1 1
                                                             1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3 ])

  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                      Int64[ 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1
                                                             1 2 1 3 3 3 3 3 3 3 3 3 2 2 2 2 3 3 3 1
                                                             3 1 1 3 3 2 2 3 3 3 3 2 2 2 2 2 3 3 3 3
                                                             3 1 1 3 3 1 1 1 1 3 3 3 2 2 2 2 2 3 3 3
                                                             3 1 1 2 1 1 3 1 1 2 2 2 2 2 2 2 2 2 3 3
                                                             1 1 1 3 1 1 1 1 1 3 2 2 2 2 2 2 2 2 3 3
                                                             3 1 1 3 3 1 3 1 3 3 3 2 2 2 2 2 2 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 2 2 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 3 3
                                                             3 3 3 1 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3  ])
  is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
    Bool[false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false  true
          true  true  true false false false false false false false false false  true  true  true  true false false false  true
         false  true  true false false  true  true false false false false  true  true  true  true  true false false false false
         false  true  true false false  true  true  true  true false false false  true  true  true  true  true false false false
         false  true  true  true  true  true false  true  true  true  true  true  true  true  true  true  true  true false false
          true  true  true false  true  true  true  true  true false  true  true  true  true  true  true  true  true false false
         false  true  true false false  true false  true false false false  true  true  true  true  true  true false false false
         false false false false false false false false false false false  true  true  true  true  true  true false false false
         false false false false false false false false false false false false  true  true false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false  true  true  true  true false
         false false false false false false false false false false false false false false false  true  true  true false false
         false false false  true false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false])
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_model_parameters::LakeModelParameters =
    LakeModelParameters(grid_specific_lake_model_parameters,
                        lake_grid,
                        hd_grid,
                        surface_model_grid,
                        cell_areas_on_surface_model_grid,
                        LakeModelSettings(),
                        9,
                        is_lake)
  lake_parameters_as_array::Vector{Float64} = Float64[9.0, 16.0, 1.0, -1.0, 0.0, 16.0, 4.0, 1.0, 16.0, 4.0, 1.0, 1.0, 3.0, 1.0, -1.0, 4.0, 2.0, 0.0, 46.0, 2.0, -1.0, 0.0, 14.0, 16.0, 7.0, 14.0, 16.0, 1.0, 0.0, 2.0, 14.0, 17.0, 1.0, 0.0, 2.0, 15.0, 16.0, 1.0, 3.0, 3.0, 15.0, 17.0, 1.0, 3.0, 3.0, 15.0, 18.0, 1.0, 3.0, 3.0, 14.0, 18.0, 1.0, 3.0, 3.0, 14.0, 19.0, 1.0, 10.0, 4.0, 1.0, -1.0, 4.0, 4.0, 0.0, 16.0, 3.0, 7.0, 0.0, 8.0, 15.0, 1.0, 8.0, 15.0, 1.0, 1.0, 3.0, 1.0, 5.0, -1.0, -1.0, 1.0, 91.0, 4.0, 8.0, 0.0, 7.0, 6.0, 16.0, 7.0, 6.0, 1.0, 0.0, 1.0, 6.0, 6.0, 1.0, 2.0, 2.0, 7.0, 5.0, 1.0, 2.0, 2.0, 7.0, 7.0, 1.0, 2.0, 2.0, 8.0, 6.0, 1.0, 2.0, 2.0, 7.0, 8.0, 1.0, 2.0, 2.0, 6.0, 8.0, 1.0, 2.0, 2.0, 6.0, 5.0, 1.0, 10.0, 3.0, 5.0, 6.0, 1.0, 10.0, 3.0, 5.0, 7.0, 1.0, 10.0, 3.0, 7.0, 9.0, 1.0, 10.0, 3.0, 6.0, 9.0, 1.0, 10.0, 3.0, 4.0, 7.0, 2.0, 10.0, 3.0, 5.0, 9.0, 1.0, 23.0, 4.0, 8.0, 8.0, 1.0, 23.0, 4.0, 5.0, 8.0, 1.0, 38.0, 5.0, 1.0, 7.0, -1.0, -1.0, 1.0, 41.0, 5.0, 7.0, 0.0, 6.0, 14.0, 6.0, 6.0, 14.0, 1.0, 0.0, 2.0, 7.0, 13.0, 1.0, 0.0, 2.0, 5.0, 13.0, 1.0, 0.0, 2.0, 6.0, 12.0, 1.0, 0.0, 2.0, 8.0, 13.0, 1.0, 0.0, 2.0, 7.0, 12.0, 1.0, 6.0, 3.0, 1.0, 3.0, -1.0, -1.0, 1.0, 86.0, 6.0, 9.0, 0.0, 3.0, 1.0, 15.0, 3.0, 1.0, 1.0, 0.0, 1.0, 3.0, 20.0, 1.0, 2.0, 2.0, 2.0, 20.0, 1.0, 5.0, 3.0, 4.0, 2.0, 1.0, 5.0, 3.0, 5.0, 3.0, 1.0, 5.0, 3.0, 4.0, 3.0, 1.0, 5.0, 3.0, 3.0, 3.0, 1.0, 5.0, 3.0, 5.0, 2.0, 1.0, 5.0, 3.0, 6.0, 3.0, 1.0, 5.0, 3.0, 6.0, 2.0, 1.0, 5.0, 3.0, 7.0, 3.0, 1.0, 16.0, 4.0, 7.0, 2.0, 1.0, 16.0, 4.0, 8.0, 3.0, 1.0, 16.0, 4.0, 8.0, 2.0, 1.0, 16.0, 4.0, 7.0, 1.0, 1.0, 46.0, 6.0, 1.0, 8.0, 2.0, 1.0, 0.0, 163.0, 7.0, 8.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 8.0, 15.0, 1.0, 0.0, 3.0, 8.0, 16.0, 1.0, 0.0, 3.0, 7.0, 16.0, 1.0, 0.0, 3.0, 8.0, 14.0, 1.0, 0.0, 3.0, 9.0, 14.0, 1.0, 0.0, 3.0, 9.0, 15.0, 1.0, 0.0, 3.0, 7.0, 13.0, 1.0, 0.0, 3.0, 6.0, 15.0, 1.0, 0.0, 3.0, 8.0, 13.0, 1.0, 0.0, 3.0, 9.0, 13.0, 1.0, 0.0, 3.0, 7.0, 15.0, 1.0, 0.0, 3.0, 6.0, 16.0, 1.0, 0.0, 3.0, 5.0, 14.0, 1.0, 0.0, 3.0, 6.0, 12.0, 1.0, 0.0, 3.0, 4.0, 13.0, 1.0, 0.0, 3.0, 7.0, 11.0, 1.0, 0.0, 3.0, 6.0, 11.0, 1.0, 0.0, 3.0, 5.0, 15.0, 1.0, 0.0, 3.0, 4.0, 14.0, 1.0, 0.0, 3.0, 5.0, 13.0, 1.0, 0.0, 3.0, 7.0, 14.0, 1.0, 0.0, 3.0, 8.0, 12.0, 1.0, 0.0, 3.0, 7.0, 12.0, 1.0, 0.0, 3.0, 6.0, 14.0, 1.0, 0.0, 3.0, 6.0, 13.0, 1.0, 25.0, 4.0, 6.0, 17.0, 1.0, 25.0, 4.0, 7.0, 17.0, 1.0, 25.0, 4.0, 5.0, 16.0, 1.0, 25.0, 4.0, 4.0, 15.0, 1.0, 25.0, 4.0, 9.0, 16.0, 1.0, 55.0, 5.0, 1.0, 4.0, -1.0, -1.0, 1.0, 298.0, 8.0, 9.0, 2.0, 4.0, 7.0, 7.0, 6.0, 57.0, 7.0, 6.0, 1.0, 0.0, 5.0, 7.0, 7.0, 1.0, 0.0, 5.0, 7.0, 5.0, 1.0, 0.0, 5.0, 8.0, 6.0, 1.0, 0.0, 5.0, 6.0, 8.0, 1.0, 0.0, 5.0, 7.0, 8.0, 1.0, 0.0, 5.0, 7.0, 9.0, 1.0, 0.0, 5.0, 6.0, 9.0, 1.0, 0.0, 5.0, 5.0, 7.0, 1.0, 0.0, 5.0, 6.0, 6.0, 1.0, 0.0, 5.0, 6.0, 10.0, 2.0, 0.0, 5.0, 4.0, 7.0, 2.0, 0.0, 5.0, 6.0, 11.0, 1.0, 0.0, 5.0, 5.0, 6.0, 1.0, 0.0, 5.0, 7.0, 12.0, 1.0, 0.0, 5.0, 6.0, 12.0, 1.0, 0.0, 5.0, 8.0, 13.0, 1.0, 0.0, 5.0, 7.0, 11.0, 1.0, 0.0, 5.0, 8.0, 8.0, 1.0, 0.0, 5.0, 6.0, 5.0, 1.0, 0.0, 5.0, 5.0, 13.0, 1.0, 0.0, 5.0, 9.0, 13.0, 1.0, 0.0, 5.0, 9.0, 12.0, 1.0, 0.0, 5.0, 5.0, 14.0, 1.0, 0.0, 5.0, 4.0, 14.0, 1.0, 0.0, 5.0, 4.0, 15.0, 1.0, 0.0, 5.0, 3.0, 15.0, 1.0, 0.0, 5.0, 5.0, 16.0, 1.0, 0.0, 5.0, 4.0, 16.0, 1.0, 0.0, 5.0, 6.0, 17.0, 1.0, 0.0, 5.0, 6.0, 16.0, 1.0, 0.0, 5.0, 5.0, 17.0, 1.0, 0.0, 5.0, 7.0, 18.0, 1.0, 0.0, 5.0, 7.0, 17.0, 1.0, 0.0, 5.0, 7.0, 16.0, 1.0, 0.0, 5.0, 7.0, 15.0, 1.0, 0.0, 5.0, 6.0, 18.0, 1.0, 0.0, 5.0, 8.0, 16.0, 1.0, 0.0, 5.0, 8.0, 15.0, 1.0, 0.0, 5.0, 10.0, 13.0, 1.0, 0.0, 5.0, 4.0, 13.0, 1.0, 0.0, 5.0, 6.0, 15.0, 1.0, 0.0, 5.0, 5.0, 15.0, 1.0, 0.0, 5.0, 9.0, 15.0, 1.0, 0.0, 5.0, 9.0, 16.0, 1.0, 0.0, 5.0, 10.0, 14.0, 1.0, 0.0, 5.0, 8.0, 14.0, 1.0, 0.0, 5.0, 7.0, 14.0, 1.0, 0.0, 5.0, 6.0, 14.0, 1.0, 0.0, 5.0, 5.0, 9.0, 1.0, 0.0, 5.0, 5.0, 8.0, 1.0, 0.0, 5.0, 9.0, 14.0, 1.0, 0.0, 5.0, 8.0, 12.0, 1.0, 0.0, 5.0, 9.0, 17.0, 1.0, 0.0, 5.0, 7.0, 13.0, 1.0, 0.0, 5.0, 6.0, 13.0, 1.0, 0.0, 5.0, 8.0, 17.0, 1.0, 55.0, 6.0, 1.0, 6.0, 2.0, 1.0, 0.0, 418.0, 9.0, -1.0, 2.0, 6.0, 8.0, 3.0, 1.0, 81.0, 3.0, 1.0, 1.0, 0.0, 6.0, 4.0, 2.0, 1.0, 0.0, 6.0, 3.0, 20.0, 1.0, 0.0, 6.0, 2.0, 20.0, 1.0, 0.0, 6.0, 3.0, 3.0, 1.0, 0.0, 6.0, 5.0, 3.0, 1.0, 0.0, 6.0, 4.0, 3.0, 1.0, 0.0, 6.0, 5.0, 2.0, 1.0, 0.0, 6.0, 6.0, 3.0, 1.0, 0.0, 6.0, 6.0, 2.0, 1.0, 0.0, 6.0, 6.0, 4.0, 1.0, 0.0, 6.0, 7.0, 3.0, 1.0, 0.0, 6.0, 6.0, 5.0, 1.0, 0.0, 6.0, 8.0, 3.0, 1.0, 0.0, 6.0, 8.0, 2.0, 1.0, 0.0, 6.0, 6.0, 6.0, 1.0, 0.0, 6.0, 5.0, 6.0, 1.0, 0.0, 6.0, 7.0, 6.0, 1.0, 0.0, 6.0, 7.0, 2.0, 1.0, 0.0, 6.0, 7.0, 1.0, 1.0, 0.0, 6.0, 7.0, 5.0, 1.0, 0.0, 6.0, 7.0, 7.0, 1.0, 0.0, 6.0, 8.0, 6.0, 1.0, 0.0, 6.0, 5.0, 7.0, 1.0, 0.0, 6.0, 6.0, 8.0, 1.0, 0.0, 6.0, 4.0, 7.0, 2.0, 0.0, 6.0, 5.0, 8.0, 1.0, 0.0, 6.0, 7.0, 9.0, 1.0, 0.0, 6.0, 6.0, 9.0, 1.0, 0.0, 6.0, 6.0, 10.0, 2.0, 0.0, 6.0, 5.0, 9.0, 1.0, 0.0, 6.0, 7.0, 11.0, 1.0, 0.0, 6.0, 6.0, 11.0, 1.0, 0.0, 6.0, 8.0, 12.0, 1.0, 0.0, 6.0, 6.0, 12.0, 1.0, 0.0, 6.0, 7.0, 12.0, 1.0, 0.0, 6.0, 9.0, 13.0, 1.0, 0.0, 6.0, 9.0, 12.0, 1.0, 0.0, 6.0, 6.0, 13.0, 1.0, 0.0, 6.0, 5.0, 13.0, 1.0, 0.0, 6.0, 10.0, 14.0, 1.0, 0.0, 6.0, 9.0, 14.0, 1.0, 0.0, 6.0, 8.0, 14.0, 1.0, 0.0, 6.0, 5.0, 14.0, 1.0, 0.0, 6.0, 8.0, 13.0, 1.0, 0.0, 6.0, 7.0, 13.0, 1.0, 0.0, 6.0, 4.0, 15.0, 1.0, 0.0, 6.0, 8.0, 8.0, 1.0, 0.0, 6.0, 7.0, 8.0, 1.0, 0.0, 6.0, 8.0, 15.0, 1.0, 0.0, 6.0, 7.0, 15.0, 1.0, 0.0, 6.0, 3.0, 15.0, 1.0, 0.0, 6.0, 3.0, 14.0, 1.0, 0.0, 6.0, 5.0, 15.0, 1.0, 0.0, 6.0, 9.0, 16.0, 1.0, 0.0, 6.0, 3.0, 13.0, 1.0, 0.0, 6.0, 8.0, 16.0, 1.0, 0.0, 6.0, 9.0, 17.0, 1.0, 0.0, 6.0, 8.0, 17.0, 1.0, 0.0, 6.0, 7.0, 17.0, 1.0, 0.0, 6.0, 7.0, 16.0, 1.0, 0.0, 6.0, 7.0, 18.0, 1.0, 0.0, 6.0, 6.0, 18.0, 1.0, 0.0, 6.0, 6.0, 17.0, 1.0, 0.0, 6.0, 6.0, 16.0, 1.0, 0.0, 6.0, 5.0, 17.0, 1.0, 0.0, 6.0, 4.0, 16.0, 1.0, 0.0, 6.0, 3.0, 16.0, 1.0, 0.0, 6.0, 6.0, 15.0, 1.0, 0.0, 6.0, 5.0, 16.0, 1.0, 0.0, 6.0, 10.0, 13.0, 1.0, 0.0, 6.0, 4.0, 13.0, 1.0, 0.0, 6.0, 9.0, 15.0, 1.0, 0.0, 6.0, 4.0, 12.0, 1.0, 0.0, 6.0, 7.0, 14.0, 1.0, 0.0, 6.0, 6.0, 14.0, 1.0, 0.0, 6.0, 4.0, 14.0, 1.0, 75.0, 7.0, 4.0, 7.0, 1.0, 75.0, 7.0, 4.0, 6.0, 2.0, 151.0, 8.0, 3.0, 2.0, 1.0, 151.0, 8.0, 4.0, 6.0, 1.0, 229.0, 9.0, 1.0, -1.0, 2.0, 4.0, 0.0]
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
  drainages::Array{Field{Float64},1} = repeat(drainage,10000,false)
  runoffs::Array{Field{Float64},1} = drainages
  evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,180,false)
  initial_water_to_lake_centers = LatLonField{Float64}(lake_grid,
    Float64[  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 440.0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0 ])
  initial_spillover_to_rivers = LatLonField{Float64}(hd_grid,
                                                     Float64[ 0.0 0.0 0.0 0.0
                                                              0.0 0.0 0.0 0.0
                                                              0.0 0.0 0.0 0.0
                                                              0.0 0.0 0.0 0.0 ])
  expected_river_inflow::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                               Float64[ 0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0 ])
  expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                                 Float64[ 0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0 ])
  expected_water_to_hd::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                              Float64[ 0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0 ])
  expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
              6 9 6 0 0 0 0 0 0 0 0 0 9 9 8 9 0 0 0 6
              0 6 6 0 0 9 9 0 0 0 0 9 7 7 7 8 0 0 0 0
              0 6 6 0 0 4 4 4 4 0 0 0 5 7 7 7 8 0 0 0
              0 6 6 9 4 4 0 4 4 0 7 5 7 5 7 7 7 8 0 0
              6 6 6 0 4 4 4 4 4 0 7 5 5 7 7 7 7 8 0 0
              0 6 6 0 0 4 0 4 0 0 0 7 5 7 3 7 8 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 8 7 7 7 7 8 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 8 8 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 0 0
              0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3
             3 2 3 0 0 0 0 0 0 0 0 0 2 2 3 2 0 0 0 3
             0 3 3 0 0 2 2 0 0 0 0 2 3 3 3 3 0 0 0 0
             0 3 3 0 0 3 3 3 3 0 0 0 3 3 3 3 3 0 0 0
             0 3 3 2 3 3 0 3 3 0 3 3 3 3 3 3 3 3 0 0
             3 3 3 0 3 3 3 3 3 0 3 3 3 3 3 3 3 3 0 0
             0 3 3 0 0 3 0 3 0 0 0 3 3 3 3 3 3 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 3 3 3 3 3 3 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 3 3 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
  expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     430.0
                   430.0 430.0 430.0 0     0     0     0     0     0     0     #=
                =# 0     0     430.0 430.0 430.0 430.0 0     0     0     430.0
                   0     430.0 430.0 0     0     430.0 430.0 0     0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 0     0     0     0
                   0     430.0 430.0 0     0     430.0 430.0 430.0 430.0 0     #=
                =# 0     0     430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     430.0 430.0 430.0 430.0 430.0 0     430.0 430.0 0     #=
                =# 430.0 430.0 430.0 430.0 430.0 430.0 430.0 430.0 0     0
                   430.0 430.0 430.0 0     430.0 430.0 430.0 430.0 430.0 0     #=
                =# 430.0 430.0 430.0 430.0 430.0 430.0 430.0 430.0 0     0
                   0     430.0 430.0 0     0     430.0 0     430.0 0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     430.0 430.0 0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0  10.0  10.0  10.0  10.0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0  10.0  10.0  10.0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0 ])
  expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 1.0 0.0
                 1.0 0.976744186 0.0
                 1.0 1.0 0.0 ])
  expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15  6 0
                1 42 0
               15  7 0 ])
  expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 6  311
                1  43 1
                15 7  1  ])
  expected_lake_volumes::Array{Float64} = Float64[0.0, 10.0, 1.0, 38.0, 6.0, 46.0, 55.0, 55.0, 229.0]
  # expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 7.0
  #               8.0 1.0 6.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 4.0 3.0 0.0 0.0 0.0 8.0
  #               0.0 6.0 6.0 0.0 0.0 1.0 2.0 0.0 0.0 0.0 0.0 3.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0 0.0
  #               0.0 6.0 6.0 0.0 0.0 6.0 6.0 5.0 6.0 0.0 0.0 0.0 7.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0
  #               0.0 6.0 6.0 3.0 7.0 8.0 0.0 7.0 6.0 0.0 6.0 7.0 6.0 7.0 6.0 6.0 5.0 4.0 0.0 0.0
  #               5.0 5.0 6.0 0.0 7.0 8.0 7.0 7.0 6.0 0.0 6.0 7.0 7.0 6.0 6.0 6.0 5.0 4.0 0.0 0.0
  #               0.0 5.0 5.0 0.0 0.0 7.0 0.0 5.0 0.0 0.0 0.0 6.0 7.0 6.0 7.0 6.0 4.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0 6.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0 4.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 2.0 1.0 1.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 1.0 1.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])
  river_fields::RiverPrognosticFields,
    lake_model_prognostics::LakeModelPrognostics,_ =
    drive_hd_and_lake_model(river_parameters,lake_model_parameters,
                            lake_parameters_as_array,
                            drainages,runoffs,evaporations,
                            4,true,initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes::Field{Float64} =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test expected_river_inflow == river_fields.river_inflow
  @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test expected_lake_types == lake_types
  @test expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  @test isapprox(expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
  @test expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test expected_true_lake_depths == lake_model_parameters.true_lake_depths
end

@testset "Lake model tests 15" begin
  hd_grid = LatLonGrid(4,4,true)
  surface_model_grid = LatLonGrid(3,3,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
                                                Int64[-2  4  2  2
                                                       6 -2 -2  2
                                                       6  2  3 -2
                                                      -2  0  6  0 ]))
  river_reservoir_nums = LatLonField{Int64}(hd_grid,5)
  overland_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  base_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  river_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  overland_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  base_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  landsea_mask = LatLonField{Bool}(hd_grid,fill(false,4,4))
  set!(river_reservoir_nums,LatLonCoords(4,4),0)
  set!(overland_reservoir_nums,LatLonCoords(4,4),0)
  set!(base_reservoir_nums,LatLonCoords(4,4),0)
  set!(landsea_mask,LatLonCoords(4,4),true)
  set!(river_reservoir_nums,LatLonCoords(4,2),0)
  set!(overland_reservoir_nums,LatLonCoords(4,2),0)
  set!(base_reservoir_nums,LatLonCoords(4,2),0)
  set!(landsea_mask,LatLonCoords(4,2),true)
  river_parameters = RiverParameters(flow_directions,
                                     river_reservoir_nums,
                                     overland_reservoir_nums,
                                     base_reservoir_nums,
                                     river_retention_coefficients,
                                     overland_retention_coefficients,
                                     base_retention_coefficients,
                                     landsea_mask,
                                     hd_grid,86400.0,86400.0)
  lake_grid = LatLonGrid(20,20,true)
  cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
    Float64[ 2.5 3.0 2.5
             3.0 4.0 3.0
             2.5 3.0 2.5 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 2 1 1 1 1 1 1 1 1 1 1 2 2 2 2 1 1 1 1
                                                             1 1 1 1 1 2 2 1 1 1 1 2 2 2 2 2 1 1 1 1
                                                             1 1 1 1 1 3 3 3 3 1 1 1 1 2 2 2 2 1 1 1
                                                             1 1 1 2 3 3 1 3 3 2 2 1 2 1 2 2 2 2 1 1
                                                             1 1 1 1 3 3 3 3 3 1 2 1 1 2 2 2 2 2 1 1
                                                             1 1 1 1 1 3 1 3 1 1 1 2 1 2 2 2 2 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 2 2 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 1 1
                                                             1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3 ])

  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                      Int64[ 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1
                                                             1 2 1 3 3 3 3 3 3 3 3 3 2 2 2 2 3 3 3 1
                                                             3 1 1 3 3 2 2 3 3 3 3 2 2 2 2 2 3 3 3 3
                                                             3 1 1 3 3 1 1 1 1 3 3 3 2 2 2 2 2 3 3 3
                                                             3 1 1 2 1 1 3 1 1 2 2 2 2 2 2 2 2 2 3 3
                                                             1 1 1 3 1 1 1 1 1 3 2 2 2 2 2 2 2 2 3 3
                                                             3 1 1 3 3 1 3 1 3 3 3 2 2 2 2 2 2 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 2 2 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 3 3
                                                             3 3 3 1 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3  ])
  is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
    Bool[false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false  true
          true  true  true false false false false false false false false false  true  true  true  true false false false  true
         false  true  true false false  true  true false false false false  true  true  true  true  true false false false false
         false  true  true false false  true  true  true  true false false false  true  true  true  true  true false false false
         false  true  true  true  true  true false  true  true  true  true  true  true  true  true  true  true  true false false
          true  true  true false  true  true  true  true  true false  true  true  true  true  true  true  true  true false false
         false  true  true false false  true false  true false false false  true  true  true  true  true  true false false false
         false false false false false false false false false false false  true  true  true  true  true  true false false false
         false false false false false false false false false false false false  true  true false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false  true  true  true  true false
         false false false false false false false false false false false false false false false  true  true  true false false
         false false false  true false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false])
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_model_parameters::LakeModelParameters =
    LakeModelParameters(grid_specific_lake_model_parameters,
                        lake_grid,
                        hd_grid,
                        surface_model_grid,
                        cell_areas_on_surface_model_grid,
                        LakeModelSettings(),
                        9,
                        is_lake)
  lake_parameters_as_array::Vector{Float64} = Float64[9.0, 16.0, 1.0, -1.0, 0.0, 16.0, 4.0, 1.0, 16.0, 4.0, 1.0, 1.0, 3.0, 1.0, -1.0, 4.0, 2.0, 0.0, 46.0, 2.0, -1.0, 0.0, 14.0, 16.0, 7.0, 14.0, 16.0, 1.0, 0.0, 2.0, 14.0, 17.0, 1.0, 0.0, 2.0, 15.0, 16.0, 1.0, 3.0, 3.0, 15.0, 17.0, 1.0, 3.0, 3.0, 15.0, 18.0, 1.0, 3.0, 3.0, 14.0, 18.0, 1.0, 3.0, 3.0, 14.0, 19.0, 1.0, 10.0, 4.0, 1.0, -1.0, 4.0, 4.0, 0.0, 16.0, 3.0, 7.0, 0.0, 8.0, 15.0, 1.0, 8.0, 15.0, 1.0, 1.0, 3.0, 1.0, 5.0, -1.0, -1.0, 1.0, 91.0, 4.0, 8.0, 0.0, 7.0, 6.0, 16.0, 7.0, 6.0, 1.0, 0.0, 1.0, 6.0, 6.0, 1.0, 2.0, 2.0, 7.0, 5.0, 1.0, 2.0, 2.0, 7.0, 7.0, 1.0, 2.0, 2.0, 8.0, 6.0, 1.0, 2.0, 2.0, 7.0, 8.0, 1.0, 2.0, 2.0, 6.0, 8.0, 1.0, 2.0, 2.0, 6.0, 5.0, 1.0, 10.0, 3.0, 5.0, 6.0, 1.0, 10.0, 3.0, 5.0, 7.0, 1.0, 10.0, 3.0, 7.0, 9.0, 1.0, 10.0, 3.0, 6.0, 9.0, 1.0, 10.0, 3.0, 4.0, 7.0, 2.0, 10.0, 3.0, 5.0, 9.0, 1.0, 23.0, 4.0, 8.0, 8.0, 1.0, 23.0, 4.0, 5.0, 8.0, 1.0, 38.0, 5.0, 1.0, 7.0, -1.0, -1.0, 1.0, 41.0, 5.0, 7.0, 0.0, 6.0, 14.0, 6.0, 6.0, 14.0, 1.0, 0.0, 2.0, 7.0, 13.0, 1.0, 0.0, 2.0, 5.0, 13.0, 1.0, 0.0, 2.0, 6.0, 12.0, 1.0, 0.0, 2.0, 8.0, 13.0, 1.0, 0.0, 2.0, 7.0, 12.0, 1.0, 6.0, 3.0, 1.0, 3.0, -1.0, -1.0, 1.0, 86.0, 6.0, 9.0, 0.0, 3.0, 1.0, 15.0, 3.0, 1.0, 1.0, 0.0, 1.0, 3.0, 20.0, 1.0, 2.0, 2.0, 2.0, 20.0, 1.0, 5.0, 3.0, 4.0, 2.0, 1.0, 5.0, 3.0, 5.0, 3.0, 1.0, 5.0, 3.0, 4.0, 3.0, 1.0, 5.0, 3.0, 3.0, 3.0, 1.0, 5.0, 3.0, 5.0, 2.0, 1.0, 5.0, 3.0, 6.0, 3.0, 1.0, 5.0, 3.0, 6.0, 2.0, 1.0, 5.0, 3.0, 7.0, 3.0, 1.0, 16.0, 4.0, 7.0, 2.0, 1.0, 16.0, 4.0, 8.0, 3.0, 1.0, 16.0, 4.0, 8.0, 2.0, 1.0, 16.0, 4.0, 7.0, 1.0, 1.0, 46.0, 6.0, 1.0, 8.0, 2.0, 1.0, 0.0, 163.0, 7.0, 8.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 8.0, 15.0, 1.0, 0.0, 3.0, 8.0, 16.0, 1.0, 0.0, 3.0, 7.0, 16.0, 1.0, 0.0, 3.0, 8.0, 14.0, 1.0, 0.0, 3.0, 9.0, 14.0, 1.0, 0.0, 3.0, 9.0, 15.0, 1.0, 0.0, 3.0, 7.0, 13.0, 1.0, 0.0, 3.0, 6.0, 15.0, 1.0, 0.0, 3.0, 8.0, 13.0, 1.0, 0.0, 3.0, 9.0, 13.0, 1.0, 0.0, 3.0, 7.0, 15.0, 1.0, 0.0, 3.0, 6.0, 16.0, 1.0, 0.0, 3.0, 5.0, 14.0, 1.0, 0.0, 3.0, 6.0, 12.0, 1.0, 0.0, 3.0, 4.0, 13.0, 1.0, 0.0, 3.0, 7.0, 11.0, 1.0, 0.0, 3.0, 6.0, 11.0, 1.0, 0.0, 3.0, 5.0, 15.0, 1.0, 0.0, 3.0, 4.0, 14.0, 1.0, 0.0, 3.0, 5.0, 13.0, 1.0, 0.0, 3.0, 7.0, 14.0, 1.0, 0.0, 3.0, 8.0, 12.0, 1.0, 0.0, 3.0, 7.0, 12.0, 1.0, 0.0, 3.0, 6.0, 14.0, 1.0, 0.0, 3.0, 6.0, 13.0, 1.0, 25.0, 4.0, 6.0, 17.0, 1.0, 25.0, 4.0, 7.0, 17.0, 1.0, 25.0, 4.0, 5.0, 16.0, 1.0, 25.0, 4.0, 4.0, 15.0, 1.0, 25.0, 4.0, 9.0, 16.0, 1.0, 55.0, 5.0, 1.0, 4.0, -1.0, -1.0, 1.0, 298.0, 8.0, 9.0, 2.0, 4.0, 7.0, 7.0, 6.0, 57.0, 7.0, 6.0, 1.0, 0.0, 5.0, 7.0, 7.0, 1.0, 0.0, 5.0, 7.0, 5.0, 1.0, 0.0, 5.0, 8.0, 6.0, 1.0, 0.0, 5.0, 6.0, 8.0, 1.0, 0.0, 5.0, 7.0, 8.0, 1.0, 0.0, 5.0, 7.0, 9.0, 1.0, 0.0, 5.0, 6.0, 9.0, 1.0, 0.0, 5.0, 5.0, 7.0, 1.0, 0.0, 5.0, 6.0, 6.0, 1.0, 0.0, 5.0, 6.0, 10.0, 2.0, 0.0, 5.0, 4.0, 7.0, 2.0, 0.0, 5.0, 6.0, 11.0, 1.0, 0.0, 5.0, 5.0, 6.0, 1.0, 0.0, 5.0, 7.0, 12.0, 1.0, 0.0, 5.0, 6.0, 12.0, 1.0, 0.0, 5.0, 8.0, 13.0, 1.0, 0.0, 5.0, 7.0, 11.0, 1.0, 0.0, 5.0, 8.0, 8.0, 1.0, 0.0, 5.0, 6.0, 5.0, 1.0, 0.0, 5.0, 5.0, 13.0, 1.0, 0.0, 5.0, 9.0, 13.0, 1.0, 0.0, 5.0, 9.0, 12.0, 1.0, 0.0, 5.0, 5.0, 14.0, 1.0, 0.0, 5.0, 4.0, 14.0, 1.0, 0.0, 5.0, 4.0, 15.0, 1.0, 0.0, 5.0, 3.0, 15.0, 1.0, 0.0, 5.0, 5.0, 16.0, 1.0, 0.0, 5.0, 4.0, 16.0, 1.0, 0.0, 5.0, 6.0, 17.0, 1.0, 0.0, 5.0, 6.0, 16.0, 1.0, 0.0, 5.0, 5.0, 17.0, 1.0, 0.0, 5.0, 7.0, 18.0, 1.0, 0.0, 5.0, 7.0, 17.0, 1.0, 0.0, 5.0, 7.0, 16.0, 1.0, 0.0, 5.0, 7.0, 15.0, 1.0, 0.0, 5.0, 6.0, 18.0, 1.0, 0.0, 5.0, 8.0, 16.0, 1.0, 0.0, 5.0, 8.0, 15.0, 1.0, 0.0, 5.0, 10.0, 13.0, 1.0, 0.0, 5.0, 4.0, 13.0, 1.0, 0.0, 5.0, 6.0, 15.0, 1.0, 0.0, 5.0, 5.0, 15.0, 1.0, 0.0, 5.0, 9.0, 15.0, 1.0, 0.0, 5.0, 9.0, 16.0, 1.0, 0.0, 5.0, 10.0, 14.0, 1.0, 0.0, 5.0, 8.0, 14.0, 1.0, 0.0, 5.0, 7.0, 14.0, 1.0, 0.0, 5.0, 6.0, 14.0, 1.0, 0.0, 5.0, 5.0, 9.0, 1.0, 0.0, 5.0, 5.0, 8.0, 1.0, 0.0, 5.0, 9.0, 14.0, 1.0, 0.0, 5.0, 8.0, 12.0, 1.0, 0.0, 5.0, 9.0, 17.0, 1.0, 0.0, 5.0, 7.0, 13.0, 1.0, 0.0, 5.0, 6.0, 13.0, 1.0, 0.0, 5.0, 8.0, 17.0, 1.0, 55.0, 6.0, 1.0, 6.0, 2.0, 1.0, 0.0, 418.0, 9.0, -1.0, 2.0, 6.0, 8.0, 3.0, 1.0, 81.0, 3.0, 1.0, 1.0, 0.0, 6.0, 4.0, 2.0, 1.0, 0.0, 6.0, 3.0, 20.0, 1.0, 0.0, 6.0, 2.0, 20.0, 1.0, 0.0, 6.0, 3.0, 3.0, 1.0, 0.0, 6.0, 5.0, 3.0, 1.0, 0.0, 6.0, 4.0, 3.0, 1.0, 0.0, 6.0, 5.0, 2.0, 1.0, 0.0, 6.0, 6.0, 3.0, 1.0, 0.0, 6.0, 6.0, 2.0, 1.0, 0.0, 6.0, 6.0, 4.0, 1.0, 0.0, 6.0, 7.0, 3.0, 1.0, 0.0, 6.0, 6.0, 5.0, 1.0, 0.0, 6.0, 8.0, 3.0, 1.0, 0.0, 6.0, 8.0, 2.0, 1.0, 0.0, 6.0, 6.0, 6.0, 1.0, 0.0, 6.0, 5.0, 6.0, 1.0, 0.0, 6.0, 7.0, 6.0, 1.0, 0.0, 6.0, 7.0, 2.0, 1.0, 0.0, 6.0, 7.0, 1.0, 1.0, 0.0, 6.0, 7.0, 5.0, 1.0, 0.0, 6.0, 7.0, 7.0, 1.0, 0.0, 6.0, 8.0, 6.0, 1.0, 0.0, 6.0, 5.0, 7.0, 1.0, 0.0, 6.0, 6.0, 8.0, 1.0, 0.0, 6.0, 4.0, 7.0, 2.0, 0.0, 6.0, 5.0, 8.0, 1.0, 0.0, 6.0, 7.0, 9.0, 1.0, 0.0, 6.0, 6.0, 9.0, 1.0, 0.0, 6.0, 6.0, 10.0, 2.0, 0.0, 6.0, 5.0, 9.0, 1.0, 0.0, 6.0, 7.0, 11.0, 1.0, 0.0, 6.0, 6.0, 11.0, 1.0, 0.0, 6.0, 8.0, 12.0, 1.0, 0.0, 6.0, 6.0, 12.0, 1.0, 0.0, 6.0, 7.0, 12.0, 1.0, 0.0, 6.0, 9.0, 13.0, 1.0, 0.0, 6.0, 9.0, 12.0, 1.0, 0.0, 6.0, 6.0, 13.0, 1.0, 0.0, 6.0, 5.0, 13.0, 1.0, 0.0, 6.0, 10.0, 14.0, 1.0, 0.0, 6.0, 9.0, 14.0, 1.0, 0.0, 6.0, 8.0, 14.0, 1.0, 0.0, 6.0, 5.0, 14.0, 1.0, 0.0, 6.0, 8.0, 13.0, 1.0, 0.0, 6.0, 7.0, 13.0, 1.0, 0.0, 6.0, 4.0, 15.0, 1.0, 0.0, 6.0, 8.0, 8.0, 1.0, 0.0, 6.0, 7.0, 8.0, 1.0, 0.0, 6.0, 8.0, 15.0, 1.0, 0.0, 6.0, 7.0, 15.0, 1.0, 0.0, 6.0, 3.0, 15.0, 1.0, 0.0, 6.0, 3.0, 14.0, 1.0, 0.0, 6.0, 5.0, 15.0, 1.0, 0.0, 6.0, 9.0, 16.0, 1.0, 0.0, 6.0, 3.0, 13.0, 1.0, 0.0, 6.0, 8.0, 16.0, 1.0, 0.0, 6.0, 9.0, 17.0, 1.0, 0.0, 6.0, 8.0, 17.0, 1.0, 0.0, 6.0, 7.0, 17.0, 1.0, 0.0, 6.0, 7.0, 16.0, 1.0, 0.0, 6.0, 7.0, 18.0, 1.0, 0.0, 6.0, 6.0, 18.0, 1.0, 0.0, 6.0, 6.0, 17.0, 1.0, 0.0, 6.0, 6.0, 16.0, 1.0, 0.0, 6.0, 5.0, 17.0, 1.0, 0.0, 6.0, 4.0, 16.0, 1.0, 0.0, 6.0, 3.0, 16.0, 1.0, 0.0, 6.0, 6.0, 15.0, 1.0, 0.0, 6.0, 5.0, 16.0, 1.0, 0.0, 6.0, 10.0, 13.0, 1.0, 0.0, 6.0, 4.0, 13.0, 1.0, 0.0, 6.0, 9.0, 15.0, 1.0, 0.0, 6.0, 4.0, 12.0, 1.0, 0.0, 6.0, 7.0, 14.0, 1.0, 0.0, 6.0, 6.0, 14.0, 1.0, 0.0, 6.0, 4.0, 14.0, 1.0, 75.0, 7.0, 4.0, 7.0, 1.0, 75.0, 7.0, 4.0, 6.0, 2.0, 151.0, 8.0, 3.0, 2.0, 1.0, 151.0, 8.0, 4.0, 6.0, 1.0, 229.0, 9.0, 1.0, -1.0, 2.0, 4.0, 0.0]
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
  drainages::Array{Field{Float64},1} = repeat(drainage,10000,false)
  runoffs::Array{Field{Float64},1} = drainages
  evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,180,false)
  initial_water_to_lake_centers = LatLonField{Float64}(lake_grid,
    Float64[ 0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
            120.0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 20.0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0 ])
  initial_spillover_to_rivers = LatLonField{Float64}(hd_grid,
                                                     Float64[ 0.0 0.0 0.0 0.0
                                                              0.0 0.0 0.0 0.0
                                                              0.0 0.0 0.0 0.0
                                                              0.0 0.0 0.0 0.0 ])
  expected_river_inflow::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                               Float64[ 0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0 ])
  expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                                 Float64[ 0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0 ])
  expected_water_to_hd::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                              Float64[ 0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0 ])
  expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
              6 0 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
              0 6 6 0 0 0 0 0 0 0 0 0 7 7 7 0 0 0 0 0
              0 6 6 0 0 4 4 4 4 0 0 0 5 7 7 7 0 0 0 0
              0 6 6 0 4 4 0 4 4 0 7 5 7 5 7 7 7 0 0 0
              6 6 6 0 4 4 4 4 4 0 7 5 5 7 7 7 7 0 0 0
              0 6 6 0 0 4 0 4 0 0 0 7 5 7 3 7 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 7 7 7 7 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2
             2 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2
             0 2 2 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0
             0 2 2 0 0 2 2 2 2 0 0 0 3 1 1 1 0 0 0 0
             0 2 2 0 2 2 0 2 2 0 1 3 1 3 1 1 1 0 0 0
             2 2 2 0 2 2 2 2 2 0 1 3 3 1 1 1 1 0 0 0
             0 2 2 0 0 2 0 2 0 0 0 1 3 1 3 1 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 46.0
                 46.0 0 46.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 46.0
                 0 46.0 46.0 0 0 0 0 0 0 0 0 0 56.0 56.0 56.0 0 0 0 0 0
                 0 46.0 46.0 0 0 38.0 38.0 38.0 38.0 0 0 0 56.0 56.0 56.0 56.0 0 0 0 0
                 0 46.0 46.0 0 38.0 38.0 0 38.0 38.0 0 56.0 56.0 56.0 56.0 56.0 56.0 56.0 0 0 0
                 46.0 46.0 46.0 0 38.0 38.0 38.0 38.0 38.0 0 56.0 56.0 56.0 56.0 56.0 56.0 56.0 0 0 0
                 0 46.0 46.0 0 0 38.0 0 38.0 0 0 0 56.0 56.0 56.0 56.0 56.0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 56.0 56.0 56.0 56.0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 1.0 0.0
                 1.0 0.5581395349 0.0
                 1.0 0.1428571429 0.0])
  expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 6  0
                1 24 0
               15 1  0 ])
  expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 6  311
                1  43 1
                15 7  1  ])
  expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 1.0, 38.0, 6.0, 46.0, 49.0, 0.0, 0.0]
  # expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0
  #               5.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 5.0
  #               0.0 3.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.8 1.8 0.8 0.0 0.0 0.0 0.0 0.0
  #               0.0 3.0 3.0 0.0 0.0 2.0 2.0 1.0 2.0 0.0 0.0 0.0 2.8 1.8 1.8 0.8 0.0 0.0 0.0 0.0
  #               0.0 3.0 3.0 0.0 3.0 4.0 0.0 3.0 2.0 0.0 1.8 2.8 1.8 2.8 1.8 1.8 0.8 0.0 0.0 0.0
  #               2.0 2.0 3.0 0.0 3.0 4.0 3.0 3.0 2.0 0.0 1.8 2.8 2.8 1.8 1.8 1.8 0.8 0.0 0.0 0.0
  #               0.0 2.0 2.0 0.0 0.0 3.0 0.0 1.0 0.0 0.0 0.0 1.8 2.8 1.8 2.8 1.8 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.8 1.8 1.8 0.8 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])
  river_fields::RiverPrognosticFields,
    lake_model_prognostics::LakeModelPrognostics,_ =
    drive_hd_and_lake_model(river_parameters,lake_model_parameters,
                            lake_parameters_as_array,
                            drainages,runoffs,evaporations,
                            5,true,initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes::Field{Float64} =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test expected_river_inflow == river_fields.river_inflow
  @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test expected_lake_types == lake_types
  @test expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  @test isapprox(expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
  @test expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test isapprox(expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.0000001)
end

@testset "Lake model tests 16" begin
  #Spill off to river plus water to lake centers - no drainage, runoff, evaporation
  hd_grid = LatLonGrid(4,4,true)
  surface_model_grid = LatLonGrid(3,3,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
                                                Int64[-2  4  2  2
                                                       6 -2 -2  2
                                                       6  2  3 -2
                                                      -2  0  6  0 ]))
  river_reservoir_nums = LatLonField{Int64}(hd_grid,5)
  overland_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  base_reservoir_nums = LatLonField{Int64}(hd_grid,1)
  river_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  overland_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  base_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
  landsea_mask = LatLonField{Bool}(hd_grid,fill(false,4,4))
  set!(river_reservoir_nums,LatLonCoords(4,4),0)
  set!(overland_reservoir_nums,LatLonCoords(4,4),0)
  set!(base_reservoir_nums,LatLonCoords(4,4),0)
  set!(landsea_mask,LatLonCoords(4,4),true)
  set!(river_reservoir_nums,LatLonCoords(4,2),0)
  set!(overland_reservoir_nums,LatLonCoords(4,2),0)
  set!(base_reservoir_nums,LatLonCoords(4,2),0)
  set!(landsea_mask,LatLonCoords(4,2),true)
  river_parameters = RiverParameters(flow_directions,
                                     river_reservoir_nums,
                                     overland_reservoir_nums,
                                     base_reservoir_nums,
                                     river_retention_coefficients,
                                     overland_retention_coefficients,
                                     base_retention_coefficients,
                                     landsea_mask,
                                     hd_grid,86400.0,86400.0)
  lake_grid = LatLonGrid(20,20,true)
  cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
    Float64[ 2.5 3.0 2.5
             3.0 4.0 3.0
             2.5 3.0 2.5 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 2 1 1 1 1 1 1 1 1 1 1 2 2 2 2 1 1 1 1
                                                             1 1 1 1 1 2 2 1 1 1 1 2 2 2 2 2 1 1 1 1
                                                             1 1 1 1 1 3 3 3 3 1 1 1 1 2 2 2 2 1 1 1
                                                             1 1 1 2 3 3 1 3 3 2 2 1 2 1 2 2 2 2 1 1
                                                             1 1 1 1 3 3 3 3 3 1 2 1 1 2 2 2 2 2 1 1
                                                             1 1 1 1 1 3 1 3 1 1 1 2 1 2 2 2 2 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 2 2 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 1 1
                                                             1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
                                                             1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3 ])
  corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                      Int64[ 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1
                                                             1 2 1 3 3 3 3 3 3 3 3 3 2 2 2 2 3 3 3 1
                                                             3 1 1 3 3 2 2 3 3 3 3 2 2 2 2 2 3 3 3 3
                                                             3 1 1 3 3 1 1 1 1 3 3 3 2 2 2 2 2 3 3 3
                                                             3 1 1 2 1 1 3 1 1 2 2 2 2 2 2 2 2 2 3 3
                                                             1 1 1 3 1 1 1 1 1 3 2 2 2 2 2 2 2 2 3 3
                                                             3 1 1 3 3 1 3 1 3 3 3 2 2 2 2 2 2 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 2 2 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 3 3
                                                             3 3 3 1 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
                                                             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3  ])
  is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
    Bool[false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false  true
          true  true  true false false false false false false false false false  true  true  true  true false false false  true
         false  true  true false false  true  true false false false false  true  true  true  true  true false false false false
         false  true  true false false  true  true  true  true false false false  true  true  true  true  true false false false
         false  true  true  true  true  true false  true  true  true  true  true  true  true  true  true  true  true false false
          true  true  true false  true  true  true  true  true false  true  true  true  true  true  true  true  true false false
         false  true  true false false  true false  true false false false  true  true  true  true  true  true false false false
         false false false false false false false false false false false  true  true  true  true  true  true false false false
         false false false false false false false false false false false false  true  true false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false  true  true  true  true false
         false false false false false false false false false false false false false false false  true  true  true false false
         false false false  true false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false
         false false false false false false false false false false false false false false false false false false false false])
  grid_specific_lake_model_parameters::LatLonLakeModelParameters =
    LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
  lake_model_parameters::LakeModelParameters =
    LakeModelParameters(grid_specific_lake_model_parameters,
                        lake_grid,
                        hd_grid,
                        surface_model_grid,
                        cell_areas_on_surface_model_grid,
                        LakeModelSettings(),
                        9,
                        is_lake)
  lake_parameters_as_array::Vector{Float64} = Float64[9.0, 16.0, 1.0, -1.0, 0.0, 16.0, 4.0, 1.0, 16.0, 4.0, 1.0, 1.0, 3.0, 1.0, -1.0, 4.0, 2.0, 0.0, 46.0, 2.0, -1.0, 0.0, 14.0, 16.0, 7.0, 14.0, 16.0, 1.0, 0.0, 2.0, 14.0, 17.0, 1.0, 0.0, 2.0, 15.0, 16.0, 1.0, 3.0, 3.0, 15.0, 17.0, 1.0, 3.0, 3.0, 15.0, 18.0, 1.0, 3.0, 3.0, 14.0, 18.0, 1.0, 3.0, 3.0, 14.0, 19.0, 1.0, 10.0, 4.0, 1.0, -1.0, 4.0, 4.0, 0.0, 16.0, 3.0, 7.0, 0.0, 8.0, 15.0, 1.0, 8.0, 15.0, 1.0, 1.0, 3.0, 1.0, 5.0, -1.0, -1.0, 1.0, 91.0, 4.0, 8.0, 0.0, 7.0, 6.0, 16.0, 7.0, 6.0, 1.0, 0.0, 1.0, 6.0, 6.0, 1.0, 2.0, 2.0, 7.0, 5.0, 1.0, 2.0, 2.0, 7.0, 7.0, 1.0, 2.0, 2.0, 8.0, 6.0, 1.0, 2.0, 2.0, 7.0, 8.0, 1.0, 2.0, 2.0, 6.0, 8.0, 1.0, 2.0, 2.0, 6.0, 5.0, 1.0, 10.0, 3.0, 5.0, 6.0, 1.0, 10.0, 3.0, 5.0, 7.0, 1.0, 10.0, 3.0, 7.0, 9.0, 1.0, 10.0, 3.0, 6.0, 9.0, 1.0, 10.0, 3.0, 4.0, 7.0, 2.0, 10.0, 3.0, 5.0, 9.0, 1.0, 23.0, 4.0, 8.0, 8.0, 1.0, 23.0, 4.0, 5.0, 8.0, 1.0, 38.0, 5.0, 1.0, 7.0, -1.0, -1.0, 1.0, 41.0, 5.0, 7.0, 0.0, 6.0, 14.0, 6.0, 6.0, 14.0, 1.0, 0.0, 2.0, 7.0, 13.0, 1.0, 0.0, 2.0, 5.0, 13.0, 1.0, 0.0, 2.0, 6.0, 12.0, 1.0, 0.0, 2.0, 8.0, 13.0, 1.0, 0.0, 2.0, 7.0, 12.0, 1.0, 6.0, 3.0, 1.0, 3.0, -1.0, -1.0, 1.0, 86.0, 6.0, 9.0, 0.0, 3.0, 1.0, 15.0, 3.0, 1.0, 1.0, 0.0, 1.0, 3.0, 20.0, 1.0, 2.0, 2.0, 2.0, 20.0, 1.0, 5.0, 3.0, 4.0, 2.0, 1.0, 5.0, 3.0, 5.0, 3.0, 1.0, 5.0, 3.0, 4.0, 3.0, 1.0, 5.0, 3.0, 3.0, 3.0, 1.0, 5.0, 3.0, 5.0, 2.0, 1.0, 5.0, 3.0, 6.0, 3.0, 1.0, 5.0, 3.0, 6.0, 2.0, 1.0, 5.0, 3.0, 7.0, 3.0, 1.0, 16.0, 4.0, 7.0, 2.0, 1.0, 16.0, 4.0, 8.0, 3.0, 1.0, 16.0, 4.0, 8.0, 2.0, 1.0, 16.0, 4.0, 7.0, 1.0, 1.0, 46.0, 6.0, 1.0, 8.0, 2.0, 1.0, 0.0, 163.0, 7.0, 8.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 8.0, 15.0, 1.0, 0.0, 3.0, 8.0, 16.0, 1.0, 0.0, 3.0, 7.0, 16.0, 1.0, 0.0, 3.0, 8.0, 14.0, 1.0, 0.0, 3.0, 9.0, 14.0, 1.0, 0.0, 3.0, 9.0, 15.0, 1.0, 0.0, 3.0, 7.0, 13.0, 1.0, 0.0, 3.0, 6.0, 15.0, 1.0, 0.0, 3.0, 8.0, 13.0, 1.0, 0.0, 3.0, 9.0, 13.0, 1.0, 0.0, 3.0, 7.0, 15.0, 1.0, 0.0, 3.0, 6.0, 16.0, 1.0, 0.0, 3.0, 5.0, 14.0, 1.0, 0.0, 3.0, 6.0, 12.0, 1.0, 0.0, 3.0, 4.0, 13.0, 1.0, 0.0, 3.0, 7.0, 11.0, 1.0, 0.0, 3.0, 6.0, 11.0, 1.0, 0.0, 3.0, 5.0, 15.0, 1.0, 0.0, 3.0, 4.0, 14.0, 1.0, 0.0, 3.0, 5.0, 13.0, 1.0, 0.0, 3.0, 7.0, 14.0, 1.0, 0.0, 3.0, 8.0, 12.0, 1.0, 0.0, 3.0, 7.0, 12.0, 1.0, 0.0, 3.0, 6.0, 14.0, 1.0, 0.0, 3.0, 6.0, 13.0, 1.0, 25.0, 4.0, 6.0, 17.0, 1.0, 25.0, 4.0, 7.0, 17.0, 1.0, 25.0, 4.0, 5.0, 16.0, 1.0, 25.0, 4.0, 4.0, 15.0, 1.0, 25.0, 4.0, 9.0, 16.0, 1.0, 55.0, 5.0, 1.0, 4.0, -1.0, -1.0, 1.0, 298.0, 8.0, 9.0, 2.0, 4.0, 7.0, 7.0, 6.0, 57.0, 7.0, 6.0, 1.0, 0.0, 5.0, 7.0, 7.0, 1.0, 0.0, 5.0, 7.0, 5.0, 1.0, 0.0, 5.0, 8.0, 6.0, 1.0, 0.0, 5.0, 6.0, 8.0, 1.0, 0.0, 5.0, 7.0, 8.0, 1.0, 0.0, 5.0, 7.0, 9.0, 1.0, 0.0, 5.0, 6.0, 9.0, 1.0, 0.0, 5.0, 5.0, 7.0, 1.0, 0.0, 5.0, 6.0, 6.0, 1.0, 0.0, 5.0, 6.0, 10.0, 2.0, 0.0, 5.0, 4.0, 7.0, 2.0, 0.0, 5.0, 6.0, 11.0, 1.0, 0.0, 5.0, 5.0, 6.0, 1.0, 0.0, 5.0, 7.0, 12.0, 1.0, 0.0, 5.0, 6.0, 12.0, 1.0, 0.0, 5.0, 8.0, 13.0, 1.0, 0.0, 5.0, 7.0, 11.0, 1.0, 0.0, 5.0, 8.0, 8.0, 1.0, 0.0, 5.0, 6.0, 5.0, 1.0, 0.0, 5.0, 5.0, 13.0, 1.0, 0.0, 5.0, 9.0, 13.0, 1.0, 0.0, 5.0, 9.0, 12.0, 1.0, 0.0, 5.0, 5.0, 14.0, 1.0, 0.0, 5.0, 4.0, 14.0, 1.0, 0.0, 5.0, 4.0, 15.0, 1.0, 0.0, 5.0, 3.0, 15.0, 1.0, 0.0, 5.0, 5.0, 16.0, 1.0, 0.0, 5.0, 4.0, 16.0, 1.0, 0.0, 5.0, 6.0, 17.0, 1.0, 0.0, 5.0, 6.0, 16.0, 1.0, 0.0, 5.0, 5.0, 17.0, 1.0, 0.0, 5.0, 7.0, 18.0, 1.0, 0.0, 5.0, 7.0, 17.0, 1.0, 0.0, 5.0, 7.0, 16.0, 1.0, 0.0, 5.0, 7.0, 15.0, 1.0, 0.0, 5.0, 6.0, 18.0, 1.0, 0.0, 5.0, 8.0, 16.0, 1.0, 0.0, 5.0, 8.0, 15.0, 1.0, 0.0, 5.0, 10.0, 13.0, 1.0, 0.0, 5.0, 4.0, 13.0, 1.0, 0.0, 5.0, 6.0, 15.0, 1.0, 0.0, 5.0, 5.0, 15.0, 1.0, 0.0, 5.0, 9.0, 15.0, 1.0, 0.0, 5.0, 9.0, 16.0, 1.0, 0.0, 5.0, 10.0, 14.0, 1.0, 0.0, 5.0, 8.0, 14.0, 1.0, 0.0, 5.0, 7.0, 14.0, 1.0, 0.0, 5.0, 6.0, 14.0, 1.0, 0.0, 5.0, 5.0, 9.0, 1.0, 0.0, 5.0, 5.0, 8.0, 1.0, 0.0, 5.0, 9.0, 14.0, 1.0, 0.0, 5.0, 8.0, 12.0, 1.0, 0.0, 5.0, 9.0, 17.0, 1.0, 0.0, 5.0, 7.0, 13.0, 1.0, 0.0, 5.0, 6.0, 13.0, 1.0, 0.0, 5.0, 8.0, 17.0, 1.0, 55.0, 6.0, 1.0, 6.0, 2.0, 1.0, 0.0, 418.0, 9.0, -1.0, 2.0, 6.0, 8.0, 3.0, 1.0, 81.0, 3.0, 1.0, 1.0, 0.0, 6.0, 4.0, 2.0, 1.0, 0.0, 6.0, 3.0, 20.0, 1.0, 0.0, 6.0, 2.0, 20.0, 1.0, 0.0, 6.0, 3.0, 3.0, 1.0, 0.0, 6.0, 5.0, 3.0, 1.0, 0.0, 6.0, 4.0, 3.0, 1.0, 0.0, 6.0, 5.0, 2.0, 1.0, 0.0, 6.0, 6.0, 3.0, 1.0, 0.0, 6.0, 6.0, 2.0, 1.0, 0.0, 6.0, 6.0, 4.0, 1.0, 0.0, 6.0, 7.0, 3.0, 1.0, 0.0, 6.0, 6.0, 5.0, 1.0, 0.0, 6.0, 8.0, 3.0, 1.0, 0.0, 6.0, 8.0, 2.0, 1.0, 0.0, 6.0, 6.0, 6.0, 1.0, 0.0, 6.0, 5.0, 6.0, 1.0, 0.0, 6.0, 7.0, 6.0, 1.0, 0.0, 6.0, 7.0, 2.0, 1.0, 0.0, 6.0, 7.0, 1.0, 1.0, 0.0, 6.0, 7.0, 5.0, 1.0, 0.0, 6.0, 7.0, 7.0, 1.0, 0.0, 6.0, 8.0, 6.0, 1.0, 0.0, 6.0, 5.0, 7.0, 1.0, 0.0, 6.0, 6.0, 8.0, 1.0, 0.0, 6.0, 4.0, 7.0, 2.0, 0.0, 6.0, 5.0, 8.0, 1.0, 0.0, 6.0, 7.0, 9.0, 1.0, 0.0, 6.0, 6.0, 9.0, 1.0, 0.0, 6.0, 6.0, 10.0, 2.0, 0.0, 6.0, 5.0, 9.0, 1.0, 0.0, 6.0, 7.0, 11.0, 1.0, 0.0, 6.0, 6.0, 11.0, 1.0, 0.0, 6.0, 8.0, 12.0, 1.0, 0.0, 6.0, 6.0, 12.0, 1.0, 0.0, 6.0, 7.0, 12.0, 1.0, 0.0, 6.0, 9.0, 13.0, 1.0, 0.0, 6.0, 9.0, 12.0, 1.0, 0.0, 6.0, 6.0, 13.0, 1.0, 0.0, 6.0, 5.0, 13.0, 1.0, 0.0, 6.0, 10.0, 14.0, 1.0, 0.0, 6.0, 9.0, 14.0, 1.0, 0.0, 6.0, 8.0, 14.0, 1.0, 0.0, 6.0, 5.0, 14.0, 1.0, 0.0, 6.0, 8.0, 13.0, 1.0, 0.0, 6.0, 7.0, 13.0, 1.0, 0.0, 6.0, 4.0, 15.0, 1.0, 0.0, 6.0, 8.0, 8.0, 1.0, 0.0, 6.0, 7.0, 8.0, 1.0, 0.0, 6.0, 8.0, 15.0, 1.0, 0.0, 6.0, 7.0, 15.0, 1.0, 0.0, 6.0, 3.0, 15.0, 1.0, 0.0, 6.0, 3.0, 14.0, 1.0, 0.0, 6.0, 5.0, 15.0, 1.0, 0.0, 6.0, 9.0, 16.0, 1.0, 0.0, 6.0, 3.0, 13.0, 1.0, 0.0, 6.0, 8.0, 16.0, 1.0, 0.0, 6.0, 9.0, 17.0, 1.0, 0.0, 6.0, 8.0, 17.0, 1.0, 0.0, 6.0, 7.0, 17.0, 1.0, 0.0, 6.0, 7.0, 16.0, 1.0, 0.0, 6.0, 7.0, 18.0, 1.0, 0.0, 6.0, 6.0, 18.0, 1.0, 0.0, 6.0, 6.0, 17.0, 1.0, 0.0, 6.0, 6.0, 16.0, 1.0, 0.0, 6.0, 5.0, 17.0, 1.0, 0.0, 6.0, 4.0, 16.0, 1.0, 0.0, 6.0, 3.0, 16.0, 1.0, 0.0, 6.0, 6.0, 15.0, 1.0, 0.0, 6.0, 5.0, 16.0, 1.0, 0.0, 6.0, 10.0, 13.0, 1.0, 0.0, 6.0, 4.0, 13.0, 1.0, 0.0, 6.0, 9.0, 15.0, 1.0, 0.0, 6.0, 4.0, 12.0, 1.0, 0.0, 6.0, 7.0, 14.0, 1.0, 0.0, 6.0, 6.0, 14.0, 1.0, 0.0, 6.0, 4.0, 14.0, 1.0, 75.0, 7.0, 4.0, 7.0, 1.0, 75.0, 7.0, 4.0, 6.0, 2.0, 151.0, 8.0, 3.0, 2.0, 1.0, 151.0, 8.0, 4.0, 6.0, 1.0, 229.0, 9.0, 1.0, -1.0, 2.0, 4.0, 0.0]
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
  drainages::Array{Field{Float64},1} = repeat(drainage,10000,false)
  runoffs::Array{Field{Float64},1} = drainages
  evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,180,false)
  initial_water_to_lake_centers = LatLonField{Float64}(lake_grid,
    Float64[  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
            430.0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0 ])
  initial_spillover_to_rivers = LatLonField{Float64}(hd_grid,
                                                     Float64[ 0.0 0.0 0.0 0.0
                                                              0.0 0.0 0.0 0.0
                                                              0.0 2.0*86400.0 0.0 0.0
                                                              0.0 0.0 1.0*86400.0 0.0 ])
  expected_river_inflow::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                               Float64[ 0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0 ])
  expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                                 Float64[ 0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0 ])
  expected_water_to_hd::Field{Float64} = LatLonField{Float64}(hd_grid,
                                                              Float64[ 0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0 ])
  expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
              6 9 6 0 0 0 0 0 0 0 0 0 9 9 8 9 0 0 0 6
              0 6 6 0 0 9 9 0 0 0 0 9 7 7 7 8 0 0 0 0
              0 6 6 0 0 4 4 4 4 0 0 0 5 7 7 7 8 0 0 0
              0 6 6 9 4 4 0 4 4 0 7 5 7 5 7 7 7 8 0 0
              6 6 6 0 4 4 4 4 4 0 7 5 5 7 7 7 7 8 0 0
              0 6 6 0 0 4 0 4 0 0 0 7 5 7 3 7 8 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 8 7 7 7 7 8 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 8 8 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3
             3 1 3 0 0 0 0 0 0 0 0 0 1 1 3 1 0 0 0 3
             0 3 3 0 0 1 1 0 0 0 0 1 3 3 3 3 0 0 0 0
             0 3 3 0 0 3 3 3 3 0 0 0 3 3 3 3 3 0 0 0
             0 3 3 1 3 3 0 3 3 0 3 3 3 3 3 3 3 3 0 0
             3 3 3 0 3 3 3 3 3 0 3 3 3 3 3 3 3 3 0 0
             0 3 3 0 0 3 0 3 0 0 0 3 3 3 3 3 3 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 3 3 3 3 3 3 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 3 3 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
  expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     430.0
                   430.0 430.0 430.0 0     0     0     0     0     0     0     #=
                =# 0     0     430.0 430.0 430.0 430.0 0     0     0     430.0
                   0     430.0 430.0 0     0     430.0 430.0 0     0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 0     0     0     0
                   0     430.0 430.0 0     0     430.0 430.0 430.0 430.0 0     #=
                =# 0     0     430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     430.0 430.0 430.0 430.0 430.0 0     430.0 430.0 0     #=
                =# 430.0 430.0 430.0 430.0 430.0 430.0 430.0 430.0 0     0
                   430.0 430.0 430.0 0     430.0 430.0 430.0 430.0 430.0 0     #=
                =# 430.0 430.0 430.0 430.0 430.0 430.0 430.0 430.0 0     0
                   0     430.0 430.0 0     0     430.0 0     430.0 0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     430.0 430.0 430.0 430.0 430.0 430.0 0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     430.0 430.0 0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0
                   0     0     0     0     0     0     0     0     0     0     #=
                =# 0     0     0     0     0     0     0     0     0     0 ])
  expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 1.0 0.0
                 1.0 0.976744186 0.0
                 1.0 0.1428571429 0.0])
  expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[  15 6  0
                1  42 0
                15 1  0 ])
  expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 6  311
                1  43 1
                15 7  1  ])
  expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 1.0, 38.0, 6.0, 46.0, 55.0, 55.0, 229.0]
  # expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 7.0
  #               8.0 1.0 6.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 4.0 3.0 0.0 0.0 0.0 8.0
  #               0.0 6.0 6.0 0.0 0.0 1.0 2.0 0.0 0.0 0.0 0.0 3.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0 0.0
  #               0.0 6.0 6.0 0.0 0.0 6.0 6.0 5.0 6.0 0.0 0.0 0.0 7.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0
  #               0.0 6.0 6.0 3.0 7.0 8.0 0.0 7.0 6.0 0.0 6.0 7.0 6.0 7.0 6.0 6.0 5.0 4.0 0.0 0.0
  #               5.0 5.0 6.0 0.0 7.0 8.0 7.0 7.0 6.0 0.0 6.0 7.0 7.0 6.0 6.0 6.0 5.0 4.0 0.0 0.0
  #               0.0 5.0 5.0 0.0 0.0 7.0 0.0 5.0 0.0 0.0 0.0 6.0 7.0 6.0 7.0 6.0 4.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0 6.0 6.0 6.0 5.0 4.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0 4.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])

  first_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  first_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  first_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                384.0 0.0 0.0 0.0
                                  0.0 2.0*86400.0 0.0 0.0
                                  0.0 0.0 1.0*86400.0 0.0 ])
  first_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
              6 0 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
              0 6 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 6 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 6 6 0 0 0 0 0 0 0 0 0 0 5 0 0 0 0 0 0
              6 6 6 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 6 6 0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  first_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2
             2 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2
             0 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 2 2 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
             2 2 2 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 2 2 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
  first_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
      Float64[  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
             =# 0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
             =# 0.0   0.0   0.0   0.0   0.0   0.0  46.0
               46.0   0.0  46.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
             =# 0.0   0.0   0.0   0.0   0.0   0.0  46.0
                0.0  46.0  46.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
             =# 0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0  46.0  46.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0  46.0  46.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
               46.0  46.0  46.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0  46.0  46.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0  ])
  first_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 0.1666666667 0.0
                 1.0 0.02325581395 0.0
                 0.06666666667 0.1428571429 0.0])
  first_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 1 0
               1 1 0
               1 1 0])
  first_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 6  311
                1  43 1
               15 7  1  ])
  first_intermediate_expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 0.0, 0.0, 0.0, 46.0, 0.0, 0.0, 0.0]
  # first_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0
  #               5.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 5.0
  #               0.0 3.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 3.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 3.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               2.0 2.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])

  second_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  second_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 2.0 0.0 1.0 ])
  second_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  second_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
              6 0 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
              0 6 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 6 6 0 0 4 4 4 4 0 0 0 0 0 0 0 0 0 0 0
              0 6 6 0 4 4 0 4 4 0 0 0 0 5 0 0 0 0 0 0
              6 6 6 0 4 4 4 4 4 0 0 0 0 0 0 0 0 0 0 0
              0 6 6 0 0 4 0 4 0 0 0 0 0 0 3 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  second_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2
             2 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2
             0 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 2 2 0 0 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0
             0 2 2 0 2 2 0 2 2 0 0 0 0 1 0 0 0 0 0 0
             2 2 2 0 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0
             0 2 2 0 0 2 0 2 0 0 0 0 0 0 2 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  second_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
      Float64[   0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0  46.0
                46.0   0   46.0 0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0  46.0
                 0    46.0 46.0 0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0    46.0 46.0 0   0   38.0 38.0 38.0 38.0  0   0   0   0   0   0   0   0   0   0   0
                 0    46.0 46.0 0  38.0 38.0  0   38.0 38.0  0   0   0   0   0   0   0   0   0   0   0
                46.0  46.0 46.0 0  38.0 38.0 38.0 38.0 38.0  0   0   0   0   0   0   0   0   0   0   0
                 0    46.0 46.0 0   0   38.0  0   38.0  0    0   0   0   0   0  346.0 0  0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0
                 0     0    0   0   0    0    0    0    0    0   0   0   0   0   0   0   0   0   0   0  ])
  second_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 1.0 1.0/6.0 0.0
                 1.0 1.0/43.0 0.0
                 1.0 1.0/7.0  0.0 ])
  second_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 1 0
               1 1 0
               15 1 0])
  second_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 15 6  311
                1  43 1
               15 7  1  ])
  second_intermediate_expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 346.0, 38.0, 0.0, 46.0, 0.0, 0.0, 0.0]
  # second_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 4.0
  #               5.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 5.0
  #               0.0 3.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 3.0 3.0 0.0 0.0 2.0 2.0 1.0 2.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 3.0 3.0 0.0 3.0 4.0 0.0 3.0 2.0 0.0 0.0 1.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               2.0 2.0 3.0 0.0 3.0 4.0 3.0 3.0 2.0 0.0 0.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 2.0 2.0 0.0 0.0 3.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])
  river_fields::RiverPrognosticFields,
    lake_model_prognostics::LakeModelPrognostics,
    lake_model_diagnostics::LakeModelDiagnostics =
    drive_hd_and_lake_model(river_parameters,lake_model_parameters,
                            lake_parameters_as_array,
                            drainages,runoffs,evaporations,
                            0,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes::Array{Float64} = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes::Field{Float64} =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test first_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(first_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test first_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test first_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test first_intermediate_expected_lake_types == lake_types
  @test first_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == first_intermediate_expected_diagnostic_lake_volumes
  @test isapprox(first_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
  @test first_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test first_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test first_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            1,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=-1)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test second_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(second_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test second_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test second_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test second_intermediate_expected_lake_types == lake_types
  @test second_intermediate_expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == second_intermediate_expected_diagnostic_lake_volumes
  @test isapprox(second_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
  @test second_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test second_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test second_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  river_fields,lake_model_prognostics,_ =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            2,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=1)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      if isa(lake,FillingLake)
        set!(lake_types,coords,1)
      elseif isa(lake,OverflowingLake)
        set!(lake_types,coords,2)
      elseif isa(lake,SubsumedLake)
        set!(lake_types,coords,3)
      else
        set!(lake_types,coords,4)
      end
    end
  end
  lake_volumes = Float64[]
  lake_fractions =
    calculate_lake_fraction_on_surface_grid(lake_model_parameters,
                                            lake_model_prognostics)
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  diagnostic_lake_volumes =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  @test expected_river_inflow == river_fields.river_inflow
  @test isapprox(expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test expected_lake_types == lake_types
  @test expected_lake_volumes == lake_volumes
  @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  @test isapprox(expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
  @test expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test expected_true_lake_depths == lake_model_parameters.true_lake_depths
end

# @testset "Lake model tests 17" begin
# #Simple tests of evaporation
#   hd_grid = LatLonGrid(4,4,true)
#   surface_model_grid = LatLonGrid(3,3,true)
#   flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
#                                                 Int64[ 3  2  2  1
#                                                        6  6 -2  4
#                                                        6  8  8  2
#                                                        9  8  8  0 ]))
#   river_reservoir_nums = LatLonField{Int64}(hd_grid,5)
#   overland_reservoir_nums = LatLonField{Int64}(hd_grid,1)
#   base_reservoir_nums = LatLonField{Int64}(hd_grid,1)
#   river_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
#   overland_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
#   base_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
#   landsea_mask = LatLonField{Bool}(hd_grid,fill(false,4,4))
#   set!(river_reservoir_nums,LatLonCoords(4,4),0)
#   set!(overland_reservoir_nums,LatLonCoords(4,4),0)
#   set!(base_reservoir_nums,LatLonCoords(4,4),0)
#   set!(landsea_mask,LatLonCoords(4,4),true)
#   river_parameters = RiverParameters(flow_directions,
#                                      river_reservoir_nums,
#                                      overland_reservoir_nums,
#                                      base_reservoir_nums,
#                                      river_retention_coefficients,
#                                      overland_retention_coefficients,
#                                      base_retention_coefficients,
#                                      landsea_mask,
#                                      hd_grid,86400.0,86400.0)
#   lake_grid = LatLonGrid(20,20,true)
#   lake_centers::Field{Bool} = LatLonField{Bool}(lake_grid,
#     Bool[ false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false true false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false ])
#   cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
#     Float64[ 2.5 3.0 2.5
#              3.0 4.0 3.0
#              2.5 3.0 2.5 ])
#   corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
#                                                     Int64[   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#                                                              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#                                                              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#                                                              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#                                                              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#                                                              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#                                                              3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#                                                              3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#                                                              3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#                                                              3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#                                                              3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 ])
#   corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
#                                                       Int64[ 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3 ])
#   cell_areas::Field{Float64} = LatLonField{Float64}(lake_grid,
#     Float64[ 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 ])
#   raw_heights::Field{Float64} = LatLonField{Float64}(lake_grid,
#     Float64[3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#             3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3
#             3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 3 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 3 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 3 3
#             3 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#             3 3 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3])
#   corrected_heights::Field{Float64}  = LatLonField{Float64}(lake_grid,
#     Float64[3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#             3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3
#             3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 3 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 3 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 3 3
#             3 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#             3 3 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 ])
#   is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
#     Bool[])
#   grid_specific_lake_model_parameters::LatLonLakeModelParameters =
#     LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
#                               corresponding_surface_cell_lon_index)
#   lake_model_parameters::LakeModelParameters =
#     LakeModelParameters(grid_specific_lake_model_parameters,
#                         lake_grid,
#                         hd_grid,
#                         surface_model_grid,
#                         cell_areas_on_surface_model_grid,
#                         LakeModelSettings(),
#                         X,
#                         is_lake)
#   lake_parameters_as_array::Vector{Float64} = Float64[]
#   drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
#   drainages::Array{Field{Float64},1} = repeat(drainage,10000,false)
#   runoffs::Array{Field{Float64},1} = drainages
#   evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,1.0)
#   evaporations::Array{Field{Float64},1} = repeat(evaporation,180,false)
#   initial_water_to_lake_centers = LatLonField{Float64}(lake_grid,
#     Float64[  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  1459.0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0 ])
#   initial_spillover_to_rivers = LatLonField{Float64}(hd_grid,
#                                                      Float64[ 0.0 0.0 0.0 0.0
#                                                               0.0 0.0 0.0 0.0
#                                                               0.0 0.0 0.0 0.0
#                                                               0.0 0.0 0.0 0.0 ])
#   expected_river_inflow::Field{Float64} = LatLonField{Float64}(hd_grid,
#                                                                Float64[ 0.0 0.0 0.0 0.0
#                                                                         0.0 0.0 0.0 0.0
#                                                                         0.0 0.0 0.0 0.0
#                                                                         0.0 0.0 0.0 0.0 ])
#   expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(hd_grid,
#                                                                  Float64[ 0.0 0.0 0.0 0.0
#                                                                           0.0 0.0 0.0 0.0
#                                                                           0.0 0.0 0.0 0.0
#                                                                           0.0 0.0 0.0 0.0 ])
#   expected_water_to_hd::Field{Float64} = LatLonField{Float64}(hd_grid,
#                                                               Float64[ 0.0 0.0 0.0 0.0
#                                                                        0.0 0.0 0.0 0.0
#                                                                        0.0 0.0 0.0 0.0
#                                                                        0.0 0.0 0.0 0.0 ])
#   expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
#        Int64[  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  1 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  ])
#   expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
#       Int64[ 0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  1 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  ])
#   expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
#         Float64[ 0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  ])
#   expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
#         Float64[ 0.0 0.0 0.0
#                  0.0 0.015625  0.0
#                  0.0 0.0 0.0])
#   expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[  0 0  0
#                 0 1  0
#                 0 0  0 ])
#   expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 36 48 36
#                48 64 48
#                36 48 36  ])
#   expected_lake_volumes::Array{Float64} = Float64[0.0]
#   expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
#         Float64[ 0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0 ])


#   first_intermediate_expected_river_inflow::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   first_intermediate_expected_water_to_ocean::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   first_intermediate_expected_water_to_hd::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   first_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
#        Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
#   first_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
#       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0   ])
#   first_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
#       Float64[  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
#              =# 0.0   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0 #=
#              =# 1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   0.0
#                 0.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0 #=
#              =# 1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   0.0
#                 0.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0 #=
#              =# 1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   0.0
#                 0.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0 #=
#             =#  1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   0.0
#                 0.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0 #=
#             =#  1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   0.0
#                 0.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0 #=
#             =#  1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   0.0
#                 0.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0 #=
#             =#  1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   0.0
#                 0.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0 #=
#             =#  1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   0.0
#                 0.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0 #=
#             =#  1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   0.0
#                 0.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0 #=
#             =#  1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   0.0
#                 0.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0 #=
#             =#  1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   0.0
#                 0.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0 #=
#             =#  1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   0.0
#                 0.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0 #=
#             =#  1459.0   0.0   0.0   0.0   0.0   1459.0   0.0
#                 0.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0 #=
#             =#  1459.0   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0 #=
#             =#  1459.0   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0 #=
#             =#  1459.0   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0 #=
#             =#  1459.0   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0   1459.0 #=
#             =#  1459.0   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
#             =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0  ])
#   first_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
#         Float64[ 0.6944444444 0.8333333333 0.6944444444
#                  0.8333333333 1.0 0.75
#                  0.6944444444 0.8333333333 0.0])
#   first_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 25 40 25
#                40 64 36
#                25 40 0])
#   first_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 36 48 36
#                48 64 48
#                36 48 36  ])
#   first_intermediate_expected_lake_volumes::Array{Float64} = Float64[1459.0]
#   first_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
#         Float64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                  0 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 0
#                  0 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 0
#                  0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
#                  0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
#                  0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
#                  0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
#                  0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
#                  0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
#                  0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
#                  0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
#                  0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
#                  0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
#                  0 4 5 5 5 5 5 5 5 5 5 5 5 5 0 0 0 0 4 0
#                  0 4 5 5 5 5 5 5 5 5 5 5 5 5 0 0 0 0 0 0
#                  0 4 5 5 5 5 5 5 5 5 5 5 5 5 0 0 0 0 0 0
#                  0 4 5 5 5 5 5 5 5 5 5 5 5 5 0 0 0 0 0 0
#                  0 4 4 4 4 4 4 4 4 4 4 4 4 4 0 0 0 0 0 0
#                  0 3 4 4 4 4 4 4 4 4 4 4 4 4 0 0 0 0 0 0
#                  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])

#   second_intermediate_expected_river_inflow::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   second_intermediate_expected_water_to_ocean::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   second_intermediate_expected_water_to_hd::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   second_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
#        Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0   ])
#   second_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
#       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
#   second_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
#       Float64[   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
#              =# 0.0   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667 #=
#              =#  1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667 #=
#              =# 1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   0.0
#                 0.0   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667 #=
#              =# 1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   0.0
#                 0.0   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667 #=
#              =# 1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   0.0
#                 0.0   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667 #=
#             =#  1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   0.0
#                 0.0   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667 #=
#             =#  1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   0.0
#                 0.0   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667 #=
#             =#  1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   0.0
#                 0.0   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667 #=
#             =#  1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   0.0
#                 0.0   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667 #=
#             =#  1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   0.0
#                 0.0   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667 #=
#             =#  1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   0.0
#                 0.0   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667 #=
#             =#  1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   0.0
#                 0.0   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667 #=
#             =#  1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   0.0
#                 0.0   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667 #=
#             =#  1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   0.0
#                 0.0   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667 #=
#             =#  1440.041667   0.0   0.0   0.0   0.0   1440.041667   0.0
#                 0.0   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667 #=
#             =#  1440.041667   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667 #=
#             =#  1440.041667   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667 #=
#             =#  1440.041667   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667 #=
#             =#  1440.041667   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667   1440.041667 #=
#             =#  1440.041667   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
#             =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0 ])
#   second_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
#         Float64[ 0.6944444444 0.8333333333 0.6944444444
#                  0.8333333333 1.0 0.75
#                  0.6944444444 0.8333333333 0.0 ])
#   second_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 25 40 25
#                40 64 36
#                25 40 0])
#   second_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 36 48 36
#                48 64 48
#                36 48 36  ])
#   second_intermediate_expected_lake_volumes::Array{Float64} = Float64[1440.041667]
#   second_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
#         Float64[   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
#                    0.00   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   0.00
#                    0.00   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   0.00
#                    0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
#                    0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
#                    0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
#                    0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
#                    0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
#                    0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
#                    0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
#                    0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
#                    0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
#                    0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
#                    0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   0.00   0.00   0.00   0.00   3.94   0.00
#                    0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   0.00   0.00   0.00   0.00   0.00   0.00
#                    0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   0.00   0.00   0.00   0.00   0.00   0.00
#                    0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   0.00   0.00   0.00   0.00   0.00   0.00
#                    0.00   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   0.00   0.00   0.00   0.00   0.00   0.00
#                    0.00   2.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   0.00   0.00   0.00   0.00   0.00   0.00
#                    0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00  ])

#   third_intermediate_expected_river_inflow::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   third_intermediate_expected_water_to_ocean::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   third_intermediate_expected_water_to_hd::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   third_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
#        Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0   ])
#   third_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
#       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
#   third_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
#       Float64[   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
#              =# 0.0   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667 #=
#              =#  586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667 #=
#              =# 586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   0.0
#                 0.0   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667 #=
#              =# 586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   0.0
#                 0.0   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667 #=
#              =# 586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   0.0
#                 0.0   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667 #=
#             =#  586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   0.0
#                 0.0   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667 #=
#             =#  586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   0.0
#                 0.0   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667 #=
#             =#  586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   0.0
#                 0.0   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667 #=
#             =#  586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   0.0
#                 0.0   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667 #=
#             =#  586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   0.0
#                 0.0   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667 #=
#             =#  586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   0.0
#                 0.0   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667 #=
#             =#  586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   0.0
#                 0.0   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667 #=
#             =#  586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   0.0
#                 0.0   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667 #=
#             =#  586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   0.0
#                 0.0   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667 #=
#             =#  586.9166667   0.0   0.0   0.0   0.0   586.9166667   0.0
#                 0.0   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667 #=
#             =#  586.9166667   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667 #=
#             =#  586.9166667   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667 #=
#             =#  586.9166667   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667 #=
#             =#  586.9166667   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667   586.9166667 #=
#             =#  586.9166667   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
#             =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0 ])
#   third_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
#         Float64[ 0.6944444444 0.8333333333 0.6944444444
#                  0.8333333333 1.0 0.75
#                  0.6944444444 0.8333333333 0.0 ])
#   third_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 25 40 25
#                40 64 36
#                25 40 0])
#   third_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 36 48 36
#                48 64 48
#                36 48 36  ])
#   third_intermediate_expected_lake_volumes::Array{Float64} = Float64[586.9166667]
#   third_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
#         Float64[   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
#   0.00   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   0.00
#   0.00   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   0.00
#   0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
#   0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
#   0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
#   0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
#   0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
#   0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
#   0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
#   0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
#   0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
#   0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
#   0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   0.00   0.00   0.00   0.00   1.04   0.00
#   0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   0.00   0.00   0.00   0.00   0.00   0.00
#   0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   0.00   0.00   0.00   0.00   0.00   0.00
#   0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   0.00   0.00   0.00   0.00   0.00   0.00
#   0.00   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   0.00   0.00   0.00   0.00   0.00   0.00
#   0.00   0.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   0.00   0.00   0.00   0.00   0.00   0.00
#   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00 ])

#   fourth_intermediate_expected_river_inflow::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   fourth_intermediate_expected_water_to_ocean::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   fourth_intermediate_expected_water_to_hd::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   fourth_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
#        Int64[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
#   fourth_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
#       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#              0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0   ])
#   fourth_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
#       Float64[  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 0 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 0 0
#                 0 0 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 0 0
#                 0 0 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 0 0
#                 0 0 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 0 0
#                 0 0 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 0 0
#                 0 0 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 0 0
#                 0 0 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 0 0
#                 0 0 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 0 0
#                 0 0 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 0 0
#                 0 0 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 0 0
#                 0 0 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 0 0 0 0 0 0
#                 0 0 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 0 0 0 0 0 0
#                 0 0 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 0 0 0 0 0 0
#                 0 0 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 #=
#             =#  567.9583333 567.9583333 567.9583333 567.9583333 567.9583333 0 0 0 0 0 0
#                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
#   fourth_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
#         Float64[ 0.3333333333 0.5 0.3333333333
#                  0.6666666667 1.0 0.5833333333
#                  0.3333333333 0.5 0.0 ])
#   fourth_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 12 24 12
#                32 64 28
#                12 24 0])
#   fourth_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 36 48 36
#                48 64 48
#                36 48 36  ])
#   fourth_intermediate_expected_lake_volumes::Array{Float64} = Float64[567.9583333]
#   fourth_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
#         Float64[   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
#                    0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
#                    0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
#                    0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
#                    0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
#                    0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
#                    0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
#                    0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
#                    0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
#                    0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
#                    0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
#                    0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
#                    0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
#                    0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00   0.00   0.00   0.00   0.00
#                    0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00   0.00   0.00   0.00   0.00
#                    0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00   0.00   0.00   0.00   0.00
#                    0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00   0.00   0.00   0.00   0.00
#                    0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
#                    0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
#                    0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00  ])

#   fifth_intermediate_expected_lake_volumes::Array{Float64} = Float64[369.2083332]
#   sixth_intermediate_expected_lake_volumes::Array{Float64} = Float64[355.9583332]

#   seventh_intermediate_expected_lake_volumes::Array{Float64} = Float64[213.625]
#   eighth_intermediate_expected_lake_volumes::Array{Float64} = Float64[203.458]

#   ninth_intermediate_expected_lake_volumes::Array{Float64} = Float64[99.0833]
#   tenth_intermediate_expected_lake_volumes::Array{Float64} = Float64[92.125]

#   eleventh_intermediate_expected_lake_volumes::Array{Float64} = Float64[39.625]
#   twelfth_intermediate_expected_lake_volumes::Array{Float64} = Float64[35.875]

#   thirteenth_intermediate_expected_lake_volumes::Array{Float64} = Float64[2.125]
  # river_fields::RiverPrognosticFields,
  #   lake_model_prognostics::LakeModelPrognostics,
  #   lake_model_diagnostics::LakeModelDiagnostics =
  #   drive_hd_and_lake_model(river_parameters,lake_model_parameters,
  #                           lake_parameters_as_array,
  #                           drainages,runoffs,evaporations,
  #                           0,true,
  #                           initial_water_to_lake_centers,
  #                           initial_spillover_to_rivers,
  #                           print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           use_realistic_surface_coupling=true)
#   lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
#   for i = 1:20
#     for j = 1:20
#       coords::LatLonCoords = LatLonCoords(i,j)
#       lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
#       if lake_number <= 0 continue end
#       lake::Lake = lake_model_prognostics.lakes[lake_number]
#       if isa(lake,FillingLake)
#         set!(lake_types,coords,1)
#       elseif isa(lake,OverflowingLake)
#         set!(lake_types,coords,2)
#       elseif isa(lake,SubsumedLake)
#         set!(lake_types,coords,3)
#       else
#         set!(lake_types,coords,4)
#       end
#     end
#   end
#   lake_volumes::Array{Float64} = Float64[]
  # lake_fractions =
  #   calculate_lake_fraction_on_surface_grid(lake_model_parameters,
  #                                           lake_model_prognostics)
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
  # diagnostic_lake_volumes::Field{Float64} =
  #   calculate_diagnostic_lake_volumes_field(lake_model_parameters,
  #                                           lake_model_prognostics)
#   @test first_intermediate_expected_river_inflow == river_fields.river_inflow
#   @test isapprox(first_intermediate_expected_water_to_ocean,
#                  river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
#   @test first_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
#   @test first_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
#   @test first_intermediate_expected_lake_types == lake_types
#   @test first_intermediate_expected_lake_volumes == lake_volumes
#   @test diagnostic_lake_volumes == first_intermediate_expected_diagnostic_lake_volumes
#   @test isapprox(first_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
#   @test first_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
#   @test first_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
#   @test first_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  # river_fields,lake_model_prognostics,lake_model_diagnostics =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           1,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=0,
#                             use_realistic_surface_coupling=true)
#   lake_types = LatLonField{Int64}(lake_grid,0)
#   for i = 1:20
#     for j = 1:20
#       coords::LatLonCoords = LatLonCoords(i,j)
#       lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
#       if lake_number <= 0 continue end
#       lake::Lake = lake_model_prognostics.lakes[lake_number]
#       if isa(lake,FillingLake)
#         set!(lake_types,coords,1)
#       elseif isa(lake,OverflowingLake)
#         set!(lake_types,coords,2)
#       elseif isa(lake,SubsumedLake)
#         set!(lake_types,coords,3)
#       else
#         set!(lake_types,coords,4)
#       end
#     end
#   end
#   lake_volumes = Float64[]
  # lake_fractions =
  #   calculate_lake_fraction_on_surface_grid(lake_model_parameters,
  #                                           lake_model_prognostics)
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
  # diagnostic_lake_volumes::Field{Float64} =
  #   calculate_diagnostic_lake_volumes_field(lake_model_parameters,
  #                                           lake_model_prognostics)
#   @test second_intermediate_expected_river_inflow == river_fields.river_inflow
#   @test isapprox(second_intermediate_expected_water_to_ocean,
#                  river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
#   @test second_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
#   @test second_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
#   @test second_intermediate_expected_lake_types == lake_types
#   @test isapprox(second_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
#   @test isapprox(diagnostic_lake_volumes,second_intermediate_expected_diagnostic_lake_volumes,
#                  rtol=0.0,atol=0.00001)
#   @test isapprox(second_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
#   @test second_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
#   @test second_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
#   @test isapprox(second_intermediate_expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.1)
  # river_fields,lake_model_prognostics,lake_model_diagnostics =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           46,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=1,
#                             use_realistic_surface_coupling=true)
#   lake_types = LatLonField{Int64}(lake_grid,0)
#   for i = 1:20
#     for j = 1:20
#       coords::LatLonCoords = LatLonCoords(i,j)
#       lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
#       if lake_number <= 0 continue end
#       lake::Lake = lake_model_prognostics.lakes[lake_number]
#       if isa(lake,FillingLake)
#         set!(lake_types,coords,1)
#       elseif isa(lake,OverflowingLake)
#         set!(lake_types,coords,2)
#       elseif isa(lake,SubsumedLake)
#         set!(lake_types,coords,3)
#       else
#         set!(lake_types,coords,4)
#       end
#     end
#   end
#   lake_volumes = Float64[]
  # lake_fractions =
  #   calculate_lake_fraction_on_surface_grid(lake_model_parameters,
  #                                           lake_model_prognostics)
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
  # diagnostic_lake_volumes::Field{Float64} =
  #   calculate_diagnostic_lake_volumes_field(lake_model_parameters,
  #                                           lake_model_prognostics)
#   @test third_intermediate_expected_river_inflow == river_fields.river_inflow
#   @test isapprox(third_intermediate_expected_water_to_ocean,
#                  river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
#   @test third_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
#   @test third_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
#   @test third_intermediate_expected_lake_types == lake_types
#   @test isapprox(third_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
#   @test isapprox(diagnostic_lake_volumes,third_intermediate_expected_diagnostic_lake_volumes,
#                  rtol=0.0,atol=0.00001)
#   @test isapprox(third_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
#   @test third_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
#   @test third_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
#   @test isapprox(third_intermediate_expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.1)
  # river_fields,lake_model_prognostics,lake_model_diagnostics =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           47,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=46,
#                             use_realistic_surface_coupling=true))
#   lake_types = LatLonField{Int64}(lake_grid,0)
#   for i = 1:20
#     for j = 1:20
#       coords::LatLonCoords = LatLonCoords(i,j)
#       lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
#       if lake_number <= 0 continue end
#       lake::Lake = lake_model_prognostics.lakes[lake_number]
#       if isa(lake,FillingLake)
#         set!(lake_types,coords,1)
#       elseif isa(lake,OverflowingLake)
#         set!(lake_types,coords,2)
#       elseif isa(lake,SubsumedLake)
#         set!(lake_types,coords,3)
#       else
#         set!(lake_types,coords,4)
#       end
#     end
#   end
#   lake_volumes = Float64[]
  # lake_fractions =
  #   calculate_lake_fraction_on_surface_grid(lake_model_parameters,
  #                                           lake_model_prognostics)
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
  # diagnostic_lake_volumes::Field{Float64} =
  #   calculate_diagnostic_lake_volumes_field(lake_model_parameters,
  #                                           lake_model_prognostics)
#   @test fourth_intermediate_expected_river_inflow == river_fields.river_inflow
#   @test isapprox(fourth_intermediate_expected_water_to_ocean,
#                  river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
#   @test fourth_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
#   @test fourth_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
#   @test fourth_intermediate_expected_lake_types == lake_types
#   @test isapprox(fourth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
#   @test isapprox(diagnostic_lake_volumes,fourth_intermediate_expected_diagnostic_lake_volumes,
#                  rtol=0.0,atol=0.00001)
#   @test isapprox(fourth_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
#   @test fourth_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
#   @test fourth_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
#   @test isapprox(fourth_intermediate_expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.1)
  # river_fields,lake_model_prognostics,lake_model_diagnostics =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           62,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=47,
#                             use_realistic_surface_coupling=true))
#   lake_volumes = Float64[]
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
#   @test isapprox(fifth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
  # river_fields,lake_model_prognostics,lake_model_diagnostics =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           63,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=62,
#                             use_realistic_surface_coupling=true))
#   lake_volumes = Float64[]
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
#   @test isapprox(sixth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
  # river_fields,lake_model_prognostics,lake_model_diagnostics =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           77,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=63,
#                             use_realistic_surface_coupling=true))
#   lake_volumes = Float64[]
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
#   @test isapprox(seventh_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
  # river_fields,lake_model_prognostics,lake_model_diagnostics =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           78,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=77,
#                             use_realistic_surface_coupling=true))
#   lake_volumes = Float64[]
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
#   @test isapprox(eighth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.001)
  # river_fields,lake_model_prognostics,lake_model_diagnostics =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           93,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=78,
#                             use_realistic_surface_coupling=true))
#   lake_volumes = Float64[]
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
#   @test isapprox(ninth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.0001)
  # river_fields,lake_model_prognostics,lake_model_diagnostics =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           94,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=93,
#                             use_realistic_surface_coupling=true))
#   lake_volumes = Float64[]
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
#   @test isapprox(tenth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
  # river_fields,lake_model_prognostics,lake_model_diagnostics =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           108,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=94,
#                             use_realistic_surface_coupling=true)
#   lake_volumes = Float64[]
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
#   @test isapprox(eleventh_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
  # river_fields,lake_model_prognostics,lake_model_diagnostics =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           109,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=108,
#                             use_realistic_surface_coupling=true)
#   lake_volumes = Float64[]
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
#   @test isapprox(twelfth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
  # river_fields,lake_model_prognostics,lake_model_diagnostics =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           124,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=109,
#                             use_realistic_surface_coupling=true)
#   lake_volumes = Float64[]
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
#   @test isapprox(thirteenth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
  # river_fields,lake_model_prognostics,_ =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           125,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=124,
#                             use_realistic_surface_coupling=true)
#   lake_types = LatLonField{Int64}(lake_grid,0)
#   for i = 1:20
#     for j = 1:20
#       coords::LatLonCoords = LatLonCoords(i,j)
#       lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
#       if lake_number <= 0 continue end
#       lake::Lake = lake_model_prognostics.lakes[lake_number]
#       if isa(lake,FillingLake)
#         set!(lake_types,coords,1)
#       elseif isa(lake,OverflowingLake)
#         set!(lake_types,coords,2)
#       elseif isa(lake,SubsumedLake)
#         set!(lake_types,coords,3)
#       else
#         set!(lake_types,coords,4)
#       end
#     end
#   end
#   lake_volumes = Float64[]
  # lake_fractions =
  #   calculate_lake_fraction_on_surface_grid(lake_model_parameters,
  #                                           lake_model_prognostics)
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
  # diagnostic_lake_volumes::Field{Float64} =
  #   calculate_diagnostic_lake_volumes_field(lake_model_parameters,
  #                                           lake_model_prognostics)
#   @test expected_river_inflow == river_fields.river_inflow
#   @test isapprox(expected_water_to_ocean,
#                  river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
#   @test expected_water_to_hd == lake_model_prognostics.water_to_hd
#   @test expected_lake_numbers == lake_model_prognostics.lake_numbers
#   @test expected_lake_types == lake_types
#   @test expected_lake_volumes == lake_volumes
#   @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
#   @test isapprox(expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
#   @test expected_number_lake_cells == lake_model_prognostics.lake_cell_count
#   @test expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
#   @test expected_true_lake_depths == lake_model_parameters.true_lake_depths
# end

# @testset "Lake model tests 18" begin
# #Simple tests of evaporation
#   grid = LatLonGrid(4,4,true)
#   surface_model_grid = LatLonGrid(3,3,true)
#   flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
#                                                 Int64[ 3  2  2  1
#                                                        6  6 -2  4
#                                                        6  8  8  2
#                                                        9  8  8  0 ]))
#   river_reservoir_nums = LatLonField{Int64}(hd_grid,5)
#   overland_reservoir_nums = LatLonField{Int64}(hd_grid,1)
#   base_reservoir_nums = LatLonField{Int64}(hd_grid,1)
#   river_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
#   overland_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
#   base_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
#   landsea_mask = LatLonField{Bool}(hd_grid,fill(false,4,4))
#   set!(river_reservoir_nums,LatLonCoords(4,4),0)
#   set!(overland_reservoir_nums,LatLonCoords(4,4),0)
#   set!(base_reservoir_nums,LatLonCoords(4,4),0)
#   set!(landsea_mask,LatLonCoords(4,4),true)
#   river_parameters = RiverParameters(flow_directions,
#                                      river_reservoir_nums,
#                                      overland_reservoir_nums,
#                                      base_reservoir_nums,
#                                      river_retention_coefficients,
#                                      overland_retention_coefficients,
#                                      base_retention_coefficients,
#                                      landsea_mask,
#                                      hd_grid,86400.0,86400.0)
#   lake_grid = LatLonGrid(20,20,true)
#   lake_centers::Field{Bool} = LatLonField{Bool}(lake_grid,
#     Bool[ false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false true false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false
#           false false false false false false false false false false false false false false false false false false false false ])
#   cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
#     Float64[ 2.5 3.0 2.5
#              3.0 4.0 3.0
#              2.5 3.0 2.5 ])
#   corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
#                                                     Int64[   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#                                                              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#                                                              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#                                                              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#                                                              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#                                                              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#                                                              3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#                                                              3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#                                                              3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#                                                              3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#                                                              3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 ])
#   corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
#                                                       Int64[ 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3 ])
#   cell_areas::Field{Float64} = LatLonField{Float64}(lake_grid,
#     Float64[ 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 ])
#   raw_heights::Field{Float64} = LatLonField{Float64}(lake_grid,
#     Float64[3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#             3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3
#             3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 3 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 3 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 3 3
#             3 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#             3 3 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3])
#   corrected_heights::Field{Float64}  = LatLonField{Float64}(lake_grid,
#     Float64[3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#             3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3
#             3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 2 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 3 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 3 3
#             3 2 1 1 1 1 1 1 1 1 1 1 1 1 3 3 3 3 3 3
#             3 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#             3 3 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#             3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 ])
#   is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
#     Bool[])
#   grid_specific_lake_model_parameters::LatLonLakeModelParameters =
#     LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
#                               corresponding_surface_cell_lon_index)
#   lake_model_parameters::LakeModelParameters =
#     LakeModelParameters(grid_specific_lake_model_parameters,
#                         lake_grid,
#                         hd_grid,
#                         surface_model_grid,
#                         cell_areas_on_surface_model_grid,
#                         LakeModelSettings(),
#                         X,
#                         is_lake)
#   lake_parameters_as_array::Vector{Float64} = Float64[]
#   drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
#   drainages::Array{Field{Float64},1} = repeat(drainage,10000,false)
#   runoff::Field{Float64} =  LatLonField{Float64}(hd_grid,
#                                                      Float64[ 0.0 0.0 0.0 0.0
#                                                               0.0 0.0 15.0/86400.0 0.0
#                                                               0.0 0.0 0.0 0.0
#                                                               0.0 0.0 0.0 0.0 ])
#   runoffs::Array{Field{Float64},1} = repeat(runoff,10000,false)
#   evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,1.0)
#   evaporations::Array{Field{Float64},1} = repeat(evaporation,180,false)
#   initial_water_to_lake_centers = LatLonField{Float64}(lake_grid,
#     Float64[  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  1459.0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#               0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0 ])
#   initial_spillover_to_rivers = LatLonField{Float64}(hd_grid,
#                                                      Float64[ 0.0 0.0 0.0 0.0
#                                                               0.0 0.0 0.0 0.0
#                                                               0.0 0.0 0.0 0.0
#                                                               0.0 0.0 0.0 0.0 ])

#   expected_lake_volumes::Vector{Vector{Float64}} =
#     [[1455.04], [1451.08], [1447.13], [1443.17], [1439.21], [1435.25], [1431.29], #=
#   =# [1427.33], [1423.38], [1419.42], [1415.46], [1411.5], [1407.54], [1403.58], #=
#   =# [1399.63], [1395.67], [1391.71], [1387.75], [1383.79], [1379.83], [1375.88], #=
#   =# [1371.92], [1367.96], [1364.0], [1360.04], [1356.08], [1352.13], [1348.17], #=
#   =# [1344.21], [1340.25], [1336.29], [1332.33], [1328.38], [1324.42], [1320.46], #=
#   =# [1316.5], [1312.54], [1308.58], [1304.63], [1300.67], [1296.71], [1292.75], #=
#   =# [1288.79], [1284.83], [1280.88], [1276.92], [1272.96], [1269.0], [1265.04], #=
#   =# [1261.08], [1257.13], [1253.17], [1249.21], [1245.25], [1241.29], [1237.33], #=
#   =# [1233.38], [1229.42], [1225.46], [1221.5], [1217.54], [1213.58], [1209.63], #=
#   =# [1205.67], [1201.71], [1197.75], [1193.79], [1189.83], [1185.88], [1181.92], #=
#   =# [1177.96], [1174.0], [1170.04], [1166.08], [1162.13], [1158.17], [1154.21], #=
#   =# [1150.25], [1146.29], [1142.33], [1138.38], [1134.42], [1130.46], [1126.5], #=
#   =# [1122.54], [1118.58], [1114.63], [1110.67], [1106.71], [1102.75], [1098.79], #=
#   =# [1094.83], [1090.88], [1086.92], [1082.96], [1079.0], [1075.04], [1071.08], #=
#   =# [1067.13], [1063.17], [1059.21], [1055.25], [1051.29], [1047.33], [1043.38], #=
#   =# [1039.42], [1035.46], [1031.5], [1027.54], [1023.58], [1019.63], [1015.67], #=
#   =# [1011.71], [1007.75], [1003.79], [999.833], [995.875], [991.917], [987.958], #=
#   =# [984.0], [980.042], [976.083], [972.125], [968.167], [964.208], [960.25], #=
#   =# [956.292], [952.333], [948.375], [944.417], [940.458], [936.5], [932.542], #=
#   =# [928.583], [924.625], [920.667], [916.708], [912.75], [908.792], [904.833], #=
#   =# [900.875], [896.917], [892.958], [889.0], [885.042], [881.083], [877.125], #=
#   =# [873.167], [869.208], [865.25], [861.292], [857.333], [853.375], [849.417], #=
#   =# [845.458], [841.5], [837.542], [833.583], [829.625], [825.667], [821.708], #=
#   =# [817.75], [813.792], [809.833], [805.875], [801.917], [797.958], [794.0], #=
#   =# [790.042], [786.083], [782.125], [778.167], [774.208], [770.25], [766.292], #=
#   =# [762.333], [758.375], [754.417], [750.458], [746.5], [742.542], [738.583], #=
#   =# [734.625], [730.667], [726.708], [722.75], [718.792], [714.833], [710.875], #=
#   =# [706.917], [702.958], [699.0], [695.042], [691.083], [687.125], [683.167], #=
#   =# [679.208], [675.25], [671.292], [667.333], [663.375], [659.417], [655.458], #=
#   =# [651.5], [647.542], [643.583], [639.625], [635.667], [631.708], [627.75], #=
#   =# [623.792], [619.833], [615.875], [611.917], [607.958], [604.0], [600.042], #=
#   =# [596.083], [592.125], [588.167], [584.208], [580.25], [576.292], [572.333], #=
#   =# [574.083], [570.125], [571.875], [573.625], [575.375], [571.417], [573.167], #=
#   =# [574.917], [570.958], [572.708], [574.458], [570.5], [572.25], [574.0], [570.042], #=
#   =# [571.792], [573.542], [575.292], [571.333], [573.083], [574.833], [570.875], #=
#   =# [572.625], [574.375], [570.417], [572.167], [573.917], [575.667], [571.708], #=
#   =# [573.458], [575.208], [571.25], [573.0], [574.75], [570.792], [572.542], #=
#   =# [574.292], [570.333], [572.083], [573.833], [575.583], [571.625], [573.375], #=
#   =# [575.125], [571.167], [572.917], [574.667], [570.708], [572.458], [574.208], #=
#   =# [570.25], [572.0], [573.75], [575.5], [571.542], [573.292], [575.042], [571.083], #=
#   =# [572.833], [574.583], [570.625], [572.375], [574.125], [570.167], [571.917], #=
#   =# [573.667], [575.417], [571.458], [573.208], [574.958], [571.0], [572.75], #=
#   =# [574.5], [570.542], [572.292], [574.042], [570.083], [571.833], [573.583], #=
#   =# [575.333], [571.375], [573.125], [574.875], [570.917], [572.667], [574.417], #=
#   =# [570.458], [572.208], [573.958], [575.708], [571.75], [573.5], [575.25], #=
#   =# [571.292], [573.042], [574.792], [570.833], [572.583], [574.333], [570.375], #=
#   =# [572.125], [573.875], [575.625], [571.667], [573.417], [575.167], [571.208], #=
#   =# [572.958], [574.708], [570.75], [572.5], [574.25], [570.292], [572.042], [573.792], #=
#   =# [575.542], [571.583], [573.333], [575.083], [571.125], [572.875], [574.625], #=
#   =# [570.667], [572.417], [574.167], [570.208], [571.958], [573.708], [575.458], [571.5], #=
#   =# [573.25], [575.0], [571.042], [572.792], [574.542], [570.583], [572.333], [574.083], #=
#   =# [570.125], [571.875], [573.625], [575.375], [571.417], [573.167], [574.917], [570.958], #=
#   =# [572.708], [574.458], [570.5], [572.25], [574.0], [570.042], [571.792], [573.542], #=
#   =# [575.292], [571.333], [573.083], [574.833], [570.875], [572.625], [574.375], [570.417], #=
#   =# [572.167], [573.917], [575.667], [571.708], [573.458], [575.208], [571.25], [573.0], #=
#   =# [574.75], [570.792], [572.542], [574.292], [570.333], [572.083], [573.833], [575.583], #=
#   =# [571.625], [573.375], [575.125], [571.167], [572.917], [574.667], [570.708], [572.458], #=
#   =# [574.208], [570.25], [572.0], [573.75], [575.5], [571.542], [573.292], [575.042], [571.083], #=
#   =# [572.833], [574.583], [570.625], [572.375], [574.125], [570.167], [571.917], [573.667], #=
#   =# [575.417], [571.458], [573.208], [574.958], [571.0], [572.75], [574.5], [570.542], [572.292],#=
#   =#  [574.042], [570.083], [571.833], [573.583], [575.333], [571.375], [573.125], [574.875], [570.917], #=
#   =# [572.667], [574.417], [570.458], [572.208], [573.958], [575.708], [571.75], [573.5], [575.25], #=
#   =# [571.292], [573.042], [574.792], [570.833], [572.583], [574.333], [570.375], [572.125], [573.875],#=
#   =#  [575.625], [571.667], [573.417], [575.167], [571.208], [572.958], [574.708], [570.75], [572.5], #=
#   =# [574.25], [570.292], [572.042], [573.792], [575.542], [571.583], [573.333], [575.083], [571.125], #=
#   =# [572.875], [574.625], [570.667], [572.417], [574.167], [570.208], [571.958], [573.708], [575.458], #=
#   =# [571.5], [573.25], [575.0], [571.042], [572.792], [574.542], [570.583], [572.333], [574.083], [570.125]]
#   lake_volumes::Vector{Vector{Float64}} = []
#   @time river_fields,lake_model_prognostics,lake_fields =
#   drive_hd_and_lake_model(river_parameters,lake_parameters,
#                           drainages,runoffs,evaporations,
#                           500,true,
#                           initial_water_to_lake_centers,
#                           initial_spillover_to_rivers,
#                           print_timestep_results=false,
#                           write_output=false,return_output=true,
#                           use_realistic_surface_coupling=true,
#                           return_lake_volumes=true,
#                           diagnostic_lake_volumes=lake_volumes)
#   for (lake_volumes_slice,expected_lake_volumes_slice) in zip(lake_volumes,expected_lake_volumes)
#     @test isapprox(lake_volumes_slice,expected_lake_volumes_slice,
#                    rtol=0.0,atol=0.01)
#   end
# end

# @testset "Lake model tests 19" begin
# #Simple tests of evaporation with multiple lakes
#   grid = LatLonGrid(4,4,true)
#   surface_model_grid = LatLonGrid(3,3,true)
#   flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
#                                                 Int64[ 5 5 5 5
#                                                        2 2 5 5
#                                                        5 4 8 8
#                                                        8 7 4 0 ]))
#   river_reservoir_nums = LatLonField{Int64}(hd_grid,5)
#   overland_reservoir_nums = LatLonField{Int64}(hd_grid,1)
#   base_reservoir_nums = LatLonField{Int64}(hd_grid,1)
#   river_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
#   overland_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
#   base_retention_coefficients = LatLonField{Float64}(hd_grid,0.0)
#   landsea_mask = LatLonField{Bool}(hd_grid,fill(false,4,4))
#   set!(river_reservoir_nums,LatLonCoords(4,4),0)
#   set!(overland_reservoir_nums,LatLonCoords(4,4),0)
#   set!(base_reservoir_nums,LatLonCoords(4,4),0)
#   set!(landsea_mask,LatLonCoords(4,4),true)
#   river_parameters = RiverParameters(flow_directions,
#                                      river_reservoir_nums,
#                                      overland_reservoir_nums,
#                                      base_reservoir_nums,
#                                      river_retention_coefficients,
#                                      overland_retention_coefficients,
#                                      base_retention_coefficients,
#                                      landsea_mask,
#                                      hd_grid,86400.0,86400.0)
#   lake_grid = LatLonGrid(20,20,true)
#   lake_centers::Field{Bool} = LatLonField{Bool}(lake_grid,
#     Bool[ false false false false false  false false false false false  false false false false false  #=
#       =#  false false false false false
#           false false false  true false  false false  true false false  false false false false false  #=
#       =#  false  true false false false
#           false  true false false false  false false false false false  false false false false false  #=
#       =#  false false false false false
#           false false false false false  false false false false false  false false false false false  #=
#       =#  false false false false false
#           false false false false false  false false false false false  false false false false false  #=
#       =#  false false false false false
#           false false false false false  false false false false false  false false false false false  #=
#       =#  false false false false false
#           false false false false false  false false false false false  false false false false false  #=
#       =#  false false false false false
#           false false false false false  false false false false false  false  true false false false  #=
#       =#  false true false false false
#           false false false false false  false false false false false  false false false false false  #=
#       =#  false false false false false
#           false false false false false  false false false false false  false false false false false  #=
#       =#  false false false false false
#           false false false false false  false false false false false  false false false false false  #=
#       =#  false false false false false
#           false false false false false  false false false false false  false false false false false  #=
#       =#  false false false false false
#           false false false false false  false false false false false  false false false false false  #=
#       =#  false false false false false
#           false false false false false  false false false false false  false false false false false  #=
#       =#  false false false false false
#           false  true false false false  false false false false false  false false false false false  #=
#       =#  false false false false false
#           false false false false false  false false false false false  false false false false false  #=
#       =#  false false false false false
#           false false false false false  false false false false false  false false false false false  #=
#       =#  false false false false false
#           false false false false false  false false false false false  false false false false false  #=
#       =#  false false false false false
#           false false false false false  false false false false false  false false false false false  #=
#       =#  false false false false false
#           false false false false false  false false false false false  false false false false false  #=
#       =#  false false false false false ])
#   cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
#     Float64[ 2.5 3.0 2.5
#              3.0 4.0 3.0
#              2.5 3.0 2.5 ])
#   corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
#                                                     Int64[   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#                                                              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#                                                              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#                                                              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#                                                              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#                                                              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#                                                              3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#                                                              3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#                                                              3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#                                                              3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#                                                              3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
#                                                              3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 ])
#   corresponding_surface_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
#                                                       Int64[ 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3
#                                                              1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3 ])
#   cell_areas::Field{Float64} = LatLonField{Float64}(lake_grid,
#     Float64[ 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
#              1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 ])
#   raw_heights::Field{Float64} = LatLonField{Float64}(lake_grid,
#     Float64[9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0
#             9.0 9.0 9.0 3.0 3.0  3.0 9.0 1.0 1.0 9.0  1.0 1.0 1.0 1.0 9.0  9.0 2.0 2.0 2.0 9.0
#             9.0 1.0 9.0 3.0 3.0  3.0 6.0 1.0 1.0 1.0  1.0 1.0 1.0 1.0 7.0  7.0 2.0 2.0 2.0 9.0
#             9.0 1.0 9.0 3.0 3.0  3.0 9.0 1.0 1.0 1.0  1.0 1.0 1.0 1.0 9.0  9.0 2.0 2.0 2.0 9.0
#             9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 1.0 1.0 1.0 9.0  9.0 2.0 2.0 2.0 9.0

#             9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 1.0 1.0 1.0 9.0  9.0 9.0 6.0 9.0 9.0
#             9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 5.0 9.0 9.0  9.0 3.0 3.0 3.0 9.0
#             9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
#             9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
#             9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0

#             9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
#             9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
#             9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
#             9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 8.0 9.0
#             9.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 6.0 6.0  9.0 9.0 9.0 9.0 7.0  7.0 7.0 7.0 7.0 7.0

#             9.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 9.0 7.0  0.0 0.0 0.0 0.0 0.0
#             9.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 9.0 7.0  0.0 0.0 0.0 0.0 0.0
#             9.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 8.0 7.0  0.0 0.0 0.0 0.0 0.0
#             9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 7.0  0.0 0.0 0.0 0.0 0.0
#             9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 7.0  0.0 0.0 0.0 0.0 0.0])
#   corrected_heights::Field{Float64}  = LatLonField{Float64}(lake_grid,
#     Float64[9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0
#             9.0 9.0 9.0 3.0 3.0  3.0 9.0 1.0 1.0 9.0  1.0 1.0 1.0 1.0 9.0  9.0 2.0 2.0 2.0 9.0
#             9.0 1.0 9.0 3.0 3.0  3.0 6.0 1.0 1.0 1.0  1.0 1.0 1.0 1.0 7.0  7.0 2.0 2.0 2.0 9.0
#             9.0 1.0 9.0 3.0 3.0  3.0 9.0 1.0 1.0 1.0  1.0 1.0 1.0 1.0 9.0  9.0 2.0 2.0 2.0 9.0
#             9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 1.0 1.0 1.0 9.0  9.0 2.0 2.0 2.0 9.0

#             9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 1.0 1.0 1.0 9.0  9.0 9.0 6.0 9.0 9.0
#             9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 5.0 9.0 9.0  9.0 3.0 3.0 3.0 9.0
#             9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
#             9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
#             9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0

#             9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
#             9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
#             9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
#             9.0 7.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 8.0 9.0
#             9.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 6.0 6.0  9.0 9.0 9.0 9.0 7.0  7.0 7.0 7.0 7.0 7.0

#             9.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 9.0 7.0  0.0 0.0 0.0 0.0 0.0
#             9.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 9.0 7.0  0.0 0.0 0.0 0.0 0.0
#             9.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 8.0 7.0  0.0 0.0 0.0 0.0 0.0
#             9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 7.0  0.0 0.0 0.0 0.0 0.0
#             9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 7.0  0.0 0.0 0.0 0.0 0.0])
#   is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
#     Bool[])
#   grid_specific_lake_model_parameters::LatLonLakeModelParameters =
#     LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
#                               corresponding_surface_cell_lon_index)
#   lake_model_parameters::LakeModelParameters =
#     LakeModelParameters(grid_specific_lake_model_parameters,
#                         lake_grid,
#                         hd_grid,
#                         surface_model_grid,
#                         cell_areas_on_surface_model_grid,
#                         LakeModelSettings(),
#                         X,
#                         is_lake)
#   lake_parameters_as_array::Vector{Float64} = Float64[]
#   drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
#   drainages::Array{Field{Float64},1} = repeat(drainage,10000,false)
#   runoffs::Array{Field{Float64},1} = drainages
#   evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,1.0)
#   evaporations::Array{Field{Float64},1} = repeat(evaporation,180,false)
#   initial_water_to_lake_centers = LatLonField{Float64}(lake_grid,
#     Float64[ 0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#              0 0 0 27.1 0  0 0 204.1 0 0  0 0 0 0 0  0 173.1 0 0 0
#              0 66.1 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#              0 0 0 0 0  0 0 0 0 0  0 18.1 0 0 0  0 63.1 0 0 0
#              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#              0 101.1 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  ])
#   initial_spillover_to_rivers = LatLonField{Float64}(hd_grid,
#                                                      Float64[ 0.0 0.0 0.0 0.0
#                                                               0.0 0.0 0.0 0.0
#                                                               0.0 0.0 0.0 0.0
#                                                               0.0 0.0 0.0 0.0 ])
#   expected_river_inflow::Field{Float64} = LatLonField{Float64}(hd_grid,
#                                                                Float64[ 0.0 0.0 0.0 0.0
#                                                                         0.0 0.0 0.0 0.0
#                                                                         0.0 0.0 0.0 0.0
#                                                                         0.0 0.0 0.0 0.0 ])
#   expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(hd_grid,
#                                                                  Float64[ 0.0 0.0 0.0 0.0
#                                                                           0.0 0.0 0.0 0.0
#                                                                           0.0 0.0 0.0 0.0
#                                                                           0.0 0.0 0.0 0.0 ])
#   expected_water_to_hd::Field{Float64} = LatLonField{Float64}(hd_grid,
#                                                               Float64[ 0.0 0.0 0.0 0.0
#                                                                        0.0 0.0 0.0 0.0
#                                                                        0.0 0.0 0.0 0.0
#                                                                        0.0 0.0 0.0 0.0 ])
#   expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
#        Int64[  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 0 0 3 0 0 0 4 0 0 0 0 0 0 0 0 6 0 0 0
#                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 0 0 0 0 0 0 0 0 0 0 5 0 0 0 0 7 0 0 0
#                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0   ])
#   expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
#       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0
#              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
#   expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
#         Float64[ 0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  ])
#   expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
#         Float64[ 0.06   0.02   0.03
#                  0.0    0.02   0.02
#                  0.03   0.0    0.00 ])
#   expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[  2    1    1
#                 0    1    1
#                 1    0    0 ])
#   expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 36 48 36
#                48 64 48
#                36 48 36  ])
#   expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#   expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
#         Float64[ 0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
#                  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0 ])
#   expected_lake_volumes_all_timesteps::Vector{Vector{Float64}} =
#     [[66.0, 97.3333, 27.0, 204.0, 18.0, 167.146, 63.0], [66.0, 93.6667, 27.0, 204.0, 18.0, 161.292, 63.0],
#      [66.0, 90.0, 27.0, 204.0, 18.0, 155.437, 63.0], [66.0, 86.3333, 27.0, 204.0, 18.0, 149.583, 63.0],
#      [66.0, 82.6667, 27.0, 204.0, 18.0, 143.729, 63.0], [66.0, 79.0, 27.0, 204.0, 18.0, 137.875, 63.0],
#      [66.0, 75.3333, 27.0, 204.0, 18.0, 132.021, 63.0], [66.0, 71.6667, 27.0, 204.0, 18.0, 126.167, 63.0],
#      [66.0, 68.0, 27.0, 204.0, 18.0, 120.312, 63.0], [66.0, 64.3333, 27.0, 204.0, 18.0, 114.458, 63.0],
#      [66.0, 60.6667, 27.0, 204.0, 18.0, 108.604, 63.0], [66.0, 57.0, 27.0, 204.0, 18.0, 102.75, 63.0],
#      [66.0, 53.3333, 27.0, 204.0, 18.0, 96.8958, 63.0], [66.0, 49.6667, 27.0, 204.0, 18.0, 91.0417, 63.0],
#      [66.0, 46.0, 27.0, 204.0, 18.0, 85.1875, 63.0], [64.6667, 43.6667, 27.0, 203.809, 18.0, 79.5243, 63.0],
#      [63.9514, 40.7153, 27.0, 200.309, 18.0, 77.309, 63.0], [63.2361, 37.7639, 27.0, 196.809, 18.0, 75.0937, 63.0],
#      [62.5208, 34.8125, 27.0, 193.309, 18.0, 72.8785, 63.0], [61.8056, 31.8611, 27.0, 189.809, 18.0, 70.6632, 63.0],
#      [61.0903, 28.9097, 27.0, 186.309, 18.0, 68.4479, 63.0], [60.375, 25.9583, 27.0, 182.809, 18.0, 66.2326, 63.0],
#      [59.6597, 23.0069, 27.0, 179.309, 18.0, 64.0174, 63.0], [58.9444, 20.0556, 27.0, 175.809, 18.0, 61.8021, 63.0],
#      [58.2292, 17.1042, 27.0, 172.309, 18.0, 59.5868, 63.0], [57.5139, 14.1528, 27.0, 168.809, 18.0, 57.3715, 63.0],
#      [56.7986, 11.2014, 27.0, 165.309, 18.0, 55.1562, 63.0], [56.0833, 8.25, 27.0, 161.809, 18.0, 52.941, 63.0],
#      [55.3681, 5.29861, 27.0, 158.309, 18.0, 50.7257, 63.0], [54.6528, 2.34722, 27.0, 154.809, 18.0, 48.5104, 63.0],
#      [53.9375, 0.0, 27.0, 151.309, 18.0, 47.8038, 61.4913], [53.2222, 0.0, 26.967, 147.842, 18.0, 46.9705, 60.1788],
#      [52.5069, 0.0, 26.342, 145.03, 18.0, 46.1372, 58.8663], [51.7917, 0.0, 25.717, 142.217, 18.0, 45.3038, 57.5538],
#      [51.0764, 0.0, 25.092, 139.405, 18.0, 44.4705, 56.2413], [50.3611, 0.0, 24.467, 136.592, 18.0, 43.6372, 54.9288],
#      [49.6458, 0.0, 23.842, 133.78, 18.0, 42.8038, 53.6163], [48.9306, 0.0, 23.217, 130.967, 18.0, 41.9705, 52.3038],
#      [48.2153, 0.0, 22.592, 128.155, 18.0, 41.1372, 50.9913], [47.5, 0.0, 21.967, 125.342, 18.0, 40.3038, 49.6788],
#      [46.7847, 0.0, 21.342, 122.53, 18.0, 39.4705, 48.3663], [46.0694, 0.0, 20.717, 119.717, 18.0, 38.6372, 47.0538],
#      [45.3542, 0.0, 20.092, 116.905, 18.0, 37.8038, 45.7413], [44.6389, 0.0, 19.467, 114.092, 18.0, 36.9705, 44.4288],
#      [43.9236, 0.0, 18.842, 111.28, 18.0, 36.1372, 43.1163], [43.2083, 0.0, 18.217, 108.467, 18.0, 35.3038, 41.8038],
#      [42.4931, 0.0, 17.592, 105.655, 18.0, 34.4705, 40.4913], [41.7778, 0.0, 16.967, 103.984, 16.8585, 33.6372, 39.1788],
#      [41.0625, 0.0, 16.342, 102.359, 15.7335, 32.8038, 37.8663], [40.3472, 0.0, 15.717, 100.734, 14.6085, 31.9705, 36.5538],
#      [39.6319, 0.0, 15.092, 99.1085, 13.4835, 31.1372, 35.2413], [38.9167, 0.0, 14.467, 97.4835, 12.3585, 30.3038, 33.9288],
#      [38.2014, 0.0, 13.842, 95.8585, 11.2335, 29.4705, 32.6163], [37.4861, 0.0, 13.217, 94.2335, 10.1085, 28.6372, 31.3038],
#      [36.7708, 0.0, 12.592, 92.6085, 8.98351, 27.8038, 29.9913], [36.0556, 0.0, 11.967, 90.9835, 7.85851, 26.9705, 28.6788],
#      [35.3403, 0.0, 11.342, 89.3585, 6.73351, 26.1372, 27.3663], [34.625, 0.0, 10.717, 87.7335, 5.60851, 25.3038, 26.0538],
#      [33.9097, 0.0, 10.092, 86.1085, 4.48351, 24.4705, 24.7413], [33.1944, 0.0, 9.46701, 84.4835, 3.35851, 23.6372, 23.4288],
#      [32.4792, 0.0, 8.84201, 82.8585, 2.23351, 22.8038, 22.1163], [31.7639, 0.0, 8.21701, 81.2335, 1.10851, 21.9705, 20.8038],
#      [31.0486, 0.0, 7.59201, 79.6085, 0.0, 21.1372, 19.4913], [30.3333, 0.0, 6.96701, 77.9835, 0.0, 20.3038, 18.1788],
#      [29.6181, 0.0, 6.34201, 76.3585, 0.0, 19.4705, 16.8663], [28.9028, 0.0, 5.71701, 74.7335, 0.0, 18.6372, 15.5538],
#      [28.1875, 0.0, 5.09201, 73.1085, 0.0, 17.8038, 14.2413], [27.4722, 0.0, 4.46701, 71.4835, 0.0, 16.9705, 12.9288],
#      [26.7569, 0.0, 3.84201, 69.8585, 0.0, 16.1372, 11.6163], [26.0417, 0.0, 3.21701, 68.2335, 0.0, 15.3038, 10.3038],
#      [25.3264, 0.0, 2.59201, 66.6085, 0.0, 14.4705, 8.99132], [24.6111, 0.0, 1.96701, 64.9835, 0.0, 13.6372, 7.67882],
#      [23.8958, 0.0, 1.34201, 63.3585, 0.0, 12.8038, 6.36632], [23.1806, 0.0, 0.717014, 61.7335, 0.0, 11.9705, 5.05382],
#      [22.4653, 0.0, 0.0920139, 60.1085, 0.0, 11.1372, 3.74132], [21.75, 0.0, 0.0, 58.4835, 0.0, 10.3038, 2.42882],
#      [21.0347, 0.0, 0.0, 56.8585, 0.0, 9.47049, 1.11632], [20.3194, 0.0, 0.0, 55.2335, 0.0, 8.63715, 6.66134e-16],
#      [19.6042, 0.0, 0.0, 53.6085, 0.0, 7.80382, 1.97215e-31], [18.8889, 0.0, 0.0, 51.9835, 0.0, 6.97049, 0.0],
#      [18.1736, 0.0, 0.0, 50.3585, 0.0, 6.13715, 0.0], [17.4583, 0.0, 0.0, 48.7335, 0.0, 5.30382, 0.0],
#      [16.7431, 0.0, 0.0, 47.1085, 0.0, 4.47049, 0.0], [16.0278, 0.0, 0.0, 45.4835, 0.0, 3.63715, 0.0],
#      [15.3125, 0.0, 0.0, 43.8585, 0.0, 2.80382, 0.0], [14.5972, 0.0, 0.0, 42.2335, 0.0, 1.97049, 0.0],
#      [13.8819, 0.0, 0.0, 40.6085, 0.0, 1.13715, 0.0], [13.1667, 0.0, 0.0, 38.9835, 0.0, 0.303819, 0.0],
#      [12.4514, 0.0, 0.0, 37.3585, 0.0, 0.0, 0.0], [11.7361, 0.0, 0.0, 35.7335, 0.0, 0.0, 0.0],
#      [11.0208, 0.0, 0.0, 34.1085, 0.0, 0.0, 0.0], [10.3056, 0.0, 0.0, 32.4835, 0.0, 0.0, 0.0],
#      [9.59028, 0.0, 0.0, 30.8585, 0.0, 0.0, 0.0], [8.875, 0.0, 0.0, 29.2335, 0.0, 0.0, 0.0],
#      [8.15972, 0.0, 0.0, 27.6085, 0.0, 0.0, 0.0], [7.44444, 0.0, 0.0, 25.9835, 0.0, 0.0, 0.0],
#      [6.72917, 0.0, 0.0, 24.3585, 0.0, 0.0, 0.0], [6.01389, 0.0, 0.0, 22.7335, 0.0, 0.0, 0.0],
#      [5.29861, 0.0, 0.0, 21.1085, 0.0, 0.0, 0.0], [4.58333, 0.0, 0.0, 19.4835, 0.0, 0.0, 0.0],
#      [3.86806, 0.0, 0.0, 17.8585, 0.0, 0.0, 0.0], [3.15278, 0.0, 0.0, 16.2335, 0.0, 0.0, 0.0],
#      [2.4375, 0.0, 0.0, 14.6085, 0.0, 0.0, 0.0], [1.72222, 0.0, 0.0, 12.9835, 0.0, 0.0, 0.0],
#      [1.00694, 0.0, 0.0, 11.3585, 0.0, 0.0, 0.0], [0.291667, 0.0, 0.0, 9.73351, 0.0, 0.0, 0.0],
#      [0.0212121, 0.0, 0.0, 8.10851, 0.0, 0.0, 0.0], [0.0015427, 0.0, 0.0, 6.48351, 0.0, 0.0, 0.0],
#      [0.000112196, 0.0, 0.0, 4.85851, 0.0, 0.0, 0.0], [8.15973e-6, 0.0, 0.0, 3.23351, 0.0, 0.0, 0.0],
#      [5.93435e-7, 0.0, 0.0, 1.60851, 0.0, 0.0, 0.0], [4.31589e-8, 0.0, 0.0, 1.33227e-15, 0.0, 0.0, 0.0],
#      [3.13883e-9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2.28279e-10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#      [1.66021e-11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

#   first_intermediate_expected_river_inflow::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   first_intermediate_expected_water_to_ocean::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   first_intermediate_expected_water_to_hd::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.7  ])
#   first_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
#        Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 0 0 3 3 3 0 4 4 0 4 4 4 4 0 0 6 6 6 0
#               0 1 0 3 3 3 4 4 4 4 4 4 4 4 4 6 6 6 6 0
#               0 1 0 3 3 3 0 4 4 4 4 4 4 4 0 0 6 6 6 0
#               0 1 0 0 0 0 0 0 0 0 0 4 4 4 0 0 6 6 6 0
#               0 1 0 0 0 0 0 0 0 0 0 4 4 4 0 0 0 6 0 0
#               0 1 0 0 0 0 0 0 0 0 0 0 4 0 0 0 7 7 7 0
#               0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#               0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#               0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#               0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#               0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#               0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#               0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0
#               0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
#               0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
#               0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
#   first_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
#       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 3 3 3 0 3 3 0 3 3 3 3 0 0 2 2 2 0
#              0 3 0 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 0
#              0 3 0 3 3 3 0 3 3 3 3 3 3 3 0 0 2 2 2 0
#              0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 2 2 2 0
#              0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 0 2 0 0
#              0 3 0 0 0 0 0 0 0 0 0 0 3 0 0 0 3 3 3 0
#              0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0
#              0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
#              0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
#              0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0   ])
#   first_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
#       Float64[  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   0.0   0.0 485.0 485.0 485.0   0.0 485.0 485.0   0.0 485.0 485.0 485.0 485.0   0.0   0.0 485.0 485.0 485.0   0.0
#                 0.0 167.0   0.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0   0.0
#                 0.0 167.0   0.0 485.0 485.0 485.0   0.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0   0.0   0.0 485.0 485.0 485.0   0.0
#                 0.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 485.0 485.0 485.0   0.0   0.0 485.0 485.0 485.0   0.0
#                 0.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 485.0 485.0 485.0   0.0   0.0   0.0 485.0   0.0   0.0
#                 0.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 485.0   0.0   0.0   0.0 485.0 485.0 485.0   0.0
#                 0.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 485.0 485.0 485.0   0.0   0.0 485.0 485.0 485.0   0.0
#                 0.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 485.0 485.0 485.0   0.0   0.0 485.0 485.0 485.0   0.0
#                 0.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 485.0 485.0 485.0   0.0   0.0 485.0 485.0 485.0   0.0
#                 0.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 485.0 485.0 485.0   0.0   0.0 485.0 485.0 485.0   0.0
#                 0.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 485.0 485.0 485.0   0.0   0.0 485.0 485.0 485.0   0.0
#                 0.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 485.0 485.0 485.0   0.0   0.0 485.0 485.0 485.0   0.0
#                 0.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
#                 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   ])
#   first_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
#         Float64[ 0.3611111111 0.5625 0.4166666667
#                  0.1458333333 0.296875 0.4375
#                  0.5555555556 0.5208333333 0.0])
#   first_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 13 27 15
#                 7 19 21
#                20 25 0])
#   first_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 36 48 36
#                48 64 48
#                36 48 36  ])
#   first_intermediate_expected_lake_volumes::Array{Float64} = Float64[66.0, 101.0, 27.0, 204.0, 18.0, 173.0, 63.0]
#   first_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
#         Float64[ 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 0.0 0.0 4.0 4.0 4.0 0.0 6.0 6.0 0.0 6.0 6.0 6.0 6.0 0.0 0.0 5.0 5.0 5.0 0.0
#                  0.0 7.0 0.0 4.0 4.0 4.0 1.0 6.0 6.0 6.0 6.0 6.0 6.0 6.0 0.0 0.0 5.0 5.0 5.0 0.0
#                  0.0 7.0 0.0 4.0 4.0 4.0 0.0 6.0 6.0 6.0 6.0 6.0 6.0 6.0 0.0 0.0 5.0 5.0 5.0 0.0
#                  0.0 7.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 6.0 6.0 6.0 0.0 0.0 5.0 5.0 5.0 0.0
#                  0.0 7.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 6.0 6.0 6.0 0.0 0.0 0.0 1.0 0.0 0.0
#                  0.0 7.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 4.0 4.0 4.0 0.0
#                  0.0 7.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 3.0 0.0 0.0 4.0 4.0 4.0 0.0
#                  0.0 7.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 3.0 0.0 0.0 4.0 4.0 4.0 0.0
#                  0.0 7.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 3.0 0.0 0.0 4.0 4.0 4.0 0.0
#                  0.0 7.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 3.0 0.0 0.0 4.0 4.0 4.0 0.0
#                  0.0 7.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 3.0 0.0 0.0 4.0 4.0 4.0 0.0
#                  0.0 7.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 3.0 0.0 0.0 4.0 4.0 4.0 0.0
#                  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ])

#   second_intermediate_expected_river_inflow::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   second_intermediate_expected_water_to_ocean::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   second_intermediate_expected_water_to_hd::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   second_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
#        Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 0 0 3 3 3 0 4 4 0 4 4 4 4 0 0 6 6 6 0
#               0 1 0 3 3 3 4 4 4 4 4 4 4 4 4 6 6 6 6 0
#               0 1 0 3 3 3 0 4 4 4 4 4 4 4 0 0 6 6 6 0
#               0 1 0 0 0 0 0 0 0 0 0 4 4 4 0 0 6 6 6 0
#               0 1 0 0 0 0 0 0 0 0 0 4 4 4 0 0 0 6 0 0
#               0 1 0 0 0 0 0 0 0 0 0 0 4 0 0 0 7 7 7 0
#               0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#               0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#               0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#               0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#               0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#               0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#               0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0
#               0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
#               0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
#               0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
#   second_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
#       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 3 3 3 0 3 3 0 3 3 3 3 0 0 1 1 1 0
#              0 3 0 3 3 3 3 3 3 3 3 3 3 3 3 1 1 1 1 0
#              0 3 0 3 3 3 0 3 3 3 3 3 3 3 0 0 1 1 1 0
#              0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 1 1 1 0
#              0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 0 1 0 0
#              0 3 0 0 0 0 0 0 0 0 0 0 3 0 0 0 3 3 3 0
#              0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
#   second_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
#       Float64[ 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
#                0.0   0.0   0.0 397.19 397.19 397.19   0.00 397.19 397.19   0.00 397.19 397.19 397.19 397.19   0.0   0.0  397.19 397.19 397.19   0.0
#                0.0 112.0   0.0 397.19 397.19 397.19 397.19 397.19 397.19 397.19 397.19 397.19 397.19 397.19 397.19 397.19 397.19 397.19 397.19   0.0
#                0.0 112.0   0.0 397.19 397.19 397.19   0.00 397.19 397.19 397.19 397.19 397.19 397.19 397.19  0.0   0.0  397.19 397.19 397.19   0.0
#                0.0 112.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 397.19 397.19 397.19   0.0    0.0 397.19 397.19 397.19   0.0
#                0.0 112.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 397.19 397.19 397.19   0.0    0.0   0.0  397.19    0.0   0.0
#                0.0 112.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 397.19    0.0    0.0    0.0 397.19 397.19 397.19   0.0
#                0.0 112.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 397.19 397.19 397.19   0.0    0.0 397.19 397.19 397.19   0.0
#                0.0 112.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 397.19 397.19 397.19   0.0    0.0 397.19 397.19 397.19   0.0
#                0.0 112.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 397.19 397.19 397.19   0.0    0.0 397.19 397.19 397.19   0.0
#                0.0 112.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 397.19 397.19 397.19   0.0    0.0 397.19 397.19 397.19   0.0
#                0.0 112.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 397.19 397.19 397.19   0.0    0.0 397.19 397.19 397.19   0.0
#                0.0 112.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 397.19 397.19 397.19   0.0    0.0 397.19 397.19 397.19   0.0
#                0.0 112.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0   0.0
#                0.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0   0.0   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0   0.0
#                0.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0  112.0    0.0    0.0    0.0    0.0    0.0    0.0   0.0
#                0.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0  112.0    0.0    0.0    0.0    0.0    0.0    0.0   0.0
#                0.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0  112.0    0.0    0.0    0.0    0.0    0.0    0.0   0.0
#                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0   0.0
#                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0   0.0  ])
#   second_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
#         Float64[ 0.3611111111 0.5625 0.4166666667
#                  0.1458333333 0.296875 0.4375
#                  0.5555555556 0.5208333333 0.0 ])
#   second_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 13 27 15
#                 7 19 21
#                20 25 0])
#   second_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 36 48 36
#                48 64 48
#                36 48 36  ])
#   second_intermediate_expected_lake_volumes::Array{Float64} = Float64[66.0, 46.0, 27.0, 204.0, 18.0, 85.1875, 63.0]
#   second_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
#         Float64[ 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
#                  0.0 0.00 0.00 4.04 4.04 4.04 0.00 6.04 6.04 0.00 6.04 6.04 6.04 6.04 0.00 0.00 5.04 5.04 5.04 0.00
#                  0.0 6.02 0.00 4.04 4.04 4.04 1.04 6.04 6.04 6.04 6.04 6.04 6.04 6.04 0.04 0.04 5.04 5.04 5.04 0.00
#                  0.0 6.02 0.00 4.04 4.04 4.04 0.00 6.04 6.04 6.04 6.04 6.04 6.04 6.04 0.00 0.00 5.04 5.04 5.04 0.00
#                  0.0 6.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 6.04 6.04 6.04 0.00 0.00 5.04 5.04 5.04 0.00
#                  0.0 6.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 6.04 6.04 6.04 0.00 0.00 0.00 1.04 0.00 0.00
#                  0.0 6.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 2.04 0.00 0.00 0.00 4.04 4.04 4.04 0.00
#                  0.0 6.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.04 3.04 3.04 0.00 0.00 4.04 4.04 4.04 0.00
#                  0.0 6.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.04 3.04 3.04 0.00 0.00 4.04 4.04 4.04 0.00
#                  0.0 6.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.04 3.04 3.04 0.00 0.00 4.04 4.04 4.04 0.00
#                  0.0 6.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.04 3.04 3.04 0.00 0.00 4.04 4.04 4.04 0.00
#                  0.0 6.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.04 3.04 3.04 0.00 0.00 4.04 4.04 4.04 0.00
#                  0.0 6.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.04 3.04 3.04 0.00 0.00 4.04 4.04 4.04 0.00
#                  0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
#                  0.0 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
#                  0.0 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00
#                  0.0 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00
#                  0.0 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00
#                  0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
#                  0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 ])

#   third_intermediate_expected_river_inflow::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   third_intermediate_expected_water_to_ocean::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   third_intermediate_expected_water_to_hd::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   third_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
#        Int64[  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 0 0 3 3 3 0 4 4 0 4 4 4 4 0 0 6 6 6 0
#                0 1 0 3 3 3 4 4 4 4 4 4 4 4 0 0 6 6 6 0
#                0 1 0 3 3 3 0 4 4 4 4 4 4 4 0 0 6 6 6 0
#                0 1 0 0 0 0 0 0 0 0 0 4 4 4 0 0 6 6 6 0
#                0 1 0 0 0 0 0 0 0 0 0 4 4 4 0 0 0 6 0 0
#                0 1 0 0 0 0 0 0 0 0 0 0 4 0 0 0 7 7 7 0
#                0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#                0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#                0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#                0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#                0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#                0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0
#                0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
#                0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
#                0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
#                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
#   third_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
#       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 3 3 3 0 1 1 0 1 1 1 1 0 0 1 1 1 0
#              0 1 0 3 3 3 1 1 1 1 1 1 1 1 0 0 1 1 1 0
#              0 1 0 3 3 3 0 1 1 1 1 1 1 1 0 0 1 1 1 0
#              0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
#              0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 1 0 0
#              0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 3 3 3 0
#              0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
#   third_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
#       Float64[     0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0   0.0
#                    0.0  0.0   0.0 248.81 248.81 248.81   0.0 248.81 248.81   0.00 248.81 248.81 248.81 248.81   0.0   0.0 142.52 142.52 142.52   0.0
#                    0.0 64.67  0.0 248.81 248.81 248.81 248.81 248.81 248.81 248.81 248.81 248.81 248.81 248.81   0.0   0.0 142.52 142.52 142.52   0.0
#                    0.0 64.67  0.0 248.81 248.81 248.81   0.0 248.81 248.81 248.81 248.81 248.81 248.81 248.81   0.0   0.0 142.52 142.52 142.52   0.0
#                    0.0 64.67  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 248.81 248.81 248.81   0.0   0.0 142.52 142.52 142.52   0.0
#                    0.0 64.67  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 248.81 248.81 248.81   0.0   0.0   0.0 142.52   0.0     0.0
#                    0.0 64.67  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 248.81   0.0     0.0   0.0 142.52 142.52 142.52   0.0
#                    0.0 64.67  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 248.81 248.81 248.81   0.0   0.0 142.52 142.52 142.52   0.0
#                    0.0 64.67  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 248.81 248.81 248.81   0.0   0.0 142.52 142.52 142.52   0.0
#                    0.0 64.67  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 248.81 248.81 248.81   0.0   0.0 142.52 142.52 142.52   0.0
#                    0.0 64.67  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 248.81 248.81 248.81   0.0   0.0 142.52 142.52 142.52   0.0
#                    0.0 64.67  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 248.81 248.81 248.81   0.0   0.0 142.52 142.52 142.52   0.0
#                    0.0 64.67  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 248.81 248.81 248.81   0.0   0.0 142.52 142.52 142.52   0.0
#                    0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
#                    0.0 43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67   0.0   0.0   0.0      0.0   0.0   0.0   0.0   0.0   0.0   0.0
#                    0.0 43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67   0.0   0.0   0.0   0.0   0.0   0.0   0.0
#                    0.0 43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67   0.0   0.0   0.0   0.0   0.0   0.0   0.0
#                    0.0 43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67   0.0   0.0   0.0   0.0   0.0   0.0   0.0
#                    0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
#                    0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   ])
#   third_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
#         Float64[ 0.3611111111 0.5625   0.3611111111
#                  0.1458333333 0.296875 0.4375
#                  0.5555555556 0.5208333333 0.0 ])
#   third_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 13 27 13
#                 7 19 21
#                20 25 0])
#   third_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 36 48 36
#                48 64 48
#                36 48 36  ])
#   third_intermediate_expected_lake_volumes::Array{Float64} = Float64[64.6667, 43.6667, 27.0, 203.809, 18.0, 79.5243, 63.0]
#   third_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
#         Float64[ 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.00 4.00 4.00 4.00 0.00 6.00 6.00 0.00 6.00 6.00 6.00 6.00 0.0 0.0 4.93 4.93 4.93 0.0
#                  0.0 5.88 0.00 4.00 4.00 4.00 1.00 6.00 6.00 6.00 6.00 6.00 6.00 6.00 0.0 0.0 4.93 4.93 4.93 0.0
#                  0.0 5.88 0.00 4.00 4.00 4.00 0.00 6.00 6.00 6.00 6.00 6.00 6.00 6.00 0.0 0.0 4.93 4.93 4.93 0.0
#                  0.0 5.88 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 6.00 6.00 6.00 0.0 0.0 4.93 4.93 4.93 0.0
#                  0.0 5.88 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 6.00 6.00 6.00 0.0 0.0 0.00 0.93 0.00 0.0
#                  0.0 5.88 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 2.00 0.00 0.0 0.0 3.93 3.93 3.93 0.0
#                  0.0 5.88 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.00 3.00 3.00 0.0 0.0 3.93 3.93 3.93 0.0
#                  0.0 5.88 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.00 3.00 3.00 0.0 0.0 3.93 3.93 3.93 0.0
#                  0.0 5.88 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.00 3.00 3.00 0.0 0.0 3.93 3.93 3.93 0.0
#                  0.0 5.88 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.00 3.00 3.00 0.0 0.0 3.93 3.93 3.93 0.0
#                  0.0 5.88 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.00 3.00 3.00 0.0 0.0 3.93 3.93 3.93 0.0
#                  0.0 5.88 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.00 3.00 3.00 0.0 0.0 3.93 3.93 3.93 0.0
#                  0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0  0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0  ])

#   fourth_intermediate_expected_river_inflow::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   fourth_intermediate_expected_water_to_ocean::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   fourth_intermediate_expected_water_to_hd::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   fourth_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
#        Int64[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 3 3 3 0 4 4 0 4 4 4 4 0 0 6 6 6 0
#              0 1 0 3 3 3 4 4 4 4 4 4 4 4 0 0 6 6 6 0
#              0 1 0 3 3 3 0 4 4 4 4 4 4 4 0 0 6 6 6 0
#              0 1 0 0 0 0 0 0 0 0 0 4 4 4 0 0 6 6 6 0
#              0 1 0 0 0 0 0 0 0 0 0 4 4 4 0 0 0 6 0 0
#              0 1 0 0 0 0 0 0 0 0 0 0 4 0 0 0 7 7 7 0
#              0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#              0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#              0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#              0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#              0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#              0 1 0 0 0 0 0 0 0 0 0 5 5 5 0 0 7 7 7 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0
#              0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
#              0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
#              0 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
#   fourth_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
#       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 3 3 3 0 1 1 0 1 1 1 1 0 0 1 1 1 0
#              0 1 0 3 3 3 1 1 1 1 1 1 1 1 0 0 1 1 1 0
#              0 1 0 3 3 3 0 1 1 1 1 1 1 1 0 0 1 1 1 0
#              0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
#              0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 1 0 0
#              0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 3 3 3 0
#              0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
#              0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0   ])
#   fourth_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
#       Float64[     0.0   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
#                    0.0   0.0    0.0  245.31 245.31 245.31   0.0  245.31 245.31   0.0  245.31 245.31 245.31 245.31   0.0    0.0  140.31 140.31 140.31   0.0
#                    0.0  63.95   0.0  245.31 245.31 245.31 245.31 245.31 245.31 245.31 245.31 245.31 245.31 245.31   0.0    0.0  140.31 140.31 140.31   0.0
#                    0.0  63.95   0.0  245.31 245.31 245.31   0.00 245.31 245.31 245.31 245.31 245.31 245.31 245.31   0.0    0.0  140.31 140.31 140.31   0.0
#                    0.0  63.95   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  245.31 245.31 245.31   0.0    0.0  140.31 140.31 140.31   0.0
#                    0.0  63.95   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  245.31 245.31 245.31   0.0    0.0    0.0  140.31   0.0    0.0
#                    0.0  63.95   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  245.31   0.0    0.0    0.0  140.31 140.31 140.31   0.0
#                    0.0  63.95   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  245.31 245.31 245.31   0.0    0.0  140.31 140.31 140.31   0.0
#                    0.0  63.95   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  245.31 245.31 245.31   0.0    0.0  140.31 140.31 140.31   0.0
#                    0.0  63.95   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  245.31 245.31 245.31   0.0    0.0  140.31 140.31 140.31   0.0
#                    0.0  63.95   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  245.31 245.31 245.31   0.0    0.0  140.31 140.31 140.31   0.0
#                    0.0  63.95   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  245.31 245.31 245.31   0.0    0.0  140.31 140.31 140.31   0.0
#                    0.0  63.95   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  245.31 245.31 245.31   0.0    0.0  140.31 140.31 140.31   0.0
#                    0.0   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
#                    0.0  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
#                    0.0  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72   0.0    0.0    0.0    0.0    0.0    0.0    0.0
#                    0.0  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72   0.0    0.0    0.0    0.0    0.0    0.0    0.0
#                    0.0  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72   0.0    0.0    0.0    0.0    0.0    0.0    0.0
#                    0.0   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
#                    0.0   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  ])
#   fourth_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
#         Float64[ 0.3611111111 0.5625   0.3611111111
#                  0.1458333333 0.296875 0.4375
#                  0.5555555556 0.5208333333 0.0 ])
#   fourth_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 13 27 13
#                 7 19 21
#                20 25 0])
#   fourth_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 36 48 36
#                48 64 48
#                36 48 36  ])
#   fourth_intermediate_expected_lake_volumes::Array{Float64} = Float64[63.9514, 40.7153, 27.0, 200.309, 18.0, 77.309, 63.0]
#   fourth_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
#         Float64[ 0.0 0.00 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.0 3.93 3.93 3.93 0.00 5.93 5.93 0.00 5.93 5.93 5.93 5.93 0.0 0.0 4.86 4.86 4.86 0.0
#                  0.0 5.81 0.0 3.93 3.93 3.93 0.93 5.93 5.93 5.93 5.93 5.93 5.93 5.93 0.0 0.0 4.86 4.86 4.86 0.0
#                  0.0 5.81 0.0 3.93 3.93 3.93 0.00 5.93 5.93 5.93 5.93 5.93 5.93 5.93 0.0 0.0 4.86 4.86 4.86 0.0
#                  0.0 5.81 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 5.93 5.93 5.93 0.0 0.0 4.86 4.86 4.86 0.0
#                  0.0 5.81 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 5.93 5.93 5.93 0.0 0.0 0.00 0.86 0.00 0.0
#                  0.0 5.81 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.93 0.00 0.0 0.0 3.86 3.86 3.86 0.0
#                  0.0 5.81 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 2.93 2.93 2.93 0.0 0.0 3.86 3.86 3.86 0.0
#                  0.0 5.81 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 2.93 2.93 2.93 0.0 0.0 3.86 3.86 3.86 0.0
#                  0.0 5.81 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 2.93 2.93 2.93 0.0 0.0 3.86 3.86 3.86 0.0
#                  0.0 5.81 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 2.93 2.93 2.93 0.0 0.0 3.86 3.86 3.86 0.0
#                  0.0 5.81 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 2.93 2.93 2.93 0.0 0.0 3.86 3.86 3.86 0.0
#                  0.0 5.81 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 2.93 2.93 2.93 0.0 0.0 3.86 3.86 3.86 0.0
#                  0.0 0.00 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.90 0.9 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.90 0.9 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.90 0.9 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.90 0.9 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0  ])

#   fifth_intermediate_expected_river_inflow::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   fifth_intermediate_expected_water_to_ocean::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   fifth_intermediate_expected_water_to_hd::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   fifth_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
#        Int64[   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 0 0 3 0 0 0 4 4 0 4 4 4 4 0 0 6 6 6 0
#                 0 1 0 0 0 0 0 4 4 4 4 4 4 4 0 0 6 6 6 0
#                 0 1 0 0 0 0 0 4 4 4 4 4 4 4 0 0 6 6 6 0
#                 0 1 0 0 0 0 0 0 0 0 0 4 4 4 0 0 6 6 6 0
#                 0 1 0 0 0 0 0 0 0 0 0 4 4 4 0 0 0 0 0 0
#                 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 1 0 0 0 0 0 0 0 0 0 5 0 0 0 0 7 0 0 0
#                 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
#   fifth_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
#       Int64[    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 0 0 1 0 0 0 1 1 0 1 1 1 1 0 0 1 1 1 0
#                 0 1 0 0 0 0 0 1 1 1 1 1 1 1 0 0 1 1 1 0
#                 0 1 0 0 0 0 0 1 1 1 1 1 1 1 0 0 1 1 1 0
#                 0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
#                 0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0
#                 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0
#                 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#                 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
#   fifth_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
#       Float64[  0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0  0.0    0.0   0.0  0.0   0.0 0.0 0.0  0.0  0.0  0.0
#                 0.0  0.0  0.0 0.0 0.0 0.0 0.0 47.11 47.11  0.0  47.11 47.11 47.11 47.11 0.0 0.0 4.47 4.47 4.47 0.0
#                 0.0 16.74 0.0 0.0 0.0 0.0 0.0 47.11 47.11 47.11 47.11 47.11 47.11 47.11 0.0 0.0 4.47 4.47 4.47 0.0
#                 0.0 16.74 0.0 0.0 0.0 0.0 0.0 47.11 47.11 47.11 47.11 47.11 47.11 47.11 0.0 0.0 4.47 4.47 4.47 0.0
#                 0.0 16.74 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0  0.0   47.11 47.11 47.11 0.0 0.0 4.47 4.47 4.47 0.0
#                 0.0 16.74 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0  0.0   47.11 47.11 47.11 0.0 0.0 0.0  0.0  0.0  0.0
#                 0.0 16.74 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0  0.0    0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
#                 0.0 16.74 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0  0.0    0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
#                 0.0 16.74 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0  0.0    0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
#                 0.0 16.74 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0  0.0    0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
#                 0.0 16.74 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0  0.0    0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
#                 0.0 16.74 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0  0.0    0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
#                 0.0 16.74 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0  0.0    0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
#                 0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0  0.0    0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
#                 0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0  0.0    0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
#                 0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0  0.0    0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
#                 0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0  0.0    0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
#                 0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0  0.0    0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
#                 0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0  0.0    0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
#                 0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0  0.0    0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0  ])
#   fifth_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
#         Float64[ 0.1388888889   0.5416666667   0.3333333333
#                  0.1458333333   0.015625       0.02083333333
#                  0.02777777778  0.0            0.0 ])
#   fifth_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[  5 26 12
#                 7  1  1
#                 1  0  0])
#   fifth_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 36 48 36
#                48 64 48
#                36 48 36  ])
#   fifth_intermediate_expected_lake_volumes::Array{Float64} = Float64[16.7431, 0.0, 0.0, 47.1085, 0.0, 4.47049, 0.0]
#   fifth_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
#         Float64[ 0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 1.81 1.81 0.00 1.81 1.81 1.81 1.81 0.0 0.0 0.37 0.37 0.37 0.0
#                  0.0 1.52 0.0 0.0 0.0 0.0 0.0 1.81 1.81 1.81 1.81 1.81 1.81 1.81 0.0 0.0 0.37 0.37 0.37 0.0
#                  0.0 1.52 0.0 0.0 0.0 0.0 0.0 1.81 1.81 1.81 1.81 1.81 1.81 1.81 0.0 0.0 0.37 0.37 0.37 0.0
#                  0.0 1.52 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 1.81 1.81 1.81 0.0 0.0 0.37 0.37 0.37 0.0
#                  0.0 1.52 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 1.81 1.81 1.81 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 1.52 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 1.52 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 1.52 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 1.52 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 1.52 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 1.52 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 1.52 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0 ])

#   sixth_intermediate_expected_river_inflow::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   sixth_intermediate_expected_water_to_ocean::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   sixth_intermediate_expected_water_to_hd::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   sixth_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
#        Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 0 0 3 0 0 0 4 4 0 4 4 4 4 0 0 6 6 6 0
#               0 1 0 0 0 0 0 4 4 4 4 4 4 4 0 0 6 6 6 0
#               0 1 0 0 0 0 0 4 4 4 4 4 4 4 0 0 6 6 6 0
#               0 1 0 0 0 0 0 0 0 0 0 4 4 4 0 0 6 6 6 0
#               0 1 0 0 0 0 0 0 0 0 0 4 4 4 0 0 0 0 0 0
#               0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 1 0 0 0 0 0 0 0 0 0 5 0 0 0 0 7 0 0 0
#               0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
#   sixth_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
#       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 1 0 0 0 1 1 0 1 1 1 1 0 0 1 1 1 0
#              0 1 0 0 0 0 0 1 1 1 1 1 1 1 0 0 1 1 1 0
#              0 1 0 0 0 0 0 1 1 1 1 1 1 1 0 0 1 1 1 0
#              0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
#              0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0
#              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0
#              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
#   sixth_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
#       Float64[ 0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
#                0.0  0.0  0.0 0.0 0.0 0.0 0.0 45.48 45.48  0.0  45.48 45.48 45.48 45.48 0.0 0.0 3.64  3.64 3.64 0.0
#                0.0 16.03 0.0 0.0 0.0 0.0 0.0 45.48 45.48 45.48 45.48 45.48 45.48 45.48 0.0 0.0 3.64  3.64 3.64 0.0
#                0.0 16.03 0.0 0.0 0.0 0.0 0.0 45.48 45.48 45.48 45.48 45.48 45.48 45.48 0.0 0.0 3.64  3.64 3.64 0.0
#                0.0 16.03 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0  45.48 45.48 45.48 0.0 0.0 3.64  3.64 3.64 0.0
#                0.0 16.03 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0  45.48 45.48 45.48 0.0 0.0 0.0   0.0  0.0  0.0
#                0.0 16.03 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
#                0.0 16.03 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
#                0.0 16.03 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
#                0.0 16.03 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
#                0.0 16.03 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
#                0.0 16.03 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
#                0.0 16.03 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
#                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
#                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
#                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
#                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
#                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
#                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
#                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0  ])
#   sixth_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
#         Float64[ 0.1388888889 0.5416666667 0.3333333333
#                  0.1458333333 0.015625 0.02083333333
#                  0.02777777778  0.0    0.0 ])
#   sixth_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 5 26 12
#                7  1 1
#                1  0 0 ])
#   sixth_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 36 48 36
#                48 64 48
#                36 48 36  ])
#   sixth_intermediate_expected_lake_volumes::Array{Float64} = Float64[16.0278, 0.0, 0.0, 45.4835, 0.0, 3.63715, 0.0]
#   sixth_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
#         Float64[ 0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 1.75 1.75 0.00 1.75 1.75 1.75 1.75 0.0 0.0 0.3 0.3 0.3 0.0
#                  0.0 1.46 0.0 0.0 0.0 0.0 0.0 1.75 1.75 1.75 1.75 1.75 1.75 1.75 0.0 0.0 0.3 0.3 0.3 0.0
#                  0.0 1.46 0.0 0.0 0.0 0.0 0.0 1.75 1.75 1.75 1.75 1.75 1.75 1.75 0.0 0.0 0.3 0.3 0.3 0.0
#                  0.0 1.46 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 1.75 1.75 1.75 0.0 0.0 0.3 0.3 0.3 0.0
#                  0.0 1.46 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 1.75 1.75 1.75 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 1.46 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 1.46 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 1.46 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 1.46 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 1.46 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 1.46 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 1.46 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0 ])
#   seventh_intermediate_expected_river_inflow::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   seventh_intermediate_expected_water_to_ocean::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   seventh_intermediate_expected_water_to_hd::Field{Float64} =
#     LatLonField{Float64}(hd_grid,
#                          Float64[ 0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0
#                                   0.0 0.0 0.0 0.0 ])
#   seventh_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
#        Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 0 0 3 0 0 0 4 4 0 4 4 4 4 0 0 6 6 6 0
#               0 1 0 0 0 0 0 4 4 4 4 4 4 4 0 0 6 6 6 0
#               0 1 0 0 0 0 0 4 4 4 4 4 4 4 0 0 6 6 6 0
#               0 1 0 0 0 0 0 0 0 0 0 4 4 4 0 0 6 6 6 0
#               0 1 0 0 0 0 0 0 0 0 0 4 4 4 0 0 0 0 0 0
#               0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 1 0 0 0 0 0 0 0 0 0 5 0 0 0 0 7 0 0 0
#               0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
#   seventh_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
#       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 1 0 0 0 1 1 0 1 1 1 1 0 0 1 1 1 0
#              0 1 0 0 0 0 0 1 1 1 1 1 1 1 0 0 1 1 1 0
#              0 1 0 0 0 0 0 1 1 1 1 1 1 1 0 0 1 1 1 0
#              0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
#              0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0
#              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0
#              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
#   seventh_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
#       Float64[ 0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
#                0.0  0.0  0.0 0.0 0.0 0.0 0.0 43.86 43.86  0.00 43.86 43.86 43.86 43.86 0.0 0.0 2.8 2.8 2.8 0.0
#                0.0 15.31 0.0 0.0 0.0 0.0 0.0 43.86 43.86 43.86 43.86 43.86 43.86 43.86 0.0 0.0 2.8 2.8 2.8 0.0
#                0.0 15.31 0.0 0.0 0.0 0.0 0.0 43.86 43.86 43.86 43.86 43.86 43.86 43.86 0.0 0.0 2.8 2.8 2.8 0.0
#                0.0 15.31 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0  43.86 43.86 43.86 0.0 0.0 2.8 2.8 2.8 0.0
#                0.0 15.31 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0  43.86 43.86 43.86 0.0 0.0 0.0 0.0 0.0 0.0
#                0.0 15.31 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
#                0.0 15.31 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
#                0.0 15.31 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
#                0.0 15.31 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
#                0.0 15.31 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
#                0.0 15.31 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
#                0.0 15.31 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
#                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
#                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
#                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
#                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
#                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
#                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
#                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0  ])
#   seventh_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
#         Float64[ 0.1388888889 0.5416666667   0.3333333333
#                  0.1458333333 0.015625 0.02083333333
#                  0.02777777778  0.0    0.0 ])
#   seventh_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[  5 26 12
#                 7  1  1
#                 1  0  0])
#   seventh_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
#         Int64[ 36 48 36
#                48 64 48
#                36 48 36  ])
#   seventh_intermediate_expected_lake_volumes::Array{Float64} = Float64[15.3125, 0.0, 0.0, 43.8585, 0.0, 2.80382, 0.0]
#   seventh_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
#         Float64[ 0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 1.69 1.69 0.00 1.69 1.69 1.69 1.69 0.0 0.0 0.23 0.23 0.23 0.0
#                  0.0 1.39 0.0 0.0 0.0 0.0 0.0 1.69 1.69 1.69 1.69 1.69 1.69 1.69 0.0 0.0 0.23 0.23 0.23 0.0
#                  0.0 1.39 0.0 0.0 0.0 0.0 0.0 1.69 1.69 1.69 1.69 1.69 1.69 1.69 0.0 0.0 0.23 0.23 0.23 0.0
#                  0.0 1.39 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 1.69 1.69 1.69 0.0 0.0 0.23 0.23 0.23 0.0
#                  0.0 1.39 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 1.69 1.69 1.69 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 1.39 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 1.39 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 1.39 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 1.39 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 1.39 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 1.39 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 1.39 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
#                  0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0 ])
  # river_fields::RiverPrognosticFields,
  #   lake_model_prognostics::LakeModelPrognostics,
  #   lake_model_diagnostics::LakeModelDiagnostics =
  #   drive_hd_and_lake_model(river_parameters,lake_model_parameters,
  #                           lake_parameters_as_array,
  #                           drainages,runoffs,evaporations,
  #                           0,true,
  #                           initial_water_to_lake_centers,
  #                           initial_spillover_to_rivers,
  #                           print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           use_realistic_surface_coupling=true)
#   lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
#   for i = 1:20
#     for j = 1:20
#       coords::LatLonCoords = LatLonCoords(i,j)
#       lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
#       if lake_number <= 0 continue end
#       lake::Lake = lake_model_prognostics.lakes[lake_number]
#       if isa(lake,FillingLake)
#         set!(lake_types,coords,1)
#       elseif isa(lake,OverflowingLake)
#         set!(lake_types,coords,2)
#       elseif isa(lake,SubsumedLake)
#         set!(lake_types,coords,3)
#       else
#         set!(lake_types,coords,4)
#       end
#     end
#   end
#   lake_volumes::Array{Float64} = Float64[]
  # lake_fractions =
  #   calculate_lake_fraction_on_surface_grid(lake_model_parameters,
  #                                           lake_model_prognostics)
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
  # diagnostic_lake_volumes::Field{Float64} =
  #   calculate_diagnostic_lake_volumes_field(lake_model_parameters,
  #                                           lake_model_prognostics)
#   @test first_intermediate_expected_river_inflow == river_fields.river_inflow
#   @test isapprox(first_intermediate_expected_water_to_ocean,
#                  river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
#   @test isapprox(first_intermediate_expected_water_to_hd,lake_model_prognostics.water_to_hd,
#                  rtol=0.0,atol=0.00001)
#   @test first_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
#   @test first_intermediate_expected_lake_types == lake_types
#   @test first_intermediate_expected_lake_volumes == lake_volumes
#   @test diagnostic_lake_volumes == first_intermediate_expected_diagnostic_lake_volumes
#   @test isapprox(first_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
#   @test first_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
#   @test first_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
#   @test first_intermediate_expected_true_lake_depths == lake_model_parameters.true_lake_depths
  # river_fields,lake_model_prognostics,lake_model_diagnostics =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           15,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=0,
#                             use_realistic_surface_coupling=true)
#   lake_types = LatLonField{Int64}(lake_grid,0)
#   for i = 1:20
#     for j = 1:20
#       coords::LatLonCoords = LatLonCoords(i,j)
#       lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
#       if lake_number <= 0 continue end
#       lake::Lake = lake_model_prognostics.lakes[lake_number]
#       if isa(lake,FillingLake)
#         set!(lake_types,coords,1)
#       elseif isa(lake,OverflowingLake)
#         set!(lake_types,coords,2)
#       elseif isa(lake,SubsumedLake)
#         set!(lake_types,coords,3)
#       else
#         set!(lake_types,coords,4)
#       end
#     end
#   end
#   lake_volumes = Float64[]
  # lake_fractions =
  #   calculate_lake_fraction_on_surface_grid(lake_model_parameters,
  #                                           lake_model_prognostics)
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
  # diagnostic_lake_volumes::Field{Float64} =
  #   calculate_diagnostic_lake_volumes_field(lake_model_parameters,
  #                                           lake_model_prognostics)
#   @test second_intermediate_expected_river_inflow == river_fields.river_inflow
#   @test isapprox(second_intermediate_expected_water_to_ocean,
#                  river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
#   @test second_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
#   @test second_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
#   @test second_intermediate_expected_lake_types == lake_types
#   @test isapprox(second_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
#   @test isapprox(diagnostic_lake_volumes,second_intermediate_expected_diagnostic_lake_volumes,
#                  rtol=0.0,atol=0.05)
#   @test isapprox(second_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
#   @test second_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
#   @test second_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
#   @test isapprox(second_intermediate_expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.1)
  # river_fields,lake_model_prognostics,lake_model_diagnostics =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           16,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=15,
#                             use_realistic_surface_coupling=true)
#   lake_types = LatLonField{Int64}(lake_grid,0)
#   for i = 1:20
#     for j = 1:20
#       coords::LatLonCoords = LatLonCoords(i,j)
#       lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
#       if lake_number <= 0 continue end
#       lake::Lake = lake_model_prognostics.lakes[lake_number]
#       if isa(lake,FillingLake)
#         set!(lake_types,coords,1)
#       elseif isa(lake,OverflowingLake)
#         set!(lake_types,coords,2)
#       elseif isa(lake,SubsumedLake)
#         set!(lake_types,coords,3)
#       else
#         set!(lake_types,coords,4)
#       end
#     end
#   end
#   lake_volumes = Float64[]
  # lake_fractions =
  #   calculate_lake_fraction_on_surface_grid(lake_model_parameters,
  #                                           lake_model_prognostics)
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
  # diagnostic_lake_volumes::Field{Float64} =
  #   calculate_diagnostic_lake_volumes_field(lake_model_parameters,
  #                                           lake_model_prognostics)
#   @test third_intermediate_expected_river_inflow == river_fields.river_inflow
#   @test isapprox(third_intermediate_expected_water_to_ocean,
#                  river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
#   @test third_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
#   @test third_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
#   @test third_intermediate_expected_lake_types == lake_types
#   @test isapprox(third_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.01)
#   @test isapprox(diagnostic_lake_volumes,third_intermediate_expected_diagnostic_lake_volumes,
#                  rtol=0.0,atol=0.05)
#   @test isapprox(third_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
#   @test third_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
#   @test third_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
#   @test isapprox(third_intermediate_expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.1)
  # river_fields,lake_model_prognostics,lake_model_diagnostics =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           17,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=16,
#                             use_realistic_surface_coupling=true)
#   lake_types = LatLonField{Int64}(lake_grid,0)
#   for i = 1:20
#     for j = 1:20
#       coords::LatLonCoords = LatLonCoords(i,j)
#       lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
#       if lake_number <= 0 continue end
#       lake::Lake = lake_model_prognostics.lakes[lake_number]
#       if isa(lake,FillingLake)
#         set!(lake_types,coords,1)
#       elseif isa(lake,OverflowingLake)
#         set!(lake_types,coords,2)
#       elseif isa(lake,SubsumedLake)
#         set!(lake_types,coords,3)
#       else
#         set!(lake_types,coords,4)
#       end
#     end
#   end
#   lake_volumes = Float64[]
  # lake_fractions =
  #   calculate_lake_fraction_on_surface_grid(lake_model_parameters,
  #                                           lake_model_prognostics)
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
  # diagnostic_lake_volumes::Field{Float64} =
  #   calculate_diagnostic_lake_volumes_field(lake_model_parameters,
  #                                           lake_model_prognostics)
#   @test fourth_intermediate_expected_river_inflow == river_fields.river_inflow
#   @test isapprox(fourth_intermediate_expected_water_to_ocean,
#                  river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
#   @test fourth_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
#   @test fourth_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
#   @test fourth_intermediate_expected_lake_types == lake_types
#   @test isapprox(fourth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.01)
#   @test isapprox(diagnostic_lake_volumes,fourth_intermediate_expected_diagnostic_lake_volumes,
#                  rtol=0.0,atol=0.05)
#   @test isapprox(fourth_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
#   @test fourth_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
#   @test fourth_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
#   @test isapprox(fourth_intermediate_expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.1)
  # river_fields,lake_model_prognostics,lake_model_diagnostics =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           83,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=17,
#                             use_realistic_surface_coupling=true)
#   lake_types = LatLonField{Int64}(lake_grid,0)
#   for i = 1:20
#     for j = 1:20
#       coords::LatLonCoords = LatLonCoords(i,j)
#       lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
#       if lake_number <= 0 continue end
#       lake::Lake = lake_model_prognostics.lakes[lake_number]
#       if isa(lake,FillingLake)
#         set!(lake_types,coords,1)
#       elseif isa(lake,OverflowingLake)
#         set!(lake_types,coords,2)
#       elseif isa(lake,SubsumedLake)
#         set!(lake_types,coords,3)
#       else
#         set!(lake_types,coords,4)
#       end
#     end
#   end
#   lake_volumes = Float64[]
  # lake_fractions =
  #   calculate_lake_fraction_on_surface_grid(lake_model_parameters,
  #                                           lake_model_prognostics)
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
  # diagnostic_lake_volumes::Field{Float64} =
  #   calculate_diagnostic_lake_volumes_field(lake_model_parameters,
  #                                           lake_model_prognostics)
#   @test fifth_intermediate_expected_river_inflow == river_fields.river_inflow
#   @test isapprox(fifth_intermediate_expected_water_to_ocean,
#                  river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
#   @test fifth_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
#   @test fifth_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
#   @test fifth_intermediate_expected_lake_types == lake_types
#   @test isapprox(fifth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.01)
#   @test isapprox(diagnostic_lake_volumes,fifth_intermediate_expected_diagnostic_lake_volumes,
#                  rtol=0.0,atol=0.05)
#   @test isapprox(fifth_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
#   @test fifth_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
#   @test fifth_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
#   @test isapprox(fifth_intermediate_expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.1)
  # river_fields,lake_model_prognostics,lake_model_diagnostics =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           84,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=83,
#                             use_realistic_surface_coupling=true)
#   lake_types = LatLonField{Int64}(lake_grid,0)
#   for i = 1:20
#     for j = 1:20
#       coords::LatLonCoords = LatLonCoords(i,j)
#       lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
#       if lake_number <= 0 continue end
#       lake::Lake = lake_model_prognostics.lakes[lake_number]
#       if isa(lake,FillingLake)
#         set!(lake_types,coords,1)
#       elseif isa(lake,OverflowingLake)
#         set!(lake_types,coords,2)
#       elseif isa(lake,SubsumedLake)
#         set!(lake_types,coords,3)
#       else
#         set!(lake_types,coords,4)
#       end
#     end
#   end
#   lake_volumes = Float64[]
  # lake_fractions =
  #   calculate_lake_fraction_on_surface_grid(lake_model_parameters,
  #                                           lake_model_prognostics)
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
  # diagnostic_lake_volumes::Field{Float64} =
  #   calculate_diagnostic_lake_volumes_field(lake_model_parameters,
  #                                           lake_model_prognostics)
#   @test sixth_intermediate_expected_river_inflow == river_fields.river_inflow
#   @test isapprox(sixth_intermediate_expected_water_to_ocean,
#                  river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
#   @test sixth_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
#   @test sixth_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
#   @test sixth_intermediate_expected_lake_types == lake_types
#   @test isapprox(sixth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.01)
#   @test isapprox(diagnostic_lake_volumes,sixth_intermediate_expected_diagnostic_lake_volumes,
#                  rtol=0.0,atol=0.05)
#   @test isapprox(sixth_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
#   @test sixth_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
#   @test sixth_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
#   @test isapprox(sixth_intermediate_expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.1)
  # river_fields,lake_model_prognostics,lake_model_diagnostics =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           85,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=84,
#                             use_realistic_surface_coupling=true)
#   lake_types = LatLonField{Int64}(lake_grid,0)
#   for i = 1:20
#     for j = 1:20
#       coords::LatLonCoords = LatLonCoords(i,j)
#       lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
#       if lake_number <= 0 continue end
#       lake::Lake = lake_model_prognostics.lakes[lake_number]
#       if isa(lake,FillingLake)
#         set!(lake_types,coords,1)
#       elseif isa(lake,OverflowingLake)
#         set!(lake_types,coords,2)
#       elseif isa(lake,SubsumedLake)
#         set!(lake_types,coords,3)
#       else
#         set!(lake_types,coords,4)
#       end
#     end
#   end
#   lake_volumes = Float64[]
  # lake_fractions =
  #   calculate_lake_fraction_on_surface_grid(lake_model_parameters,
  #                                           lake_model_prognostics)
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
  # diagnostic_lake_volumes::Field{Float64} =
  #   calculate_diagnostic_lake_volumes_field(lake_model_parameters,
  #                                           lake_model_prognostics)
#   @test seventh_intermediate_expected_river_inflow == river_fields.river_inflow
#   @test isapprox(seventh_intermediate_expected_water_to_ocean,
#                  river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
#   @test seventh_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
#   @test seventh_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
#   @test seventh_intermediate_expected_lake_types == lake_types
#   @test isapprox(seventh_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.01)
#   @test isapprox(diagnostic_lake_volumes,seventh_intermediate_expected_diagnostic_lake_volumes,
#                  rtol=0.0,atol=0.05)
#   @test isapprox(seventh_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
#   @test seventh_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
#   @test seventh_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
#   @test isapprox(seventh_intermediate_expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.1)
#   lake_volumes_all_timesteps::Vector{Vector{Float64}} = []
  # river_fields,lake_model_prognostics,_ =
  #   drive_hd_and_lake_model(river_parameters,river_fields,
  #                           lake_model_parameters,
  #                           lake_model_prognostics,
  #                           drainages,runoffs,evaporations,
  #                           115,true,
#                             initial_water_to_lake_centers,
#                             initial_spillover_to_rivers,
#                             print_timestep_results=false,
  #                           write_output=false,return_output=true,
  #                           lake_model_diagnostics=
  #                           lake_model_diagnostics,
  #                           forcing_timesteps_to_skip=85,
#                             use_realistic_surface_coupling=true)
#   for (lake_volumes_slice,expected_lake_volumes_slice) in zip(lake_volumes_all_timesteps,expected_lake_volumes_all_timesteps)
#     @test isapprox(lake_volumes_slice,expected_lake_volumes_slice,
#                    rtol=0.0,atol=0.01)
#   end
#   lake_types = LatLonField{Int64}(lake_grid,0)
#   for i = 1:20
#     for j = 1:20
#       coords::LatLonCoords = LatLonCoords(i,j)
#       lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
#       if lake_number <= 0 continue end
#       lake::Lake = lake_model_prognostics.lakes[lake_number]
#       if isa(lake,FillingLake)
#         set!(lake_types,coords,1)
#       elseif isa(lake,OverflowingLake)
#         set!(lake_types,coords,2)
#       elseif isa(lake,SubsumedLake)
#         set!(lake_types,coords,3)
#       else
#         set!(lake_types,coords,4)
#       end
#     end
#   end
#   lake_volumes = Float64[]
  # lake_fractions =
  #   calculate_lake_fraction_on_surface_grid(lake_model_parameters,
  #                                           lake_model_prognostics)
  # for lake::Lake in lake_model_prognostics.lakes
  #   append!(lake_volumes,get_lake_volume(lake))
  # end
  # diagnostic_lake_volumes::Field{Float64} =
  #   calculate_diagnostic_lake_volumes_field(lake_model_parameters,
  #                                           lake_model_prognostics)
#   @test expected_river_inflow == river_fields.river_inflow
#   @test isapprox(expected_water_to_ocean,
#                  river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
#   @test expected_water_to_hd == lake_model_prognostics.water_to_hd
#   @test expected_lake_numbers == lake_model_prognostics.lake_numbers
#   @test expected_lake_types == lake_types
#   @test isapprox(expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
#   @test isapprox(diagnostic_lake_volumes,expected_diagnostic_lake_volumes,rtol=0.0,atol=0.00001)
#   @test isapprox(expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.01)
#   @test expected_number_lake_cells == lake_model_prognostics.lake_cell_count
#   @test expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
#   @test expected_true_lake_depths == lake_model_parameters.true_lake_depths
# end
