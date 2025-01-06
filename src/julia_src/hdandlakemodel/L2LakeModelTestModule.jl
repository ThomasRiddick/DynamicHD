using Profile
using Test: @test, @testset
using L2HDDriverModule: drive_hd_and_lake_model
using HDModule: RiverParameters,RiverPrognosticFields
using GridModule: LatLonGrid
using FieldModule: Field,LatLonField,LatLonDirectionIndicators,set!,repeat
using CoordsModule: LatLonCoords
using L2LakeModule: Lake, OverflowingLake, FillingLake, SubsumedLake
using L2LakeModule: get_lake_volume, get_lake_filled_cells
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
    if lake.variables.active_lake
      println("lake number")
      println(lake.parameters.lake_number)
      println(typeof(lake))
      println(get_lake_filled_cells(lake))
    end
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
  @test isapprox(expected_water_to_hd,lake_model_prognostics.water_to_hd,rtol=0.0,
                 atol=0.0000000001)
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
  @test isapprox(expected_intermediate_water_to_hd,
                 lake_model_prognostics.water_to_hd,rtol=0.0,atol=0.00001)
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
  @test isapprox(expected_water_to_hd,lake_model_prognostics.water_to_hd,
                 rtol=0.0,atol=0.000000000001)
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
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 0 0
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
  @test isapprox(expected_water_to_hd,lake_model_prognostics.water_to_hd,rtol=0.0,
                 atol=0.0000000000001)
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

@testset "Lake model tests 17" begin
  #Simple tests of evaporation and initial water to lake center
  #No runoff or drainage, single large lake, using realistic
  #surface coupling
  hd_grid = LatLonGrid(4,4,true)
  surface_model_grid = LatLonGrid(3,3,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
                                                Int64[ 3  2  2  1
                                                       6  6 -2  4
                                                       6  8  8  2
                                                       9  8  8  0 ]))
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
  is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
    Bool[ false false false false false false false false false #=
       =# false false false false false false false false false #=
       =# false false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true false false false false #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true false false false false #=
       =# false false
          false true true true true true true true true #=
       =# true true true true true false false false false #=
       =# false false
          false true true true true true true true true #=
       =# true true true true true false false false false #=
       =# false false
          false true true true true true true true true #=
       =# true true true true true false false false false #=
       =# false false
          false false true true true true true true true #=
       =# true true true true true false false false false #=
       =# false false
          false false false false false false false false false #=
       =# false false false false false false false false false #=
       =# false false ])
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
  lake_parameters_as_array::Vector{Float64} = Float64[1.0, 1481.0, 1.0, -1.0, 0.0, 10.0, 11.0, 294.0, 10.0, 11.0, 1.0, 0.0, 1.0, 11.0, 12.0, 1.0, 0.0, 1.0, 9.0, 12.0, 1.0, 0.0, 1.0, 10.0, 10.0, 1.0, 0.0, 1.0, 12.0, 11.0, 1.0, 0.0, 1.0, 11.0, 10.0, 1.0, 0.0, 1.0, 8.0, 12.0, 1.0, 0.0, 1.0, 11.0, 9.0, 1.0, 0.0, 1.0, 9.0, 9.0, 1.0, 0.0, 1.0, 12.0, 10.0, 1.0, 0.0, 1.0, 12.0, 9.0, 1.0, 0.0, 1.0, 7.0, 11.0, 1.0, 0.0, 1.0, 10.0, 8.0, 1.0, 0.0, 1.0, 8.0, 8.0, 1.0, 0.0, 1.0, 13.0, 9.0, 1.0, 0.0, 1.0, 13.0, 8.0, 1.0, 0.0, 1.0, 6.0, 10.0, 1.0, 0.0, 1.0, 10.0, 7.0, 1.0, 0.0, 1.0, 9.0, 7.0, 1.0, 0.0, 1.0, 7.0, 7.0, 1.0, 0.0, 1.0, 14.0, 8.0, 1.0, 0.0, 1.0, 12.0, 7.0, 1.0, 0.0, 1.0, 5.0, 9.0, 1.0, 0.0, 1.0, 9.0, 6.0, 1.0, 0.0, 1.0, 8.0, 6.0, 1.0, 0.0, 1.0, 6.0, 6.0, 1.0, 0.0, 1.0, 15.0, 7.0, 1.0, 0.0, 1.0, 12.0, 6.0, 1.0, 0.0, 1.0, 4.0, 8.0, 1.0, 0.0, 1.0, 8.0, 5.0, 1.0, 0.0, 1.0, 7.0, 5.0, 1.0, 0.0, 1.0, 5.0, 5.0, 1.0, 0.0, 1.0, 14.0, 6.0, 1.0, 0.0, 1.0, 12.0, 5.0, 1.0, 0.0, 1.0, 11.0, 5.0, 1.0, 0.0, 1.0, 13.0, 5.0, 1.0, 0.0, 1.0, 7.0, 4.0, 1.0, 0.0, 1.0, 6.0, 4.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0, 0.0, 1.0, 14.0, 5.0, 1.0, 0.0, 1.0, 11.0, 4.0, 1.0, 0.0, 1.0, 10.0, 4.0, 1.0, 0.0, 1.0, 14.0, 4.0, 1.0, 0.0, 1.0, 6.0, 3.0, 1.0, 0.0, 1.0, 5.0, 3.0, 1.0, 0.0, 1.0, 15.0, 4.0, 1.0, 0.0, 1.0, 10.0, 3.0, 1.0, 0.0, 1.0, 9.0, 3.0, 1.0, 0.0, 1.0, 13.0, 3.0, 1.0, 0.0, 1.0, 11.0, 7.0, 1.0, 0.0, 1.0, 15.0, 6.0, 1.0, 0.0, 1.0, 16.0, 6.0, 1.0, 0.0, 1.0, 16.0, 3.0, 1.0, 0.0, 1.0, 8.0, 11.0, 1.0, 0.0, 1.0, 7.0, 10.0, 1.0, 0.0, 1.0, 16.0, 7.0, 1.0, 0.0, 1.0, 16.0, 8.0, 1.0, 0.0, 1.0, 17.0, 5.0, 1.0, 0.0, 1.0, 6.0, 11.0, 1.0, 0.0, 1.0, 6.0, 5.0, 1.0, 0.0, 1.0, 5.0, 6.0, 1.0, 0.0, 1.0, 17.0, 8.0, 1.0, 0.0, 1.0, 16.0, 9.0, 1.0, 0.0, 1.0, 5.0, 12.0, 1.0, 0.0, 1.0, 11.0, 13.0, 1.0, 0.0, 1.0, 8.0, 13.0, 1.0, 0.0, 1.0, 6.0, 12.0, 1.0, 0.0, 1.0, 5.0, 7.0, 1.0, 0.0, 1.0, 9.0, 5.0, 1.0, 0.0, 1.0, 15.0, 10.0, 1.0, 0.0, 1.0, 4.0, 11.0, 1.0, 0.0, 1.0, 10.0, 14.0, 1.0, 0.0, 1.0, 7.0, 14.0, 1.0, 0.0, 1.0, 8.0, 14.0, 1.0, 0.0, 1.0, 9.0, 14.0, 1.0, 0.0, 1.0, 6.0, 15.0, 1.0, 0.0, 1.0, 6.0, 14.0, 1.0, 0.0, 1.0, 11.0, 14.0, 1.0, 0.0, 1.0, 6.0, 16.0, 1.0, 0.0, 1.0, 5.0, 16.0, 1.0, 0.0, 1.0, 5.0, 15.0, 1.0, 0.0, 1.0, 5.0, 14.0, 1.0, 0.0, 1.0, 12.0, 15.0, 1.0, 0.0, 1.0, 5.0, 17.0, 1.0, 0.0, 1.0, 4.0, 17.0, 1.0, 0.0, 1.0, 4.0, 15.0, 1.0, 0.0, 1.0, 13.0, 15.0, 1.0, 0.0, 1.0, 13.0, 14.0, 1.0, 0.0, 1.0, 6.0, 18.0, 1.0, 0.0, 1.0, 5.0, 18.0, 1.0, 0.0, 1.0, 4.0, 18.0, 1.0, 0.0, 1.0, 4.0, 14.0, 1.0, 0.0, 1.0, 4.0, 16.0, 1.0, 0.0, 1.0, 12.0, 16.0, 1.0, 0.0, 1.0, 11.0, 16.0, 1.0, 0.0, 1.0, 13.0, 13.0, 1.0, 0.0, 1.0, 7.0, 18.0, 1.0, 0.0, 1.0, 13.0, 16.0, 1.0, 0.0, 1.0, 7.0, 17.0, 1.0, 0.0, 1.0, 6.0, 17.0, 1.0, 0.0, 1.0, 11.0, 17.0, 1.0, 0.0, 1.0, 10.0, 16.0, 1.0, 0.0, 1.0, 14.0, 12.0, 1.0, 0.0, 1.0, 8.0, 17.0, 1.0, 0.0, 1.0, 8.0, 16.0, 1.0, 0.0, 1.0, 8.0, 18.0, 1.0, 0.0, 1.0, 10.0, 18.0, 1.0, 0.0, 1.0, 9.0, 16.0, 1.0, 0.0, 1.0, 15.0, 12.0, 1.0, 0.0, 1.0, 9.0, 18.0, 1.0, 0.0, 1.0, 15.0, 13.0, 1.0, 0.0, 1.0, 14.0, 11.0, 1.0, 0.0, 1.0, 9.0, 15.0, 1.0, 0.0, 1.0, 8.0, 15.0, 1.0, 0.0, 1.0, 7.0, 15.0, 1.0, 0.0, 1.0, 7.0, 16.0, 1.0, 0.0, 1.0, 16.0, 12.0, 1.0, 0.0, 1.0, 16.0, 13.0, 1.0, 0.0, 1.0, 15.0, 14.0, 1.0, 0.0, 1.0, 16.0, 14.0, 1.0, 0.0, 1.0, 9.0, 17.0, 1.0, 0.0, 1.0, 11.0, 18.0, 1.0, 0.0, 1.0, 12.0, 18.0, 1.0, 0.0, 1.0, 17.0, 11.0, 1.0, 0.0, 1.0, 17.0, 14.0, 1.0, 0.0, 1.0, 11.0, 15.0, 1.0, 0.0, 1.0, 10.0, 15.0, 1.0, 0.0, 1.0, 9.0, 8.0, 1.0, 0.0, 1.0, 10.0, 5.0, 1.0, 0.0, 1.0, 13.0, 18.0, 1.0, 0.0, 1.0, 5.0, 8.0, 1.0, 0.0, 1.0, 15.0, 11.0, 1.0, 0.0, 1.0, 16.0, 11.0, 1.0, 0.0, 1.0, 9.0, 13.0, 1.0, 0.0, 1.0, 8.0, 9.0, 1.0, 0.0, 1.0, 4.0, 9.0, 1.0, 0.0, 1.0, 12.0, 14.0, 1.0, 0.0, 1.0, 4.0, 12.0, 1.0, 0.0, 1.0, 17.0, 12.0, 1.0, 0.0, 1.0, 17.0, 13.0, 1.0, 0.0, 1.0, 10.0, 17.0, 1.0, 0.0, 1.0, 12.0, 17.0, 1.0, 0.0, 1.0, 4.0, 10.0, 1.0, 0.0, 1.0, 4.0, 13.0, 1.0, 0.0, 1.0, 5.0, 13.0, 1.0, 0.0, 1.0, 14.0, 13.0, 1.0, 0.0, 1.0, 8.0, 10.0, 1.0, 0.0, 1.0, 13.0, 6.0, 1.0, 0.0, 1.0, 6.0, 13.0, 1.0, 0.0, 1.0, 16.0, 10.0, 1.0, 0.0, 1.0, 15.0, 8.0, 1.0, 0.0, 1.0, 17.0, 10.0, 1.0, 0.0, 1.0, 9.0, 10.0, 1.0, 0.0, 1.0, 10.0, 13.0, 1.0, 0.0, 1.0, 11.0, 11.0, 1.0, 0.0, 1.0, 11.0, 8.0, 1.0, 0.0, 1.0, 15.0, 9.0, 1.0, 0.0, 1.0, 17.0, 9.0, 1.0, 0.0, 1.0, 7.0, 6.0, 1.0, 0.0, 1.0, 17.0, 3.0, 1.0, 0.0, 1.0, 17.0, 4.0, 1.0, 0.0, 1.0, 12.0, 8.0, 1.0, 0.0, 1.0, 6.0, 7.0, 1.0, 0.0, 1.0, 17.0, 6.0, 1.0, 0.0, 1.0, 17.0, 7.0, 1.0, 0.0, 1.0, 6.0, 8.0, 1.0, 0.0, 1.0, 10.0, 12.0, 1.0, 0.0, 1.0, 7.0, 12.0, 1.0, 0.0, 1.0, 10.0, 6.0, 1.0, 0.0, 1.0, 16.0, 4.0, 1.0, 0.0, 1.0, 16.0, 5.0, 1.0, 0.0, 1.0, 11.0, 6.0, 1.0, 0.0, 1.0, 7.0, 13.0, 1.0, 0.0, 1.0, 6.0, 9.0, 1.0, 0.0, 1.0, 14.0, 3.0, 1.0, 0.0, 1.0, 15.0, 3.0, 1.0, 0.0, 1.0, 5.0, 10.0, 1.0, 0.0, 1.0, 11.0, 3.0, 1.0, 0.0, 1.0, 12.0, 3.0, 1.0, 0.0, 1.0, 12.0, 12.0, 1.0, 0.0, 1.0, 9.0, 11.0, 1.0, 0.0, 1.0, 13.0, 10.0, 1.0, 0.0, 1.0, 5.0, 11.0, 1.0, 0.0, 1.0, 4.0, 3.0, 1.0, 0.0, 1.0, 13.0, 7.0, 1.0, 0.0, 1.0, 7.0, 3.0, 1.0, 0.0, 1.0, 13.0, 11.0, 1.0, 0.0, 1.0, 14.0, 7.0, 1.0, 0.0, 1.0, 8.0, 3.0, 1.0, 0.0, 1.0, 12.0, 4.0, 1.0, 0.0, 1.0, 14.0, 9.0, 1.0, 0.0, 1.0, 13.0, 4.0, 1.0, 0.0, 1.0, 15.0, 5.0, 1.0, 0.0, 1.0, 12.0, 13.0, 1.0, 0.0, 1.0, 13.0, 12.0, 1.0, 0.0, 1.0, 14.0, 10.0, 1.0, 0.0, 1.0, 5.0, 4.0, 1.0, 0.0, 1.0, 4.0, 5.0, 1.0, 0.0, 1.0, 8.0, 7.0, 1.0, 0.0, 1.0, 4.0, 6.0, 1.0, 0.0, 1.0, 8.0, 4.0, 1.0, 0.0, 1.0, 10.0, 9.0, 1.0, 0.0, 1.0, 7.0, 8.0, 1.0, 0.0, 1.0, 9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 7.0, 1.0, 0.0, 1.0, 7.0, 9.0, 1.0, 0.0, 1.0, 13.0, 17.0, 1.0, 0.0, 1.0, 14.0, 14.0, 1.0, 208.0, 2.0, 18.0, 6.0, 1.0, 208.0, 2.0, 17.0, 2.0, 1.0, 208.0, 2.0, 3.0, 9.0, 1.0, 208.0, 2.0, 19.0, 7.0, 1.0, 208.0, 2.0, 3.0, 18.0, 1.0, 208.0, 2.0, 3.0, 16.0, 1.0, 208.0, 2.0, 9.0, 19.0, 1.0, 208.0, 2.0, 3.0, 4.0, 1.0, 208.0, 2.0, 3.0, 7.0, 1.0, 208.0, 2.0, 2.0, 8.0, 1.0, 208.0, 2.0, 2.0, 17.0, 1.0, 208.0, 2.0, 2.0, 15.0, 1.0, 208.0, 2.0, 3.0, 19.0, 1.0, 208.0, 2.0, 3.0, 12.0, 1.0, 208.0, 2.0, 2.0, 3.0, 1.0, 208.0, 2.0, 3.0, 6.0, 1.0, 208.0, 2.0, 4.0, 2.0, 1.0, 208.0, 2.0, 4.0, 19.0, 1.0, 208.0, 2.0, 18.0, 8.0, 1.0, 208.0, 2.0, 3.0, 2.0, 1.0, 208.0, 2.0, 2.0, 11.0, 1.0, 208.0, 2.0, 2.0, 2.0, 1.0, 208.0, 2.0, 7.0, 19.0, 1.0, 208.0, 2.0, 11.0, 2.0, 1.0, 208.0, 2.0, 19.0, 9.0, 1.0, 208.0, 2.0, 19.0, 5.0, 1.0, 208.0, 2.0, 6.0, 19.0, 1.0, 208.0, 2.0, 3.0, 14.0, 1.0, 208.0, 2.0, 18.0, 14.0, 1.0, 208.0, 2.0, 13.0, 2.0, 1.0, 208.0, 2.0, 18.0, 12.0, 1.0, 208.0, 2.0, 14.0, 2.0, 1.0, 208.0, 2.0, 19.0, 4.0, 1.0, 208.0, 2.0, 19.0, 13.0, 1.0, 208.0, 2.0, 19.0, 11.0, 1.0, 208.0, 2.0, 18.0, 13.0, 1.0, 208.0, 2.0, 19.0, 3.0, 1.0, 208.0, 2.0, 10.0, 19.0, 1.0, 208.0, 2.0, 16.0, 2.0, 1.0, 208.0, 2.0, 19.0, 12.0, 1.0, 208.0, 2.0, 18.0, 9.0, 1.0, 208.0, 2.0, 19.0, 14.0, 1.0, 208.0, 2.0, 19.0, 10.0, 1.0, 208.0, 2.0, 3.0, 8.0, 1.0, 208.0, 2.0, 3.0, 15.0, 1.0, 208.0, 2.0, 11.0, 19.0, 1.0, 208.0, 2.0, 3.0, 13.0, 1.0, 208.0, 2.0, 3.0, 3.0, 1.0, 208.0, 2.0, 18.0, 10.0, 1.0, 208.0, 2.0, 14.0, 19.0, 1.0, 208.0, 2.0, 3.0, 11.0, 1.0, 208.0, 2.0, 3.0, 10.0, 1.0, 208.0, 2.0, 19.0, 6.0, 1.0, 208.0, 2.0, 12.0, 19.0, 1.0, 208.0, 2.0, 2.0, 12.0, 1.0, 208.0, 2.0, 2.0, 13.0, 1.0, 208.0, 2.0, 15.0, 2.0, 1.0, 208.0, 2.0, 13.0, 19.0, 1.0, 208.0, 2.0, 18.0, 7.0, 1.0, 208.0, 2.0, 8.0, 19.0, 1.0, 208.0, 2.0, 2.0, 14.0, 1.0, 208.0, 2.0, 7.0, 2.0, 1.0, 208.0, 2.0, 18.0, 2.0, 1.0, 208.0, 2.0, 2.0, 6.0, 1.0, 208.0, 2.0, 8.0, 2.0, 1.0, 208.0, 2.0, 18.0, 5.0, 1.0, 208.0, 2.0, 5.0, 2.0, 1.0, 208.0, 2.0, 3.0, 5.0, 1.0, 208.0, 2.0, 2.0, 7.0, 1.0, 208.0, 2.0, 2.0, 4.0, 1.0, 208.0, 2.0, 18.0, 4.0, 1.0, 208.0, 2.0, 2.0, 5.0, 1.0, 208.0, 2.0, 9.0, 2.0, 1.0, 208.0, 2.0, 3.0, 17.0, 1.0, 208.0, 2.0, 2.0, 16.0, 1.0, 208.0, 2.0, 2.0, 19.0, 1.0, 208.0, 2.0, 2.0, 18.0, 1.0, 208.0, 2.0, 5.0, 19.0, 1.0, 208.0, 2.0, 18.0, 11.0, 1.0, 208.0, 2.0, 6.0, 2.0, 1.0, 208.0, 2.0, 19.0, 8.0, 1.0, 208.0, 2.0, 2.0, 9.0, 1.0, 208.0, 2.0, 18.0, 3.0, 1.0, 208.0, 2.0, 10.0, 2.0, 1.0, 208.0, 2.0, 2.0, 10.0, 1.0, 208.0, 2.0, 12.0, 2.0, 1.0, 502.0, 3.0, 1.0, -1.0, 4.0, 4.0, 0.0]
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
  drainages::Array{Field{Float64},1} = repeat(drainage,10000,false)
  runoffs::Array{Field{Float64},1} = drainages
  evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,1.0/2.906374502)
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
              0 0 0 0 0  0 0 0 0 0  502.0 0 0 0 0  0 0 0 0 0
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
       Int64[  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  1 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  ])
  expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  1 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
              0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  ])
  expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[ 0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
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
                 0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  ])
  expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.0 0.0 0.0
                 0.0 0.015625  0.0
                 0.0 0.0 0.0])
  expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[  0 0  0
                0 1  0
                0 0  0 ])
  expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 36 48 36
               48 64 48
               36 48 36  ])
  expected_lake_volumes::Array{Float64} = Float64[0.0]
  # expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[ 0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0 ])


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
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  first_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
  first_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0   ])
  first_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
      Float64[  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
             =# 0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0 #=
             =# 502.0   502.0   502.0   502.0   502.0   502.0   0.0
                0.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0 #=
             =# 502.0   502.0   502.0   502.0   502.0   502.0   0.0
                0.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0 #=
             =# 502.0   502.0   502.0   502.0   502.0   502.0   0.0
                0.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0 #=
            =#  502.0   502.0   502.0   502.0   502.0   502.0   0.0
                0.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0 #=
            =#  502.0   502.0   502.0   502.0   502.0   502.0   0.0
                0.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0 #=
            =#  502.0   502.0   502.0   502.0   502.0   502.0   0.0
                0.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0 #=
            =#  502.0   502.0   502.0   502.0   502.0   502.0   0.0
                0.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0 #=
            =#  502.0   502.0   502.0   502.0   502.0   502.0   0.0
                0.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0 #=
            =#  502.0   502.0   502.0   502.0   502.0   502.0   0.0
                0.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0 #=
            =#  502.0   502.0   502.0   502.0   502.0   502.0   0.0
                0.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0 #=
            =#  502.0   502.0   502.0   502.0   502.0   502.0   0.0
                0.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0 #=
            =#  502.0   502.0   502.0   502.0   502.0   502.0   0.0
                0.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0 #=
            =#  502.0   0.0   0.0   0.0   0.0   502.0   0.0
                0.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0 #=
            =#  502.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0 #=
            =#  502.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0 #=
            =#  502.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0 #=
            =#  502.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0     0.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0   502.0 #=
            =#  502.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0  ])
  first_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.6944444444 0.8333333333 0.6944444444
                 0.8333333333 1.0 0.75
                 24.0/36.0 0.8333333333 0.0])
  first_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 25 40 25
               40 64 36
               24 40 0])
  first_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 36 48 36
               48 64 48
               36 48 36  ])
  first_intermediate_expected_lake_volumes::Array{Float64} = Float64[502.0]
  # first_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  #                0 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 0
  #                0 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 0
  #                0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
  #                0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
  #                0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
  #                0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
  #                0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
  #                0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
  #                0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
  #                0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
  #                0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
  #                0 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 0
  #                0 4 5 5 5 5 5 5 5 5 5 5 5 5 0 0 0 0 4 0
  #                0 4 5 5 5 5 5 5 5 5 5 5 5 5 0 0 0 0 0 0
  #                0 4 5 5 5 5 5 5 5 5 5 5 5 5 0 0 0 0 0 0
  #                0 4 5 5 5 5 5 5 5 5 5 5 5 5 0 0 0 0 0 0
  #                0 4 4 4 4 4 4 4 4 4 4 4 4 4 0 0 0 0 0 0
  #                0 3 4 4 4 4 4 4 4 4 4 4 4 4 0 0 0 0 0 0
  #                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])

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
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0   ])
  second_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
  second_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
      Float64[   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
             =# 0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
             =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
             =# 495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   0.0
                0.0   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
             =# 495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
             =# 495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   0.0
                0.0   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
             =# 495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
             =# 495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   0.0
                0.0   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   0.0
                0.0   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
             =# 495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   0.0
                0.0   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =# 495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   0.0
                0.0   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   0.0
                0.0   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   0.0
                0.0   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   0.0
                0.0   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   0.0
                0.0   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   0.0
                0.0   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   0.0
                0.0   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   0.0   0.0   0.0   0.0   495.5008758   0.0
                0.0   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   495.5008758   495.5008758   495.5008758   495.5008758 #=
            =#  495.5008758   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0 ])
  second_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.6944444444 0.8333333333 0.6944444444
                 0.8333333333 1.0 0.75
                 24.0/36.0 0.8333333333 0.0 ])
  second_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 25 40 25
               40 64 36
               24 40 0])
  second_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 36 48 36
               48 64 48
               36 48 36  ])
  second_intermediate_expected_lake_volumes::Array{Float64} = Float64[495.5008758]
  # second_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
  #                  0.00   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   0.00
  #                  0.00   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   0.00
  #                  0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
  #                  0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
  #                  0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
  #                  0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
  #                  0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
  #                  0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
  #                  0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
  #                  0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
  #                  0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
  #                  0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   3.94   0.00
  #                  0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   0.00   0.00   0.00   0.00   3.94   0.00
  #                  0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   0.00   0.00   0.00   0.00   0.00   0.00
  #                  0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   0.00   0.00   0.00   0.00   0.00   0.00
  #                  0.00   3.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   4.94   0.00   0.00   0.00   0.00   0.00   0.00
  #                  0.00   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   0.00   0.00   0.00   0.00   0.00   0.00
  #                  0.00   2.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   3.94   0.00   0.00   0.00   0.00   0.00   0.00
  #                  0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00  ])

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
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0   ])
  third_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
  third_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
      Float64[   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
             =# 0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =#  209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   0.0
                0.0   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   0.0
                0.0   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   0.0
                0.0   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
            =#  209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   0.0
                0.0   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
            =#  209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   0.0
                0.0   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
            =#  209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   0.0
                0.0   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
            =#  209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   0.0
                0.0   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
            =#  209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   0.0
                0.0   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
            =#  209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   0.0
                0.0   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
            =#  209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   0.0
                0.0   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
            =#  209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   0.0
                0.0   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
            =#  209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   0.0
                0.0   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
            =#  209.53941055597593   0.0   0.0   0.0   0.0   209.53941055597593   0.0
                0.0   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
            =#  209.53941055597593   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
            =#  209.53941055597593   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   209.53941055597593 209.539410555975939166667   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
            =#  209.53941055597593   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
            =#  209.53941055597593   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 209.539410555975939166667 #=
             =# 209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593   209.53941055597593 #=
            =#  209.53941055597593   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 #=
            =#  0.0   0.0   0.0   0.0   0.0   0.0   0.0 ])
  third_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.6944444444 0.8333333333 0.6944444444
                 0.8333333333 1.0 0.75
                 24.0/36.0 0.8333333333 0.0 ])
  third_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 25 40 25
               40 64 36
               24 40 0])
  third_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 36 48 36
               48 64 48
               36 48 36  ])
  third_intermediate_expected_lake_volumes::Array{Float64} = Float64[209.53941055597593]
  # third_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
  # 0.00   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   0.00
  # 0.00   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   0.00
  # 0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
  # 0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
  # 0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
  # 0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
  # 0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
  # 0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
  # 0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
  # 0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
  # 0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
  # 0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   1.04   0.00
  # 0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   0.00   0.00   0.00   0.00   1.04   0.00
  # 0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   0.00   0.00   0.00   0.00   0.00   0.00
  # 0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   0.00   0.00   0.00   0.00   0.00   0.00
  # 0.00   1.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   2.04   0.00   0.00   0.00   0.00   0.00   0.00
  # 0.00   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   0.00   0.00   0.00   0.00   0.00   0.00
  # 0.00   0.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   1.04   0.00   0.00   0.00   0.00   0.00   0.00
  # 0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00 ])

  fourth_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  fourth_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  fourth_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  fourth_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
       Int64[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  ])
  fourth_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
             0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0   ])
  fourth_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
      Float64[  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 0 203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 0 0
                0 0 203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 0 0
                0 0 203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 0 0
                0 0 203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 0 0
                0 0 203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 0 0
                0 0 203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 0 0
                0 0 203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 0 0
                0 0 203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 0 0
                0 0 203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 0 0
                0 0 203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             c=# 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 0 0
                0 0 203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =# 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 0 0 0 0 0 0
                0 0 203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 0 0 0 0 0 0
                0 0 203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 0 0 0 0 0 0
                0 0 203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 203.04028634610876 203.04028634610876 #=
            =#  203.04028634610876 203.04028634610876 203.04028634610876 203.04028634610876 #=
             =# 203.04028634610876 0 0 0 0 0 0
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  fourth_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.3333333333 0.5 0.3333333333
                 0.6666666667 1.0 0.5833333333
                 0.3333333333 0.5 0.0 ])
  fourth_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 12 24 12
               32 64 28
               12 24 0])
  fourth_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 36 48 36
               48 64 48
               36 48 36  ])
  fourth_intermediate_expected_lake_volumes::Array{Float64} = Float64[203.04028634610876]
  # fourth_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
  #                  0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
  #                  0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
  #                  0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
  #                  0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
  #                  0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
  #                  0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
  #                  0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
  #                  0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
  #                  0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
  #                  0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
  #                  0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
  #                  0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00
  #                  0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00   0.00   0.00   0.00   0.00
  #                  0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00   0.00   0.00   0.00   0.00
  #                  0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00   0.00   0.00   0.00   0.00
  #                  0.00   0.00   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.97   0.00   0.00   0.00   0.00   0.00   0.00
  #                  0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
  #                  0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00
  #                  0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00  ])

  fifth_intermediate_expected_lake_volumes::Array{Float64} = Float64[130.0971746259524]
  sixth_intermediate_expected_lake_volumes::Array{Float64} = Float64[125.53823014344262]

  seventh_intermediate_expected_lake_volumes::Array{Float64} = Float64[93.62561876587421]
  eighth_intermediate_expected_lake_volumes::Array{Float64} = Float64[89.06667428336443]

  ninth_intermediate_expected_lake_volumes::Array{Float64} = Float64[48.03617394077642]
  tenth_intermediate_expected_lake_volumes::Array{Float64} = Float64[43.47722945826665]

  eleventh_intermediate_expected_lake_volumes::Array{Float64} = Float64[25.241451528227554]
  twelfth_intermediate_expected_lake_volumes::Array{Float64} = Float64[20.682507045717777]

  thirteenth_intermediate_expected_lake_volumes::Array{Float64} = Float64[2.446729115678684]
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
                            write_output=false,return_output=true,
                            use_realistic_surface_coupling=true)
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
                            forcing_timesteps_to_skip=-1,
                            use_realistic_surface_coupling=true)
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
  @test isapprox(second_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
  @test isapprox(diagnostic_lake_volumes,second_intermediate_expected_diagnostic_lake_volumes,
                 rtol=0.0,atol=0.00001)
  @test isapprox(second_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
  @test second_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test second_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test isapprox(second_intermediate_expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.1)
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            45,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=1,
                            use_realistic_surface_coupling=true)
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
  @test isapprox(third_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
  @test isapprox(diagnostic_lake_volumes,third_intermediate_expected_diagnostic_lake_volumes,
                 rtol=0.0,atol=0.00001)
  @test isapprox(third_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
  @test third_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test third_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test isapprox(third_intermediate_expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.1)
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            46,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=45,
                            use_realistic_surface_coupling=true)
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
  @test fourth_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(fourth_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test fourth_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test fourth_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test fourth_intermediate_expected_lake_types == lake_types
  @test isapprox(fourth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
  @test isapprox(diagnostic_lake_volumes,fourth_intermediate_expected_diagnostic_lake_volumes,
                 rtol=0.0,atol=0.00001)
  @test isapprox(fourth_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
  @test fourth_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test fourth_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test isapprox(fourth_intermediate_expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.1)
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            62,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=46,
                            use_realistic_surface_coupling=true)
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  @test isapprox(fifth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            63,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=62,
                            use_realistic_surface_coupling=true)
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  @test isapprox(sixth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            70,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=63,
                            use_realistic_surface_coupling=true)
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  @test isapprox(seventh_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            71,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=70,
                            use_realistic_surface_coupling=true)
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  @test isapprox(eighth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.001)
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            80,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=71,
                            use_realistic_surface_coupling=true)
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  @test isapprox(ninth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.0001)
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            81,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=80,
                            use_realistic_surface_coupling=true)
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  @test isapprox(tenth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            85,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=81,
                            use_realistic_surface_coupling=true)
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  @test isapprox(eleventh_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            86,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=85,
                            use_realistic_surface_coupling=true)
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  @test isapprox(twelfth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            90,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=86,
                            use_realistic_surface_coupling=true)
  lake_volumes = Float64[]
  for lake::Lake in lake_model_prognostics.lakes
    append!(lake_volumes,get_lake_volume(lake))
  end
  @test isapprox(thirteenth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
  river_fields,lake_model_prognostics,_ =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            91,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=90,
                            use_realistic_surface_coupling=true)
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

@testset "Lake model tests 18" begin
  #Simple tests of evaporation with runoff where lake reaches
  #equilibrium without draining completely
  hd_grid = LatLonGrid(4,4,true)
  surface_model_grid = LatLonGrid(3,3,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
                                                Int64[ 3  2  2  1
                                                       6  6 -2  4
                                                       6  8  8  2
                                                       9  8  8  0 ]))
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
  is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
    Bool[ false false false false false false false false false #=
       =# false false false false false false false false false #=
       =# false false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true true true true true #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true false false false false #=
       =# true false
          false true true true true true true true true #=
       =# true true true true true false false false false #=
       =# false false
          false true true true true true true true true #=
       =# true true true true true false false false false #=
       =# false false
          false true true true true true true true true #=
       =# true true true true true false false false false #=
       =# false false
          false true true true true true true true true #=
       =# true true true true true false false false false #=
       =# false false
          false false true true true true true true true #=
       =# true true true true true false false false false #=
       =# false false
          false false false false false false false false false #=
       =# false false false false false false false false false #=
       =# false false ])
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
  lake_parameters_as_array::Vector{Float64} = Float64[1.0, 1481.0, 1.0, -1.0, 0.0, 10.0, 11.0, 294.0, 10.0, 11.0, 1.0, 0.0, 1.0, 11.0, 12.0, 1.0, 0.0, 1.0, 9.0, 12.0, 1.0, 0.0, 1.0, 10.0, 10.0, 1.0, 0.0, 1.0, 12.0, 11.0, 1.0, 0.0, 1.0, 11.0, 10.0, 1.0, 0.0, 1.0, 8.0, 12.0, 1.0, 0.0, 1.0, 11.0, 9.0, 1.0, 0.0, 1.0, 9.0, 9.0, 1.0, 0.0, 1.0, 12.0, 10.0, 1.0, 0.0, 1.0, 12.0, 9.0, 1.0, 0.0, 1.0, 7.0, 11.0, 1.0, 0.0, 1.0, 10.0, 8.0, 1.0, 0.0, 1.0, 8.0, 8.0, 1.0, 0.0, 1.0, 13.0, 9.0, 1.0, 0.0, 1.0, 13.0, 8.0, 1.0, 0.0, 1.0, 6.0, 10.0, 1.0, 0.0, 1.0, 10.0, 7.0, 1.0, 0.0, 1.0, 9.0, 7.0, 1.0, 0.0, 1.0, 7.0, 7.0, 1.0, 0.0, 1.0, 14.0, 8.0, 1.0, 0.0, 1.0, 12.0, 7.0, 1.0, 0.0, 1.0, 5.0, 9.0, 1.0, 0.0, 1.0, 9.0, 6.0, 1.0, 0.0, 1.0, 8.0, 6.0, 1.0, 0.0, 1.0, 6.0, 6.0, 1.0, 0.0, 1.0, 15.0, 7.0, 1.0, 0.0, 1.0, 12.0, 6.0, 1.0, 0.0, 1.0, 4.0, 8.0, 1.0, 0.0, 1.0, 8.0, 5.0, 1.0, 0.0, 1.0, 7.0, 5.0, 1.0, 0.0, 1.0, 5.0, 5.0, 1.0, 0.0, 1.0, 14.0, 6.0, 1.0, 0.0, 1.0, 12.0, 5.0, 1.0, 0.0, 1.0, 11.0, 5.0, 1.0, 0.0, 1.0, 13.0, 5.0, 1.0, 0.0, 1.0, 7.0, 4.0, 1.0, 0.0, 1.0, 6.0, 4.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0, 0.0, 1.0, 14.0, 5.0, 1.0, 0.0, 1.0, 11.0, 4.0, 1.0, 0.0, 1.0, 10.0, 4.0, 1.0, 0.0, 1.0, 14.0, 4.0, 1.0, 0.0, 1.0, 6.0, 3.0, 1.0, 0.0, 1.0, 5.0, 3.0, 1.0, 0.0, 1.0, 15.0, 4.0, 1.0, 0.0, 1.0, 10.0, 3.0, 1.0, 0.0, 1.0, 9.0, 3.0, 1.0, 0.0, 1.0, 13.0, 3.0, 1.0, 0.0, 1.0, 11.0, 7.0, 1.0, 0.0, 1.0, 15.0, 6.0, 1.0, 0.0, 1.0, 16.0, 6.0, 1.0, 0.0, 1.0, 16.0, 3.0, 1.0, 0.0, 1.0, 8.0, 11.0, 1.0, 0.0, 1.0, 7.0, 10.0, 1.0, 0.0, 1.0, 16.0, 7.0, 1.0, 0.0, 1.0, 16.0, 8.0, 1.0, 0.0, 1.0, 17.0, 5.0, 1.0, 0.0, 1.0, 6.0, 11.0, 1.0, 0.0, 1.0, 6.0, 5.0, 1.0, 0.0, 1.0, 5.0, 6.0, 1.0, 0.0, 1.0, 17.0, 8.0, 1.0, 0.0, 1.0, 16.0, 9.0, 1.0, 0.0, 1.0, 5.0, 12.0, 1.0, 0.0, 1.0, 11.0, 13.0, 1.0, 0.0, 1.0, 8.0, 13.0, 1.0, 0.0, 1.0, 6.0, 12.0, 1.0, 0.0, 1.0, 5.0, 7.0, 1.0, 0.0, 1.0, 9.0, 5.0, 1.0, 0.0, 1.0, 15.0, 10.0, 1.0, 0.0, 1.0, 4.0, 11.0, 1.0, 0.0, 1.0, 10.0, 14.0, 1.0, 0.0, 1.0, 7.0, 14.0, 1.0, 0.0, 1.0, 8.0, 14.0, 1.0, 0.0, 1.0, 9.0, 14.0, 1.0, 0.0, 1.0, 6.0, 15.0, 1.0, 0.0, 1.0, 6.0, 14.0, 1.0, 0.0, 1.0, 11.0, 14.0, 1.0, 0.0, 1.0, 6.0, 16.0, 1.0, 0.0, 1.0, 5.0, 16.0, 1.0, 0.0, 1.0, 5.0, 15.0, 1.0, 0.0, 1.0, 5.0, 14.0, 1.0, 0.0, 1.0, 12.0, 15.0, 1.0, 0.0, 1.0, 5.0, 17.0, 1.0, 0.0, 1.0, 4.0, 17.0, 1.0, 0.0, 1.0, 4.0, 15.0, 1.0, 0.0, 1.0, 13.0, 15.0, 1.0, 0.0, 1.0, 13.0, 14.0, 1.0, 0.0, 1.0, 6.0, 18.0, 1.0, 0.0, 1.0, 5.0, 18.0, 1.0, 0.0, 1.0, 4.0, 18.0, 1.0, 0.0, 1.0, 4.0, 14.0, 1.0, 0.0, 1.0, 4.0, 16.0, 1.0, 0.0, 1.0, 12.0, 16.0, 1.0, 0.0, 1.0, 11.0, 16.0, 1.0, 0.0, 1.0, 13.0, 13.0, 1.0, 0.0, 1.0, 7.0, 18.0, 1.0, 0.0, 1.0, 13.0, 16.0, 1.0, 0.0, 1.0, 7.0, 17.0, 1.0, 0.0, 1.0, 6.0, 17.0, 1.0, 0.0, 1.0, 11.0, 17.0, 1.0, 0.0, 1.0, 10.0, 16.0, 1.0, 0.0, 1.0, 14.0, 12.0, 1.0, 0.0, 1.0, 8.0, 17.0, 1.0, 0.0, 1.0, 8.0, 16.0, 1.0, 0.0, 1.0, 8.0, 18.0, 1.0, 0.0, 1.0, 10.0, 18.0, 1.0, 0.0, 1.0, 9.0, 16.0, 1.0, 0.0, 1.0, 15.0, 12.0, 1.0, 0.0, 1.0, 9.0, 18.0, 1.0, 0.0, 1.0, 15.0, 13.0, 1.0, 0.0, 1.0, 14.0, 11.0, 1.0, 0.0, 1.0, 9.0, 15.0, 1.0, 0.0, 1.0, 8.0, 15.0, 1.0, 0.0, 1.0, 7.0, 15.0, 1.0, 0.0, 1.0, 7.0, 16.0, 1.0, 0.0, 1.0, 16.0, 12.0, 1.0, 0.0, 1.0, 16.0, 13.0, 1.0, 0.0, 1.0, 15.0, 14.0, 1.0, 0.0, 1.0, 16.0, 14.0, 1.0, 0.0, 1.0, 9.0, 17.0, 1.0, 0.0, 1.0, 11.0, 18.0, 1.0, 0.0, 1.0, 12.0, 18.0, 1.0, 0.0, 1.0, 17.0, 11.0, 1.0, 0.0, 1.0, 17.0, 14.0, 1.0, 0.0, 1.0, 11.0, 15.0, 1.0, 0.0, 1.0, 10.0, 15.0, 1.0, 0.0, 1.0, 9.0, 8.0, 1.0, 0.0, 1.0, 10.0, 5.0, 1.0, 0.0, 1.0, 13.0, 18.0, 1.0, 0.0, 1.0, 5.0, 8.0, 1.0, 0.0, 1.0, 15.0, 11.0, 1.0, 0.0, 1.0, 16.0, 11.0, 1.0, 0.0, 1.0, 9.0, 13.0, 1.0, 0.0, 1.0, 8.0, 9.0, 1.0, 0.0, 1.0, 4.0, 9.0, 1.0, 0.0, 1.0, 12.0, 14.0, 1.0, 0.0, 1.0, 4.0, 12.0, 1.0, 0.0, 1.0, 17.0, 12.0, 1.0, 0.0, 1.0, 17.0, 13.0, 1.0, 0.0, 1.0, 10.0, 17.0, 1.0, 0.0, 1.0, 12.0, 17.0, 1.0, 0.0, 1.0, 4.0, 10.0, 1.0, 0.0, 1.0, 4.0, 13.0, 1.0, 0.0, 1.0, 5.0, 13.0, 1.0, 0.0, 1.0, 14.0, 13.0, 1.0, 0.0, 1.0, 8.0, 10.0, 1.0, 0.0, 1.0, 13.0, 6.0, 1.0, 0.0, 1.0, 6.0, 13.0, 1.0, 0.0, 1.0, 16.0, 10.0, 1.0, 0.0, 1.0, 15.0, 8.0, 1.0, 0.0, 1.0, 17.0, 10.0, 1.0, 0.0, 1.0, 9.0, 10.0, 1.0, 0.0, 1.0, 10.0, 13.0, 1.0, 0.0, 1.0, 11.0, 11.0, 1.0, 0.0, 1.0, 11.0, 8.0, 1.0, 0.0, 1.0, 15.0, 9.0, 1.0, 0.0, 1.0, 17.0, 9.0, 1.0, 0.0, 1.0, 7.0, 6.0, 1.0, 0.0, 1.0, 17.0, 3.0, 1.0, 0.0, 1.0, 17.0, 4.0, 1.0, 0.0, 1.0, 12.0, 8.0, 1.0, 0.0, 1.0, 6.0, 7.0, 1.0, 0.0, 1.0, 17.0, 6.0, 1.0, 0.0, 1.0, 17.0, 7.0, 1.0, 0.0, 1.0, 6.0, 8.0, 1.0, 0.0, 1.0, 10.0, 12.0, 1.0, 0.0, 1.0, 7.0, 12.0, 1.0, 0.0, 1.0, 10.0, 6.0, 1.0, 0.0, 1.0, 16.0, 4.0, 1.0, 0.0, 1.0, 16.0, 5.0, 1.0, 0.0, 1.0, 11.0, 6.0, 1.0, 0.0, 1.0, 7.0, 13.0, 1.0, 0.0, 1.0, 6.0, 9.0, 1.0, 0.0, 1.0, 14.0, 3.0, 1.0, 0.0, 1.0, 15.0, 3.0, 1.0, 0.0, 1.0, 5.0, 10.0, 1.0, 0.0, 1.0, 11.0, 3.0, 1.0, 0.0, 1.0, 12.0, 3.0, 1.0, 0.0, 1.0, 12.0, 12.0, 1.0, 0.0, 1.0, 9.0, 11.0, 1.0, 0.0, 1.0, 13.0, 10.0, 1.0, 0.0, 1.0, 5.0, 11.0, 1.0, 0.0, 1.0, 4.0, 3.0, 1.0, 0.0, 1.0, 13.0, 7.0, 1.0, 0.0, 1.0, 7.0, 3.0, 1.0, 0.0, 1.0, 13.0, 11.0, 1.0, 0.0, 1.0, 14.0, 7.0, 1.0, 0.0, 1.0, 8.0, 3.0, 1.0, 0.0, 1.0, 12.0, 4.0, 1.0, 0.0, 1.0, 14.0, 9.0, 1.0, 0.0, 1.0, 13.0, 4.0, 1.0, 0.0, 1.0, 15.0, 5.0, 1.0, 0.0, 1.0, 12.0, 13.0, 1.0, 0.0, 1.0, 13.0, 12.0, 1.0, 0.0, 1.0, 14.0, 10.0, 1.0, 0.0, 1.0, 5.0, 4.0, 1.0, 0.0, 1.0, 4.0, 5.0, 1.0, 0.0, 1.0, 8.0, 7.0, 1.0, 0.0, 1.0, 4.0, 6.0, 1.0, 0.0, 1.0, 8.0, 4.0, 1.0, 0.0, 1.0, 10.0, 9.0, 1.0, 0.0, 1.0, 7.0, 8.0, 1.0, 0.0, 1.0, 9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 7.0, 1.0, 0.0, 1.0, 7.0, 9.0, 1.0, 0.0, 1.0, 13.0, 17.0, 1.0, 0.0, 1.0, 14.0, 14.0, 1.0, 574.0, 2.0, 18.0, 6.0, 1.0, 574.0, 2.0, 17.0, 2.0, 1.0, 574.0, 2.0, 3.0, 9.0, 1.0, 574.0, 2.0, 19.0, 7.0, 1.0, 574.0, 2.0, 3.0, 18.0, 1.0, 574.0, 2.0, 3.0, 16.0, 1.0, 574.0, 2.0, 9.0, 19.0, 1.0, 574.0, 2.0, 3.0, 4.0, 1.0, 574.0, 2.0, 3.0, 7.0, 1.0, 574.0, 2.0, 2.0, 8.0, 1.0, 574.0, 2.0, 2.0, 17.0, 1.0, 574.0, 2.0, 2.0, 15.0, 1.0, 574.0, 2.0, 3.0, 19.0, 1.0, 574.0, 2.0, 3.0, 12.0, 1.0, 574.0, 2.0, 2.0, 3.0, 1.0, 574.0, 2.0, 3.0, 6.0, 1.0, 574.0, 2.0, 4.0, 2.0, 1.0, 574.0, 2.0, 4.0, 19.0, 1.0, 574.0, 2.0, 18.0, 8.0, 1.0, 574.0, 2.0, 3.0, 2.0, 1.0, 574.0, 2.0, 2.0, 11.0, 1.0, 574.0, 2.0, 2.0, 2.0, 1.0, 574.0, 2.0, 7.0, 19.0, 1.0, 574.0, 2.0, 11.0, 2.0, 1.0, 574.0, 2.0, 19.0, 9.0, 1.0, 574.0, 2.0, 19.0, 5.0, 1.0, 574.0, 2.0, 6.0, 19.0, 1.0, 574.0, 2.0, 3.0, 14.0, 1.0, 574.0, 2.0, 18.0, 14.0, 1.0, 574.0, 2.0, 13.0, 2.0, 1.0, 574.0, 2.0, 18.0, 12.0, 1.0, 574.0, 2.0, 14.0, 2.0, 1.0, 574.0, 2.0, 19.0, 4.0, 1.0, 574.0, 2.0, 19.0, 13.0, 1.0, 574.0, 2.0, 19.0, 11.0, 1.0, 574.0, 2.0, 18.0, 13.0, 1.0, 574.0, 2.0, 19.0, 3.0, 1.0, 574.0, 2.0, 10.0, 19.0, 1.0, 574.0, 2.0, 16.0, 2.0, 1.0, 574.0, 2.0, 19.0, 12.0, 1.0, 574.0, 2.0, 18.0, 9.0, 1.0, 574.0, 2.0, 19.0, 14.0, 1.0, 574.0, 2.0, 19.0, 10.0, 1.0, 574.0, 2.0, 3.0, 8.0, 1.0, 574.0, 2.0, 3.0, 15.0, 1.0, 574.0, 2.0, 11.0, 19.0, 1.0, 574.0, 2.0, 3.0, 13.0, 1.0, 574.0, 2.0, 3.0, 3.0, 1.0, 574.0, 2.0, 18.0, 10.0, 1.0, 574.0, 2.0, 14.0, 19.0, 1.0, 574.0, 2.0, 3.0, 11.0, 1.0, 574.0, 2.0, 3.0, 10.0, 1.0, 574.0, 2.0, 19.0, 6.0, 1.0, 574.0, 2.0, 12.0, 19.0, 1.0, 574.0, 2.0, 2.0, 12.0, 1.0, 574.0, 2.0, 2.0, 13.0, 1.0, 574.0, 2.0, 15.0, 2.0, 1.0, 574.0, 2.0, 13.0, 19.0, 1.0, 574.0, 2.0, 18.0, 7.0, 1.0, 574.0, 2.0, 8.0, 19.0, 1.0, 574.0, 2.0, 2.0, 14.0, 1.0, 574.0, 2.0, 7.0, 2.0, 1.0, 574.0, 2.0, 18.0, 2.0, 1.0, 574.0, 2.0, 2.0, 6.0, 1.0, 574.0, 2.0, 8.0, 2.0, 1.0, 574.0, 2.0, 18.0, 5.0, 1.0, 574.0, 2.0, 5.0, 2.0, 1.0, 574.0, 2.0, 3.0, 5.0, 1.0, 574.0, 2.0, 2.0, 7.0, 1.0, 574.0, 2.0, 2.0, 4.0, 1.0, 574.0, 2.0, 18.0, 4.0, 1.0, 574.0, 2.0, 2.0, 5.0, 1.0, 574.0, 2.0, 9.0, 2.0, 1.0, 574.0, 2.0, 3.0, 17.0, 1.0, 574.0, 2.0, 2.0, 16.0, 1.0, 574.0, 2.0, 2.0, 19.0, 1.0, 574.0, 2.0, 2.0, 18.0, 1.0, 574.0, 2.0, 5.0, 19.0, 1.0, 574.0, 2.0, 18.0, 11.0, 1.0, 574.0, 2.0, 6.0, 2.0, 1.0, 574.0, 2.0, 19.0, 8.0, 1.0, 574.0, 2.0, 2.0, 9.0, 1.0, 574.0, 2.0, 18.0, 3.0, 1.0, 574.0, 2.0, 10.0, 2.0, 1.0, 574.0, 2.0, 2.0, 10.0, 1.0, 574.0, 2.0, 12.0, 2.0, 1.0, 1459.0, 3.0, 1.0, -1.0, 4.0, 4.0, 0.0]
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
  drainages::Array{Field{Float64},1} = repeat(drainage,10000,false)
  runoff::Field{Float64} =  LatLonField{Float64}(hd_grid,
                                                     Float64[ 0.0 0.0 0.0 0.0
                                                              0.0 0.0 15.0/86400.0 0.0
                                                              0.0 0.0 0.0 0.0
                                                              0.0 0.0 0.0 0.0 ])
  runoffs::Array{Field{Float64},1} = repeat(runoff,10000,false)
  evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,1.0*1.003676471)
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
              0 0 0 0 0  0 0 0 0 0  1459.0 0 0 0 0  0 0 0 0 0
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

  expected_lake_volumes::Vector{Vector{Float64}} =
    [[1455.04], [1451.08], [1447.13], [1443.17], [1439.21], [1435.25], [1431.29], #=
  =# [1427.33], [1423.38], [1419.42], [1415.46], [1411.5], [1407.54], [1403.58], #=
  =# [1399.63], [1395.67], [1391.71], [1387.75], [1383.79], [1379.83], [1375.88], #=
  =# [1371.92], [1367.96], [1364.0], [1360.04], [1356.08], [1352.13], [1348.17], #=
  =# [1344.21], [1340.25], [1336.29], [1332.33], [1328.38], [1324.42], [1320.46], #=
  =# [1316.5], [1312.54], [1308.58], [1304.63], [1300.67], [1296.71], [1292.75], #=
  =# [1288.79], [1284.83], [1280.88], [1276.92], [1272.96], [1269.0], [1265.04], #=
  =# [1261.08], [1257.13], [1253.17], [1249.21], [1245.25], [1241.29], [1237.33], #=
  =# [1233.38], [1229.42], [1225.46], [1221.5], [1217.54], [1213.58], [1209.63], #=
  =# [1205.67], [1201.71], [1197.75], [1193.79], [1189.83], [1185.88], [1181.92], #=
  =# [1177.96], [1174.0], [1170.04], [1166.08], [1162.13], [1158.17], [1154.21], #=
  =# [1150.25], [1146.29], [1142.33], [1138.38], [1134.42], [1130.46], [1126.5], #=
  =# [1122.54], [1118.58], [1114.63], [1110.67], [1106.71], [1102.75], [1098.79], #=
  =# [1094.83], [1090.88], [1086.92], [1082.96], [1079.0], [1075.04], [1071.08], #=
  =# [1067.13], [1063.17], [1059.21], [1055.25], [1051.29], [1047.33], [1043.38], #=
  =# [1039.42], [1035.46], [1031.5], [1027.54], [1023.58], [1019.63], [1015.67], #=
  =# [1011.71], [1007.75], [1003.79], [999.833], [995.875], [991.917], [987.958], #=
  =# [984.0], [980.042], [976.083], [972.125], [968.167], [964.208], [960.25], #=
  =# [956.292], [952.333], [948.375], [944.417], [940.458], [936.5], [932.542], #=
  =# [928.583], [924.625], [920.667], [916.708], [912.75], [908.792], [904.833], #=
  =# [900.875], [896.917], [892.958], [889.0], [885.042], [881.083], [877.125], #=
  =# [873.167], [869.208], [865.25], [861.292], [857.333], [853.375], [849.417], #=
  =# [845.458], [841.5], [837.542], [833.583], [829.625], [825.667], [821.708], #=
  =# [817.75], [813.792], [809.833], [805.875], [801.917], [797.958], [794.0], #=
  =# [790.042], [786.083], [782.125], [778.167], [774.208], [770.25], [766.292], #=
  =# [762.333], [758.375], [754.417], [750.458], [746.5], [742.542], [738.583], #=
  =# [734.625], [730.667], [726.708], [722.75], [718.792], [714.833], [710.875], #=
  =# [706.917], [702.958], [699.0], [695.042], [691.083], [687.125], [683.167], #=
  =# [679.208], [675.25], [671.292], [667.333], [663.375], [659.417], [655.458], #=
  =# [651.5], [647.542], [643.583], [639.625], [635.667], [631.708], [627.75], #=
  =# [623.792], [619.833], [615.875], [611.917], [607.958], [604.0], [600.042], #=
  =# [596.083], [592.125], [588.167], [584.208], [580.25], [576.292], [572.333], #=
  =# [574.0346183503599], [570.0762850092487], [571.7775717684988], [573.4788585277488], #=
  =# [575.1801452869988], [571.2218119458877], [572.9230987051377], [574.6243854643877], #=
  =# [570.6660521232766], [572.3673388825266], [574.0686256417766], [570.1102923006655], #=
  =# [571.8115790599155], [573.5128658191655], [575.2141525784156], [571.2558192373043], #=
  =# [572.9571059965543], [574.6583927558042], [570.7000594146931], [572.401346173943], #=
  =# [574.102632933193], [570.1442995920818], [571.8455863513318], [573.5468731105818], #=
  =# [575.2481598698319], [571.2898265287207], [572.9911132879707], [574.6924000472206], #=
  =# [570.7340667061095], [572.4353534653595], [574.1366402246096], [570.1783068834984], #=
  =# [571.8795936427484], [573.5808804019985], [575.2821671612485], [571.3238338201373], #=
  =# [573.0251205793874], [574.7264073386373], [570.7680739975261], [572.4693607567762], #=
  =# [574.1706475160262], [570.212314174915], [571.9136009341651], [573.6148876934151], #=
  =# [575.3161744526651], [571.357841111554], [573.059127870804], [574.760414630054], #=
  =# [570.8020812889429], [572.5033680481929], [574.2046548074429], [570.2463214663318], #=
  =# [571.9476082255818], [573.6488949848318], [575.3501817440819], [571.3918484029707], #=
  =# [573.0931351622207], [574.7944219214708], [570.8360885803596], [572.5373753396096], #=
  =# [574.2386620988597], [570.2803287577485], [571.9816155169985], [573.6829022762486], #=
  =# [575.3841890354986], [571.4258556943875], [573.1271424536375], [574.8284292128875], #=
  =# [570.8700958717764], [572.5713826310264], [574.2726693902764], [570.3143360491653], #=
  =# [572.0156228084153], [573.7169095676652], [575.4181963269152], [571.4598629858041], #=
  =# [573.161149745054],  [574.8624365043039], [570.9041031631928], [572.6053899224428], #=
  =# [574.3066766816928], [570.3483433405817], [572.0496300998317], [573.7509168590817], #=
  =# [575.4522036183317], [571.4938702772206], [573.1951570364706], [574.8964437957206], #=
  =# [570.9381104546095], [572.6393972138595], [574.3406839731095], [570.3823506319984], #=
  =# [572.0836373912484], [573.7849241504985], [575.4862109097485], [571.5278775686373], #=
  =# [573.2291643278874], [574.9304510871374], [570.9721177460262], [572.6734045052763], #=
  =# [574.3746912645263], [570.4163579234151], [572.1176446826652], [573.8189314419152], #=
  =# [575.5202182011652], [571.5618848600541], [573.2631716193041], [574.9644583785541], #=
  =# [571.0061250374429], [572.7074117966929], [574.4086985559429], [570.4503652148318], #=
  =# [572.1516519740818], [573.8529387333317], [575.5542254925816], [571.5958921514705], #=
  =# [573.2971789107205], [574.9984656699705], [571.0401323288594], [572.7414190881094], #=
  =# [574.4427058473594], [570.4843725062483], [572.1856592654983], [573.8869460247483], #=
  =# [575.5882327839984], [571.6298994428871], [573.331186202137], [575.0324729613869], #=
  =# [571.0741396202758], [572.7754263795258], [574.4767131387758], [570.5183797976646], #=
  =# [572.2196665569146], [573.9209533161646], [575.6222400754145], [571.6639067343034], #=
  =# [573.3651934935533], [575.0664802528033], [571.1081469116922], [572.8094336709422], #=
  =# [574.5107204301922], [570.5523870890811], [572.2536738483311], [573.9549606075811], #=
  =# [575.6562473668312], [571.69791402572], [573.39920078497], [575.1004875442201], #=
  =# [571.1421542031089], [572.843440962359], [574.544727721609], [570.5863943804978], #=
  =# [572.2876811397479], [573.9889678989979], [575.6902546582479], [571.7319213171368], #=
  =# [573.4332080763868], [575.1344948356368], [571.1761614945257], [572.8774482537757], #=
  =# [574.5787350130257], [570.6204016719146], [572.3216884311646], [574.0229751904146], #=
  =# [570.0646418493035], [571.7659286085535], [573.4672153678035], [575.1685021270536], #=
  =# [571.2101687859424], [572.9114555451924], [574.6127423044425], [570.6544089633313], #=
  =# [572.3556957225813], [574.0569824818314], [570.0986491407202], [571.7999358999701], #=
  =# [573.50122265922], [575.2025094184701], [571.2441760773588], [572.9454628366088], #=
  =# [574.6467495958589], [570.6884162547477], [572.3897030139977], [574.0909897732477], #=
  =# [570.1326564321365], [571.8339431913865], [573.5352299506366], [575.2365167098865], #=
  =# [571.2781833687753], [572.9794701280254], [574.6807568872753], [570.7224235461641], #=
  =# [572.4237103054141], [574.1249970646641], [570.1666637235529], [571.8679504828029], #=
  =# [573.569237242053], [575.270524001303], [571.3121906601918], [573.0134774194419], #=
  =# [574.7147641786919], [570.7564308375808], [572.4577175968308], [574.1590043560808], #=
  =# [570.2006710149697], [571.9019577742197], [573.6032445334697], [575.3045312927197], #=
  =# [571.3461979516085], [573.0474847108585], [574.7487714701085], [570.7904381289974], #=
  =# [572.4917248882473], [574.1930116474973], [570.2346783063862], [571.9359650656362], #=
  =# [573.6372518248862], [575.3385385841362], [571.3802052430251], [573.0814920022751], #=
  =# [574.7827787615252], [570.8244454204139], [572.5257321796639], [574.227018938914], #=
  =# [570.2686855978028], [571.9699723570528], [573.6712591163028], [575.3725458755529], #=
  =# [571.4142125344417], [573.1154992936918], [574.8167860529418], [570.8584527118306], #=
  =# [572.5597394710807], [574.2610262303307], [570.3026928892195], [572.0039796484696], #=
  =# [573.7052664077196], [575.4065531669696], [571.4482198258585], [573.1495065851085], #=
  =# [574.8507933443585], [570.8924600032473], [572.5937467624973], [574.2950335217473], #=
  =# [570.3367001806362], [572.0379869398861], [573.739273699136], [575.440560458386], #=
  =# [571.4822271172749], [573.1835138765249], [574.8848006357749], [570.9264672946638], #=
  =# [572.6277540539138], [574.3290408131638], [570.3707074720527], [572.0719942313027], #=
  =# [573.7732809905527], [575.4745677498028], [571.5162344086916], [573.2175211679416], #=
  =# [574.9188079271917], [570.9604745860805], [572.6617613453305], [574.3630481045806], #=
  =# [570.4047147634694], [572.1060015227195], [573.8072882819695], [575.5085750412195], #=
  =# [571.5502417001084], [573.2515284593584], [574.9528152186084], [570.9944818774973], #=
  =# [572.6957686367473], [574.3970553959973], [570.4387220548862], [572.1400088141362]]
  lake_volumes::Vector{Vector{Float64}} = []
  river_fields::RiverPrognosticFields,
    lake_model_prognostics::LakeModelPrognostics,_ =
    drive_hd_and_lake_model(river_parameters,lake_model_parameters,
                            lake_parameters_as_array,
                            drainages,runoffs,evaporations,
                            500,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            return_lake_volumes=true,
                            diagnostic_lake_volumes=lake_volumes,
                            use_realistic_surface_coupling=true)
  for (lake_volumes_slice,expected_lake_volumes_slice) in zip(lake_volumes,expected_lake_volumes)
    @test isapprox(lake_volumes_slice,expected_lake_volumes_slice,
                   rtol=0.0,atol=0.01)
  end
end

@testset "Lake model tests 19" begin
  #Simple tests of evaporation with multiple lakes
  hd_grid = LatLonGrid(4,4,true)
  surface_model_grid = LatLonGrid(3,3,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(hd_grid,
                                                Int64[ 5 5 5 5
                                                       2 2 5 5
                                                       5 4 8 8
                                                       8 7 4 0 ]))
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
  lake_centers::Field{Bool} = LatLonField{Bool}(lake_grid,
    Bool[ false false false false false  false false false false false  false false false false false  #=
      =#  false false false false false
          false false false  true false  false false  true false false  false false false false false  #=
      =#  false  true false false false
          false  true false false false  false false false false false  false false false false false  #=
      =#  false false false false false
          false false false false false  false false false false false  false false false false false  #=
      =#  false false false false false
          false false false false false  false false false false false  false false false false false  #=
      =#  false false false false false
          false false false false false  false false false false false  false false false false false  #=
      =#  false false false false false
          false false false false false  false false false false false  false false false false false  #=
      =#  false false false false false
          false false false false false  false false false false false  false  true false false false  #=
      =#  false true false false false
          false false false false false  false false false false false  false false false false false  #=
      =#  false false false false false
          false false false false false  false false false false false  false false false false false  #=
      =#  false false false false false
          false false false false false  false false false false false  false false false false false  #=
      =#  false false false false false
          false false false false false  false false false false false  false false false false false  #=
      =#  false false false false false
          false false false false false  false false false false false  false false false false false  #=
      =#  false false false false false
          false false false false false  false false false false false  false false false false false  #=
      =#  false false false false false
          false  true false false false  false false false false false  false false false false false  #=
      =#  false false false false false
          false false false false false  false false false false false  false false false false false  #=
      =#  false false false false false
          false false false false false  false false false false false  false false false false false  #=
      =#  false false false false false
          false false false false false  false false false false false  false false false false false  #=
      =#  false false false false false
          false false false false false  false false false false false  false false false false false  #=
      =#  false false false false false
          false false false false false  false false false false false  false false false false false  #=
      =#  false false false false false ])
  cell_areas_on_surface_model_grid::Field{Float64} = LatLonField{Float64}(surface_model_grid,
    Float64[ 2.5 3.0 2.5
             3.0 4.0 3.0
             2.5 3.0 2.5 ])
  corresponding_surface_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
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
  cell_areas::Field{Float64} = LatLonField{Float64}(lake_grid,
    Float64[ 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
             1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
             1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
             1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
             1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
             1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
             1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
             1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
             1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
             1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
             1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
             1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
             1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
             1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
             1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
             1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
             1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
             1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
             1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
             1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 ])
  raw_heights::Field{Float64} = LatLonField{Float64}(lake_grid,
    Float64[9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0
            9.0 9.0 9.0 3.0 3.0  3.0 9.0 1.0 1.0 9.0  1.0 1.0 1.0 1.0 9.0  9.0 2.0 2.0 2.0 9.0
            9.0 1.0 9.0 3.0 3.0  3.0 6.0 1.0 1.0 1.0  1.0 1.0 1.0 1.0 7.0  7.0 2.0 2.0 2.0 9.0
            9.0 1.0 9.0 3.0 3.0  3.0 9.0 1.0 1.0 1.0  1.0 1.0 1.0 1.0 9.0  9.0 2.0 2.0 2.0 9.0
            9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 1.0 1.0 1.0 9.0  9.0 2.0 2.0 2.0 9.0

            9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 1.0 1.0 1.0 9.0  9.0 9.0 6.0 9.0 9.0
            9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 5.0 9.0 9.0  9.0 3.0 3.0 3.0 9.0
            9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
            9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
            9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0

            9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
            9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
            9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
            9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 8.0 9.0
            9.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 6.0 6.0  9.0 9.0 9.0 9.0 7.0  7.0 7.0 7.0 7.0 7.0

            9.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 9.0 7.0  0.0 0.0 0.0 0.0 0.0
            9.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 9.0 7.0  0.0 0.0 0.0 0.0 0.0
            9.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 8.0 7.0  0.0 0.0 0.0 0.0 0.0
            9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 7.0  0.0 0.0 0.0 0.0 0.0
            9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 7.0  0.0 0.0 0.0 0.0 0.0])
  corrected_heights::Field{Float64}  = LatLonField{Float64}(lake_grid,
    Float64[9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0
            9.0 9.0 9.0 3.0 3.0  3.0 9.0 1.0 1.0 9.0  1.0 1.0 1.0 1.0 9.0  9.0 2.0 2.0 2.0 9.0
            9.0 1.0 9.0 3.0 3.0  3.0 6.0 1.0 1.0 1.0  1.0 1.0 1.0 1.0 7.0  7.0 2.0 2.0 2.0 9.0
            9.0 1.0 9.0 3.0 3.0  3.0 9.0 1.0 1.0 1.0  1.0 1.0 1.0 1.0 9.0  9.0 2.0 2.0 2.0 9.0
            9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 1.0 1.0 1.0 9.0  9.0 2.0 2.0 2.0 9.0

            9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 1.0 1.0 1.0 9.0  9.0 9.0 6.0 9.0 9.0
            9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 5.0 9.0 9.0  9.0 3.0 3.0 3.0 9.0
            9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
            9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
            9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0

            9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
            9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
            9.0 1.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 4.0 4.0 4.0 9.0  9.0 3.0 3.0 3.0 9.0
            9.0 7.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 8.0 9.0
            9.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 6.0 6.0  9.0 9.0 9.0 9.0 7.0  7.0 7.0 7.0 7.0 7.0

            9.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 9.0 7.0  0.0 0.0 0.0 0.0 0.0
            9.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 9.0 7.0  0.0 0.0 0.0 0.0 0.0
            9.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 6.0 6.0  6.0 6.0 6.0 8.0 7.0  0.0 0.0 0.0 0.0 0.0
            9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 7.0  0.0 0.0 0.0 0.0 0.0
            9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 9.0  9.0 9.0 9.0 9.0 7.0  0.0 0.0 0.0 0.0 0.0])
  is_lake::Field{Bool} =  LatLonField{Bool}(lake_grid,
    Bool[ false false false false false false false false false #=
    =#    false false false false false false false false false #=
    =#    false false
          false false false true true true false true true #=
    =#    false true true true true false false true true #=
    =#   true false
          false true false  true true true true true true #=
    =#    true  true  true  true  true  true  true  true  true #=
    =#    true false
          false true false true true true false true true #=
    =#    true true true true true false false true true #=
    =#    true false
          false true false false false false false false false #=
    =#    false false true true true false false true true #=
    =#    true false
          false true false false false false false false false #=
    =#    false false  true  true  true false false false  true #=
    =#    false false
          false  true false false false false false false false #=
    =#    false false false  true false false false  true  true #=
    =#    true false
          false  true false false false false false false false #=
    =#    false false  true  true  true false false  true  true #=
    =#    true false
          false  true false false false false false false false #=
    =#    false false  true  true  true false false  true  true #=
    =#    true false
          false  true false false false false false false false #=
    =#    false false  true  true  true false false  true  true #=
    =#    true false
          false  true false false false false false false false #=
    =#    false false  true  true  true false false  true  true #=
    =#    true false
          false  true false false false false false false false #=
    =#    false false  true  true  true false false  true  true #=
    =#    true false
          false  true false false false false false false false #=
    =#    false false  true  true  true false false  true  true #=
    =#    true false
          false  true false false false false false false false #=
    =#    false false false false false false false false false #=
    =#    false false
          false  true  true  true  true  true  true  true  true #=
    =#    true false false false false false false false false #=
    =#    false false
          false  true  true  true  true  true  true  true  true #=
    =#    true  true  true  true false false false false false #=
    =#    false false
          false  true  true  true  true  true  true  true  true #=
    =#    true  true  true  true false false false false false #=
    =#    false false
          false  true  true  true  true  true  true  true  true #=
    =#    true  true  true  true false false false false false #=
    =#    false false
          false false false false false false false false false #=
    =#    false false false false false false false false false #=
    =#    false false
          false false false false false false false false false #=
    =#    false false false false false false false false false #=
    =#    false false ])
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
                        12,
                        is_lake)
  lake_parameters_as_array::Vector{Float64} = Float64[12.0, 236.0, 1.0, 8.0, 0.0, 15.0, 2.0, 45.0, 15.0, 2.0, 1.0, 0.0, 6.0, 16.0, 3.0, 1.0, 0.0, 6.0, 15.0, 3.0, 1.0, 0.0, 6.0, 16.0, 4.0, 1.0, 0.0, 6.0, 16.0, 2.0, 1.0, 0.0, 6.0, 17.0, 4.0, 1.0, 0.0, 6.0, 15.0, 5.0, 1.0, 0.0, 6.0, 16.0, 5.0, 1.0, 0.0, 6.0, 18.0, 3.0, 1.0, 0.0, 6.0, 16.0, 6.0, 1.0, 0.0, 6.0, 17.0, 6.0, 1.0, 0.0, 6.0, 15.0, 6.0, 1.0, 0.0, 6.0, 18.0, 2.0, 1.0, 0.0, 6.0, 15.0, 7.0, 1.0, 0.0, 6.0, 18.0, 6.0, 1.0, 0.0, 6.0, 18.0, 7.0, 1.0, 0.0, 6.0, 15.0, 4.0, 1.0, 0.0, 6.0, 18.0, 8.0, 1.0, 0.0, 6.0, 17.0, 8.0, 1.0, 0.0, 6.0, 18.0, 9.0, 1.0, 0.0, 6.0, 17.0, 9.0, 1.0, 0.0, 6.0, 16.0, 9.0, 1.0, 0.0, 6.0, 18.0, 10.0, 1.0, 0.0, 6.0, 17.0, 10.0, 1.0, 0.0, 6.0, 16.0, 10.0, 1.0, 0.0, 6.0, 15.0, 9.0, 1.0, 0.0, 6.0, 18.0, 11.0, 1.0, 0.0, 6.0, 15.0, 10.0, 1.0, 0.0, 6.0, 17.0, 2.0, 1.0, 0.0, 6.0, 18.0, 4.0, 1.0, 0.0, 6.0, 18.0, 5.0, 1.0, 0.0, 6.0, 15.0, 8.0, 1.0, 0.0, 6.0, 17.0, 12.0, 1.0, 0.0, 6.0, 16.0, 8.0, 1.0, 0.0, 6.0, 18.0, 13.0, 1.0, 0.0, 6.0, 16.0, 12.0, 1.0, 0.0, 6.0, 16.0, 13.0, 1.0, 0.0, 6.0, 17.0, 3.0, 1.0, 0.0, 6.0, 17.0, 5.0, 1.0, 0.0, 6.0, 16.0, 7.0, 1.0, 0.0, 6.0, 17.0, 7.0, 1.0, 0.0, 6.0, 17.0, 11.0, 1.0, 0.0, 6.0, 18.0, 12.0, 1.0, 0.0, 6.0, 16.0, 11.0, 1.0, 0.0, 6.0, 17.0, 13.0, 1.0, 45.0, 7.0, 1.0, 4.0, 3.0, 1.0, 0.0, 116.0, 2.0, 9.0, 0.0, 8.0, 17.0, 21.0, 8.0, 17.0, 1.0, 0.0, 3.0, 9.0, 18.0, 1.0, 0.0, 3.0, 7.0, 18.0, 1.0, 0.0, 3.0, 10.0, 17.0, 1.0, 0.0, 3.0, 7.0, 17.0, 1.0, 0.0, 3.0, 8.0, 19.0, 1.0, 0.0, 3.0, 11.0, 18.0, 1.0, 0.0, 3.0, 7.0, 19.0, 1.0, 0.0, 3.0, 8.0, 18.0, 1.0, 0.0, 3.0, 10.0, 19.0, 1.0, 0.0, 3.0, 9.0, 17.0, 1.0, 0.0, 3.0, 12.0, 18.0, 1.0, 0.0, 3.0, 12.0, 17.0, 1.0, 0.0, 3.0, 13.0, 17.0, 1.0, 0.0, 3.0, 12.0, 19.0, 1.0, 0.0, 3.0, 11.0, 19.0, 1.0, 0.0, 3.0, 13.0, 19.0, 1.0, 0.0, 3.0, 13.0, 18.0, 1.0, 0.0, 3.0, 10.0, 18.0, 1.0, 0.0, 3.0, 9.0, 19.0, 1.0, 0.0, 3.0, 11.0, 17.0, 1.0, 63.0, 6.0, 1.0, 5.0, -1.0, -1.0, 1.0, 101.0, 3.0, 10.0, 0.0, 8.0, 12.0, 18.0, 8.0, 12.0, 1.0, 0.0, 4.0, 9.0, 13.0, 1.0, 0.0, 4.0, 8.0, 13.0, 1.0, 0.0, 4.0, 10.0, 12.0, 1.0, 0.0, 4.0, 9.0, 12.0, 1.0, 0.0, 4.0, 11.0, 13.0, 1.0, 0.0, 4.0, 11.0, 12.0, 1.0, 0.0, 4.0, 12.0, 14.0, 1.0, 0.0, 4.0, 11.0, 14.0, 1.0, 0.0, 4.0, 12.0, 12.0, 1.0, 0.0, 4.0, 13.0, 13.0, 1.0, 0.0, 4.0, 8.0, 14.0, 1.0, 0.0, 4.0, 10.0, 14.0, 1.0, 0.0, 4.0, 9.0, 14.0, 1.0, 0.0, 4.0, 13.0, 14.0, 1.0, 0.0, 4.0, 13.0, 12.0, 1.0, 0.0, 4.0, 10.0, 13.0, 1.0, 0.0, 4.0, 12.0, 13.0, 1.0, 18.0, 5.0, 1.0, 6.0, 2.0, 3.0, 0.0, 66.0, 4.0, 8.0, 0.0, 3.0, 2.0, 11.0, 3.0, 2.0, 1.0, 0.0, 1.0, 4.0, 2.0, 1.0, 0.0, 1.0, 5.0, 2.0, 1.0, 0.0, 1.0, 6.0, 2.0, 1.0, 0.0, 1.0, 7.0, 2.0, 1.0, 0.0, 1.0, 8.0, 2.0, 1.0, 0.0, 1.0, 9.0, 2.0, 1.0, 0.0, 1.0, 10.0, 2.0, 1.0, 0.0, 1.0, 11.0, 2.0, 1.0, 0.0, 1.0, 12.0, 2.0, 1.0, 0.0, 1.0, 13.0, 2.0, 1.0, 66.0, 7.0, 1.0, 1.0, -1.0, -1.0, 1.0, 71.0, 5.0, 9.0, 0.0, 2.0, 17.0, 12.0, 2.0, 17.0, 1.0, 0.0, 2.0, 3.0, 18.0, 1.0, 0.0, 2.0, 2.0, 18.0, 1.0, 0.0, 2.0, 4.0, 17.0, 1.0, 0.0, 2.0, 3.0, 17.0, 1.0, 0.0, 2.0, 5.0, 18.0, 1.0, 0.0, 2.0, 5.0, 17.0, 1.0, 0.0, 2.0, 5.0, 19.0, 1.0, 0.0, 2.0, 2.0, 19.0, 1.0, 0.0, 2.0, 4.0, 19.0, 1.0, 0.0, 2.0, 3.0, 19.0, 1.0, 0.0, 2.0, 4.0, 18.0, 1.0, 48.0, 6.0, 1.0, 2.0, -1.0, -1.0, 1.0, 141.0, 6.0, 10.0, 0.0, 2.0, 8.0, 26.0, 2.0, 8.0, 1.0, 0.0, 1.0, 3.0, 9.0, 1.0, 0.0, 1.0, 2.0, 9.0, 1.0, 0.0, 1.0, 4.0, 8.0, 1.0, 0.0, 1.0, 3.0, 8.0, 1.0, 0.0, 1.0, 4.0, 10.0, 1.0, 0.0, 1.0, 3.0, 10.0, 1.0, 0.0, 1.0, 4.0, 11.0, 1.0, 0.0, 1.0, 4.0, 9.0, 1.0, 0.0, 1.0, 3.0, 11.0, 1.0, 0.0, 1.0, 2.0, 11.0, 1.0, 0.0, 1.0, 4.0, 12.0, 1.0, 0.0, 1.0, 3.0, 12.0, 1.0, 0.0, 1.0, 2.0, 12.0, 1.0, 0.0, 1.0, 4.0, 13.0, 1.0, 0.0, 1.0, 3.0, 13.0, 1.0, 0.0, 1.0, 2.0, 13.0, 1.0, 0.0, 1.0, 5.0, 14.0, 1.0, 0.0, 1.0, 4.0, 14.0, 1.0, 0.0, 1.0, 6.0, 14.0, 1.0, 0.0, 1.0, 6.0, 13.0, 1.0, 0.0, 1.0, 3.0, 14.0, 1.0, 0.0, 1.0, 6.0, 12.0, 1.0, 0.0, 1.0, 2.0, 14.0, 1.0, 0.0, 1.0, 5.0, 12.0, 1.0, 0.0, 1.0, 5.0, 13.0, 1.0, 104.0, 5.0, 1.0, 3.0, -1.0, -1.0, 1.0, 56.0, 7.0, 11.0, 0.0, 2.0, 4.0, 9.0, 2.0, 4.0, 1.0, 0.0, 3.0, 3.0, 5.0, 1.0, 0.0, 3.0, 2.0, 5.0, 1.0, 0.0, 3.0, 4.0, 6.0, 1.0, 0.0, 3.0, 4.0, 5.0, 1.0, 0.0, 3.0, 3.0, 4.0, 1.0, 0.0, 3.0, 3.0, 6.0, 1.0, 0.0, 3.0, 4.0, 4.0, 1.0, 0.0, 3.0, 2.0, 6.0, 1.0, 27.0, 6.0, 1.0, 10.0, 1.0, 2.0, 0.0, 298.0, 8.0, -1.0, 2.0, 1.0, 4.0, 15.0, 2.0, 57.0, 15.0, 2.0, 1.0, 0.0, 7.0, 16.0, 3.0, 1.0, 0.0, 7.0, 15.0, 3.0, 1.0, 0.0, 7.0, 14.0, 2.0, 2.0, 0.0, 7.0, 16.0, 4.0, 1.0, 0.0, 7.0, 16.0, 2.0, 1.0, 0.0, 7.0, 17.0, 3.0, 1.0, 0.0, 7.0, 16.0, 5.0, 1.0, 0.0, 7.0, 15.0, 5.0, 1.0, 0.0, 7.0, 18.0, 3.0, 1.0, 0.0, 7.0, 18.0, 2.0, 1.0, 0.0, 7.0, 17.0, 6.0, 1.0, 0.0, 7.0, 15.0, 6.0, 1.0, 0.0, 7.0, 18.0, 4.0, 1.0, 0.0, 7.0, 18.0, 5.0, 1.0, 0.0, 7.0, 15.0, 4.0, 1.0, 0.0, 7.0, 17.0, 2.0, 1.0, 0.0, 7.0, 17.0, 5.0, 1.0, 0.0, 7.0, 15.0, 7.0, 1.0, 0.0, 7.0, 18.0, 6.0, 1.0, 0.0, 7.0, 13.0, 2.0, 1.0, 0.0, 7.0, 16.0, 7.0, 1.0, 0.0, 7.0, 12.0, 2.0, 1.0, 0.0, 7.0, 17.0, 8.0, 1.0, 0.0, 7.0, 11.0, 2.0, 1.0, 0.0, 7.0, 18.0, 9.0, 1.0, 0.0, 7.0, 17.0, 9.0, 1.0, 0.0, 7.0, 18.0, 8.0, 1.0, 0.0, 7.0, 10.0, 2.0, 1.0, 0.0, 7.0, 16.0, 9.0, 1.0, 0.0, 7.0, 15.0, 8.0, 1.0, 0.0, 7.0, 17.0, 7.0, 1.0, 0.0, 7.0, 16.0, 8.0, 1.0, 0.0, 7.0, 17.0, 4.0, 1.0, 0.0, 7.0, 18.0, 7.0, 1.0, 0.0, 7.0, 15.0, 9.0, 1.0, 0.0, 7.0, 16.0, 10.0, 1.0, 0.0, 7.0, 9.0, 2.0, 1.0, 0.0, 7.0, 15.0, 10.0, 1.0, 0.0, 7.0, 16.0, 6.0, 1.0, 0.0, 7.0, 17.0, 10.0, 1.0, 0.0, 7.0, 8.0, 2.0, 1.0, 0.0, 7.0, 17.0, 11.0, 1.0, 0.0, 7.0, 16.0, 11.0, 1.0, 0.0, 7.0, 18.0, 11.0, 1.0, 0.0, 7.0, 17.0, 12.0, 1.0, 0.0, 7.0, 16.0, 12.0, 1.0, 0.0, 7.0, 7.0, 2.0, 1.0, 0.0, 7.0, 18.0, 13.0, 1.0, 0.0, 7.0, 16.0, 13.0, 1.0, 0.0, 7.0, 17.0, 13.0, 1.0, 0.0, 7.0, 6.0, 2.0, 1.0, 0.0, 7.0, 18.0, 12.0, 1.0, 0.0, 7.0, 5.0, 2.0, 1.0, 0.0, 7.0, 18.0, 10.0, 1.0, 0.0, 7.0, 4.0, 2.0, 1.0, 0.0, 7.0, 3.0, 2.0, 1.0, 56.0, 8.0, 1.0, -1.0, 4.0, 4.0, 0.0, 183.0, 9.0, 12.0, 2.0, 2.0, 5.0, 8.0, 17.0, 34.0, 8.0, 17.0, 1.0, 0.0, 6.0, 9.0, 18.0, 1.0, 0.0, 6.0, 7.0, 18.0, 1.0, 0.0, 6.0, 10.0, 17.0, 1.0, 0.0, 6.0, 7.0, 17.0, 1.0, 0.0, 6.0, 8.0, 19.0, 1.0, 0.0, 6.0, 6.0, 18.0, 1.0, 0.0, 6.0, 11.0, 18.0, 1.0, 0.0, 6.0, 7.0, 19.0, 1.0, 0.0, 6.0, 5.0, 17.0, 1.0, 0.0, 6.0, 12.0, 18.0, 1.0, 0.0, 6.0, 12.0, 17.0, 1.0, 0.0, 6.0, 4.0, 18.0, 1.0, 0.0, 6.0, 13.0, 19.0, 1.0, 0.0, 6.0, 13.0, 17.0, 1.0, 0.0, 6.0, 13.0, 18.0, 1.0, 0.0, 6.0, 4.0, 19.0, 1.0, 0.0, 6.0, 4.0, 17.0, 1.0, 0.0, 6.0, 8.0, 18.0, 1.0, 0.0, 6.0, 10.0, 19.0, 1.0, 0.0, 6.0, 9.0, 17.0, 1.0, 0.0, 6.0, 11.0, 19.0, 1.0, 0.0, 6.0, 12.0, 19.0, 1.0, 0.0, 6.0, 5.0, 19.0, 1.0, 0.0, 6.0, 5.0, 18.0, 1.0, 0.0, 6.0, 10.0, 18.0, 1.0, 0.0, 6.0, 9.0, 19.0, 1.0, 0.0, 6.0, 11.0, 17.0, 1.0, 0.0, 6.0, 3.0, 17.0, 1.0, 0.0, 6.0, 3.0, 19.0, 1.0, 0.0, 6.0, 3.0, 18.0, 1.0, 0.0, 6.0, 2.0, 18.0, 1.0, 0.0, 6.0, 2.0, 17.0, 1.0, 0.0, 6.0, 2.0, 19.0, 1.0, 34.0, 7.0, 1.0, 11.0, 1.0, 3.0, 0.0, 238.0, 10.0, 11.0, 2.0, 3.0, 6.0, 8.0, 12.0, 45.0, 8.0, 12.0, 1.0, 0.0, 5.0, 9.0, 13.0, 1.0, 0.0, 5.0, 7.0, 13.0, 1.0, 0.0, 5.0, 10.0, 12.0, 1.0, 0.0, 5.0, 8.0, 13.0, 1.0, 0.0, 5.0, 8.0, 14.0, 1.0, 0.0, 5.0, 6.0, 13.0, 1.0, 0.0, 5.0, 11.0, 13.0, 1.0, 0.0, 5.0, 6.0, 12.0, 1.0, 0.0, 5.0, 6.0, 14.0, 1.0, 0.0, 5.0, 5.0, 12.0, 1.0, 0.0, 5.0, 12.0, 12.0, 1.0, 0.0, 5.0, 4.0, 13.0, 1.0, 0.0, 5.0, 4.0, 11.0, 1.0, 0.0, 5.0, 13.0, 13.0, 1.0, 0.0, 5.0, 13.0, 12.0, 1.0, 0.0, 5.0, 4.0, 14.0, 1.0, 0.0, 5.0, 3.0, 10.0, 1.0, 0.0, 5.0, 4.0, 12.0, 1.0, 0.0, 5.0, 10.0, 13.0, 1.0, 0.0, 5.0, 10.0, 14.0, 1.0, 0.0, 5.0, 9.0, 14.0, 1.0, 0.0, 5.0, 12.0, 13.0, 1.0, 0.0, 5.0, 2.0, 9.0, 1.0, 0.0, 5.0, 3.0, 9.0, 1.0, 0.0, 5.0, 11.0, 14.0, 1.0, 0.0, 5.0, 5.0, 13.0, 1.0, 0.0, 5.0, 12.0, 14.0, 1.0, 0.0, 5.0, 4.0, 8.0, 1.0, 0.0, 5.0, 3.0, 8.0, 1.0, 0.0, 5.0, 2.0, 8.0, 1.0, 0.0, 5.0, 4.0, 9.0, 1.0, 0.0, 5.0, 9.0, 12.0, 1.0, 0.0, 5.0, 5.0, 14.0, 1.0, 0.0, 5.0, 2.0, 11.0, 1.0, 0.0, 5.0, 4.0, 10.0, 1.0, 0.0, 5.0, 13.0, 14.0, 1.0, 0.0, 5.0, 2.0, 12.0, 1.0, 0.0, 5.0, 11.0, 12.0, 1.0, 0.0, 5.0, 3.0, 12.0, 1.0, 0.0, 5.0, 2.0, 13.0, 1.0, 0.0, 5.0, 3.0, 11.0, 1.0, 0.0, 5.0, 3.0, 14.0, 1.0, 0.0, 5.0, 2.0, 14.0, 1.0, 0.0, 5.0, 3.0, 13.0, 1.0, 45.0, 6.0, 1.0, 7.0, 1.0, 2.0, 0.0, 288.0, 11.0, 12.0, 2.0, 10.0, 7.0, 8.0, 12.0, 55.0, 8.0, 12.0, 1.0, 0.0, 6.0, 9.0, 13.0, 1.0, 0.0, 6.0, 7.0, 13.0, 1.0, 0.0, 6.0, 10.0, 12.0, 1.0, 0.0, 6.0, 8.0, 13.0, 1.0, 0.0, 6.0, 8.0, 14.0, 1.0, 0.0, 6.0, 6.0, 13.0, 1.0, 0.0, 6.0, 11.0, 13.0, 1.0, 0.0, 6.0, 6.0, 12.0, 1.0, 0.0, 6.0, 6.0, 14.0, 1.0, 0.0, 6.0, 5.0, 12.0, 1.0, 0.0, 6.0, 12.0, 12.0, 1.0, 0.0, 6.0, 4.0, 13.0, 1.0, 0.0, 6.0, 4.0, 11.0, 1.0, 0.0, 6.0, 13.0, 13.0, 1.0, 0.0, 6.0, 13.0, 12.0, 1.0, 0.0, 6.0, 4.0, 14.0, 1.0, 0.0, 6.0, 3.0, 10.0, 1.0, 0.0, 6.0, 4.0, 12.0, 1.0, 0.0, 6.0, 10.0, 13.0, 1.0, 0.0, 6.0, 10.0, 14.0, 1.0, 0.0, 6.0, 9.0, 14.0, 1.0, 0.0, 6.0, 12.0, 13.0, 1.0, 0.0, 6.0, 2.0, 9.0, 1.0, 0.0, 6.0, 3.0, 9.0, 1.0, 0.0, 6.0, 11.0, 14.0, 1.0, 0.0, 6.0, 5.0, 13.0, 1.0, 0.0, 6.0, 12.0, 14.0, 1.0, 0.0, 6.0, 4.0, 8.0, 1.0, 0.0, 6.0, 3.0, 8.0, 1.0, 0.0, 6.0, 2.0, 8.0, 1.0, 0.0, 6.0, 4.0, 9.0, 1.0, 0.0, 6.0, 3.0, 7.0, 1.0, 0.0, 6.0, 9.0, 12.0, 1.0, 0.0, 6.0, 2.0, 6.0, 1.0, 0.0, 6.0, 5.0, 14.0, 1.0, 0.0, 6.0, 3.0, 5.0, 1.0, 0.0, 6.0, 2.0, 5.0, 1.0, 0.0, 6.0, 4.0, 5.0, 1.0, 0.0, 6.0, 4.0, 4.0, 1.0, 0.0, 6.0, 2.0, 4.0, 1.0, 0.0, 6.0, 3.0, 4.0, 1.0, 0.0, 6.0, 4.0, 6.0, 1.0, 0.0, 6.0, 3.0, 6.0, 1.0, 0.0, 6.0, 2.0, 11.0, 1.0, 0.0, 6.0, 4.0, 10.0, 1.0, 0.0, 6.0, 2.0, 12.0, 1.0, 0.0, 6.0, 13.0, 14.0, 1.0, 0.0, 6.0, 2.0, 13.0, 1.0, 0.0, 6.0, 11.0, 12.0, 1.0, 0.0, 6.0, 2.0, 14.0, 1.0, 0.0, 6.0, 3.0, 12.0, 1.0, 0.0, 6.0, 3.0, 11.0, 1.0, 0.0, 6.0, 3.0, 14.0, 1.0, 0.0, 6.0, 3.0, 13.0, 1.0, 55.0, 7.0, 1.0, 9.0, 1.0, 4.0, 0.0, 468.0, 12.0, -1.0, 2.0, 9.0, 11.0, 8.0, 17.0, 91.0, 8.0, 17.0, 1.0, 0.0, 7.0, 9.0, 18.0, 1.0, 0.0, 7.0, 7.0, 18.0, 1.0, 0.0, 7.0, 10.0, 17.0, 1.0, 0.0, 7.0, 7.0, 17.0, 1.0, 0.0, 7.0, 8.0, 19.0, 1.0, 0.0, 7.0, 6.0, 18.0, 1.0, 0.0, 7.0, 11.0, 18.0, 1.0, 0.0, 7.0, 7.0, 19.0, 1.0, 0.0, 7.0, 5.0, 17.0, 1.0, 0.0, 7.0, 12.0, 18.0, 1.0, 0.0, 7.0, 12.0, 17.0, 1.0, 0.0, 7.0, 4.0, 18.0, 1.0, 0.0, 7.0, 13.0, 19.0, 1.0, 0.0, 7.0, 13.0, 17.0, 1.0, 0.0, 7.0, 13.0, 18.0, 1.0, 0.0, 7.0, 4.0, 19.0, 1.0, 0.0, 7.0, 4.0, 17.0, 1.0, 0.0, 7.0, 8.0, 18.0, 1.0, 0.0, 7.0, 10.0, 19.0, 1.0, 0.0, 7.0, 9.0, 17.0, 1.0, 0.0, 7.0, 11.0, 19.0, 1.0, 0.0, 7.0, 12.0, 19.0, 1.0, 0.0, 7.0, 5.0, 19.0, 1.0, 0.0, 7.0, 5.0, 18.0, 1.0, 0.0, 7.0, 3.0, 16.0, 1.0, 0.0, 7.0, 10.0, 18.0, 1.0, 0.0, 7.0, 9.0, 19.0, 1.0, 0.0, 7.0, 11.0, 17.0, 1.0, 0.0, 7.0, 3.0, 15.0, 1.0, 0.0, 7.0, 3.0, 17.0, 1.0, 0.0, 7.0, 2.0, 17.0, 1.0, 0.0, 7.0, 3.0, 19.0, 1.0, 0.0, 7.0, 4.0, 14.0, 1.0, 0.0, 7.0, 2.0, 14.0, 1.0, 0.0, 7.0, 2.0, 19.0, 1.0, 0.0, 7.0, 5.0, 14.0, 1.0, 0.0, 7.0, 4.0, 13.0, 1.0, 0.0, 7.0, 6.0, 14.0, 1.0, 0.0, 7.0, 6.0, 13.0, 1.0, 0.0, 7.0, 5.0, 12.0, 1.0, 0.0, 7.0, 4.0, 12.0, 1.0, 0.0, 7.0, 6.0, 12.0, 1.0, 0.0, 7.0, 4.0, 11.0, 1.0, 0.0, 7.0, 3.0, 11.0, 1.0, 0.0, 7.0, 3.0, 12.0, 1.0, 0.0, 7.0, 4.0, 10.0, 1.0, 0.0, 7.0, 3.0, 10.0, 1.0, 0.0, 7.0, 2.0, 12.0, 1.0, 0.0, 7.0, 2.0, 11.0, 1.0, 0.0, 7.0, 4.0, 9.0, 1.0, 0.0, 7.0, 3.0, 9.0, 1.0, 0.0, 7.0, 2.0, 9.0, 1.0, 0.0, 7.0, 7.0, 13.0, 1.0, 0.0, 7.0, 2.0, 8.0, 1.0, 0.0, 7.0, 3.0, 13.0, 1.0, 0.0, 7.0, 2.0, 13.0, 1.0, 0.0, 7.0, 5.0, 13.0, 1.0, 0.0, 7.0, 2.0, 18.0, 1.0, 0.0, 7.0, 3.0, 14.0, 1.0, 0.0, 7.0, 3.0, 18.0, 1.0, 0.0, 7.0, 8.0, 13.0, 1.0, 0.0, 7.0, 8.0, 12.0, 1.0, 0.0, 7.0, 3.0, 7.0, 1.0, 0.0, 7.0, 8.0, 14.0, 1.0, 0.0, 7.0, 4.0, 8.0, 1.0, 0.0, 7.0, 3.0, 8.0, 1.0, 0.0, 7.0, 9.0, 13.0, 1.0, 0.0, 7.0, 4.0, 6.0, 1.0, 0.0, 7.0, 3.0, 6.0, 1.0, 0.0, 7.0, 2.0, 6.0, 1.0, 0.0, 7.0, 10.0, 13.0, 1.0, 0.0, 7.0, 10.0, 12.0, 1.0, 0.0, 7.0, 3.0, 5.0, 1.0, 0.0, 7.0, 2.0, 5.0, 1.0, 0.0, 7.0, 4.0, 5.0, 1.0, 0.0, 7.0, 4.0, 4.0, 1.0, 0.0, 7.0, 3.0, 4.0, 1.0, 0.0, 7.0, 2.0, 4.0, 1.0, 0.0, 7.0, 11.0, 14.0, 1.0, 0.0, 7.0, 11.0, 13.0, 1.0, 0.0, 7.0, 11.0, 12.0, 1.0, 0.0, 7.0, 12.0, 14.0, 1.0, 0.0, 7.0, 12.0, 13.0, 1.0, 0.0, 7.0, 12.0, 12.0, 1.0, 0.0, 7.0, 13.0, 14.0, 1.0, 0.0, 7.0, 13.0, 13.0, 1.0, 0.0, 7.0, 13.0, 12.0, 1.0, 0.0, 7.0, 9.0, 12.0, 1.0, 0.0, 7.0, 10.0, 14.0, 1.0, 0.0, 7.0, 9.0, 14.0, 1.0, 91.0, 8.0, 1.0, -1.0, 4.0, 4.0, 0.0]
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
  drainages::Array{Field{Float64},1} = repeat(drainage,10000,false)
  runoffs::Array{Field{Float64},1} = drainages
  evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,1.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,180,false)
  initial_water_to_lake_centers = LatLonField{Float64}(lake_grid,
    Float64[ 0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 27.1 0  0 0 204.1 0 0  0 0 0 0 0  0 173.1 0 0 0
             0 66.1 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 18.1 0 0 0  0 63.1 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 101.1 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
             0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  ])
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
       Int64[  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
               0 0 0 7 0 0 0 6 0 0 0 0 0 0 0 0 5 0 0 0
               0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
               0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 2 0 0 0
               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
               0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0   ])
  expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
        Float64[ 0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
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
                 0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  ])
  expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.06   0.02   0.03
                 0.0    0.02   0.02
                 0.03   0.0    0.00 ])
  expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[  2    1    1
                0    1    1
                1    0    0 ])
  expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 36 48 36
               48 64 48
               36 48 36  ])
  expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.0, 0.0, 0.0, 0.0, 0.0]
  # expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[ 0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0
  #                0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0 0 ])
  expected_lake_volumes_all_timesteps::Vector{Vector{Float64}} =
    [[45.0, 63.0, 18.0, 66.0, 48.0, 104.0, 27.0, 52.333333333333336, 34.0, 45.0, 55.0, 85.14583333333331],
     [45.0, 63.0, 18.0, 66.0, 48.0, 104.0, 27.0, 48.666666666666664, 34.0, 45.0, 55.0, 79.29166666666666],
     [45.0, 63.0, 18.0, 66.0, 48.0, 104.0, 27.0, 45.00000000000001, 34.0, 45.0, 55.0, 73.43749999999999],
     [45.0, 63.0, 18.0, 66.0, 48.0, 104.0, 27.0, 41.333333333333336, 34.0, 45.0, 55.0, 67.58333333333331],
     [45.0, 63.0, 18.0, 66.0, 48.0, 104.0, 27.0, 37.666666666666664, 34.0, 45.0, 55.0, 61.72916666666665],
     [45.0, 63.0, 18.0, 66.0, 48.0, 104.0, 27.0, 34.00000000000001, 34.0, 45.0, 55.0, 55.87499999999998],
     [45.0, 63.0, 18.0, 66.0, 48.0, 104.0, 27.0, 30.33333333333335, 34.0, 45.0, 55.0, 50.020833333333314],
     [45.0, 63.0, 18.0, 66.0, 48.0, 104.0, 27.0, 26.66666666666668, 34.0, 45.0, 55.0, 44.16666666666665],
     [45.0, 63.0, 18.0, 66.0, 48.0, 104.0, 27.0, 23.000000000000025, 34.0, 45.0, 55.0, 38.312499999999986],
     [45.0, 63.0, 18.0, 66.0, 48.0, 104.0, 27.0, 19.333333333333353, 34.0, 45.0, 55.0, 32.45833333333332],
     [45.0, 63.0, 18.0, 66.0, 48.0, 104.0, 27.0, 15.666666666666693, 34.0, 45.0, 55.0, 26.60416666666666],
     [45.0, 63.0, 18.0, 66.0, 48.0, 104.0, 27.0, 12.000000000000021, 34.0, 45.0, 55.0, 20.749999999999993],
     [45.0, 63.0, 18.0, 66.0, 48.0, 104.0, 27.0, 8.333333333333353, 34.0, 45.0, 55.0, 14.895833333333325],
     [45.0, 63.0, 18.0, 66.0, 48.0, 104.0, 27.0, 4.666666666666693, 34.0, 45.0, 55.0, 9.041666666666659],
     [45.0, 63.0, 18.0, 66.0, 48.0, 104.0, 27.0, 1.0000000000000329, 34.0, 45.0, 55.0, 3.187499999999992],
     [43.666666666666686, 63.0, 18.0, 64.66666666666669, 48.0, 104.0, 27.0, 0.0, 32.66666666666666, 45.0, 53.66666666666666, 0.0],
     [40.7152777777778, 63.0, 18.0, 63.95138888888891, 48.0, 104.0, 27.0, 0.0, 30.45138888888888, 45.0, 50.16666666666666, 0.0],
     [37.763888888888914, 63.0, 18.0, 63.23611111111112, 48.0, 104.0, 27.0, 0.0, 28.2361111111111, 45.0, 46.66666666666666, 0.0],
     [34.81250000000003, 63.0, 18.0, 62.520833333333336, 48.0, 104.0, 27.0, 0.0, 26.02083333333333, 45.0, 43.16666666666666, 0.0],
     [31.86111111111114, 63.0, 18.0, 61.80555555555556, 48.0, 104.0, 27.0, 0.0, 23.805555555555557, 45.0, 39.66666666666666, 0.0],
     [28.90972222222225, 63.0, 18.0, 61.09027777777778, 48.0, 104.0, 27.0, 0.0, 21.59027777777778, 45.0, 36.16666666666664, 0.0],
     [25.95833333333336, 63.0, 18.0, 60.375, 48.0, 104.0, 27.0, 0.0, 19.375, 45.0, 32.66666666666663, 0.0],
     [23.00694444444447, 63.0, 18.0, 59.65972222222222, 48.0, 104.0, 27.0, 0.0, 17.15972222222222, 45.0, 29.166666666666632, 0.0],
     [20.055555555555582, 63.0, 18.0, 58.94444444444444, 48.0, 104.0, 27.0, 0.0, 14.94444444444445, 45.0, 25.666666666666618, 0.0],
     [17.104166666666693, 63.0, 18.0, 58.229166666666664, 48.0, 104.0, 27.0, 0.0, 12.729166666666671, 45.0, 22.166666666666636, 0.0],
     [14.152777777777803, 63.0, 18.0, 57.513888888888886, 48.0, 104.0, 27.0, 0.0, 10.5138888888889, 45.0, 18.666666666666636, 0.0],
     [11.201388888888914, 63.0, 18.0, 56.79861111111111, 48.0, 104.0, 27.0, 0.0, 8.298611111111128, 45.0, 15.16666666666664, 0.0],
     [8.250000000000025, 63.0, 18.0, 56.08333333333333, 48.0, 104.0, 27.0, 0.0, 6.083333333333357, 45.0, 11.666666666666652, 0.0],
     [5.2986111111111365, 63.0, 18.0, 55.36805555555555, 48.0, 104.0, 27.0, 0.0, 3.868055555555578, 45.0, 8.166666666666657, 0.0],
     [2.3472222222222476, 63.0, 18.0, 54.65277777777777, 48.0, 104.0, 27.0, 0.0, 1.652777777777799, 45.0, 4.666666666666656, 0.0],
     [0.0, 62.71875000000001, 18.0, 53.93749999999999, 47.71875000000001, 104.0, 27.0, 0.0, 0.0, 45.0, 1.166666666666647, 0.0],
     [0.0, 61.40625000000001, 18.0, 53.222222222222214, 46.88541666666667, 104.0, 25.83333333333333, 0.0, 0.0, 43.83333333333333, 0.0, 0.0],
     [0.0, 60.09375000000001, 18.0, 52.506944444444436, 46.052083333333336, 104.0, 25.20833333333333, 0.0, 0.0, 41.020833333333314, 0.0, 0.0],
     [0.0, 58.78125000000001, 18.0, 51.79166666666666, 45.21875000000001, 104.0, 24.58333333333333, 0.0, 0.0, 38.2083333333333, 0.0, 0.0],
     [0.0, 57.46875000000001, 18.0, 51.07638888888888, 44.38541666666667, 104.0, 23.958333333333332, 0.0, 0.0, 35.3958333333333, 0.0, 0.0],
     [0.0, 56.15625000000001, 18.0, 50.3611111111111, 43.55208333333334, 104.0, 23.333333333333332, 0.0, 0.0, 32.5833333333333, 0.0, 0.0],
     [0.0, 54.84375000000001, 18.0, 49.64583333333332, 42.718750000000014, 104.0, 22.708333333333336, 0.0, 0.0, 29.7708333333333, 0.0, 0.0],
     [0.0, 53.53125000000001, 18.0, 48.93055555555554, 41.885416666666686, 104.0, 22.08333333333334, 0.0, 0.0, 26.9583333333333, 0.0, 0.0],
     [0.0, 52.21875000000001, 18.0, 48.215277777777764, 41.05208333333336, 104.0, 21.45833333333334, 0.0, 0.0, 24.1458333333333, 0.0, 0.0],
     [0.0, 50.90625000000001, 18.0, 47.499999999999986, 40.21875000000002, 104.0, 20.83333333333334, 0.0, 0.0, 21.3333333333333, 0.0, 0.0],
     [0.0, 49.59375000000001, 18.0, 46.78472222222221, 39.385416666666686, 104.0, 20.20833333333334, 0.0, 0.0, 18.5208333333333, 0.0, 0.0],
     [0.0, 48.28125000000001, 18.0, 46.06944444444443, 38.55208333333335, 104.0, 19.58333333333334, 0.0, 0.0, 15.7083333333333, 0.0, 0.0],
     [0.0, 46.96875000000001, 18.0, 45.35416666666665, 37.718750000000014, 104.0, 18.95833333333334, 0.0, 0.0, 12.8958333333333, 0.0, 0.0],
     [0.0, 45.65625000000001, 18.0, 44.63888888888887, 36.88541666666668, 104.0, 18.333333333333343, 0.0, 0.0, 10.083333333333286, 0.0, 0.0],
     [0.0, 44.34375000000001, 18.0, 43.92361111111109, 36.05208333333334, 104.0, 17.708333333333343, 0.0, 0.0, 7.270833333333286, 0.0, 0.0],
     [0.0, 43.03125000000001, 18.0, 43.208333333333314, 35.21875000000001, 104.0, 17.083333333333343, 0.0, 0.0, 4.458333333333286, 0.0, 0.0],
     [0.0, 41.71875000000001, 18.0, 42.493055555555536, 34.38541666666667, 104.0, 16.458333333333346, 0.0, 0.0, 1.645833333333286, 0.0, 0.0],
     [0.0, 40.40625000000001, 17.416666666666643, 41.77777777777776, 33.552083333333336, 103.41666666666664, 15.833333333333346, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 39.09375000000001, 16.291666666666643, 41.06249999999998, 32.71875, 101.79166666666666, 15.208333333333348, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 37.78125000000001, 15.166666666666643, 40.3472222222222, 31.885416666666664, 100.16666666666666, 14.583333333333348, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 36.46875000000001, 14.041666666666643, 39.63194444444442, 31.05208333333333, 98.54166666666666, 13.95833333333335, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 35.15625000000001, 12.916666666666643, 38.91666666666664, 30.218749999999993, 96.91666666666667, 13.33333333333335, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 33.84375000000001, 11.791666666666643, 38.201388888888864, 29.38541666666666, 95.29166666666667, 12.708333333333352, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 32.53125000000001, 10.666666666666643, 37.48611111111109, 28.55208333333333, 93.66666666666666, 12.083333333333353, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 31.218750000000007, 9.541666666666643, 36.770833333333314, 27.718749999999993, 92.04166666666666, 11.458333333333355, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 29.906250000000004, 8.416666666666643, 36.055555555555536, 26.885416666666657, 90.41666666666666, 10.833333333333355, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 28.593750000000004, 7.291666666666643, 35.34027777777776, 26.052083333333325, 88.79166666666666, 10.208333333333355, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 27.281250000000004, 6.166666666666643, 34.62499999999998, 25.21874999999999, 87.16666666666666, 9.583333333333355, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 25.96875, 5.041666666666643, 33.9097222222222, 24.385416666666654, 85.54166666666666, 8.958333333333355, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 24.65625, 3.916666666666643, 33.19444444444442, 23.55208333333332, 83.91666666666666, 8.333333333333357, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 23.34375, 2.791666666666643, 32.47916666666664, 22.71874999999999, 82.29166666666666, 7.708333333333356, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 22.03125, 1.666666666666643, 31.763888888888864, 21.885416666666657, 80.66666666666666, 7.083333333333355, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 20.71875, 0.541666666666643, 31.048611111111086, 21.05208333333332, 79.04166666666666, 6.458333333333354, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 19.40625, 0.0, 30.333333333333307, 20.218749999999986, 77.41666666666666, 5.833333333333356, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 18.09375, 0.0, 29.61805555555553, 19.385416666666654, 75.79166666666666, 5.208333333333358, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 16.78125, 0.0, 28.90277777777775, 18.55208333333332, 74.16666666666666, 4.583333333333359, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 15.46875, 0.0, 28.18749999999997, 17.71874999999999, 72.54166666666666, 3.9583333333333597, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 14.15625, 0.0, 27.472222222222193, 16.885416666666657, 70.91666666666666, 3.3333333333333606, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 12.843749999999998, 0.0, 26.756944444444414, 16.052083333333325, 69.29166666666666, 2.7083333333333615, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 11.531249999999998, 0.0, 26.041666666666636, 15.218749999999993, 67.66666666666666, 2.0833333333333623, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 10.218749999999998, 0.0, 25.326388888888857, 14.385416666666659, 66.04166666666666, 1.458333333333363, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 8.906249999999998, 0.0, 24.61111111111108, 13.552083333333327, 64.41666666666666, 0.8333333333333623, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 7.593749999999998, 0.0, 23.8958333333333, 12.718749999999993, 62.79166666666665, 0.20833333333336168, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 6.281249999999998, 0.0, 23.180555555555525, 11.88541666666666, 61.16666666666664, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 4.968749999999998, 0.0, 22.465277777777747, 11.052083333333329, 59.541666666666636, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 3.6562499999999982, 0.0, 21.74999999999997, 10.218749999999996, 57.916666666666636, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 2.3437499999999982, 0.0, 21.034722222222193, 9.385416666666663, 56.29166666666663, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 1.0312499999999982, 0.0, 20.319444444444414, 8.552083333333329, 54.66666666666662, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 19.60416666666664, 7.718749999999996, 53.04166666666662, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 18.888888888888864, 6.8854166666666625, 51.416666666666615, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 18.173611111111086, 6.052083333333329, 49.791666666666615, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 17.458333333333307, 5.218749999999995, 48.166666666666615, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 16.74305555555553, 4.385416666666662, 46.541666666666615, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 16.02777777777775, 3.5520833333333286, 44.916666666666615, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 15.312499999999972, 2.7187499999999956, 43.291666666666615, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 14.597222222222193, 1.8854166666666625, 41.666666666666615, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 13.881944444444414, 1.052083333333329, 40.041666666666615, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 13.166666666666636, 0.2187499999999959, 38.41666666666662, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 12.451388888888857, 0.0, 36.79166666666662, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 11.736111111111079, 0.0, 35.16666666666662, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 11.0208333333333, 0.0, 33.54166666666663, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 10.305555555555522, 0.0, 31.91666666666663, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 9.590277777777743, 0.0, 30.291666666666632, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 8.874999999999964, 0.0, 28.666666666666632, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 8.159722222222186, 0.0, 27.041666666666632, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 7.444444444444408, 0.0, 25.416666666666636, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 6.7291666666666305, 0.0, 23.791666666666632, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 6.013888888888853, 0.0, 22.16666666666663, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 5.298611111111075, 0.0, 20.54166666666663, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 4.5833333333332975, 0.0, 18.91666666666663, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 3.86805555555552, 0.0, 17.29166666666663, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 3.1527777777777413, 0.0, 15.666666666666629, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 2.4374999999999636, 0.0, 14.041666666666629, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.722222222222186, 0.0, 12.416666666666627, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0069444444444082, 0.0, 10.791666666666627, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.29166666666663044, 0.0, 9.166666666666627, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.021212121212118573, 0.0, 7.541666666666627, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0015426997245177147, 0.0, 5.916666666666627, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.000112196343601288, 0.0, 4.291666666666627, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 8.159734080093683e-6, 0.0, 2.666666666666627, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 5.934352058249947e-7, 0.0, 1.041666666666627, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

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
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.7  ])
  first_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
       Int64[ 0 0 0 0 0 0  0 0 0 0 0 0  0 0  0  0 0 0 0 0
              0 0 0 7 7 7  0 6 6 0 6 6  6 6  0  0 5 5 5 0
              0 4 0 7 7 7 11 6 6 6 6 6  6 6 12 12 5 5 5 0
              0 4 0 7 7 7  0 6 6 6 6 6  6 6  0  0 5 5 5 0
              0 4 0 0 0 0  0 0 0 0 0 6  6 6  0  0 5 5 5 0
              0 4 0 0 0 0  0 0 0 0 0 6  6 6  0  0 0 9 0 0
              0 4 0 0 0 0  0 0 0 0 0 0 10 0  0  0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3  0  0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3  0  0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3  0  0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3  0  0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3  0  0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3  0  0 2 2 2 0
              0 0 0 0 0 0  0 0 0 0 0 0  0 0  0  0 0 0 0 0
              0 1 1 1 1 1  1 1 1 1 0 0  0 0  0  0 0 0 0 0
              0 1 1 1 1 1  1 1 1 1 1 1  1 0  0  0 0 0 0 0
              0 1 1 1 1 1  1 1 1 1 1 1  1 0  0  0 0 0 0 0
              0 1 1 1 1 1  1 1 1 1 1 1  1 0  0  0 0 0 0 0
              0 0 0 0 0 0  0 0 0 0 0 0  0 0  0  0 0 0 0 0
              0 0 0 0 0 0  0 0 0 0 0 0  0 0  0  0 0 0 0 0 ])
  first_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 3 3 3 0 3 3 0 3 3 3 3 0 0 3 3 3 0
             0 3 0 3 3 3 3 3 3 3 3 3 3 3 2 2 3 3 3 0
             0 3 0 3 3 3 0 3 3 3 3 3 3 3 0 0 3 3 3 0
             0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 0 3 0 0
             0 3 0 0 0 0 0 0 0 0 0 0 3 0 0 0 3 3 3 0
             0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 3 3 3 3 3 3 3 3 3 0 0 0 0 0 0 0 0 0 0
             0 3 3 3 3 3 3 3 3 3 3 3 3 0 0 0 0 0 0 0
             0 3 3 3 3 3 3 3 3 3 3 3 3 0 0 0 0 0 0 0
             0 3 3 3 3 3 3 3 3 3 3 3 3 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0   ])
  first_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
      Float64[  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0 485.0 485.0 485.0   0.0 485.0 485.0   0.0 485.0 485.0 485.0 485.0   0.0   0.0 485.0 485.0 485.0   0.0
                0.0 167.0   0.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0   0.0
                0.0 167.0   0.0 485.0 485.0 485.0   0.0 485.0 485.0 485.0 485.0 485.0 485.0 485.0   0.0   0.0 485.0 485.0 485.0   0.0
                0.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 485.0 485.0 485.0   0.0   0.0 485.0 485.0 485.0   0.0
                0.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 485.0 485.0 485.0   0.0   0.0   0.0 485.0   0.0   0.0
                0.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 485.0   0.0   0.0   0.0 485.0 485.0 485.0   0.0
                0.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 485.0 485.0 485.0   0.0   0.0 485.0 485.0 485.0   0.0
                0.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 485.0 485.0 485.0   0.0   0.0 485.0 485.0 485.0   0.0
                0.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 485.0 485.0 485.0   0.0   0.0 485.0 485.0 485.0   0.0
                0.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 485.0 485.0 485.0   0.0   0.0 485.0 485.0 485.0   0.0
                0.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 485.0 485.0 485.0   0.0   0.0 485.0 485.0 485.0   0.0
                0.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 485.0 485.0 485.0   0.0   0.0 485.0 485.0 485.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0 167.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   ])
  first_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.3611111111 0.5625 0.4166666667
                 0.1458333333 0.296875 0.4375
                 0.5555555556 0.5208333333 0.0])
  first_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 13 27 15
                7 19 21
               20 25 0])
  first_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 36 48 36
               48 64 48
               36 48 36  ])
  first_intermediate_expected_lake_volumes::Array{Float64} = Float64[45.0, 63.0, 18.0, 66.0, 48.0, 104.0,
                                                                     27.0, 56.0, 34.0, 45.0, 55.0, 91.0]
  # first_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[ 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.0 0.0 4.0 4.0 4.0 0.0 6.0 6.0 0.0 6.0 6.0 6.0 6.0 0.0 0.0 5.0 5.0 5.0 0.0
  #                0.0 7.0 0.0 4.0 4.0 4.0 1.0 6.0 6.0 6.0 6.0 6.0 6.0 6.0 0.0 0.0 5.0 5.0 5.0 0.0
  #                0.0 7.0 0.0 4.0 4.0 4.0 0.0 6.0 6.0 6.0 6.0 6.0 6.0 6.0 0.0 0.0 5.0 5.0 5.0 0.0
  #                0.0 7.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 6.0 6.0 6.0 0.0 0.0 5.0 5.0 5.0 0.0
  #                0.0 7.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 6.0 6.0 6.0 0.0 0.0 0.0 1.0 0.0 0.0
  #                0.0 7.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 4.0 4.0 4.0 0.0
  #                0.0 7.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 3.0 0.0 0.0 4.0 4.0 4.0 0.0
  #                0.0 7.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 3.0 0.0 0.0 4.0 4.0 4.0 0.0
  #                0.0 7.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 3.0 0.0 0.0 4.0 4.0 4.0 0.0
  #                0.0 7.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 3.0 0.0 0.0 4.0 4.0 4.0 0.0
  #                0.0 7.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 3.0 0.0 0.0 4.0 4.0 4.0 0.0
  #                0.0 7.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 3.0 3.0 0.0 0.0 4.0 4.0 4.0 0.0
  #                0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ])

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
       Int64[ 0 0 0 0 0 0  0 0 0 0 0 0  0 0  0  0 0 0 0 0
              0 0 0 7 7 7  0 6 6 0 6 6  6 6  0  0 5 5 5 0
              0 4 0 7 7 7 11 6 6 6 6 6  6 6 12 12 5 5 5 0
              0 4 0 7 7 7  0 6 6 6 6 6  6 6  0  0 5 5 5 0
              0 4 0 0 0 0  0 0 0 0 0 6  6 6  0  0 5 5 5 0
              0 4 0 0 0 0  0 0 0 0 0 6  6 6  0  0 0 9 0 0
              0 4 0 0 0 0  0 0 0 0 0 0 10 0  0  0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3  0  0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3  0  0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3  0  0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3  0  0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3  0  0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3  0  0 2 2 2 0
              0 0 0 0 0 0  0 0 0 0 0 0  0 0  0  0 0 0 0 0
              0 1 1 1 1 1  1 1 1 1 0 0  0 0  0  0 0 0 0 0
              0 1 1 1 1 1  1 1 1 1 1 1  1 0  0  0 0 0 0 0
              0 1 1 1 1 1  1 1 1 1 1 1  1 0  0  0 0 0 0 0
              0 1 1 1 1 1  1 1 1 1 1 1  1 0  0  0 0 0 0 0
              0 0 0 0 0 0  0 0 0 0 0 0  0 0  0  0 0 0 0 0
              0 0 0 0 0 0  0 0 0 0 0 0  0 0  0  0 0 0 0 0 ])
  second_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 3 3 3 0 3 3 0 3 3 3 3 0 0 3 3 3 0
             0 3 0 3 3 3 3 3 3 3 3 3 3 3 1 1 3 3 3 0
             0 3 0 3 3 3 0 3 3 3 3 3 3 3 0 0 3 3 3 0
             0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 0 3 0 0
             0 3 0 0 0 0 0 0 0 0 0 0 3 0 0 0 3 3 3 0
             0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 3 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 3 3 3 3 3 3 3 3 3 0 0 0 0 0 0 0 0 0 0
             0 3 3 3 3 3 3 3 3 3 3 3 3 0 0 0 0 0 0 0
             0 3 3 3 3 3 3 3 3 3 3 3 3 0 0 0 0 0 0 0
             0 3 3 3 3 3 3 3 3 3 3 3 3 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  second_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
      Float64[ 0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
               0.0   0.0   0.0 397.19 397.19 397.19   0.00 397.19 397.19   0.00 397.19 397.19 397.19 397.19   0.0   0.0  397.19 397.19 397.19   0.0
               0.0 112.0   0.0 397.19 397.19 397.19 397.19 397.19 397.19 397.19 397.19 397.19 397.19 397.19 397.19 397.19 397.19 397.19 397.19   0.0
               0.0 112.0   0.0 397.19 397.19 397.19   0.00 397.19 397.19 397.19 397.19 397.19 397.19 397.19  0.0   0.0  397.19 397.19 397.19   0.0
               0.0 112.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 397.19 397.19 397.19   0.0    0.0 397.19 397.19 397.19   0.0
               0.0 112.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 397.19 397.19 397.19   0.0    0.0   0.0  397.19    0.0   0.0
               0.0 112.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 397.19    0.0    0.0    0.0 397.19 397.19 397.19   0.0
               0.0 112.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 397.19 397.19 397.19   0.0    0.0 397.19 397.19 397.19   0.0
               0.0 112.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 397.19 397.19 397.19   0.0    0.0 397.19 397.19 397.19   0.0
               0.0 112.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 397.19 397.19 397.19   0.0    0.0 397.19 397.19 397.19   0.0
               0.0 112.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 397.19 397.19 397.19   0.0    0.0 397.19 397.19 397.19   0.0
               0.0 112.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 397.19 397.19 397.19   0.0    0.0 397.19 397.19 397.19   0.0
               0.0 112.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 397.19 397.19 397.19   0.0    0.0 397.19 397.19 397.19   0.0
               0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0   0.0
               0.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0   0.0   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0   0.0
               0.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0  112.0    0.0    0.0    0.0    0.0    0.0    0.0   0.0
               0.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0  112.0    0.0    0.0    0.0    0.0    0.0    0.0   0.0
               0.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0 112.0  112.0    0.0    0.0    0.0    0.0    0.0    0.0   0.0
               0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0   0.0
               0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0   0.0  ])
  second_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.3611111111 0.5625 0.4166666667
                 0.1458333333 0.296875 0.4375
                 0.5555555556 0.5208333333 0.0 ])
  second_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 13 27 15
                7 19 21
               20 25 0])
  second_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 36 48 36
               48 64 48
               36 48 36  ])
  second_intermediate_expected_lake_volumes::Array{Float64} = Float64[45.0, 63.0, 18.0, 66.0, 48.0, 104.0, 27.0, 1.0,
                                                                      34.0, 45.0, 55.0, 3.1875]
  # second_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[ 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
  #                0.0 0.00 0.00 4.04 4.04 4.04 0.00 6.04 6.04 0.00 6.04 6.04 6.04 6.04 0.00 0.00 5.04 5.04 5.04 0.00
  #                0.0 6.02 0.00 4.04 4.04 4.04 1.04 6.04 6.04 6.04 6.04 6.04 6.04 6.04 0.04 0.04 5.04 5.04 5.04 0.00
  #                0.0 6.02 0.00 4.04 4.04 4.04 0.00 6.04 6.04 6.04 6.04 6.04 6.04 6.04 0.00 0.00 5.04 5.04 5.04 0.00
  #                0.0 6.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 6.04 6.04 6.04 0.00 0.00 5.04 5.04 5.04 0.00
  #                0.0 6.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 6.04 6.04 6.04 0.00 0.00 0.00 1.04 0.00 0.00
  #                0.0 6.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 2.04 0.00 0.00 0.00 4.04 4.04 4.04 0.00
  #                0.0 6.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.04 3.04 3.04 0.00 0.00 4.04 4.04 4.04 0.00
  #                0.0 6.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.04 3.04 3.04 0.00 0.00 4.04 4.04 4.04 0.00
  #                0.0 6.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.04 3.04 3.04 0.00 0.00 4.04 4.04 4.04 0.00
  #                0.0 6.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.04 3.04 3.04 0.00 0.00 4.04 4.04 4.04 0.00
  #                0.0 6.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.04 3.04 3.04 0.00 0.00 4.04 4.04 4.04 0.00
  #                0.0 6.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.04 3.04 3.04 0.00 0.00 4.04 4.04 4.04 0.00
  #                0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
  #                0.0 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
  #                0.0 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00
  #                0.0 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00
  #                0.0 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 1.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00
  #                0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
  #                0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 ])

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
       Int64[ 0 0 0 0 0 0  0 0 0 0 0 0  0 0 0 0 0 0 0 0
              0 0 0 7 7 7  0 6 6 0 6 6  6 6 0 0 5 5 5 0
              0 4 0 7 7 7 11 6 6 6 6 6  6 6 0 0 5 5 5 0
              0 4 0 7 7 7  0 6 6 6 6 6  6 6 0 0 5 5 5 0
              0 4 0 0 0 0  0 0 0 0 0 6  6 6 0 0 5 5 5 0
              0 4 0 0 0 0  0 0 0 0 0 6  6 6 0 0 0 9 0 0
              0 4 0 0 0 0  0 0 0 0 0 0 10 0 0 0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3 0 0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3 0 0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3 0 0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3 0 0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3 0 0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3 0 0 2 2 2 0
              0 0 0 0 0 0  0 0 0 0 0 0  0 0 0 0 0 0 0 0
              0 1 1 1 1 1  1 1 1 1 0 0  0 0 0 0 0 0 0 0
              0 1 1 1 1 1  1 1 1 1 1 1  1 0 0 0 0 0 0 0
              0 1 1 1 1 1  1 1 1 1 1 1  1 0 0 0 0 0 0 0
              0 1 1 1 1 1  1 1 1 1 1 1  1 0 0 0 0 0 0 0
              0 0 0 0 0 0  0 0 0 0 0 0  0 0 0 0 0 0 0 0
              0 0 0 0 0 0  0 0 0 0 0 0  0 0 0 0 0 0 0 0 ])
  third_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 3 3 3 0 3 3 0 3 3 3 3 0 0 3 3 3 0
             0 1 0 3 3 3 1 3 3 3 3 3 3 3 0 0 3 3 3 0
             0 1 0 3 3 3 0 3 3 3 3 3 3 3 0 0 3 3 3 0
             0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 0 1 0 0
             0 1 0 0 0 0 0 0 0 0 0 0 3 0 0 0 3 3 3 0
             0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  third_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
      Float64[     0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0   0.0
                   0.0  0.0   0.0 247.67 247.67 247.67   0.0 247.67 247.67   0.00 247.67 247.67 247.67 247.67   0.0   0.0 143.67 143.67 143.67   0.0
                   0.0 64.67  0.0 247.67 247.67 247.67 247.67 247.67 247.67 247.67 247.67 247.67 247.67 247.67   0.0   0.0 143.67 143.67 143.67   0.0
                   0.0 64.67  0.0 247.67 247.67 247.67   0.0 247.67 247.67 247.67 247.67 247.67 247.67 247.67   0.0   0.0 143.67 143.67 143.67   0.0
                   0.0 64.67  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 247.67 247.67 247.67   0.0   0.0 143.67 143.67 143.67   0.0
                   0.0 64.67  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 247.67 247.67 247.67   0.0   0.0   0.0 143.67   0.0     0.0
                   0.0 64.67  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 247.67   0.0     0.0   0.0 143.67 143.67 143.67   0.0
                   0.0 64.67  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 247.67 247.67 247.67   0.0   0.0 143.67 143.67 143.67   0.0
                   0.0 64.67  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 247.67 247.67 247.67   0.0   0.0 143.67 143.67 143.67   0.0
                   0.0 64.67  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 247.67 247.67 247.67   0.0   0.0 143.67 143.67 143.67   0.0
                   0.0 64.67  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 247.67 247.67 247.67   0.0   0.0 143.67 143.67 143.67   0.0
                   0.0 64.67  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 247.67 247.67 247.67   0.0   0.0 143.67 143.67 143.67   0.0
                   0.0 64.67  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0 247.67 247.67 247.67   0.0   0.0 143.67 143.67 143.67   0.0
                   0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                   0.0 43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67   0.0   0.0   0.0      0.0   0.0   0.0   0.0   0.0   0.0   0.0
                   0.0 43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                   0.0 43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                   0.0 43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67  43.67   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                   0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                   0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   ])
  third_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.3611111111 0.5625   0.3611111111
                 0.1458333333 0.296875 0.4375
                 0.5555555556 0.5208333333 0.0 ])
  third_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 13 27 13
                7 19 21
               20 25 0])
  third_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 36 48 36
               48 64 48
               36 48 36  ])
  third_intermediate_expected_lake_volumes::Array{Float64} = Float64[43.666666666666686, 63.0, 18.0, 64.66666666666669,
                                                                     48.0, 104.0, 27.0, 0.0, 32.66666666666666, 45.0,
                                                                     53.66666666666666, 0.0]
  # third_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[ 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.00 4.00 4.00 4.00 0.00 6.00 6.00 0.00 6.00 6.00 6.00 6.00 0.0 0.0 4.93 4.93 4.93 0.0
  #                0.0 5.88 0.00 4.00 4.00 4.00 1.00 6.00 6.00 6.00 6.00 6.00 6.00 6.00 0.0 0.0 4.93 4.93 4.93 0.0
  #                0.0 5.88 0.00 4.00 4.00 4.00 0.00 6.00 6.00 6.00 6.00 6.00 6.00 6.00 0.0 0.0 4.93 4.93 4.93 0.0
  #                0.0 5.88 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 6.00 6.00 6.00 0.0 0.0 4.93 4.93 4.93 0.0
  #                0.0 5.88 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 6.00 6.00 6.00 0.0 0.0 0.00 0.93 0.00 0.0
  #                0.0 5.88 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 2.00 0.00 0.0 0.0 3.93 3.93 3.93 0.0
  #                0.0 5.88 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.00 3.00 3.00 0.0 0.0 3.93 3.93 3.93 0.0
  #                0.0 5.88 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.00 3.00 3.00 0.0 0.0 3.93 3.93 3.93 0.0
  #                0.0 5.88 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.00 3.00 3.00 0.0 0.0 3.93 3.93 3.93 0.0
  #                0.0 5.88 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.00 3.00 3.00 0.0 0.0 3.93 3.93 3.93 0.0
  #                0.0 5.88 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.00 3.00 3.00 0.0 0.0 3.93 3.93 3.93 0.0
  #                0.0 5.88 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.00 3.00 3.00 0.0 0.0 3.93 3.93 3.93 0.0
  #                0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.97 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0  0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0  ])

  fourth_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  fourth_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  fourth_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  fourth_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
       Int64[ 0 0 0 0 0 0  0 0 0 0 0 0  0 0 0 0 0 0 0 0
              0 0 0 7 7 7  0 6 6 0 6 6  6 6 0 0 5 5 5 0
              0 4 0 7 7 7 11 6 6 6 6 6  6 6 0 0 5 5 5 0
              0 4 0 7 7 7  0 6 6 6 6 6  6 6 0 0 5 5 5 0
              0 4 0 0 0 0  0 0 0 0 0 6  6 6 0 0 5 5 5 0
              0 4 0 0 0 0  0 0 0 0 0 6  6 6 0 0 0 9 0 0
              0 4 0 0 0 0  0 0 0 0 0 0 10 0 0 0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3 0 0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3 0 0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3 0 0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3 0 0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3 0 0 2 2 2 0
              0 4 0 0 0 0  0 0 0 0 0 3  3 3 0 0 2 2 2 0
              0 0 0 0 0 0  0 0 0 0 0 0  0 0 0 0 0 0 0 0
              0 1 1 1 1 1  1 1 1 1 0 0  0 0 0 0 0 0 0 0
              0 1 1 1 1 1  1 1 1 1 1 1  1 0 0 0 0 0 0 0
              0 1 1 1 1 1  1 1 1 1 1 1  1 0 0 0 0 0 0 0
              0 1 1 1 1 1  1 1 1 1 1 1  1 0 0 0 0 0 0 0
              0 0 0 0 0 0  0 0 0 0 0 0  0 0 0 0 0 0 0 0
              0 0 0 0 0 0  0 0 0 0 0 0  0 0 0 0 0 0 0 0 ])
  fourth_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 3 3 3 0 3 3 0 3 3 3 3 0 0 3 3 3 0
             0 1 0 3 3 3 1 3 3 3 3 3 3 3 0 0 3 3 3 0
             0 1 0 3 3 3 0 3 3 3 3 3 3 3 0 0 3 3 3 0
             0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 0 1 0 0
             0 1 0 0 0 0 0 0 0 0 0 0 3 0 0 0 3 3 3 0
             0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 1 0 0 0 0 0 0 0 0 0 3 3 3 0 0 3 3 3 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
             0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0   ])
  fourth_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
      Float64[     0.0   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
                   0.0   0.0    0.0  244.17 244.17 244.17   0.0  244.17 244.17   0.0  244.17 244.17 244.17 244.17   0.0    0.0  141.45 141.45 141.45   0.0
                   0.0  63.95   0.0  244.17 244.17 244.17 244.17 244.17 244.17 244.17 244.17 244.17 244.17 244.17   0.0    0.0  141.45 141.45 141.45   0.0
                   0.0  63.95   0.0  244.17 244.17 244.17   0.00 244.17 244.17 244.17 244.17 244.17 244.17 244.17   0.0    0.0  141.45 141.45 141.45   0.0
                   0.0  63.95   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  244.17 244.17 244.17   0.0    0.0  141.45 141.45 141.45   0.0
                   0.0  63.95   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  244.17 244.17 244.17   0.0    0.0    0.0  141.45   0.0    0.0
                   0.0  63.95   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  244.17   0.0    0.0    0.0  141.45 141.45 141.45   0.0
                   0.0  63.95   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  244.17 244.17 244.17   0.0    0.0  141.45 141.45 141.45   0.0
                   0.0  63.95   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  244.17 244.17 244.17   0.0    0.0  141.45 141.45 141.45   0.0
                   0.0  63.95   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  244.17 244.17 244.17   0.0    0.0  141.45 141.45 141.45   0.0
                   0.0  63.95   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  244.17 244.17 244.17   0.0    0.0  141.45 141.45 141.45   0.0
                   0.0  63.95   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  244.17 244.17 244.17   0.0    0.0  141.45 141.45 141.45   0.0
                   0.0  63.95   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  244.17 244.17 244.17   0.0    0.0  141.45 141.45 141.45   0.0
                   0.0   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
                   0.0  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
                   0.0  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72   0.0    0.0    0.0    0.0    0.0    0.0    0.0
                   0.0  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72   0.0    0.0    0.0    0.0    0.0    0.0    0.0
                   0.0  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72  40.72   0.0    0.0    0.0    0.0    0.0    0.0    0.0
                   0.0   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
                   0.0   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0  ])
  fourth_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.3611111111 0.5625   0.3611111111
                 0.1458333333 0.296875 0.4375
                 0.5555555556 0.5208333333 0.0 ])
  fourth_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 13 27 13
                7 19 21
               20 25 0])
  fourth_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 36 48 36
               48 64 48
               36 48 36  ])
  fourth_intermediate_expected_lake_volumes::Array{Float64} = Float64[40.7152777777778, 63.0, 18.0, 63.95138888888891,
                                                                      48.0, 104.0, 27.0, 0.0, 30.45138888888888,
                                                                      45.0, 50.16666666666666, 0.0]
  # fourth_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[ 0.0 0.00 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.0 3.93 3.93 3.93 0.00 5.93 5.93 0.00 5.93 5.93 5.93 5.93 0.0 0.0 4.86 4.86 4.86 0.0
  #                0.0 5.81 0.0 3.93 3.93 3.93 0.93 5.93 5.93 5.93 5.93 5.93 5.93 5.93 0.0 0.0 4.86 4.86 4.86 0.0
  #                0.0 5.81 0.0 3.93 3.93 3.93 0.00 5.93 5.93 5.93 5.93 5.93 5.93 5.93 0.0 0.0 4.86 4.86 4.86 0.0
  #                0.0 5.81 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 5.93 5.93 5.93 0.0 0.0 4.86 4.86 4.86 0.0
  #                0.0 5.81 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 5.93 5.93 5.93 0.0 0.0 0.00 0.86 0.00 0.0
  #                0.0 5.81 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.93 0.00 0.0 0.0 3.86 3.86 3.86 0.0
  #                0.0 5.81 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 2.93 2.93 2.93 0.0 0.0 3.86 3.86 3.86 0.0
  #                0.0 5.81 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 2.93 2.93 2.93 0.0 0.0 3.86 3.86 3.86 0.0
  #                0.0 5.81 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 2.93 2.93 2.93 0.0 0.0 3.86 3.86 3.86 0.0
  #                0.0 5.81 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 2.93 2.93 2.93 0.0 0.0 3.86 3.86 3.86 0.0
  #                0.0 5.81 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 2.93 2.93 2.93 0.0 0.0 3.86 3.86 3.86 0.0
  #                0.0 5.81 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 2.93 2.93 2.93 0.0 0.0 3.86 3.86 3.86 0.0
  #                0.0 0.00 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.90 0.9 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.90 0.9 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.90 0.9 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.90 0.9 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.90 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0  ])

  fifth_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  fifth_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  fifth_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  fifth_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 7 0 0 0 6 6 0 6 6 6 6 0 0 5 5 5 0
              0 4 0 0 0 0 0 6 6 6 6 6 6 6 0 0 5 5 5 0
              0 4 0 0 0 0 0 6 6 6 6 6 6 6 0 0 5 5 5 0
              0 4 0 0 0 0 0 0 0 0 0 6 6 6 0 0 5 5 5 0
              0 4 0 0 0 0 0 0 0 0 0 6 6 6 0 0 0 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 3 0 0 0 0 2 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  fifth_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 0 0 1 0 0 0 1 1 0 1 1 1 1 0 0 1 1 1 0
                0 1 0 0 0 0 0 1 1 1 1 1 1 1 0 0 1 1 1 0
                0 1 0 0 0 0 0 1 1 1 1 1 1 1 0 0 1 1 1 0
                0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
                0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0
                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0
                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  fifth_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
      Float64[  0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
                0.0  0.0  0.0 0.0 0.0 0.0 0.0 46.54 46.54  0.0  46.54 46.54 46.54 46.54 0.0 0.0 4.39 4.39 4.39 0.0
                0.0 16.74 0.0 0.0 0.0 0.0 0.0 46.54 46.54 46.54 46.54 46.54 46.54 46.54 0.0 0.0 4.39 4.39 4.39 0.0
                0.0 16.74 0.0 0.0 0.0 0.0 0.0 46.54 46.54 46.54 46.54 46.54 46.54 46.54 0.0 0.0 4.39 4.39 4.39 0.0
                0.0 16.74 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0  46.54 46.54 46.54 0.0 0.0 4.39 4.39 4.39 0.0
                0.0 16.74 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0  46.54 46.54 46.54 0.0 0.0 0.0  0.0  0.0  0.0
                0.0 16.74 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
                0.0 16.74 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
                0.0 16.74 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
                0.0 16.74 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
                0.0 16.74 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
                0.0 16.74 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
                0.0 16.74 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0
                0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0  0.0  0.0  0.0 ])
  fifth_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.1388888889   0.5416666667   0.3333333333
                 0.1458333333   0.015625       0.02083333333
                 0.02777777778  0.0            0.0 ])
  fifth_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[  5 26 12
                7  1  1
                1  0  0])
  fifth_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 36 48 36
               48 64 48
               36 48 36  ])
  fifth_intermediate_expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 0.0, 16.74305555555553,
                                                                     4.385416666666662, 46.541666666666615,
                                                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  # fifth_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[ 0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 1.81 1.81 0.00 1.81 1.81 1.81 1.81 0.0 0.0 0.37 0.37 0.37 0.0
  #                0.0 1.52 0.0 0.0 0.0 0.0 0.0 1.81 1.81 1.81 1.81 1.81 1.81 1.81 0.0 0.0 0.37 0.37 0.37 0.0
  #                0.0 1.52 0.0 0.0 0.0 0.0 0.0 1.81 1.81 1.81 1.81 1.81 1.81 1.81 0.0 0.0 0.37 0.37 0.37 0.0
  #                0.0 1.52 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 1.81 1.81 1.81 0.0 0.0 0.37 0.37 0.37 0.0
  #                0.0 1.52 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 1.81 1.81 1.81 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 1.52 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 1.52 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 1.52 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 1.52 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 1.52 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 1.52 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 1.52 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0 ])

  sixth_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  sixth_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  sixth_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  sixth_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 7 0 0 0 6 6 0 6 6 6 6 0 0 5 5 5 0
              0 4 0 0 0 0 0 6 6 6 6 6 6 6 0 0 5 5 5 0
              0 4 0 0 0 0 0 6 6 6 6 6 6 6 0 0 5 5 5 0
              0 4 0 0 0 0 0 0 0 0 0 6 6 6 0 0 5 5 5 0
              0 4 0 0 0 0 0 0 0 0 0 6 6 6 0 0 0 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 3 0 0 0 0 2 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  sixth_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 1 0 0 0 1 1 0 1 1 1 1 0 0 1 1 1 0
             0 1 0 0 0 0 0 1 1 1 1 1 1 1 0 0 1 1 1 0
             0 1 0 0 0 0 0 1 1 1 1 1 1 1 0 0 1 1 1 0
             0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
             0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  sixth_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
      Float64[ 0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
               0.0  0.0  0.0 0.0 0.0 0.0 0.0 44.92 44.92  0.0  44.92 44.92 44.92 44.92 0.0 0.0 3.55  3.55 3.55 0.0
               0.0 16.03 0.0 0.0 0.0 0.0 0.0 44.92 44.92 44.92 44.92 44.92 44.92 44.92 0.0 0.0 3.55  3.55 3.55 0.0
               0.0 16.03 0.0 0.0 0.0 0.0 0.0 44.92 44.92 44.92 44.92 44.92 44.92 44.92 0.0 0.0 3.55  3.55 3.55 0.0
               0.0 16.03 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0  44.92 44.92 44.92 0.0 0.0 3.55  3.55 3.55 0.0
               0.0 16.03 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0  44.92 44.92 44.92 0.0 0.0 0.0   0.0  0.0  0.0
               0.0 16.03 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
               0.0 16.03 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
               0.0 16.03 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
               0.0 16.03 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
               0.0 16.03 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
               0.0 16.03 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
               0.0 16.03 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
               0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
               0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
               0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
               0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
               0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
               0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0
               0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0   0.0  0.0  0.0  ])
  sixth_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.1388888889 0.5416666667 0.3333333333
                 0.1458333333 0.015625 0.02083333333
                 0.02777777778  0.0    0.0 ])
  sixth_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 5 26 12
               7  1 1
               1  0 0 ])
  sixth_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 36 48 36
               48 64 48
               36 48 36  ])
  sixth_intermediate_expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 0.0, 16.02777777777775,
                                                                     3.5520833333333286, 44.916666666666615,
                                                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  # sixth_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[ 0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 1.75 1.75 0.00 1.75 1.75 1.75 1.75 0.0 0.0 0.3 0.3 0.3 0.0
  #                0.0 1.46 0.0 0.0 0.0 0.0 0.0 1.75 1.75 1.75 1.75 1.75 1.75 1.75 0.0 0.0 0.3 0.3 0.3 0.0
  #                0.0 1.46 0.0 0.0 0.0 0.0 0.0 1.75 1.75 1.75 1.75 1.75 1.75 1.75 0.0 0.0 0.3 0.3 0.3 0.0
  #                0.0 1.46 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 1.75 1.75 1.75 0.0 0.0 0.3 0.3 0.3 0.0
  #                0.0 1.46 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 1.75 1.75 1.75 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 1.46 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 1.46 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 1.46 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 1.46 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 1.46 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 1.46 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 1.46 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.0 0.0 0.0 0.0 ])
  seventh_intermediate_expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  seventh_intermediate_expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  seventh_intermediate_expected_water_to_hd::Field{Float64} =
    LatLonField{Float64}(hd_grid,
                         Float64[ 0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0
                                  0.0 0.0 0.0 0.0 ])
  seventh_intermediate_expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
       Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 7 0 0 0 6 6 0 6 6 6 6 0 0 5 5 5 0
              0 4 0 0 0 0 0 6 6 6 6 6 6 6 0 0 5 5 5 0
              0 4 0 0 0 0 0 6 6 6 6 6 6 6 0 0 5 5 5 0
              0 4 0 0 0 0 0 0 0 0 0 6 6 6 0 0 5 5 5 0
              0 4 0 0 0 0 0 0 0 0 0 6 6 6 0 0 0 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 3 0 0 0 0 2 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  seventh_intermediate_expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 1 0 0 0 1 1 0 1 1 1 1 0 0 1 1 1 0
             0 1 0 0 0 0 0 1 1 1 1 1 1 1 0 0 1 1 1 0
             0 1 0 0 0 0 0 1 1 1 1 1 1 1 0 0 1 1 1 0
             0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0
             0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  seventh_intermediate_expected_diagnostic_lake_volumes::Field{Float64} = LatLonField{Float64}(lake_grid,
      Float64[ 0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
               0.0  0.0  0.0 0.0 0.0 0.0 0.0 43.29 43.29  0.00 43.29 43.29 43.29 43.29 0.0 0.0 2.72 2.72 2.72 0.0
               0.0 15.31 0.0 0.0 0.0 0.0 0.0 43.29 43.29 43.29 43.29 43.29 43.29 43.29 0.0 0.0 2.72 2.72 2.72 0.0
               0.0 15.31 0.0 0.0 0.0 0.0 0.0 43.29 43.29 43.29 43.29 43.29 43.29 43.29 0.0 0.0 2.72 2.72 2.72 0.0
               0.0 15.31 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0  43.29 43.29 43.29 0.0 0.0 2.72 2.72 2.72 0.0
               0.0 15.31 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0  43.29 43.29 43.29 0.0 0.0 0.0 0.0 0.0 0.0
               0.0 15.31 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
               0.0 15.31 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
               0.0 15.31 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
               0.0 15.31 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
               0.0 15.31 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
               0.0 15.31 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
               0.0 15.31 0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
               0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
               0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
               0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
               0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
               0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
               0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0
               0.0  0.0  0.0 0.0 0.0 0.0 0.0  0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0 0.0 0.0 0.0 0.0 0.0  ])
  seventh_intermediate_expected_lake_fractions::Field{Float64} = LatLonField{Float64}(surface_model_grid,
        Float64[ 0.1388888889 0.5416666667   0.3333333333
                 0.1458333333 0.015625 0.02083333333
                 0.02777777778  0.0    0.0 ])
  seventh_intermediate_expected_number_lake_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[  5 26 12
                7  1  1
                1  0  0])
  seventh_intermediate_expected_number_fine_grid_cells::Field{Int64} = LatLonField{Int64}(surface_model_grid,
        Int64[ 36 48 36
               48 64 48
               36 48 36  ])
  seventh_intermediate_expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 0.0, 15.312499999999972,
                                                                       2.7187499999999956, 43.291666666666615,
                                                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  # seventh_intermediate_expected_true_lake_depths::Field{Float64} = LatLonField{Float64}(lake_grid,
  #       Float64[ 0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 1.69 1.69 0.00 1.69 1.69 1.69 1.69 0.0 0.0 0.23 0.23 0.23 0.0
  #                0.0 1.39 0.0 0.0 0.0 0.0 0.0 1.69 1.69 1.69 1.69 1.69 1.69 1.69 0.0 0.0 0.23 0.23 0.23 0.0
  #                0.0 1.39 0.0 0.0 0.0 0.0 0.0 1.69 1.69 1.69 1.69 1.69 1.69 1.69 0.0 0.0 0.23 0.23 0.23 0.0
  #                0.0 1.39 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 1.69 1.69 1.69 0.0 0.0 0.23 0.23 0.23 0.0
  #                0.0 1.39 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 1.69 1.69 1.69 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 1.39 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 1.39 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 1.39 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 1.39 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 1.39 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 1.39 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 1.39 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0
  #                0.0 0.00 0.0 0.0 0.0 0.0 0.0 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.0 0.0 0.00 0.00 0.00 0.0 ])
  lake_volumes_all_timesteps::Vector{Vector{Float64}} = []
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
                            write_output=false,return_output=true,
                            return_lake_volumes=true,
                            diagnostic_lake_volumes=lake_volumes_all_timesteps,
                            use_realistic_surface_coupling=true)
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
  @test isapprox(first_intermediate_expected_water_to_hd,lake_model_prognostics.water_to_hd,
                 rtol=0.0,atol=0.00001)
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
                            15,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=-1,
                            return_lake_volumes=true,
                            diagnostic_lake_volumes=lake_volumes_all_timesteps,
                            use_realistic_surface_coupling=true)
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
  @test isapprox(second_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
  @test isapprox(diagnostic_lake_volumes,second_intermediate_expected_diagnostic_lake_volumes,
                 rtol=0.0,atol=0.05)
  @test isapprox(second_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
  @test second_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test second_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test isapprox(second_intermediate_expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.1)
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            16,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=15,
                            return_lake_volumes=true,
                            diagnostic_lake_volumes=lake_volumes_all_timesteps,
                            use_realistic_surface_coupling=true)
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
  @test isapprox(third_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.01)
  @test isapprox(diagnostic_lake_volumes,third_intermediate_expected_diagnostic_lake_volumes,
                 rtol=0.0,atol=0.05)
  @test isapprox(third_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
  @test third_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test third_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test isapprox(third_intermediate_expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.1)
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            17,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=16,
                            return_lake_volumes=true,
                            diagnostic_lake_volumes=lake_volumes_all_timesteps,
                            use_realistic_surface_coupling=true)
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
  @test fourth_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(fourth_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test fourth_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test fourth_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test fourth_intermediate_expected_lake_types == lake_types
  @test isapprox(fourth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.01)
  @test isapprox(diagnostic_lake_volumes,fourth_intermediate_expected_diagnostic_lake_volumes,
                 rtol=0.0,atol=0.05)
  @test isapprox(fourth_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
  @test fourth_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test fourth_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test isapprox(fourth_intermediate_expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.1)
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            83,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=17,
                            return_lake_volumes=true,
                            diagnostic_lake_volumes=lake_volumes_all_timesteps,
                            use_realistic_surface_coupling=true)
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
  @test fifth_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(fifth_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test fifth_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test fifth_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test fifth_intermediate_expected_lake_types == lake_types
  @test isapprox(fifth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.01)
  @test isapprox(diagnostic_lake_volumes,fifth_intermediate_expected_diagnostic_lake_volumes,
                 rtol=0.0,atol=0.05)
  @test isapprox(fifth_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
  @test fifth_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test fifth_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test isapprox(fifth_intermediate_expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.1)
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            84,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=83,
                            return_lake_volumes=true,
                            diagnostic_lake_volumes=lake_volumes_all_timesteps,
                            use_realistic_surface_coupling=true)
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
  @test sixth_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(sixth_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test sixth_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test sixth_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test sixth_intermediate_expected_lake_types == lake_types
  @test isapprox(sixth_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.01)
  @test isapprox(diagnostic_lake_volumes,sixth_intermediate_expected_diagnostic_lake_volumes,
                 rtol=0.0,atol=0.05)
  @test isapprox(sixth_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
  @test sixth_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test sixth_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test isapprox(sixth_intermediate_expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.1)
  river_fields,lake_model_prognostics,lake_model_diagnostics =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            85,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=84,
                            return_lake_volumes=true,
                            diagnostic_lake_volumes=lake_volumes_all_timesteps,
                            use_realistic_surface_coupling=true)
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
  @test seventh_intermediate_expected_river_inflow == river_fields.river_inflow
  @test isapprox(seventh_intermediate_expected_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test seventh_intermediate_expected_water_to_hd == lake_model_prognostics.water_to_hd
  @test seventh_intermediate_expected_lake_numbers == lake_model_prognostics.lake_numbers
  @test seventh_intermediate_expected_lake_types == lake_types
  @test isapprox(seventh_intermediate_expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.01)
  @test isapprox(diagnostic_lake_volumes,seventh_intermediate_expected_diagnostic_lake_volumes,
                 rtol=0.0,atol=0.05)
  @test isapprox(seventh_intermediate_expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.00001)
  @test seventh_intermediate_expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test seventh_intermediate_expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test isapprox(seventh_intermediate_expected_true_lake_depths,lake_model_parameters.true_lake_depths,rtol=0.0,atol=0.1)
  river_fields,lake_model_prognostics,_ =
    drive_hd_and_lake_model(river_parameters,river_fields,
                            lake_model_parameters,
                            lake_model_prognostics,
                            drainages,runoffs,evaporations,
                            112,true,
                            initial_water_to_lake_centers,
                            initial_spillover_to_rivers,
                            print_timestep_results=false,
                            write_output=false,return_output=true,
                            lake_model_diagnostics=
                            lake_model_diagnostics,
                            forcing_timesteps_to_skip=85,
                            return_lake_volumes=true,
                            diagnostic_lake_volumes=lake_volumes_all_timesteps,
                            use_realistic_surface_coupling=true)
  for (lake_volumes_slice,expected_lake_volumes_slice) in zip(lake_volumes_all_timesteps,expected_lake_volumes_all_timesteps)
    @test isapprox(lake_volumes_slice,expected_lake_volumes_slice,
                   rtol=0.0,atol=0.01)
  end
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
  @test isapprox(expected_lake_volumes,lake_volumes,rtol=0.0,atol=0.00001)
  @test isapprox(diagnostic_lake_volumes,expected_diagnostic_lake_volumes,rtol=0.0,atol=0.00001)
  @test isapprox(expected_lake_fractions,lake_fractions,rtol=0.0,atol=0.01)
  @test expected_number_lake_cells == lake_model_prognostics.lake_cell_count
  @test expected_number_fine_grid_cells == lake_model_parameters.number_fine_grid_cells
  #@test expected_true_lake_depths == lake_model_parameters.true_lake_depths
end
