using Profile
using Test: @test, @testset
using L2HDDriverModule: drive_hd_and_lake_model
using HDModule: RiverParameters,RiverPrognosticFields
using GridModule: LatLonGrid
using FieldModule: Field,LatLonField,LatLonDirectionIndicators,set!,repeat
using CoordsModule: LatLonCoords
using L2LakeModelDefsModule: LatLonLakeModelParameters,LakeModelParameters
using L2LakeModelDefsModule: LakeModelSettings,LakeModelPrognostics

@testset "Lake model tests 1" begin
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
  lake_centers_list::Vector{CartesianIndex} = CartesianIndex[CartesianIndex(4,3)]
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
                        lake_centers_list,
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
    lake_model_prognostics::LakeModelPrognostics =
    drive_hd_and_lake_model(river_parameters,lake_model_parameters,
                            lake_parameters_as_array,
                            drainages,runoffs,evaporations,
                            1100,print_timestep_results=false,
                            write_output=false,return_output=true)
  print(lake_model_prognostics)
  # lake_fractions::Field{Float64} = calculate_lake_fraction_on_surface_grid(lake_parameters,lake_fields)
  # lake_effective_volumes::Field{Float64} = calculate_effective_lake_height_on_surface_grid(lake_parameters,
  #                                                                                          lake_fields)
  # lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
  # for i = 1:9
  #   for j = 1:9
  #     coords::LatLonCoords = LatLonCoords(i,j)
  #     lake_number::Int64 = lake_fields.lake_numbers(coords)
  #     if lake_number <= 0 continue end
  #     lake::Lake = lake_prognostics.lakes[lake_number]
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
  # for lake::Lake in lake_prognostics.lakes
  #   append!(lake_volumes,get_lake_variables(lake).lake_volume)
  # end
  # diagnostic_lake_volumes::Field{Float64} =
  #   calculate_diagnostic_lake_volumes_field(lake_parameters,lake_fields,
  #                                           lake_prognostics)
  # @test expected_river_inflow == river_fields.river_inflow
  # @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  # @test expected_water_to_hd    == lake_fields.water_to_hd
  # @test expected_lake_numbers == lake_fields.lake_numbers
  # @test expected_lake_types == lake_types
  # @test expected_lake_volumes == lake_volumes
  # @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  # @test expected_lake_fractions == lake_fractions
  # @test expected_number_lake_cells == lake_fields.number_lake_cells
  # @test expected_number_fine_grid_cells == lake_parameters.number_fine_grid_cells
  # @test expected_true_lake_depths == lake_fields.true_lake_depths
end

@testset "Lake model tests 2" begin
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
  lake_centers_list::Vector{CartesianIndex} =
    CartesianIndex[CartesianIndex(4,5),CartesianIndex(4,2),CartesianIndex(4,5)]
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
                        lake_centers_list,
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
    lake_model_prognostics::LakeModelPrognostics =
    drive_hd_and_lake_model(river_parameters,lake_model_parameters,
                            lake_parameters_as_array,
                            drainages,runoffs,evaporations,
                            1100,print_timestep_results=false,
                            write_output=false,return_output=true)
  println(lake_model_prognostics)
  println(lake_model_prognostics.lake_cell_count)
  println(lake_model_prognostics.water_to_hd)
  # lake_fractions::Field{Float64} = calculate_lake_fraction_on_surface_grid(lake_parameters,lake_fields)
  # lake_effective_volumes::Field{Float64} = calculate_effective_lake_height_on_surface_grid(lake_parameters,
  #                                                                                          lake_fields)
  # lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
  # for i = 1:9
  #   for j = 1:9
  #     coords::LatLonCoords = LatLonCoords(i,j)
  #     lake_number::Int64 = lake_fields.lake_numbers(coords)
  #     if lake_number <= 0 continue end
  #     lake::Lake = lake_prognostics.lakes[lake_number]
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
  # for lake::Lake in lake_prognostics.lakes
  #   append!(lake_volumes,get_lake_variables(lake).lake_volume)
  # end
  # diagnostic_lake_volumes::Field{Float64} =
  #   calculate_diagnostic_lake_volumes_field(lake_parameters,lake_fields,
  #                                           lake_prognostics)
  # @test expected_river_inflow == river_fields.river_inflow
  # @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  # @test expected_water_to_hd    == lake_fields.water_to_hd
  # @test expected_lake_numbers == lake_fields.lake_numbers
  # @test expected_lake_types == lake_types
  # @test expected_lake_volumes == lake_volumes
  # @test diagnostic_lake_volumes == expected_diagnostic_lake_volumes
  # @test expected_lake_fractions == lake_fractions
  # @test expected_number_lake_cells == lake_fields.number_lake_cells
  # @test expected_number_fine_grid_cells == lake_parameters.number_fine_grid_cells
  # @test expected_true_lake_depths == lake_fields.true_lake_depths
end
