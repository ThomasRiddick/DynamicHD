module HDTestModule

using Serialization
using Profile
using Test: @test, @testset
using HDDriverModule: drive_hd_model,drive_hd_and_lake_model
using HDModule: RiverParameters,RiverPrognosticFields
using GridModule: LatLonGrid, UnstructuredGrid
using CoordsModule: LatLonCoords, Generic1DCoords
using FieldModule: Field,LatLonField,LatLonDirectionIndicators,set!,repeat,add_offset,==,isequal
using FieldModule: UnstructuredField,UnstructuredDirectionIndicators
using LakeModule: GridSpecificLakeParameters,LakeParameters,LatLonLakeParameters,LakeFields
using LakeModule: UnstructuredLakeParameters
using LakeModule: LakePrognostics,Lake,FillingLake,OverflowingLake,SubsumedLake
using LakeModule: get_lake_variables
using MergeTypesModule

@testset "HD model tests" begin
  grid = LatLonGrid(4,4,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(grid,
                                                Int64[ 2 2 2 2
                                                       4 6 2 2
                                                       6 6 0 4
                                                       9 8 8 7 ] ))
  river_reservoir_nums = LatLonField{Int64}(grid,5)
  overland_reservoir_nums = LatLonField{Int64}(grid,1)
  base_reservoir_nums = LatLonField{Int64}(grid,1)
  river_retention_coefficients = LatLonField{Float64}(grid,0.7)
  overland_retention_coefficients = LatLonField{Float64}(grid,0.5)
  base_retention_coefficients = LatLonField{Float64}(grid,0.1)
  landsea_mask = LatLonField{Bool}(grid,fill(false,4,4))
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
                                     grid)
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,Float64[ 1.0 1.0 1.0 1.0
                                                                                 1.0 1.0 1.0 1.0
                                                                                 1.0 1.0 0.0 1.0
                                                                                 1.0 1.0 1.0 1.0 ])
  drainages::Array{Field{Float64},1} = repeat(drainage,200)
  runoffs::Array{Field{Float64},1} = deepcopy(drainages)
  evaporation::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,200)
  expected_water_to_ocean::Field{Float64} =
    LatLonField{Float64}(river_parameters.grid,Float64[ 0.0 0.0 0.0 0.0
                                                        0.0 0.0 0.0 0.0
                                                        0.0 0.0 30.0 0.0
                                                        0.0 0.0 0.0 0.0 ])

  expected_river_inflow::Field{Float64} =
    LatLonField{Float64}(river_parameters.grid,Float64[ 0.0 0.0 0.0 0.0
                                                        2.0 2.0 6.0 6.0
                                                        0.0 6.0 0.0 8.0
                                                        0.0 0.0 0.0 0.0 ])

  @time water_to_ocean::Field{Float64},river_inflow::Field{Float64} =
      drive_hd_model(river_parameters,drainages,runoffs,evaporations,200,
                     print_timestep_results=false,write_output=false,
                     return_output=true)
  # Profile.clear()
  # Profile.print()
  # Profile.init(delay=0.01)
  # function timing(river_parameters)
  #   for i in 1:50000
  #     drainagesl::Array{Field{Float64},1} = repeat(drainage,20)
  #     runoffsl::Array{Field{Float64},1} = deepcopy(drainages)
  #     drive_hd_model(river_parameters,drainagesl,runoffsl,20,print_timestep_results=false)
  #   end
  # end
  # @time timing(river_parameters)
  # r = Profile.retrieve();
  # f = open("/Users/thomasriddick/Downloads/profile.bin", "w")
  # Serialization.serialize(f, r)
  # close(f)
  @test expected_water_to_ocean == water_to_ocean
  @test expected_river_inflow   == river_inflow
end

@testset "Lake model tests 1" begin
  grid = LatLonGrid(3,3,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(grid,
                                                Int64[-2 6 2
                                                       8 7 2
                                                       8 7 0 ] ))
  river_reservoir_nums = LatLonField{Int64}(grid,5)
  overland_reservoir_nums = LatLonField{Int64}(grid,1)
  base_reservoir_nums = LatLonField{Int64}(grid,1)
  river_retention_coefficients = LatLonField{Float64}(grid,0.7)
  overland_retention_coefficients = LatLonField{Float64}(grid,0.5)
  base_retention_coefficients = LatLonField{Float64}(grid,0.1)
  landsea_mask = LatLonField{Bool}(grid,fill(false,3,3))
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
                                     grid)
  lake_grid = LatLonGrid(9,9,true)
  lake_centers::Field{Bool} = LatLonField{Bool}(lake_grid,
                                                Bool[ false false false false false false false false false
                                                      false false false false false false false false false
                                                      false false  true false false false false false false
                                                      false false false false false false false false false
                                                      false false false false false false false false false
                                                      false false false false false false false false false
                                                      false false false false false false false false false
                                                      false false false false false false false false false
                                                      false false false false false false false false false ])
  connection_volume_thresholds::Field{Float64} = LatLonField{Float64}(lake_grid,-1.0)
  flood_volume_threshold::Field{Float64} = LatLonField{Float64}(lake_grid,
                                                    Float64[ -1.0 -1.0 -1.0 80.0 -1.0 -1.0 -1.0 -1.0 -1.0
                                                             -1.0 -1.0 -1.0 80.0 -1.0 -1.0 -1.0 -1.0 -1.0
                                                             -1.0 -1.0  1.0 80.0 -1.0 -1.0 -1.0 -1.0 -1.0
                                                             -1.0 -1.0 10.0 10.0 -1.0 -1.0 -1.0 -1.0 -1.0
                                                             -1.0 -1.0 10.0 10.0 -1.0 -1.0 -1.0 -1.0 -1.0
                                                             -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
                                                             -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
                                                             -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
                                                             -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 ])
  flood_local_redirect::Field{Bool} = LatLonField{Bool}(lake_grid,false)
  connect_local_redirect::Field{Bool} = LatLonField{Bool}(lake_grid,false)
  additional_flood_local_redirect::Field{Bool} = LatLonField{Bool}(lake_grid,false)
  additional_connect_local_redirect::Field{Bool} = LatLonField{Bool}(lake_grid,false)
  merge_points::Field{MergeTypes} = LatLonField{MergeTypes}(lake_grid,
    MergeTypes[ no_merge_mtype no_merge_mtype no_merge_mtype connection_merge_not_set_flood_merge_as_secondary #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype ])
  flood_next_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 0 0 0 1 0 0 0 0 0
                                                           0 0 0 1 0 0 0 0 0
                                                           0 0 4 2 0 0 0 0 0
                                                           0 0 5 5 0 0 0 0 0
                                                           0 0 4 3 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0 ])
  flood_next_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 0 0 0 5 0 0 0 0 0
                                                           0 0 0 4 0 0 0 0 0
                                                           0 0 3 4 0 0 0 0 0
                                                           0 0 3 4 0 0 0 0 0
                                                           0 0 4 4 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0 ])
  connect_next_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0 ])
  connect_next_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0 ])
  flood_force_merge_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0 ])
  flood_force_merge_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0 ])
  connect_force_merge_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0 ])
  connect_force_merge_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0 ])
  flood_redirect_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 0 0 0 1 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0 ])
  flood_redirect_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 0 0 0 3 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0 ])
  connect_redirect_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0 ])
  connect_redirect_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
                                                    Int64[ 0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0
                                                           0 0 0 0 0 0 0 0 0 ])
  additional_flood_redirect_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,0)
  additional_flood_redirect_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,0)
  additional_connect_redirect_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,0)
  additional_connect_redirect_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,0)
  grid_specific_lake_parameters::GridSpecificLakeParameters =
    LatLonLakeParameters(flood_next_cell_lat_index,
                         flood_next_cell_lon_index,
                         connect_next_cell_lat_index,
                         connect_next_cell_lon_index,
                         flood_force_merge_lat_index,
                         flood_force_merge_lon_index,
                         connect_force_merge_lat_index,
                         connect_force_merge_lon_index,
                         flood_redirect_lat_index,
                         flood_redirect_lon_index,
                         connect_redirect_lat_index,
                         connect_redirect_lon_index,
                         additional_flood_redirect_lat_index,
                         additional_flood_redirect_lon_index,
                         additional_connect_redirect_lat_index,
                         additional_connect_redirect_lon_index)
  lake_parameters = LakeParameters(lake_centers,
                                   connection_volume_thresholds,
                                   flood_volume_threshold,
                                   flood_local_redirect,
                                   connect_local_redirect,
                                   additional_flood_local_redirect,
                                   additional_connect_local_redirect,
                                   merge_points,
                                   lake_grid,
                                   grid,
                                   grid_specific_lake_parameters)
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,Float64[ 1.0 1.0 1.0
                                                                                 1.0 1.0 1.0
                                                                                 1.0 1.0 0.0 ])
  drainages::Array{Field{Float64},1} = repeat(drainage,1000)
  runoffs::Array{Field{Float64},1} = deepcopy(drainages)
  evaporation::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,1000)
  expected_river_inflow::Field{Float64} = LatLonField{Float64}(grid,
                                                               Float64[ 0.0 0.0  2.0
                                                                        4.0 0.0 14.0
                                                                        0.0 0.0  0.0 ])
  expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(grid,
                                                                 Float64[ 0.0 0.0  0.0
                                                                          0.0 0.0  0.0
                                                                          0.0 0.0 16.0 ])
  expected_water_to_hd::Field{Float64} = LatLonField{Float64}(grid,
                                                              Float64[ 0.0 0.0 10.0
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
  expected_lake_volumes::Array{Float64} = Float64[80.0]
  @time river_fields::RiverPrognosticFields,lake_prognostics::LakePrognostics,lake_fields::LakeFields =
    drive_hd_and_lake_model(river_parameters,lake_parameters,
                            drainages,runoffs,evaporations,
                            1000,print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
  for i = 1:9
    for j = 1:9
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_fields.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_prognostics.lakes[lake_number]
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
  for lake::Lake in lake_prognostics.lakes
    append!(lake_volumes,get_lake_variables(lake).lake_volume)
  end
  @test expected_river_inflow == river_fields.river_inflow
  @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_water_to_hd    == lake_fields.water_to_hd
  @test expected_lake_numbers == lake_fields.lake_numbers
  @test expected_lake_types == lake_types
  @test expected_lake_volumes == lake_volumes
end

@testset "Lake model tests 2" begin
  grid = LatLonGrid(4,4,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(grid,
                                                Int64[-2  4  2  2
                                                       6 -2 -2  2
                                                       6  2  3 -2
                                                      -2  0  6  0 ]))
  river_reservoir_nums = LatLonField{Int64}(grid,5)
  overland_reservoir_nums = LatLonField{Int64}(grid,1)
  base_reservoir_nums = LatLonField{Int64}(grid,1)
  river_retention_coefficients = LatLonField{Float64}(grid,0.7)
  overland_retention_coefficients = LatLonField{Float64}(grid,0.5)
  base_retention_coefficients = LatLonField{Float64}(grid,0.1)
  landsea_mask = LatLonField{Bool}(grid,fill(false,4,4))
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
                                     grid)
  lake_grid = LatLonGrid(20,20,true)
  lake_centers::Field{Bool} = LatLonField{Bool}(lake_grid,
    Bool[ false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false false  #=
    =# false false false false
   true  false false false false false false false false false false false false false false false  #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false true  false false #=
    =# false false false false
   false false false false false true  false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false true  false #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false false  #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false true #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false true  false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false ])
  connection_volume_thresholds::Field{Float64} = LatLonField{Float64}(lake_grid,
    Float64[ -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1 186.0 23.0  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1 56.0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
                                                                      ])
  flood_volume_threshold::Field{Float64} = LatLonField{Float64}(lake_grid,
    Float64[  -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1 5.0
             0.0 262.0  5.0    -1   -1    -1    -1   -1   -1  -1  -1    -1 111.0 111.0 56.0 111.0   -1    -1   -1 2.0
              -1   5.0  5.0    -1   -1 340.0 262.0   -1   -1  -1  -1 111.0   1.0   1.0 56.0  56.0   -1    -1   -1  -1
              -1   5.0  5.0    -1   -1  10.0  10.0 38.0 10.0  -1  -1    -1   0.0   1.0  1.0  26.0 56.0    -1   -1  -1
              -1   5.0  5.0 186.0  2.0   2.0    -1 10.0 10.0  -1 1.0   6.0   1.0   0.0  1.0  26.0 26.0 111.0   -1  -1
            16.0  16.0 16.0    -1  2.0   0.0   2.0  2.0 10.0  -1 1.0   0.0   0.0   1.0  1.0   1.0 26.0  56.0   -1  -1
              -1  46.0 16.0    -1   -1   2.0    -1 23.0   -1  -1  -1   1.0   0.0   1.0  1.0   1.0 56.0    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1  56.0   1.0   1.0  1.0  26.0 56.0    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1  56.0  56.0   -1    -1   -1    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1   0.0  3.0   3.0 10.0  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1   0.0  3.0   3.0   -1  -1
              -1    -1   -1   1.0   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 ])
  flood_local_redirect::Field{Bool} = LatLonField{Bool}(lake_grid,
    Bool[ false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false  false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false true false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false  true false false  false false #=
      =#  false false false false
          false false false false false false  false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false  false #=
      =#  true false false false
          false false false false false false false false false false false false false false true false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false  false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false ])
  connect_local_redirect::Field{Bool} = LatLonField{Bool}(lake_grid,
    Bool[ false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false  false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false  false false #=
      =#  false false false false
          false false false false false false  false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false  false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false  false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false ])
  additional_flood_local_redirect::Field{Bool} = LatLonField{Bool}(lake_grid,false)
  additional_connect_local_redirect::Field{Bool} = LatLonField{Bool}(lake_grid,false)
  merge_points::Field{MergeTypes} = LatLonField{MergeTypes}(lake_grid,
    MergeTypes[ no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          connection_merge_not_set_flood_merge_as_primary no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          connection_merge_not_set_flood_merge_as_secondary no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype no_merge_mtype  no_merge_mtype #=
    =#          no_merge_mtype  no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype connection_merge_not_set_flood_merge_as_primary no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype connection_merge_not_set_flood_merge_as_primary  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype ])
  flood_next_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  3
            2  3  5 -1 -1 -1 -1 -1 -1 -1 -1 -1  2  2  4  5 -1 -1 -1  1
           -1  4  2 -1 -1  8  2 -1 -1 -1 -1  2  5  3  9  2 -1 -1 -1 -1
           -1  3  4 -1 -1  6  4  6  3 -1 -1 -1  7  3  4  3  6 -1 -1 -1
           -1  6  5  3  6  7 -1  4  4 -1  5  7  6  6  4  8  4  3 -1 -1
            7  6  6 -1  5  5  6  5  5 -1  5  5  4  8  6  6  5  5 -1 -1
           -1  6  7 -1 -1  6 -1  4 -1 -1 -1  5  6  6  8  7  5 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  8  7  7  8  6  7 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  8  9 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 14 14 13 16 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 13 14 13 -1 -1
           -1 -1 -1 17 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  flood_next_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  1
           19  5  2 -1 -1 -1 -1 -1 -1 -1 -1 -1 15 12 16  3 -1 -1 -1 19
           -1  2  2 -1 -1 18  1 -1 -1 -1 -1 13 15 12 13 14 -1 -1 -1 -1
           -1  2  1 -1 -1  8  5 10  6 -1 -1 -1 12 13 13 14 17 -1 -1 -1
           -1  2  1  5  7  5 -1  6  8 -1 14 14 10 12 14 15 15 11 -1 -1
            2  0  1 -1  4  5  4  7  8 -1 10 11 12 12 13 14 16 17 -1 -1
           -1  4  1 -1 -1  6 -1  7 -1 -1 -1 12 11 15 14 13  9 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 16 11 15 13 16 16 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 11 12 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 15 16 18 15 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 16 17 17 -1 -1
           -1 -1 -1  3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  connect_next_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1  3  7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1  3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  connect_next_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1  6  7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 15 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  flood_force_merge_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  6 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  6 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  7 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  flood_force_merge_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  2 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  8 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 12 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  connect_force_merge_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  connect_force_merge_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  flood_redirect_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  1 -1 -1 -1 -1
           -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1  7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  7 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  6 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  5 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  3 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1  3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  flood_redirect_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0 -1 -1 -1 -1
           -1 -1 -1 -1 -1  3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 14 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 14 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1  0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  5 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 13 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  3 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  connect_redirect_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  connect_redirect_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  add_offset(flood_next_cell_lat_index,1,Int64[-1])
  add_offset(flood_next_cell_lon_index,1,Int64[-1])
  add_offset(connect_next_cell_lat_index,1,Int64[-1])
  add_offset(connect_next_cell_lon_index,1,Int64[-1])
  add_offset(flood_force_merge_lat_index,1,Int64[-1])
  add_offset(flood_force_merge_lon_index,1,Int64[-1])
  add_offset(connect_force_merge_lat_index,1,Int64[-1])
  add_offset(connect_force_merge_lon_index,1,Int64[-1])
  add_offset(flood_redirect_lat_index,1,Int64[-1])
  add_offset(flood_redirect_lon_index,1,Int64[-1])
  add_offset(connect_redirect_lat_index,1,Int64[-1])
  add_offset(connect_redirect_lon_index,1,Int64[-1])
  additional_flood_redirect_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,0)
  additional_flood_redirect_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,0)
  additional_connect_redirect_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,0)
  additional_connect_redirect_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,0)
  grid_specific_lake_parameters::GridSpecificLakeParameters =
    LatLonLakeParameters(flood_next_cell_lat_index,
                         flood_next_cell_lon_index,
                         connect_next_cell_lat_index,
                         connect_next_cell_lon_index,
                         flood_force_merge_lat_index,
                         flood_force_merge_lon_index,
                         connect_force_merge_lat_index,
                         connect_force_merge_lon_index,
                         flood_redirect_lat_index,
                         flood_redirect_lon_index,
                         connect_redirect_lat_index,
                         connect_redirect_lon_index,
                         additional_flood_redirect_lat_index,
                         additional_flood_redirect_lon_index,
                         additional_connect_redirect_lat_index,
                         additional_connect_redirect_lon_index)
  lake_parameters = LakeParameters(lake_centers,
                                   connection_volume_thresholds,
                                   flood_volume_threshold,
                                   flood_local_redirect,
                                   connect_local_redirect,
                                   additional_flood_local_redirect,
                                   additional_connect_local_redirect,
                                   merge_points,
                                   lake_grid,
                                   grid,
                                   grid_specific_lake_parameters)
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,Float64[ 1.0 1.0 1.0 1.0
                                                                                 1.0 1.0 1.0 1.0
                                                                                 1.0 1.0 1.0 1.0
                                                                                 1.0 0.0 1.0 0.0 ])
  drainages::Array{Field{Float64},1} = repeat(drainage,10000)
  runoffs::Array{Field{Float64},1} = deepcopy(drainages)
  evaporation::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,10000)
  expected_river_inflow::Field{Float64} = LatLonField{Float64}(grid,
                                                               Float64[ 0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 2.0
                                                                        0.0 2.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0 ])
  expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(grid,
                                                                 Float64[ 0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0
                                                                          0.0 0.0 0.0 0.0
                                                                          0.0 6.0 0.0 22.0 ])
  expected_water_to_hd::Field{Float64} = LatLonField{Float64}(grid,
                                                              Float64[ 0.0 0.0 0.0  0.0
                                                                       0.0 0.0 0.0 12.0
                                                                       0.0 0.0 0.0  0.0
                                                                       0.0 2.0 0.0 18.0 ])
  expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
             1 5 1 0 0 0 0 0 0 0 0 0 5 5 5 5 0 0 0 1
             0 1 1 0 0 5 5 0 0 0 0 5 5 5 5 5 0 0 0 0
             0 1 1 0 0 3 3 3 3 0 0 0 4 5 5 5 5 0 0 0
             0 1 1 5 3 3 0 3 3 5 5 4 5 4 5 5 5 5 0 0
             1 1 1 0 3 3 3 3 3 0 5 4 4 5 5 5 5 5 0 0
             0 1 1 0 0 3 0 3 0 0 0 5 4 5 5 5 5 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 5 5 5 5 5 5 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 5 5 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6 6 6 6 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6 6 6 0 0
             0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  expected_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3
             3 2 3 0 0 0 0 0 0 0 0 0 2 2 2 2 0 0 0 3
             0 3 3 0 0 2 2 0 0 0 0 2 2 2 2 2 0 0 0 0
             0 3 3 0 0 3 3 3 3 0 0 0 3 2 2 2 2 0 0 0
             0 3 3 2 3 3 0 3 3 2 2 3 2 3 2 2 2 2 0 0
             3 3 3 0 3 3 3 3 3 0 2 3 3 2 2 2 2 2 0 0
             0 3 3 0 0 3 0 3 0 0 0 2 3 2 2 2 2 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 2 2 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 2 2 0 0 0 0 0 0
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
  expected_lake_volumes::Array{Float64} = Float64[46.0, 1.0, 38.0, 6.0, 340.0, 10.0]
  @time river_fields::RiverPrognosticFields,lake_prognostics::LakePrognostics,lake_fields::LakeFields =
    drive_hd_and_lake_model(river_parameters,lake_parameters,
                            drainages,runoffs,evaporations,
                            10000,print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_fields.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_prognostics.lakes[lake_number]
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
  for lake::Lake in lake_prognostics.lakes
    append!(lake_volumes,get_lake_variables(lake).lake_volume)
  end
  @test expected_river_inflow == river_fields.river_inflow
  @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_water_to_hd    == lake_fields.water_to_hd
  @test expected_lake_numbers == lake_fields.lake_numbers
  @test expected_lake_types == lake_types
  @test isapprox(expected_lake_volumes,lake_volumes,atol=0.00001)
  # function timing2(river_parameters,lake_parameters)
  #   for i in 1:50000
  #     drainagesl = repeat(drainage,20)
  #     runoffsl = deepcopy(drainages)
  #     drive_hd_and_lake_model(river_parameters,lake_parameters,
  #                             drainagesl,runoffsl,20,print_timestep_results=false)
  #   end
  # end
  #@time timing2(river_parameters,lake_parameters)
end

@testset "Lake model tests 3" begin
  grid = LatLonGrid(4,4,true)
  flow_directions =  LatLonDirectionIndicators(LatLonField{Int64}(grid,
                                                Int64[-2  4  2  2
                                                       6 -2 -2  2
                                                       6  2  3 -2
                                                      -2  0  6  0 ]))
  river_reservoir_nums = LatLonField{Int64}(grid,5)
  overland_reservoir_nums = LatLonField{Int64}(grid,1)
  base_reservoir_nums = LatLonField{Int64}(grid,1)
  river_retention_coefficients = LatLonField{Float64}(grid,0.7)
  overland_retention_coefficients = LatLonField{Float64}(grid,0.5)
  base_retention_coefficients = LatLonField{Float64}(grid,0.1)
  landsea_mask = LatLonField{Bool}(grid,fill(false,4,4))
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
                                     grid)
  lake_grid = LatLonGrid(20,20,true)
  lake_centers::Field{Bool} = LatLonField{Bool}(lake_grid,
    Bool[ false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false false  #=
    =# false false false false
   true  false false false false false false false false false false false false false false false  #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false true  false false #=
    =# false false false false
   false false false false false true  false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false true  false #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false false  #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false true #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false true  false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false
   false false false false false false false false false false false false false false false false #=
    =# false false false false ])
  connection_volume_thresholds::Field{Float64} = LatLonField{Float64}(lake_grid,
    Float64[ -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1 186.0 23.0  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1 56.0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
             -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
                                                                      ])
  flood_volume_threshold::Field{Float64} = LatLonField{Float64}(lake_grid,
    Float64[  -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1 5.0
             0.0 262.0  5.0    -1   -1    -1    -1   -1   -1  -1  -1    -1 111.0 111.0 56.0 111.0   -1    -1   -1 2.0
              -1   5.0  5.0    -1   -1 340.0 262.0   -1   -1  -1  -1 111.0   1.0   1.0 56.0  56.0   -1    -1   -1  -1
              -1   5.0  5.0    -1   -1  10.0  10.0 38.0 10.0  -1  -1    -1   0.0   1.0  1.0  26.0 56.0    -1   -1  -1
              -1   5.0  5.0 186.0  2.0   2.0    -1 10.0 10.0  -1 1.0   6.0   1.0   0.0  1.0  26.0 26.0 111.0   -1  -1
            16.0  16.0 16.0    -1  2.0   0.0   2.0  2.0 10.0  -1 1.0   0.0   0.0   1.0  1.0   1.0 26.0  56.0   -1  -1
              -1  46.0 16.0    -1   -1   2.0    -1 23.0   -1  -1  -1   1.0   0.0   1.0  1.0   1.0 56.0    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1  56.0   1.0   1.0  1.0  26.0 56.0    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1  56.0  56.0   -1    -1   -1    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1   0.0  3.0   3.0 10.0  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1   0.0  3.0   3.0   -1  -1
              -1    -1   -1   1.0   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1
              -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 ])
  flood_local_redirect::Field{Bool} = LatLonField{Bool}(lake_grid,
    Bool[ false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false  false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false true false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false  true false false  false false #=
      =#  false false false false
          false false false false false false  false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false  false #=
      =#  true false false false
          false false false false false false false false false false false false false false true false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false  false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false ])
  connect_local_redirect::Field{Bool} = LatLonField{Bool}(lake_grid,
    Bool[ false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false  false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false  false false #=
      =#  false false false false
          false false false false false false  false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false  false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false  false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false
          false false false false false false false false false false false false false false false false #=
      =#  false false false false ])
  additional_flood_local_redirect::Field{Bool} = LatLonField{Bool}(lake_grid,false)
  additional_connect_local_redirect::Field{Bool} = LatLonField{Bool}(lake_grid,false)
  merge_points::Field{MergeTypes} = LatLonField{MergeTypes}(lake_grid,
    MergeTypes[ no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          connection_merge_not_set_flood_merge_as_primary no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          connection_merge_not_set_flood_merge_as_secondary no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype no_merge_mtype  no_merge_mtype #=
    =#          no_merge_mtype  no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype connection_merge_not_set_flood_merge_as_primary no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype connection_merge_not_set_flood_merge_as_primary  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype
                no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype ])
  flood_next_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  3
            2  3  5 -1 -1 -1 -1 -1 -1 -1 -1 -1  2  2  4  5 -1 -1 -1  1
           -1  4  2 -1 -1  8  2 -1 -1 -1 -1  2  5  3  9  2 -1 -1 -1 -1
           -1  3  4 -1 -1  6  4  6  3 -1 -1 -1  7  3  4  3  6 -1 -1 -1
           -1  6  5  3  6  7 -1  4  4 -1  5  7  6  6  4  8  4  3 -1 -1
            7  6  6 -1  5  5  6  5  5 -1  5  5  4  8  6  6  5  5 -1 -1
           -1  6  7 -1 -1  6 -1  4 -1 -1 -1  5  6  6  8  7  5 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  8  7  7  8  6  7 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  8  9 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 14 14 13 16 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 13 14 13 -1 -1
           -1 -1 -1 17 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  flood_next_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  1
           19  5  2 -1 -1 -1 -1 -1 -1 -1 -1 -1 15 12 16  3 -1 -1 -1 19
           -1  2  2 -1 -1 18  1 -1 -1 -1 -1 13 15 12 13 14 -1 -1 -1 -1
           -1  2  1 -1 -1  8  5 10  6 -1 -1 -1 12 13 13 14 17 -1 -1 -1
           -1  2  1  5  7  5 -1  6  8 -1 14 14 10 12 14 15 15 11 -1 -1
            2  0  1 -1  4  5  4  7  8 -1 10 11 12 12 13 14 16 17 -1 -1
           -1  4  1 -1 -1  6 -1  7 -1 -1 -1 12 11 15 14 13  9 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 16 11 15 13 16 16 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 11 12 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 15 16 18 15 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 16 17 17 -1 -1
           -1 -1 -1  3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  connect_next_cell_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1  3  7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1  3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  connect_next_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1  6  7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 15 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  flood_force_merge_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  6 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  6 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  7 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  flood_force_merge_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  2 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  8 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 12 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  connect_force_merge_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  connect_force_merge_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  flood_redirect_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  1 -1 -1 -1 -1
           -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1  7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  7 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  6 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  5 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  3 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1  3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  flood_redirect_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0 -1 -1 -1 -1
           -1 -1 -1 -1 -1  3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 14 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 14 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1  0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  5 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 13 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  3 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  connect_redirect_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  connect_redirect_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  add_offset(flood_next_cell_lat_index,1,Int64[-1])
  add_offset(flood_next_cell_lon_index,1,Int64[-1])
  add_offset(connect_next_cell_lat_index,1,Int64[-1])
  add_offset(connect_next_cell_lon_index,1,Int64[-1])
  add_offset(flood_force_merge_lat_index,1,Int64[-1])
  add_offset(flood_force_merge_lon_index,1,Int64[-1])
  add_offset(connect_force_merge_lat_index,1,Int64[-1])
  add_offset(connect_force_merge_lon_index,1,Int64[-1])
  add_offset(flood_redirect_lat_index,1,Int64[-1])
  add_offset(flood_redirect_lon_index,1,Int64[-1])
  add_offset(connect_redirect_lat_index,1,Int64[-1])
  add_offset(connect_redirect_lon_index,1,Int64[-1])
  additional_flood_redirect_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,0)
  additional_flood_redirect_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,0)
  additional_connect_redirect_lat_index::Field{Int64} = LatLonField{Int64}(lake_grid,0)
  additional_connect_redirect_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,0)
  grid_specific_lake_parameters::GridSpecificLakeParameters =
    LatLonLakeParameters(flood_next_cell_lat_index,
                         flood_next_cell_lon_index,
                         connect_next_cell_lat_index,
                         connect_next_cell_lon_index,
                         flood_force_merge_lat_index,
                         flood_force_merge_lon_index,
                         connect_force_merge_lat_index,
                         connect_force_merge_lon_index,
                         flood_redirect_lat_index,
                         flood_redirect_lon_index,
                         connect_redirect_lat_index,
                         connect_redirect_lon_index,
                         additional_flood_redirect_lat_index,
                         additional_flood_redirect_lon_index,
                         additional_connect_redirect_lat_index,
                         additional_connect_redirect_lon_index)
  lake_parameters = LakeParameters(lake_centers,
                                   connection_volume_thresholds,
                                   flood_volume_threshold,
                                   flood_local_redirect,
                                   connect_local_redirect,
                                   additional_flood_local_redirect,
                                   additional_connect_local_redirect,
                                   merge_points,
                                   lake_grid,
                                   grid,
                                   grid_specific_lake_parameters)
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,Float64[ 1.0 1.0 1.0 1.0
                                                                                 1.0 1.0 1.0 1.0
                                                                                 1.0 1.0 1.0 1.0
                                                                                 1.0 0.0 1.0 0.0 ])
  drainages::Array{Field{Float64},1} = repeat(drainage,10000)
  runoffs::Array{Field{Float64},1} = deepcopy(drainages)
  evaporation::Field{Float64} = LatLonField{Float64}(river_parameters.grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,180)
  evaporation = LatLonField{Float64}(river_parameters.grid,100.0)
  additional_evaporations::Array{Field{Float64},1} = repeat(evaporation,200)
  append!(evaporations,additional_evaporations)
  expected_river_inflow::Field{Float64} = LatLonField{Float64}(grid,
                                                               Float64[ 0.0 0.0 0.0 0.0
                                                                        0.0 0.0 0.0 2.0
                                                                        0.0 2.0 0.0 0.0
                                                                        0.0 0.0 0.0 0.0 ])
  expected_water_to_ocean::Field{Float64} = LatLonField{Float64}(grid,
                                                                 Float64[ -96.0   0.0   0.0   0.0
                                                                            0.0 -96.0 -96.0   0.0
                                                                            0.0   0.0   0.0 -94.0
                                                                          -98.0   4.0   0.0   4.0 ])
  expected_water_to_hd::Field{Float64} = LatLonField{Float64}(grid,
                                                              Float64[ 0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0
                                                                       0.0 0.0 0.0 0.0 ])
  expected_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0
             0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
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
  expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  expected_intermediate_river_inflow::Field{Float64} = LatLonField{Float64}(grid,
                                                         Float64[ 0.0 0.0 0.0 0.0
                                                                  0.0 0.0 0.0 2.0
                                                                  0.0 2.0 0.0 0.0
                                                                  0.0 0.0 0.0 0.0 ])
  expected_intermediate_water_to_ocean::Field{Float64} = LatLonField{Float64}(grid,
                                                           Float64[ 0.0 0.0 0.0 0.0
                                                                    0.0 0.0 0.0 0.0
                                                                    0.0 0.0 0.0 0.0
                                                                    0.0 6.0 0.0 22.0 ])
  expected_intermediate_water_to_hd::Field{Float64} = LatLonField{Float64}(grid,
                                                        Float64[ 0.0 0.0 0.0  0.0
                                                                 0.0 0.0 0.0 12.0
                                                                 0.0 0.0 0.0  0.0
                                                                 0.0 2.0 0.0 18.0 ])
  expected_intermediate_lake_numbers::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
             1 5 1 0 0 0 0 0 0 0 0 0 5 5 5 5 0 0 0 1
             0 1 1 0 0 5 5 0 0 0 0 5 5 5 5 5 0 0 0 0
             0 1 1 0 0 3 3 3 3 0 0 0 4 5 5 5 5 0 0 0
             0 1 1 5 3 3 0 3 3 5 5 4 5 4 5 5 5 5 0 0
             1 1 1 0 3 3 3 3 3 0 5 4 4 5 5 5 5 5 0 0
             0 1 1 0 0 3 0 3 0 0 0 5 4 5 5 5 5 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 5 5 5 5 5 5 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 5 5 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6 6 6 6 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6 6 6 0 0
             0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ])
  expected_intermediate_lake_types::Field{Int64} = LatLonField{Int64}(lake_grid,
      Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3
             3 2 3 0 0 0 0 0 0 0 0 0 2 2 2 2 0 0 0 3
             0 3 3 0 0 2 2 0 0 0 0 2 2 2 2 2 0 0 0 0
             0 3 3 0 0 3 3 3 3 0 0 0 3 2 2 2 2 0 0 0
             0 3 3 2 3 3 0 3 3 2 2 3 2 3 2 2 2 2 0 0
             3 3 3 0 3 3 3 3 3 0 2 3 3 2 2 2 2 2 0 0
             0 3 3 0 0 3 0 3 0 0 0 2 3 2 2 2 2 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 2 2 0 0 0
             0 0 0 0 0 0 0 0 0 0 0 0 2 2 0 0 0 0 0 0
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
  expected_intermediate_lake_volumes::Array{Float64} = Float64[46.0, 1.0, 38.0, 6.0, 340.0, 10.0]
  evaporations_copy::Array{Field{Float64},1} = deepcopy(evaporations)
  @time river_fields::RiverPrognosticFields,lake_prognostics::LakePrognostics,lake_fields::LakeFields =
    drive_hd_and_lake_model(river_parameters,lake_parameters,
                            drainages,runoffs,evaporations_copy,
                            5000,print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_types = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_fields.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_prognostics.lakes[lake_number]
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
  for lake::Lake in lake_prognostics.lakes
    append!(lake_volumes,get_lake_variables(lake).lake_volume)
  end
  @test expected_intermediate_river_inflow == river_fields.river_inflow
  @test isapprox(expected_intermediate_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_intermediate_water_to_hd    == lake_fields.water_to_hd
  @test expected_intermediate_lake_numbers == lake_fields.lake_numbers
  @test expected_intermediate_lake_types == lake_types
  @test isapprox(expected_intermediate_lake_volumes,lake_volumes,atol=0.00001)
  @time river_fields,lake_prognostics,lake_fields =
    drive_hd_and_lake_model(river_parameters,lake_parameters,
                            drainages,runoffs,evaporations,
                            10000,print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_types::LatLonField{Int64} = LatLonField{Int64}(lake_grid,0)
  for i = 1:20
    for j = 1:20
      coords::LatLonCoords = LatLonCoords(i,j)
      lake_number::Int64 = lake_fields.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_prognostics.lakes[lake_number]
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
  for lake::Lake in lake_prognostics.lakes
    append!(lake_volumes,get_lake_variables(lake).lake_volume)
  end
  @test expected_river_inflow == river_fields.river_inflow
  @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_water_to_hd    == lake_fields.water_to_hd
  @test expected_lake_numbers == lake_fields.lake_numbers
  @test expected_lake_types == lake_types
  @test isapprox(expected_lake_volumes,lake_volumes,atol=0.00001)
  # function timing2(river_parameters,lake_parameters)
  #   for i in 1:50000
  #     drainagesl = repeat(drainage,20)
  #     runoffsl = deepcopy(drainages)
  #     drive_hd_and_lake_model(river_parameters,lake_parameters,
  #                             drainagesl,runoffsl,20,print_timestep_results=false)
  #   end
  # end
  #@time timing2(river_parameters,lake_parameters)
end

@testset "Lake model tests 4" begin
  grid = UnstructuredGrid(16)
  flow_directions =  UnstructuredDirectionIndicators(UnstructuredField{Int64}(grid,
                                                     vec(Int64[-2  1  7  8 #=
                                                             =# 6 -2 -2 12 #=
                                                            =# 10 14 16 -2 #=
                                                            =# -2  0 16  0 ])))
  river_reservoir_nums = UnstructuredField{Int64}(grid,5)
  overland_reservoir_nums = UnstructuredField{Int64}(grid,1)
  base_reservoir_nums = UnstructuredField{Int64}(grid,1)
  river_retention_coefficients = UnstructuredField{Float64}(grid,0.7)
  overland_retention_coefficients = UnstructuredField{Float64}(grid,0.5)
  base_retention_coefficients = UnstructuredField{Float64}(grid,0.1)
  landsea_mask = UnstructuredField{Bool}(grid,fill(false,16))
  set!(river_reservoir_nums,Generic1DCoords(16),0)
  set!(overland_reservoir_nums,Generic1DCoords(16),0)
  set!(base_reservoir_nums,Generic1DCoords(16),0)
  set!(landsea_mask,Generic1DCoords(16),true)
  set!(river_reservoir_nums,Generic1DCoords(14),0)
  set!(overland_reservoir_nums,Generic1DCoords(14),0)
  set!(base_reservoir_nums,Generic1DCoords(14),0)
  set!(landsea_mask,Generic1DCoords(14),true)
  river_parameters = RiverParameters(flow_directions,
                                     river_reservoir_nums,
                                     overland_reservoir_nums,
                                     base_reservoir_nums,
                                     river_retention_coefficients,
                                     overland_retention_coefficients,
                                     base_retention_coefficients,
                                     landsea_mask,
                                     grid)
  mapping_to_coarse_grid::Array{Int64,1} =
    vec(Int64[1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 #=
           =# 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 #=
           =# 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 #=
           =# 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 #=
           =# 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 #=
           =# 5 5 5 5 5 6 6 6 6 6 7 7 7 7 7 8 8 8 8 8 #=
           =# 5 5 5 5 5 6 6 6 6 6 7 7 7 7 7 8 8 8 8 8 #=
           =# 5 5 5 5 5 6 6 6 6 6 7 7 7 7 7 8 8 8 8 8 #=
           =# 5 5 5 5 5 6 6 6 6 6 7 7 7 7 7 8 8 8 8 8 #=
           =# 5 5 5 5 5 6 6 6 6 6 7 7 7 7 7 8 8 8 8 8 #=
           =# 9 9 9 9 9 10 10 10 10 10 11 11 11 11 11 12 12 12 12 12 #=
           =# 9 9 9 9 9 10 10 10 10 10 11 11 11 11 11 12 12 12 12 12 #=
           =# 9 9 9 9 9 10 10 10 10 10 11 11 11 11 11 12 12 12 12 12 #=
           =# 9 9 9 9 9 10 10 10 10 10 11 11 11 11 11 12 12 12 12 12 #=
           =# 9 9 9 9 9 10 10 10 10 10 11 11 11 11 11 12 12 12 12 12 #=
           =# 13 13 13 13 13 14 14 14 14 14 15 15 15 15 15 16 16 16 16 16 #=
           =# 13 13 13 13 13 14 14 14 14 14 15 15 15 15 15 16 16 16 16 16 #=
           =# 13 13 13 13 13 14 14 14 14 14 15 15 15 15 15 16 16 16 16 16 #=
           =# 13 13 13 13 13 14 14 14 14 14 15 15 15 15 15 16 16 16 16 16 #=
           =# 13 13 13 13 13 14 14 14 14 14 15 15 15 15 15 16 16 16 16 16 ])
  lake_grid = UnstructuredGrid(400,mapping_to_coarse_grid)
  lake_centers::Field{Bool} = UnstructuredField{Bool}(lake_grid,
    vec(Bool[ false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false  #=
    =# false false false false #=
    =# true  false false false false false false false false false false false false false false false  #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false true  false false #=
    =# false false false false #=
    =# false false false false false true  false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false true  false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false  #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false true #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false true  false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false ]))
  connection_volume_thresholds::Field{Float64} = UnstructuredField{Float64}(lake_grid,
    vec(Float64[    -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1 186.0 23.0  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1 56.0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ]))
  flood_volume_threshold::Field{Float64} = UnstructuredField{Float64}(lake_grid,
    vec(Float64[  -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 #=
             =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1 5.0 #=
             =# 0.0 262.0  5.0    -1   -1    -1    -1   -1   -1  -1  -1    -1 111.0 111.0 56.0 111.0   -1    -1   -1 2.0 #=
              =# -1   5.0  5.0    -1   -1 340.0 262.0   -1   -1  -1  -1 111.0   1.0   1.0 56.0  56.0   -1    -1   -1  -1 #=
              =# -1   5.0  5.0    -1   -1  10.0  10.0 38.0 10.0  -1  -1    -1   0.0   1.0  1.0  26.0 56.0    -1   -1  -1 #=
              =# -1   5.0  5.0 186.0  2.0   2.0    -1 10.0 10.0  -1 1.0   6.0   1.0   0.0  1.0  26.0 26.0 111.0   -1  -1 #=
            =# 16.0  16.0 16.0    -1  2.0   0.0   2.0  2.0 10.0  -1 1.0   0.0   0.0   1.0  1.0   1.0 26.0  56.0   -1  -1 #=
            =# -1  46.0 16.0    -1   -1   2.0    -1 23.0   -1  -1  -1   1.0   0.0   1.0  1.0   1.0 56.0    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1  56.0   1.0   1.0  1.0  26.0 56.0    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1  56.0  56.0   -1    -1   -1    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1   0.0  3.0   3.0 10.0  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1   0.0  3.0   3.0   -1  -1 #=
            =# -1    -1   -1   1.0   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 ]))
  flood_local_redirect::Field{Bool} = UnstructuredField{Bool}(lake_grid,
    vec(Bool[ false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false  false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false true false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false  true false false  false false #=
      =#  false false false false #=
      =#  false false false false false false  false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false  false #=
      =#  true false false false #=
      =#  false false false false false false false false false false false false false false true false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =# false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false  false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false ]))
  connect_local_redirect::Field{Bool} = UnstructuredField{Bool}(lake_grid,
    vec(Bool[ false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false  false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =# false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false  false false #=
      =#  false false false false #=
      =#  false false false false false false  false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false  false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =# false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#   false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#   false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =# false false false false  false false false false false false false false false false false false #=
      =#  false false false false #=
      =# false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =# false false false false false false false false false false false false false false false false #=
      =#  false false false false ]))
  additional_flood_local_redirect::Field{Bool} = UnstructuredField{Bool}(lake_grid,false)
  additional_connect_local_redirect::Field{Bool} = UnstructuredField{Bool}(lake_grid,false)
  merge_points::Field{MergeTypes} = UnstructuredField{MergeTypes}(lake_grid,
    vec(MergeTypes[ no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          connection_merge_not_set_flood_merge_as_primary no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          connection_merge_not_set_flood_merge_as_secondary no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype no_merge_mtype  no_merge_mtype #=
    =#          no_merge_mtype  no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype connection_merge_not_set_flood_merge_as_primary no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype connection_merge_not_set_flood_merge_as_primary  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype ]))
  flood_next_cell_index::Field{Int64} = UnstructuredField{Int64}(lake_grid,
    vec(Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 61 59 65 102 -1 -1 -1 -1 -1 -1 -1 -1 -1 55 52 96 103 -1 -1 -1 39 -1 82 42 -1 -1 178 41 -1 -1 -1 -1 53 115 72 193 54 -1 -1 -1 -1 -1 62 81 -1 -1 128 85 130 66 -1 -1 -1 152 73 93 74 137 -1 -1 -1 -1 122 101 65 127 145 -1 86 88 -1 114 154 130 132 94 175 95 71 -1 -1 142 120 121 -1 104 105 124 107 108 -1 110 111 92 172 133 134 116 117 -1 -1 -1 124 141 -1 -1 126 -1 87 -1 -1 -1 112 131 135 174 153 109 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 176 151 155 173 136 156 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 171 192 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 295 296 278 335 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 276 297 277 -1 -1 -1 -1 -1 343 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ]))
  connect_next_cell_index::Field{Int64} = UnstructuredField{Int64}(lake_grid,
    vec(Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 66 147 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 75 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ]))
  flood_force_merge_index::Field{Int64} = UnstructuredField{Int64}(lake_grid,
    vec(Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 122 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 128 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 152 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ]))
  connect_force_merge_index::Field{Int64} = UnstructuredField{Int64}(lake_grid,
    vec(Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ]))
  flood_redirect_index::Field{Int64} = UnstructuredField{Int64}(lake_grid,
    vec(Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 4 -1 -1 -1 -1 -1 -1 -1 -1 -1 7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 154 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 154 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 4 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 125 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 113 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 15 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 13 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ]))
  connect_redirect_index::Field{Int64} = UnstructuredField{Int64}(lake_grid,
    vec(Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ]))
  add_offset(flood_next_cell_index,1,Int64[-1])
  add_offset(connect_next_cell_index,1,Int64[-1])
  add_offset(flood_force_merge_index,1,Int64[-1])
  add_offset(connect_force_merge_index,1,Int64[-1])
  add_offset(flood_redirect_index,1,Int64[-1])
  add_offset(connect_redirect_index,1,Int64[-1])
  additional_flood_redirect_index::Field{Int64} = UnstructuredField{Int64}(lake_grid,0)
  additional_connect_redirect_index::Field{Int64} = UnstructuredField{Int64}(lake_grid,0)
  grid_specific_lake_parameters::GridSpecificLakeParameters =
    UnstructuredLakeParameters(flood_next_cell_index,
                               connect_next_cell_index,
                               flood_force_merge_index,
                               connect_force_merge_index,
                               flood_redirect_index,
                               connect_redirect_index,
                               additional_flood_redirect_index,
                               additional_connect_redirect_index)
  lake_parameters = LakeParameters(lake_centers,
                                   connection_volume_thresholds,
                                   flood_volume_threshold,
                                   flood_local_redirect,
                                   connect_local_redirect,
                                   additional_flood_local_redirect,
                                   additional_connect_local_redirect,
                                   merge_points,
                                   lake_grid,
                                   grid,
                                   grid_specific_lake_parameters)
  drainage::Field{Float64} = UnstructuredField{Float64}(river_parameters.grid,
                                                        vec(Float64[ 1.0 1.0 1.0 1.0 #=
                                                                  =# 1.0 1.0 1.0 1.0 #=
                                                                  =# 1.0 1.0 1.0 1.0 #=
                                                                  =# 1.0 0.0 1.0 0.0 ]))
  drainages::Array{Field{Float64},1} = repeat(drainage,10000)
  runoffs::Array{Field{Float64},1} = deepcopy(drainages)
  evaporation::Field{Float64} = UnstructuredField{Float64}(river_parameters.grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,10000)
  expected_river_inflow::Field{Float64} = UnstructuredField{Float64}(grid,
                                          vec(Float64[ 0.0 0.0 0.0 0.0 #=
                                                    =# 0.0 0.0 0.0 2.0 #=
                                                    =# 0.0 2.0 0.0 0.0 #=
                                                   =#  0.0 0.0 0.0 0.0 ]))
  expected_water_to_ocean::Field{Float64} = UnstructuredField{Float64}(grid,
                                            vec(Float64[ 0.0 0.0 0.0 0.0 #=
                                                      =# 0.0 0.0 0.0 0.0 #=
                                                      =# 0.0 0.0 0.0 0.0 #=
                                                      =# 0.0 6.0 0.0 22.0 ]))
  expected_water_to_hd::Field{Float64} = UnstructuredField{Float64}(grid,
                                         vec(Float64[ 0.0 0.0 0.0  0.0 #=
                                                   =# 0.0 0.0 0.0 12.0 #=
                                                   =# 0.0 0.0 0.0  0.0 #=
                                                   =# 0.0 2.0 0.0 18.0 ]))
  expected_lake_numbers::Field{Int64} = UnstructuredField{Int64}(lake_grid,
      vec(Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 #=
              =# 1 4 1 0 0 0 0 0 0 0 0 0 4 4 4 4 0 0 0 1 #=
              =# 0 1 1 0 0 4 4 0 0 0 0 4 4 4 4 4 0 0 0 0 #=
              =# 0 1 1 0 0 3 3 3 3 0 0 0 2 4 4 4 4 0 0 0 #=
              =# 0 1 1 4 3 3 0 3 3 4 4 2 4 2 4 4 4 4 0 0 #=
              =# 1 1 1 0 3 3 3 3 3 0 4 2 2 4 4 4 4 4 0 0 #=
              =# 0 1 1 0 0 3 0 3 0 0 0 4 2 4 4 4 4 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 4 4 4 4 4 4 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 4 4 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 5 5 5 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 5 5 0 0 #=
              =# 0 0 0 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]))
  expected_lake_types::Field{Int64} = UnstructuredField{Int64}(lake_grid,
      vec(Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 #=
              =# 3 2 3 0 0 0 0 0 0 0 0 0 2 2 2 2 0 0 0 3 #=
              =# 0 3 3 0 0 2 2 0 0 0 0 2 2 2 2 2 0 0 0 0 #=
              =# 0 3 3 0 0 3 3 3 3 0 0 0 3 2 2 2 2 0 0 0 #=
              =# 0 3 3 2 3 3 0 3 3 2 2 3 2 3 2 2 2 2 0 0 #=
              =# 3 3 3 0 3 3 3 3 3 0 2 3 3 2 2 2 2 2 0 0 #=
              =# 0 3 3 0 0 3 0 3 0 0 0 2 3 2 2 2 2 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 2 2 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 2 2 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 0 0 #=
              =# 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]))
  expected_lake_volumes::Array{Float64} = Float64[46.0, 6.0, 38.0,  340.0, 10.0, 1.0]
  @time river_fields::RiverPrognosticFields,lake_prognostics::LakePrognostics,lake_fields::LakeFields =
    drive_hd_and_lake_model(river_parameters,lake_parameters,
                            drainages,runoffs,evaporations,
                            10000,print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_types::UnstructuredField{Int64} = UnstructuredField{Int64}(lake_grid,0)
  for i = 1:400
      coords::Generic1DCoords = Generic1DCoords(i)
      lake_number::Int64 = lake_fields.lake_numbers(coords)
      if lake_number <= 0 continue end
      lake::Lake = lake_prognostics.lakes[lake_number]
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
  lake_volumes::Array{Float64} = Float64[]
  for lake::Lake in lake_prognostics.lakes
    append!(lake_volumes,get_lake_variables(lake).lake_volume)
  end
  @test expected_river_inflow == river_fields.river_inflow
  @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_water_to_hd    == lake_fields.water_to_hd
  @test expected_lake_numbers == lake_fields.lake_numbers
  @test expected_lake_types == lake_types
  @test isapprox(expected_lake_volumes,lake_volumes,atol=0.00001)
  # function timing2(river_parameters,lake_parameters)
  #   for i in 1:50000
  #     drainagesl = repeat(drainage,20)
  #     runoffsl = deepcopy(drainages)
  #     drive_hd_and_lake_model(river_parameters,lake_parameters,
  #                             drainagesl,runoffsl,20,print_timestep_results=false)
  #   end
  # end
  #@time timing2(river_parameters,lake_parameters)
end

@testset "Lake model tests 5" begin
  grid = UnstructuredGrid(16)
  flow_directions =  UnstructuredDirectionIndicators(UnstructuredField{Int64}(grid,
                                                     vec(Int64[-2  1  7  8 #=
                                                             =# 6 -2 -2 12 #=
                                                            =# 10 14 16 -2 #=
                                                            =# -2  0 16  0 ])))
  river_reservoir_nums = UnstructuredField{Int64}(grid,5)
  overland_reservoir_nums = UnstructuredField{Int64}(grid,1)
  base_reservoir_nums = UnstructuredField{Int64}(grid,1)
  river_retention_coefficients = UnstructuredField{Float64}(grid,0.7)
  overland_retention_coefficients = UnstructuredField{Float64}(grid,0.5)
  base_retention_coefficients = UnstructuredField{Float64}(grid,0.1)
  landsea_mask = UnstructuredField{Bool}(grid,fill(false,16))
  set!(river_reservoir_nums,Generic1DCoords(16),0)
  set!(overland_reservoir_nums,Generic1DCoords(16),0)
  set!(base_reservoir_nums,Generic1DCoords(16),0)
  set!(landsea_mask,Generic1DCoords(16),true)
  set!(river_reservoir_nums,Generic1DCoords(14),0)
  set!(overland_reservoir_nums,Generic1DCoords(14),0)
  set!(base_reservoir_nums,Generic1DCoords(14),0)
  set!(landsea_mask,Generic1DCoords(14),true)
  river_parameters = RiverParameters(flow_directions,
                                     river_reservoir_nums,
                                     overland_reservoir_nums,
                                     base_reservoir_nums,
                                     river_retention_coefficients,
                                     overland_retention_coefficients,
                                     base_retention_coefficients,
                                     landsea_mask,
                                     grid)
  mapping_to_coarse_grid::Array{Int64,1} =
    vec(Int64[1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 #=
           =# 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 #=
           =# 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 #=
           =# 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 #=
           =# 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 #=
           =# 5 5 5 5 5 6 6 6 6 6 7 7 7 7 7 8 8 8 8 8 #=
           =# 5 5 5 5 5 6 6 6 6 6 7 7 7 7 7 8 8 8 8 8 #=
           =# 5 5 5 5 5 6 6 6 6 6 7 7 7 7 7 8 8 8 8 8 #=
           =# 5 5 5 5 5 6 6 6 6 6 7 7 7 7 7 8 8 8 8 8 #=
           =# 5 5 5 5 5 6 6 6 6 6 7 7 7 7 7 8 8 8 8 8 #=
           =# 9 9 9 9 9 10 10 10 10 10 11 11 11 11 11 12 12 12 12 12 #=
           =# 9 9 9 9 9 10 10 10 10 10 11 11 11 11 11 12 12 12 12 12 #=
           =# 9 9 9 9 9 10 10 10 10 10 11 11 11 11 11 12 12 12 12 12 #=
           =# 9 9 9 9 9 10 10 10 10 10 11 11 11 11 11 12 12 12 12 12 #=
           =# 9 9 9 9 9 10 10 10 10 10 11 11 11 11 11 12 12 12 12 12 #=
           =# 13 13 13 13 13 14 14 14 14 14 15 15 15 15 15 16 16 16 16 16 #=
           =# 13 13 13 13 13 14 14 14 14 14 15 15 15 15 15 16 16 16 16 16 #=
           =# 13 13 13 13 13 14 14 14 14 14 15 15 15 15 15 16 16 16 16 16 #=
           =# 13 13 13 13 13 14 14 14 14 14 15 15 15 15 15 16 16 16 16 16 #=
           =# 13 13 13 13 13 14 14 14 14 14 15 15 15 15 15 16 16 16 16 16 ])
  lake_grid = UnstructuredGrid(400,mapping_to_coarse_grid)
  lake_centers::Field{Bool} = UnstructuredField{Bool}(lake_grid,
    vec(Bool[ false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false  #=
    =# false false false false #=
    =# true  false false false false false false false false false false false false false false false  #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false true  false false #=
    =# false false false false #=
    =# false false false false false true  false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false true  false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false  #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false true #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false true  false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false #=
    =# false false false false false false false false false false false false false false false false #=
    =# false false false false ]))
  connection_volume_thresholds::Field{Float64} = UnstructuredField{Float64}(lake_grid,
    vec(Float64[    -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1 186.0 23.0  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1 56.0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 #=
             =# -1 -1 -1 -1 -1    -1   -1  -1 -1   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ]))
  flood_volume_threshold::Field{Float64} = UnstructuredField{Float64}(lake_grid,
    vec(Float64[  -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 #=
             =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1 5.0 #=
             =# 0.0 262.0  5.0    -1   -1    -1    -1   -1   -1  -1  -1    -1 111.0 111.0 56.0 111.0   -1    -1   -1 2.0 #=
              =# -1   5.0  5.0    -1   -1 340.0 262.0   -1   -1  -1  -1 111.0   1.0   1.0 56.0  56.0   -1    -1   -1  -1 #=
              =# -1   5.0  5.0    -1   -1  10.0  10.0 38.0 10.0  -1  -1    -1   0.0   1.0  1.0  26.0 56.0    -1   -1  -1 #=
              =# -1   5.0  5.0 186.0  2.0   2.0    -1 10.0 10.0  -1 1.0   6.0   1.0   0.0  1.0  26.0 26.0 111.0   -1  -1 #=
            =# 16.0  16.0 16.0    -1  2.0   0.0   2.0  2.0 10.0  -1 1.0   0.0   0.0   1.0  1.0   1.0 26.0  56.0   -1  -1 #=
            =# -1  46.0 16.0    -1   -1   2.0    -1 23.0   -1  -1  -1   1.0   0.0   1.0  1.0   1.0 56.0    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1  56.0   1.0   1.0  1.0  26.0 56.0    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1  56.0  56.0   -1    -1   -1    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1   0.0  3.0   3.0 10.0  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1   0.0  3.0   3.0   -1  -1 #=
            =# -1    -1   -1   1.0   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 #=
            =# -1    -1   -1    -1   -1    -1    -1   -1   -1  -1  -1    -1    -1    -1   -1    -1   -1    -1   -1  -1 ]))
  flood_local_redirect::Field{Bool} = UnstructuredField{Bool}(lake_grid,
    vec(Bool[ false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false  false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false true false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false  true false false  false false #=
      =#  false false false false #=
      =#  false false false false false false  false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false  false #=
      =#  true false false false #=
      =#  false false false false false false false false false false false false false false true false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =# false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false  false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false ]))
  connect_local_redirect::Field{Bool} = UnstructuredField{Bool}(lake_grid,
    vec(Bool[ false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false  false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =# false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false  false false #=
      =#  false false false false #=
      =#  false false false false false false  false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false  false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =# false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#   false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#   false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =# false false false false  false false false false false false false false false false false false #=
      =#  false false false false #=
      =# false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =#  false false false false false false false false false false false false false false false false #=
      =#  false false false false #=
      =# false false false false false false false false false false false false false false false false #=
      =#  false false false false ]))
  additional_flood_local_redirect::Field{Bool} = UnstructuredField{Bool}(lake_grid,false)
  additional_connect_local_redirect::Field{Bool} = UnstructuredField{Bool}(lake_grid,false)
  merge_points::Field{MergeTypes} = UnstructuredField{MergeTypes}(lake_grid,
    vec(MergeTypes[ no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          connection_merge_not_set_flood_merge_as_primary no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          connection_merge_not_set_flood_merge_as_secondary no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype no_merge_mtype  no_merge_mtype #=
    =#          no_merge_mtype  no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype connection_merge_not_set_flood_merge_as_primary no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype connection_merge_not_set_flood_merge_as_primary  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype connection_merge_not_set_flood_merge_as_secondary no_merge_mtype  #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype #=
    =#          no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype no_merge_mtype ]))
  flood_next_cell_index::Field{Int64} = UnstructuredField{Int64}(lake_grid,
    vec(Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 61 59 65 102 -1 -1 -1 -1 -1 -1 -1 -1 -1 55 52 96 103 -1 -1 -1 39 -1 82 42 -1 -1 178 41 -1 -1 -1 -1 53 115 72 193 54 -1 -1 -1 -1 -1 62 81 -1 -1 128 85 130 66 -1 -1 -1 152 73 93 74 137 -1 -1 -1 -1 122 101 65 127 145 -1 86 88 -1 114 154 130 132 94 175 95 71 -1 -1 142 120 121 -1 104 105 124 107 108 -1 110 111 92 172 133 134 116 117 -1 -1 -1 124 141 -1 -1 126 -1 87 -1 -1 -1 112 131 135 174 153 109 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 176 151 155 173 136 156 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 171 192 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 295 296 278 335 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 276 297 277 -1 -1 -1 -1 -1 343 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ]))
  connect_next_cell_index::Field{Int64} = UnstructuredField{Int64}(lake_grid,
    vec(Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 66 147 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 75 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ]))
  flood_force_merge_index::Field{Int64} = UnstructuredField{Int64}(lake_grid,
    vec(Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 122 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 128 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 152 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ]))
  connect_force_merge_index::Field{Int64} = UnstructuredField{Int64}(lake_grid,
    vec(Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ]))
  flood_redirect_index::Field{Int64} = UnstructuredField{Int64}(lake_grid,
    vec(Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 4 -1 -1 -1 -1 -1 -1 -1 -1 -1 7 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 154 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 154 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 4 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 125 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 113 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 15 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 13 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ]))
  connect_redirect_index::Field{Int64} = UnstructuredField{Int64}(lake_grid,
    vec(Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ]))
  add_offset(flood_next_cell_index,1,Int64[-1])
  add_offset(connect_next_cell_index,1,Int64[-1])
  add_offset(flood_force_merge_index,1,Int64[-1])
  add_offset(connect_force_merge_index,1,Int64[-1])
  add_offset(flood_redirect_index,1,Int64[-1])
  add_offset(connect_redirect_index,1,Int64[-1])
  additional_flood_redirect_index::Field{Int64} = UnstructuredField{Int64}(lake_grid,0)
  additional_connect_redirect_index::Field{Int64} = UnstructuredField{Int64}(lake_grid,0)
  grid_specific_lake_parameters::GridSpecificLakeParameters =
    UnstructuredLakeParameters(flood_next_cell_index,
                               connect_next_cell_index,
                               flood_force_merge_index,
                               connect_force_merge_index,
                               flood_redirect_index,
                               connect_redirect_index,
                               additional_flood_redirect_index,
                               additional_connect_redirect_index)
  lake_parameters = LakeParameters(lake_centers,
                                   connection_volume_thresholds,
                                   flood_volume_threshold,
                                   flood_local_redirect,
                                   connect_local_redirect,
                                   additional_flood_local_redirect,
                                   additional_connect_local_redirect,
                                   merge_points,
                                   lake_grid,
                                   grid,
                                   grid_specific_lake_parameters)
  drainage::Field{Float64} = UnstructuredField{Float64}(river_parameters.grid,
                                                        vec(Float64[ 1.0 1.0 1.0 1.0 #=
                                                                  =# 1.0 1.0 1.0 1.0 #=
                                                                  =# 1.0 1.0 1.0 1.0 #=
                                                                  =# 1.0 0.0 1.0 0.0 ]))
  drainages::Array{Field{Float64},1} = repeat(drainage,10000)
  runoffs::Array{Field{Float64},1} = deepcopy(drainages)
  evaporation::Field{Float64} = UnstructuredField{Float64}(river_parameters.grid,0.0)
  evaporations::Array{Field{Float64},1} = repeat(evaporation,180)
  evaporation = UnstructuredField{Float64}(river_parameters.grid,100.0)
  additional_evaporations::Array{Field{Float64},1} = repeat(evaporation,200)
  append!(evaporations,additional_evaporations)
  expected_river_inflow::Field{Float64} = UnstructuredField{Float64}(grid,
                                                            vec(Float64[ 0.0 0.0 0.0 0.0 #=
                                                                      =# 0.0 0.0 0.0 2.0 #=
                                                                      =# 0.0 2.0 0.0 0.0 #=
                                                                      =# 0.0 0.0 0.0 0.0 ]))
  expected_water_to_ocean::Field{Float64} = UnstructuredField{Float64}(grid,
                                                              vec(Float64[ -96.0   0.0   0.0   0.0 #=
                                                                        =#   0.0 -96.0 -96.0   0.0 #= =#   0.0   0.0   0.0 -94.0 #=
                                                                        =# -98.0   4.0   0.0   4.0 ]))
  expected_water_to_hd::Field{Float64} = UnstructuredField{Float64}(grid,
                                                           vec(Float64[ 0.0 0.0 0.0 0.0 #=
                                                                    =# 0.0 0.0 0.0 0.0 #=
                                                                    =# 0.0 0.0 0.0 0.0 #=
                                                                    =# 0.0 0.0 0.0 0.0 ]))
  expected_lake_numbers::Field{Int64} = UnstructuredField{Int64}(lake_grid,
      vec(Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]))
  expected_lake_types::Field{Int64} = UnstructuredField{Int64}(lake_grid,
      vec(Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
          =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]))
  expected_lake_volumes::Array{Float64} = Float64[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  expected_intermediate_river_inflow::Field{Float64} = UnstructuredField{Float64}(grid,
                                                       vec(Float64[ 0.0 0.0 0.0 0.0 #=
                                                                 =# 0.0 0.0 0.0 2.0 #=
                                                                 =# 0.0 2.0 0.0 0.0 #=
                                                                 =# 0.0 0.0 0.0 0.0 ]))
  expected_intermediate_water_to_ocean::Field{Float64} = UnstructuredField{Float64}(grid,
                                                         vec(Float64[ 0.0 0.0 0.0 0.0 #=
                                                                   =# 0.0 0.0 0.0 0.0 #=
                                                                   =# 0.0 0.0 0.0 0.0 #=
                                                                   =# 0.0 6.0 0.0 22.0 ]))
  expected_intermediate_water_to_hd::Field{Float64} = UnstructuredField{Float64}(grid,
                                                      vec(Float64[ 0.0 0.0 0.0  0.0 #=
                                                                =# 0.0 0.0 0.0 12.0 #=
                                                                =# 0.0 0.0 0.0  0.0 #=
                                                                =# 0.0 2.0 0.0 18.0 ]))
  expected_intermediate_lake_numbers::Field{Int64} = UnstructuredField{Int64}(lake_grid,
      vec(Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 #=
              =# 1 4 1 0 0 0 0 0 0 0 0 0 4 4 4 4 0 0 0 1 #=
              =# 0 1 1 0 0 4 4 0 0 0 0 4 4 4 4 4 0 0 0 0 #=
              =# 0 1 1 0 0 3 3 3 3 0 0 0 2 4 4 4 4 0 0 0 #=
              =# 0 1 1 4 3 3 0 3 3 4 4 2 4 2 4 4 4 4 0 0 #=
              =# 1 1 1 0 3 3 3 3 3 0 4 2 2 4 4 4 4 4 0 0 #=
              =# 0 1 1 0 0 3 0 3 0 0 0 4 2 4 4 4 4 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 4 4 4 4 4 4 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 4 4 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 5 5 5 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 5 5 0 0 #=
              =# 0 0 0 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]))
  expected_intermediate_lake_types::Field{Int64} = UnstructuredField{Int64}(lake_grid,
      vec(Int64[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 #=
              =# 3 2 3 0 0 0 0 0 0 0 0 0 2 2 2 2 0 0 0 3 #=
              =# 0 3 3 0 0 2 2 0 0 0 0 2 2 2 2 2 0 0 0 0 #=
              =# 0 3 3 0 0 3 3 3 3 0 0 0 3 2 2 2 2 0 0 0 #=
              =# 0 3 3 2 3 3 0 3 3 2 2 3 2 3 2 2 2 2 0 0 #=
              =# 3 3 3 0 3 3 3 3 3 0 2 3 3 2 2 2 2 2 0 0 #=
              =# 0 3 3 0 0 3 0 3 0 0 0 2 3 2 2 2 2 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 2 2 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 2 2 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 0 0 #=
              =# 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 #=
              =# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]))
  expected_intermediate_lake_volumes::Array{Float64} = Float64[46.0, 6.0, 38.0,  340.0, 10.0, 1.0]
  evaporations_copy::Array{Field{Float64},1} = deepcopy(evaporations)
  @time river_fields::RiverPrognosticFields,lake_prognostics::LakePrognostics,lake_fields::LakeFields =
    drive_hd_and_lake_model(river_parameters,lake_parameters,
                            drainages,runoffs,evaporations_copy,
                            5000,print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_types = UnstructuredField{Int64}(lake_grid,0)
  for i = 1:400
    coords::Generic1DCoords = Generic1DCoords(i)
    lake_number::Int64 = lake_fields.lake_numbers(coords)
    if lake_number <= 0 continue end
    lake::Lake = lake_prognostics.lakes[lake_number]
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
  lake_volumes = Float64[]
  for lake::Lake in lake_prognostics.lakes
    append!(lake_volumes,get_lake_variables(lake).lake_volume)
  end
  @test expected_intermediate_river_inflow == river_fields.river_inflow
  @test isapprox(expected_intermediate_water_to_ocean,
                 river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_intermediate_water_to_hd    == lake_fields.water_to_hd
  @test expected_intermediate_lake_numbers == lake_fields.lake_numbers
  @test expected_intermediate_lake_types == lake_types
  @test isapprox(expected_intermediate_lake_volumes,lake_volumes,atol=0.00001)
  @time river_fields,lake_prognostics,lake_fields =
    drive_hd_and_lake_model(river_parameters,lake_parameters,
                            drainages,runoffs,evaporations,
                            10000,print_timestep_results=false,
                            write_output=false,return_output=true)
  lake_types::UnstructuredField{Int64} = UnstructuredField{Int64}(lake_grid,0)
  for i = 1:400
    coords::Generic1DCoords = Generic1DCoords(i)
    lake_number::Int64 = lake_fields.lake_numbers(coords)
    if lake_number <= 0 continue end
    lake::Lake = lake_prognostics.lakes[lake_number]
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
  lake_volumes::Array{Float64} = Float64[]
  for lake::Lake in lake_prognostics.lakes
    append!(lake_volumes,get_lake_variables(lake).lake_volume)
  end
  @test expected_river_inflow == river_fields.river_inflow
  @test isapprox(expected_water_to_ocean,river_fields.water_to_ocean,rtol=0.0,atol=0.00001)
  @test expected_water_to_hd    == lake_fields.water_to_hd
  @test expected_lake_numbers == lake_fields.lake_numbers
  @test expected_lake_types == lake_types
  @test isapprox(expected_lake_volumes,lake_volumes,atol=0.00001)
  # function timing2(river_parameters,lake_parameters)
  #   for i in 1:50000
  #     drainagesl = repeat(drainage,20)
  #     runoffsl = deepcopy(drainages)
  #     drive_hd_and_lake_model(river_parameters,lake_parameters,
  #                             drainagesl,runoffsl,20,print_timestep_results=false)
  #   end
  # end
  #@time timing2(river_parameters,lake_parameters)
end

end
