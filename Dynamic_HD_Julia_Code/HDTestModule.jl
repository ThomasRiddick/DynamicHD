module HDTestModule

using Serialization
using Profile
using Test: @test, @testset
using HDDriverModule: drive_hd_model,drive_hd_and_lake_model
using HDModule: RiverParameters
using GridModule: LatLonGrid
using CoordsModule: LatLonCoords
using FieldModule: Field,LatLonField,LatLonDirectionIndicators,set!,repeat,add_offset
using LakeModule: GridSpecificLakeParameters,LakeParameters,LatLonLakeParameters
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
  drainages::Array{Field{Float64},1} = repeat(drainage,100)
  runoffs::Array{Field{Float64},1} = deepcopy(drainages)
  @time drive_hd_model(river_parameters,drainages,runoffs,100,print_timestep_results=false)
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
                         connect_redirect_lon_index)
  lake_parameters = LakeParameters(lake_centers,
                                   connection_volume_thresholds,
                                   flood_volume_threshold,
                                   flood_local_redirect,
                                   connect_local_redirect,
                                   merge_points,
                                   lake_grid,
                                   grid,
                                   grid_specific_lake_parameters)
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,Float64[ 1.0 1.0 1.0
                                                                                 1.0 1.0 1.0
                                                                                 1.0 1.0 0.0 ])
  drainages::Array{Field{Float64},1} = repeat(drainage,1000)
  runoffs::Array{Field{Float64},1} = deepcopy(drainages)
  #drive_hd_and_lake_model(river_parameters,lake_parameters,
  #                        drainages,runoffs,1000,print_timestep_results=true)
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
    Int64[ -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  3
            2  3  5 -1 -1 -1 -1 -1 -1 -1 -1 -1  2  2  4  5 -1 -1 -1  1
           -1  4  2 -1 -1  7  2 -1 -1 -1 -1  2  5  3  9  2 -1 -1 -1 -1
           -1  3  4 -1 -1  6  4  5  3 -1 -1 -1  7  3  4  3  6 -1 -1 -1
           -1  6  5  3  6  7 -1  4  4 -1  5  7  6  6  4  8  4  3 -1 -1
            7  6  6 -1  5  5  6  5  5 -1  5  5  4  8  6  6  5  5 -1 -1
           -1  5  7 -1 -1  6 -1  4 -1 -1 -1  5  6  6  8  7  5 -1 -1 -1
           -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1  8  7  7  8  6  7 -1 -1 -1
           -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1  8  9 -1 -1 -1 -1 -1 -1
           -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 14 14 13 15 -1
           -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 13 14 13 -1 -1
           -1 -1  1 16 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ])
  flood_next_cell_lon_index::Field{Int64} = LatLonField{Int64}(lake_grid,
    Int64[ -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  1
           19  5  2 -1 -1 -1 -1 -1 -1 -1 -1 -1 15 12 16  3 -1 -1 -1 19
           -1  2  2 -1 -1 17  1 -1 -1 -1 -1 13 15 12 13 14 -1 -1 -1 -1
           -1  2  1 -1 -1  8  5  9  6 -1 -1 -1 12 13 13 14 17 -1 -1 -1
           -1  2  1  5  7  5 -1  6  8 -1 14 14 10 12 14 15 15 11 -1 -1
            2  0  1 -1  4  5  4  7  8 -1 10 11 12 12 13 14 16 17 -1 -1
           -1  3  1 -1 -1  6 -1  7 -1 -1 -1 12 11 15 14 13  9 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 16 11 15 13 16 16 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 11 12 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 15 16 18 15 -1
           -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 16 17 17 -1 -1
           -1 -1 -1  4 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
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
                         connect_redirect_lon_index)
  lake_parameters = LakeParameters(lake_centers,
                                   connection_volume_thresholds,
                                   flood_volume_threshold,
                                   flood_local_redirect,
                                   connect_local_redirect,
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
  @time drive_hd_and_lake_model(river_parameters,lake_parameters,
                          drainages,runoffs,10000,print_timestep_results=true)
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
