module TestIdentifyBifurcatedRiverMouths

using Test: @test, @testset
using IdentifyBifurcatedRiverMouths: check_if_line_intersects_cell
using IdentifyBifurcatedRiverMouths: check_if_line_section_intersects_cell 
using IdentifyBifurcatedRiverMouths: find_cells_on_line_section
using IdentifyBifurcatedRiverMouths: search_for_river_mouth_location_on_line_section
using IdentifyBifurcatedRiverMouths: identify_bifurcated_river_mouths
using IdentifyBifurcatedRiverMouths: for_all_secondary_neighbors
using IdentifyBifurcatedRiverMouths: Cells
using IdentifyBifurcatedRiverMouths: RiverDelta
using IdentifyBifurcatedRiverMouthsInput: load_river_deltas_from_string

@testset "River mouth identification tests" begin
	@test check_if_line_intersects_cell((start_point=(lat=2.0,lon=0.0),
					       											 end_point=(lat=2.0,lon=2.0)),
					      											 [(lat=0.0,lon=0.0),
					       												(lat=0.0,lon=1.0),
					       												(lat=1.0,lon=0.5)]) == (false,-1)
	@test check_if_line_intersects_cell((start_point=(lat=-2.0,lon=0.0),
					       											 end_point=(lat=-2.0,lon=2.0)),
					      											 [(lat=0.0,lon=0.0),
					       											  (lat=0.0,lon=1.0),
					       											  (lat=1.0,lon=0.5)]) == (false,-1)
	@test check_if_line_intersects_cell((start_point=(lat=0.0,lon=2.0),
					       											 end_point=(lat=-2.0,lon=-2.0)),
					      											 [(lat=0.0,lon=0.0),
					       												(lat=0.0,lon=1.0),
					       												(lat=1.0,lon=0.5)]) == (false,-1)
	@test check_if_line_intersects_cell((start_point=(lat=0.0,lon=2.0),
					       											 end_point=(lat=2.0,lon=2.0)),
					      											 [(lat=0.0,lon=0.0),
					       												(lat=0.0,lon=1.0),
					       												(lat=1.0,lon=0.5)]) == (false,-1)
	@test check_if_line_intersects_cell((start_point=(lat=0.5,lon=0.0),
					       											 end_point=(lat=0.5,lon=2.0)),
					      											 [(lat=0.0,lon=0.0),
					       												(lat=0.0,lon=1.0),
					       												(lat=1.0,lon=0.5)]) == (true,3)
	@test check_if_line_intersects_cell((start_point=(lat=2.0,lon=0.25),
					       											 end_point=(lat=-1.0,lon=0.25)),
					      											 [(lat=0.0,lon=0.0),
					       												(lat=0.0,lon=1.0),
					       												(lat=1.0,lon=0.5)]) == (true,1)
	@test check_if_line_intersects_cell((start_point=(lat=2.0,lon=0.75),
					       											 end_point=(lat=-1.0,lon=0.75)),
					      											 [(lat=0.0,lon=0.0),
					       											  (lat=0.0,lon=1.0),
					       											  (lat=1.0,lon=0.5)]) == (true,2)
	@test check_if_line_intersects_cell((start_point=(lat=-2.0,lon=0.75),
					       											 end_point=(lat=-1.0,lon=0.75)),
					      											 [(lat=0.0,lon=0.0),
					       												(lat=0.0,lon=1.0),
					        											(lat=1.0,lon=0.5)]) == (true,2)
	@test check_if_line_section_intersects_cell((start_point=(lat=2.0,lon=0.0),
					       	       											 end_point=(lat=2.0,lon=2.0)),
																			 			  false,
													      	       			[(lat=0.0,lon=0.0),
													       	        		 (lat=0.0,lon=1.0),
													       	        		(lat=1.0,lon=0.5)],false) == false
	@test check_if_line_section_intersects_cell((start_point=(lat=-2.0,lon=0.0),
					       	       											 end_point=(lat=-2.0,lon=-2.0)),
																			 			  false,
					      	       											[(lat=0.0,lon=0.0),
					       	       											 (lat=0.0,lon=1.0),
					       	       											 (lat=1.0,lon=0.5)],false) == false
	@test check_if_line_section_intersects_cell((start_point=(lat=0.0,lon=2.0),
					       	       											 end_point=(lat=-2.0,lon=-2.0)),
																							false,
					      	       											[(lat=0.0,lon=0.0),
					       	        										 (lat=0.0,lon=1.0),
					       	        										 (lat=1.0,lon=0.5)],false) == false
	@test check_if_line_section_intersects_cell((start_point=(lat=0.0,lon=2.0),
					       	       											 end_point=(lat=2.0,lon=2.0)),
																							false,
					      	       											[(lat=0.0,lon=0.0),
					       	        										 (lat=0.0,lon=1.0),
					       	        										 (lat=1.0,lon=0.5)],false) == false
	@test check_if_line_section_intersects_cell((start_point=(lat=0.5,lon=0.0),
					       	       										   end_point=(lat=0.5,lon=2.0)),
																							false,
					      	       										  [(lat=0.0,lon=0.0),
					       	        									   (lat=0.0,lon=1.0),
					       	        								     (lat=1.0,lon=0.5)],false) == true
	@test check_if_line_section_intersects_cell((start_point=(lat=2.0,lon=0.25),
					       	       											 end_point=(lat=-1.0,lon=0.25)),
																							false,
					      	       											[(lat=0.0,lon=0.0),
					       	        										 (lat=0.0,lon=1.0),
					       	        										 (lat=1.0,lon=0.5)],false) == true
	@test check_if_line_section_intersects_cell((start_point=(lat=2.0,lon=0.75),
					       	       											 end_point=(lat=-1.0,lon=0.75)),
																							false,
					      	       											[(lat=0.0,lon=0.0),
					       	        										 (lat=0.0,lon=1.0),
					       	        										 (lat=1.0,lon=0.5)],false) == true
	@test check_if_line_section_intersects_cell((start_point=(lat=-2.0,lon=0.75),
					       	       											 end_point=(lat=-1.0,lon=0.75)),
																							false,
					      	       											[(lat=0.0,lon=0.0),
					       	        										 (lat=0.0,lon=1.0),
					       	        										 (lat=1.0,lon=0.5)],false) == false
	cell_indices = CartesianIndex[ CartesianIndex(i) for i=1:80 ]
	cell_neighbors_int = Int64[ #=
	      # 1-5
        =# 5 7 2; 1 10 3; 2 13 4; 3 16 5; 4 19 1
        # 6-10
        20 21 7; 1 6 8; 7 23 9; 8 25 10; 2 9 11
        # 11-15
        10 27 12; 11 29 13; 3 12 14; 13 31 15; 14 33 16
        #16-20
        4 15 17; 16 35 18; 17 37 19; 5 18 20; 19 39 6
				#21-25
        6 40 22; 21 41 23; 8 22 24; 23 43 25; 24 26 9
				#26-30
        25 45 27; 11 26 28; 27 47 29; 12 28 30; 29 49 31
				#31-35
        14 30 32; 31 51 33; 15 32 34; 33 53 35; 17 34 36
				#36-40
        35 55 37; 18 36 38; 37 57 39; 20 38 40; 39 59 21
        #41-45
        22 60 42; 41 61 43; 24 42 44; 43 63 45; 26 44 46
        #46-50
        45 64 47; 28 46 48; 47 66 49; 30 48 50; 49 67 51
				#51-55
        32 50 52; 51 69 53; 34 52 54; 53 70 55; 36 54 56
				#56-60
        55 72 57; 38 56 58; 57 73 59; 40 58 60; 59 75 41
				#61-65
        42 75 62; 61 76 63; 44 62 64; 46 63 65; 64 77 66
				#66-70
        48 65 67; 50 66 68; 67 78 69; 52 68 70; 54 69 71
        #71-75
        70 79 72; 56 71 73; 58 72 74; 73 80 75; 60 74 61
        #76-80
        62 80 77; 65 76 78; 68 77 79; 71 78 80; 74 79 76 ]
  cell_neighbors =
  	CartesianIndex[ CartesianIndex(cell_neighbors_int[i,j]) for i=1:80,j=1:3 ]
  expected_secondary_neighbors_int = Int64[
  		#1
      4 19 -1 6 8 20 10 3 9
      5 7 -1 9 11 8 13 4 12
      1 10 -1 12 14 11 16 5 15
      2 13 -1 15 17 14 19 1 18
      3 16 -1 18 20 17 7 2 6
			#6
      19 39 5 40 22 -1 1 8 23
      5 2 19 20 21 10 23 9 22
      1 6 2 22 24 21 25 10 -1
      7 23 1 24 26 -1 2 11 27
      1 3 7 8 25 13 27 12 26
			#11
      2 9 3 26 28 25 29 13 -1
      10 27 2 28 30 -1 3 14 31
      2 4 10 11 29 16 31 15 30
      3 12 4 30 32 29 33 16 -1
      13 31 3 32 34 -1 4 17 35
			#16
      3 5 13 14 33 19 35 18 34
      4 15 5 34 36 33 37 19 -1
      16 35 4 36 38 -1 5 20 39
      4 1 16 17 37 7 39 6 38
      5 18 1 38 40 37 21 7 -1
			#21
      20 7 -1 39 59 8 41 23 60
      6 40 7 60 42 59 8 24 43
      7 9 6 21 41 -1 43 25 42
      8 22 -1 42 44 41 26 9 45
      23 43 -1 45 27 44 8 10 11
			#26
      24 9 43 44 46 10 11 28 47
      10 12 9 25 45 -1 47 29 46
      11 26 -1 46 48 45 12 30 49
      11 13 -1 27 47 14 49 31 48
      12 28 13 48 50 47 14 32 51
			#31
      13 15 12 29 49 -1 51 33 50
      14 30 -1 50 52 49 15 34 53
      14 16 -1 31 51 17 53 35 52
      15 32 16 52 54 51 17 36 55
      16 18 15 33 53 -1 55 37 54
			#36
      17 34 -1 54 56 53 18 38 57
      17 19 -1 35 55 20 57 39 56
      18 36 19 56 58 55 20 40 59
      19 6 18 37 57 -1 59 21 58
      20 38 -1 58 60 57 6 22 41
			#41
      21 23 40 59 75 24 61 43 -1
      22 60 23 75 62 -1 24 44 63
      23 25 22 41 61 26 63 45 62
      24 42 25 62 64 61 26 46 -1
      25 27 24 43 63 28 64 47 -1
			#46
      26 44 27 63 65 -1 28 48 66
      27 29 26 45 64 30 66 49 65
      28 46 29 65 67 64 30 50 -1
      29 31 28 47 66 32 67 51 -1
      30 48 31 66 68 -1 32 52 69
			#51
      31 33 30 49 67 34 69 53 68
      32 50 33 68 70 67 34 54 -1
      33 35 32 51 69 36 70 55 -1
      34 52 35 69 71 -1 36 56 72
      35 37 34 53 70 38 72 57 71
			#56
      36 54 37 71 73 70 38 58 -1
      37 39 36 55 72 40 73 59 -1
      38 56 39 72 74 -1 40 60 75
      39 21 38 57 73 22 75 41 74
      40 58 21 74 61 73 22 42 -1
			#61
      41 43 -1 60 74 44 76 63 80
      42 75 43 80 77 74 44 64 65
      43 45 42 61 76 -1 46 65 77
      45 47 -1 44 62 48 77 66 76
      46 63 47 76 78 62 48 67 68
			#66
      47 49 46 64 77 -1 50 68 78
      49 51 -1 48 65 52 78 69 77
      50 66 51 77 79 65 52 70 71
      51 53 50 67 78 -1 54 71 79
      53 55 -1 52 68 56 79 72 78
			#71
      54 69 55 78 80 68 56 73 74
      55 57 54 70 79 -1 58 74 80
      57 59 -1 56 71 60 80 75 79
      58 72 59 79 76 71 60 61 62
      59 41 58 73 80 -1 42 62 76
			#76
      61 63 75 74 79 64 65 78 -1
      64 66 63 62 80 67 68 79 -1
      67 69 66 65 76 70 71 80 -1
      70 72 69 68 77 73 74 76 -1
      73 75 72 71 78 61 62 77 -1 ]
  expected_secondary_neighbors =
  	CartesianIndex[ CartesianIndex(expected_secondary_neighbors_int[i,j]) for i=1:80,j=1:9 ]
  for i=1:80
  	secondary_neighbors::Array{CartesianIndex} = CartesianIndex[]
  	for_all_secondary_neighbors(x->push!(secondary_neighbors,x),
																CartesianIndex(i),
																cell_neighbors)
  	if size(secondary_neighbors) == (8,)
  		push!(secondary_neighbors,CartesianIndex(-1))
  	end
  	@test size(secondary_neighbors) == (9,)
  	@test Set(secondary_neighbors) == Set(expected_secondary_neighbors[i,:])
	end
	cell_coords::@NamedTuple{lats::Array{Float64},lons::Array{Float64}} =
		(lats = Float64[ 75.0 75.0 75.0 75.0 75.0 #=
									=# 45.0 45.0 45.0 45.0 45.0  45.0 45.0 45.0 45.0 45.0  45.0 45.0 45.0 45.0 45.0 #=
									=# 15.0 15.0 15.0 15.0 15.0  15.0 15.0 15.0 15.0 15.0  #=
									=# 15.0 15.0 15.0 15.0 15.0  15.0 15.0 15.0 15.0 15.0 #=
									=# -15.0 -15.0 -15.0 -15.0 -15.0  -15.0 -15.0 -15.0 -15.0 -15.0 #=
									=# -15.0 -15.0 -15.0 -15.0 -15.0  -15.0 -15.0 -15.0 -15.0 -15.0 #=
									=# -45.0 -45.0 -45.0 -45.0 -45.0  -45.0 -45.0 -45.0 -45.0 -45.0  #=
									=# -45.0 -45.0 -45.0 -45.0 -45.0 #=
									=# -75.0 -75.0 -75.0 -75.0 -75.0 ],
		 lons = Float64[ -144.0 -72.0 0.0 72.0 144.0 #=
								  =# -162.0 -144.0 -126.0 -90.0 -72.0 -54.0 -18.0 0.0 18.0 54.0 #=
								  =#   72.0 90.0 126.0 144.0 162.0 #=
								  =# -162.0 -144.0 -126.0 -108.0 -90.0 -72.0 -54.0 -36.0 -18.0 0.0 #=
								  =# 18.0 36.0 54.0 72.0 90.0 108.0 126.0 144.0 162.0 180.0 #=
								  =# -144.0 -126.0 -108.0 -90.0 -72.0 -54.0 -36.0 -18.0 0.0 #=
								  =# 18.0 36.0 54.0 72.0 90.0 108.0 126.0 144.0 162.0 180.0 -162.0 #=
								  =# -126.0 -108.0 -90.0 -54.0 -36.0 -18.0 18.0 36.0 54.0 90.0 #=
								  =#108.0 126.0 162.0 180.0 -162.0 #=
							 	  =# -108.0 -36.0 36.0 108.0 180.0])
	cell_vertices_lats = Float64[ #=
	      # 1-5
        =# 90.0 60.0 60.0; 90.0 60.0 60.0; 90.0 60.0 60.0; 90.0 60.0 60.0; 90.0 60.0 60.0
        # 6-10
        60.0 30.0 30.0; 60.0 60.0 30.0; 60.0 60.0 30.0; 60.0 30.0 30.0; 60.0 60.0 30.0
        # 11-15
        60.0 30.0 30.0; 60.0 30.0 30.0; 60.0 60.0 30.0; 60.0 30.0 30.0; 60.0 30.0 30.0
        #16-20
        60.0 60.0 30.0; 60.0 30.0 30.0; 60.0 30.0 30.0; 60.0 60.0 30.0; 60.0 30.0 30.0
				#21-25
        30.0 30.0 0.0; 30.0 0.0 0.0; 30.0 30.0 0.0; 30.0 0.0 0.0; 30.0 30.0 0.0
				#26-30
        30.0 0.0 0.0; 30.0 30.0 0.0; 30.0 0.0 0.0; 30.0 30.0 0.0; 30.0 0.0 0.0
				#31-35
        30.0 30.0 0.0; 30.0 0.0 0.0; 30.0 30.0 0.0; 30.0 0.0 0.0; 30.0 30.0 0.0
				#36-40
        30.0 0.0 0.0; 30.0 30.0 0.0; 30.0 0.0 0.0; 30.0 30.0 0.0; 30.0 0.0 0.0
        #41-45
        0.0 0.0 -30.0; 0.0 -30.0 -30.0; 0.0 0.0 -30.0; 0.0 -30.0 -30.0; 0.0 0.0 -30.0
        #46-50
        0.0 -30.0 -30.0; 0.0 0.0 -30.0; 0.0 -30.0 -30.0; 0.0 0.0 -30.0; 0.0 -30.0 -30.0
				#51-55
        0.0 0.0 -30.0; 0.0 -30.0 -30.0; 0.0 0.0 -30.0; 0.0 -30.0 -30.0; 0.0 0.0 -30.0
				#56-60
        0.0 -30.0 -30.0; 0.0 0.0 -30.0; 0.0 -30.0 -30.0; 0.0 0.0 -30.0; 0.0 -30.0 -30.0
				#61-65
        -30.0 -30.0 -60.0; -30.0 -60.0 -60.0; -30.0 -30.0 -60.0; -30.0 -30.0 -60.0; -30.0 -60.0 -60.0
				#66-70
        -30.0 -30.0 -60.0; -30.0 -30.0 -60.0; -30.0 -60.0 -60.0; -30.0 -30.0 -60.0; -30.0 -30.0 -60.0
        #71-75
        -30.0 -60.0 -60.0; -30.0 -60.0 -60.0; -30.0 -60.0 -60.0; -30.0 -60.0 -60.0; -30.0 -60.0 -60.0
        #76-80
        -60.0 -60.0 -90.0; -60.0 -60.0 -90.0; -60.0 -60.0 -90.0; -60.0 -60.0 -90.0; -60.0 -60.0 -90.0 ]
	cell_vertices_lons = Float64[ #=
	      # 1-5
        =# -144.0 -180.0 -108.0; -90.0 -108.0 -36.0; 0.0 -36.0 36.0; 90.0 36.0 108.0; 144.0 108.0 180.0
        # 6-10
        -180.0 -180.0 -144.0; -180.0 -108.0 -144.0; -108.0 -144.0 -108.0; -108.0 -108.0 -72.0; #=
        =#-108.0 -36.0 -72.0
        # 11-15
        -36.0 -72.0 -36.0; -36.0 -36.0 0.0; -36.0 36.0 0.0; 36.0 0.0 36.0; 36.0 36.0 72.0
        #16-20
        36.0 108.0 72.0; 108.0 72.0 108.0; 108.0 108.0 144.0; 108.0 180.0 144.0; #=
        =#180.0 144.0 180.0
				#21-25
        -180.0 -144.0 -162.0; -144.0 -162.0 -126.0; -144.0 -108.0 -126.0; #=
        =#-108.0 -126.0 -90.0; -108.0 -72.0 -90.0
				#26-30
        -72.0 -90.0 -54.0; -72.0 -36.0 -54.0; -36.0 -54.0 -18.0; #=
        =# -36.0 0.0 -18.0; 0.0 -18.0 18.0
				#31-35
        0.0 36.0 18.0; 36.0 54.0 18.0; 36.0 72.0 54.0; 72.0 54.0 90.0; 72.0 108.0 90.0
				#36-40
        108.0 90.0 126.0; 108.0 144.0 126.0; 144.0 126.0 162.0; #=
        =# 144.0 180.0 162.0; 180.0 162.0 -162.0
        #41-45
        -162.0 -126.0 -144.0; -126.0 -144.0 -108.0; -126.0 -90.0 -108.0; #=
        =#-90.0 -108.0 -72.0; -90.0 -54.0 -72.0
        #46-50
        -54.0 -72.0 -36.0; -54.0 -18.0 -36.0; -18.0 -36.0 0.0; -18.0 18.0 0.0; 18.0 0.0 36.0
				#51-55
        18.0 54.0 36.0; 54.0 36.0 72.0; 54.0 90.0 72.0; 90.0 72.0 108.0; 90.0 126.0 108.0
				#56-60
        126.0 108.0 144.0; 126.0 162.0 144.0; 162.0 144.0 180.0; #=
        =#162.0 -162.0 180.0; -162.0 180.0 -144.0
				#61-65
        -144.0 -108.0 -126.0; -108.0 -126.0 -90.0; -108.0 -72.0 -90.0; #=
        =#-72.0 -36.0 -54.0; -36.0 -54.0 -18.0
				#66-70
        -36.0 0.0 -18.0; 0.0 36.0 18.0; 36.0 18.0 54.0; 36.0 72.0 54.0; #=
        =#72.0 108.0 90.0
        #71-75
        108.0 90.0 126.0; 108.0 144.0 126.0; 144.0 180.0 126.0; #=
        =#180.0 162.0 -162.0; 180.0 -144.0 -162.0
        #76-80
        -144.0 -72.0 -108.0; -72.0 0.0 -36.0; 0.0 72.0 36.0; 72.0 144.0 108.0; 144.0 -144.0 180.0 ]
	cell_vertices::@NamedTuple{lats::Array{Float64},lons::Array{Float64}} =
		(lats = cell_vertices_lats, lons = cell_vertices_lons)
	cells::Cells = Cells(cell_indices,cell_neighbors,
	             				 cell_coords,cell_vertices)
	@test find_cells_on_line_section((start_point = (lat=30.0,lon=0.0),
																		end_point = (lat=-30.0,lon=0.0)),
																	 cells) == CartesianIndex[CartesianIndex(13,),CartesianIndex(14,),
																														CartesianIndex(30,),CartesianIndex(31,),
																														CartesianIndex(49,),CartesianIndex(50,)]
	@test find_cells_on_line_section((start_point = (lat=1.0,lon=-30.0),
																		end_point = (lat=1.0,lon= 30.0)),
																	 cells) == CartesianIndex[CartesianIndex(28,), CartesianIndex(29,),
																														CartesianIndex(30,), CartesianIndex(31,),
																														CartesianIndex(32,)]
	@test find_cells_on_line_section((start_point = (lat=-1.0,lon=-30.0),
																		end_point = (lat=-1.0,lon=30.0)),
																	 cells) == CartesianIndex[CartesianIndex(47,), CartesianIndex(48,),
																														CartesianIndex(49,), CartesianIndex(50,),
																														CartesianIndex(51,)]
	@test find_cells_on_line_section((start_point = (lat=-30.0,lon=-40.0),
																		end_point = (lat=30.0,lon=40.0)),
																	 cells) == CartesianIndex[CartesianIndex(15,),
																														CartesianIndex(30,), CartesianIndex(31,),
																														CartesianIndex(32,), CartesianIndex(33,),
																														CartesianIndex(46,), CartesianIndex(47,),
																														CartesianIndex(48,), CartesianIndex(49,)]
	@test find_cells_on_line_section((start_point = (lat=30.0,lon=-40.0),
																		end_point = (lat=-30.0,lon=40.0)),
																	 cells) == CartesianIndex[CartesianIndex(11,),
																														CartesianIndex(27,), CartesianIndex(28,),
																														CartesianIndex(29,), CartesianIndex(30,),
																														CartesianIndex(49,), CartesianIndex(50,),
																														CartesianIndex(51,), CartesianIndex(52,)]
	@test find_cells_on_line_section((start_point = (lat=45.0,lon=-90.0),
																		end_point = (lat=45.0,lon=18.0)),
																	 cells) == CartesianIndex[CartesianIndex(9,), CartesianIndex(10,),
																														CartesianIndex(11,),CartesianIndex(12,),
																														CartesianIndex(13,),CartesianIndex(14,)]
	@test find_cells_on_line_section((start_point = (lat=15.0,lon=-107.0),
																		end_point = (lat=-75.0,lon=-107.0)),
																	 cells) == CartesianIndex[CartesianIndex(24,), CartesianIndex(43,),
																														CartesianIndex(44,),CartesianIndex(62,),
																														CartesianIndex(63,),CartesianIndex(76,)]
	@test find_cells_on_line_section((start_point = (lat=15.0,lon=109.0),
																		end_point = (lat=-75.0,lon=109.0)),
																	 cells) == CartesianIndex[CartesianIndex(36,), CartesianIndex(55,),
																														CartesianIndex(56,),CartesianIndex(71,),
																														CartesianIndex(72,),CartesianIndex(79,)]
	@test find_cells_on_line_section((start_point = (lat=75.0,lon=-73.0),
																		end_point = (lat=-15.0,lon=-73.0)),
																	 cells) == CartesianIndex[CartesianIndex(2,), CartesianIndex(9,),
																														CartesianIndex(10,),CartesianIndex(25,),
																														CartesianIndex(26,),CartesianIndex(45,)]
	@test find_cells_on_line_section((start_point = (lat=75.0,lon=71.0),
																		end_point = (lat=-15.0,lon=71.0)),
																	 cells) == CartesianIndex[CartesianIndex(4,), CartesianIndex(15,),
																														CartesianIndex(16,),CartesianIndex(33,),
																														CartesianIndex(34,),CartesianIndex(53,)]
	@test find_cells_on_line_section((start_point = (lat=15.0,lon=144.0),
																		end_point = (lat=15.0,lon=-126.0)),
																	 cells) == CartesianIndex[CartesianIndex(21,), CartesianIndex(22,),
																														CartesianIndex(23,),CartesianIndex(38,),
																														CartesianIndex(39,),CartesianIndex(40,)]
	@test find_cells_on_line_section((start_point = (lat=-15.0,lon=144.0),
																		end_point = (lat=-15.0,lon=-126.0)),
																	 cells) == CartesianIndex[CartesianIndex(41,), CartesianIndex(42,),
																														CartesianIndex(57,),CartesianIndex(58,),
																														CartesianIndex(59,),CartesianIndex(60,)]
	@test find_cells_on_line_section((start_point = (lat=-45.0,lon=162.0),
																		end_point = (lat=-45.0,lon=-162.0)),
																	 cells) == CartesianIndex[CartesianIndex(73,), CartesianIndex(74,),
																														CartesianIndex(75,)]
	@test find_cells_on_line_section((start_point = (lat=-75.0,lon=-36.0),
																		end_point = (lat=-45.0,lon=18.0)),
																	 cells) == CartesianIndex[CartesianIndex(67,), CartesianIndex(77,)]
	@test find_cells_on_line_section((start_point = (lat=75.0,lon=0.0),
																		end_point = (lat=45.0,lon=54.0)),
																	 cells) == CartesianIndex[CartesianIndex(3,), CartesianIndex(13,),
																														CartesianIndex(14,), CartesianIndex(15,),
																														CartesianIndex(16,)]
	@test find_cells_on_line_section((start_point = (lat=-75.0,lon=180.0),
																		end_point = (lat=-45.0,lon=-126.0)),
																	 cells) == CartesianIndex[CartesianIndex(61,),  CartesianIndex(75,),
																												    CartesianIndex(80,)]
	@test find_cells_on_line_section((start_point = (lat=75.0,lon=-144.0),
																		end_point = (lat=45.0,lon=162.0)),
																	 cells) == CartesianIndex[CartesianIndex(1,), CartesianIndex(6,),
																														CartesianIndex(7,), CartesianIndex(19,),
																														CartesianIndex(20,)]
	@test find_cells_on_line_section((start_point = (lat=-75.0,lon=-108.0),
																		end_point = (lat=-45.0,lon=180.0)),
																	 cells) == CartesianIndex[CartesianIndex(74,), CartesianIndex(75,),
																														CartesianIndex(76,), CartesianIndex(80,)]
	@test find_cells_on_line_section((start_point = (lat=75.0,lon=144.0),
																		end_point = (lat=45.0,lon=-162.0)),
																	 cells) == CartesianIndex[CartesianIndex(5,), CartesianIndex(6,),
																														CartesianIndex(7,), CartesianIndex(19,),
																														CartesianIndex(20,)]
	@test find_cells_on_line_section((start_point = (lat=-45.0,lon=162.0),
																		end_point = (lat=75.0,lon=-144.0)),
																	 cells) == CartesianIndex[CartesianIndex(1,), CartesianIndex(6,),
																														CartesianIndex(7,), CartesianIndex(21,),
																														CartesianIndex(40,), CartesianIndex(58,),
																														CartesianIndex(59,), CartesianIndex(73,)]
	@test find_cells_on_line_section((start_point = (lat=75.0,lon=-144.0),
																		end_point = (lat=75.0,lon=144.0)),
																	 cells) == CartesianIndex[CartesianIndex(1,), CartesianIndex(5,)]
	@test find_cells_on_line_section((start_point = (lat=75.0,lon=-72.0),
																		end_point = (lat=75.0,lon=72.0)),
																	 cells) == CartesianIndex[CartesianIndex(2,), CartesianIndex(3,),
																														CartesianIndex(4,),]
	@test find_cells_on_line_section((start_point = (lat=-75.0,lon=-108.0),
																		end_point = (lat=-75.0,lon=180.0)),
																	 cells) == CartesianIndex[CartesianIndex(76,), CartesianIndex(80,)]
	@test find_cells_on_line_section((start_point = (lat=-75.0,lon=-36.0),
																		end_point = (lat=-75.0,lon=108.0)),
																	 cells) == CartesianIndex[CartesianIndex(77,), CartesianIndex(78,),
																														CartesianIndex(79,),]
	lsmask::Array{Bool} = fill(false,80)
	lsmask[4] = true
	lsmask[14:19] .= true
	lsmask[31:39] .= true
	lsmask[50:59] .= true
	lsmask[62:74] .= true
	lsmask[76:79] .= true
	river_mouth_indices = CartesianIndex[CartesianIndex(99,), CartesianIndex(-99,)]
	@test search_for_river_mouth_location_on_line_section((start_point = (lat=15.0,lon=-72.0),
																												 end_point = (lat=15.0,lon=72.0)),
																	 											 cells,
																												 lsmask,
							 																					 river_mouth_indices)
	@test river_mouth_indices == CartesianIndex[CartesianIndex(99,), CartesianIndex(-99,),
																						  CartesianIndex(31,)]
	river_mouth_indices = CartesianIndex[CartesianIndex(99,), CartesianIndex(-99,)]
	@test search_for_river_mouth_location_on_line_section((start_point = (lat=15.0,lon=-144.0),
																												 end_point = (lat=15.0,lon=144.0)),
																	 											 cells,
																												 lsmask,
							 																					 river_mouth_indices)
	@test river_mouth_indices == CartesianIndex[CartesianIndex(99,), CartesianIndex(-99,),
																						  CartesianIndex(39,)]
	river_mouth_indices = CartesianIndex[CartesianIndex(99,), CartesianIndex(-99,)]
	@test search_for_river_mouth_location_on_line_section((start_point = (lat=15.0,lon=-90.0),
																												 end_point = (lat=-75.0,lon=-36.0)),
																	 											 cells,
																												 lsmask,
							 																					 river_mouth_indices)
	@test river_mouth_indices == CartesianIndex[CartesianIndex(99,), CartesianIndex(-99,),
																						  CartesianIndex(64,)]
	river_mouth_indices = CartesianIndex[CartesianIndex(99,), CartesianIndex(-99,)]
	@test ! search_for_river_mouth_location_on_line_section((start_point = (lat=45.0,lon=-90.0),
																												   end_point = (lat=5.0,lon=-36.0)),
																	 											   cells,
																												   lsmask,
							 																					   river_mouth_indices)
	@test river_mouth_indices == CartesianIndex[CartesianIndex(99,), CartesianIndex(-99,)]
  lsmask = .! lsmask
  lsmask[6]  = false
  lsmask[20] = false
  lsmask[21] = false
	lsmask[40] = false
	lsmask[60] = false
	river_mouth_indices = CartesianIndex[CartesianIndex(99,), CartesianIndex(-99,)]
	@test search_for_river_mouth_location_on_line_section((start_point = (lat=15.0,lon=144.0),
																												 end_point = (lat=15.0,lon=-144.0)),
																	 											 cells,
																												 lsmask,
							 																					 river_mouth_indices)
	@test river_mouth_indices == CartesianIndex[CartesianIndex(99,), CartesianIndex(-99,),
																						  CartesianIndex(22,)]
	lsmask = fill(false,80)
  lsmask[34:36]  .= true
  lsmask[46:57]  .= true
  lsmask[64:72]  .= true
  lsmask[77:78]  .= true
	river_deltas::Array{RiverDelta} = load_river_deltas_from_string("""
	River_A=[[[45.0,-126.0],[45.0,-90.0],[75.0,-72.0],[75.0,0.0],[-15.0,0.0],[-15.0,36.0]],
					 [[45.0,-126.0],[45.0,-90.0],[75.0,-72.0],[-15.0,-36.0],[-45.0,18.0]],
					 [[45.0,-126.0],[-15.0,-36.0]],
		       [[45.0,-126.0],[15.0,-126.0],[15.0,-108.0],[-15.0,-108.0],[-15.0,-90.0],
		        [-45.0,-90.0],[-45.0,-54.0],[-75.0,-36.0],[-75.0,36.0]]]
	River_B=[[[75.0,90.0],[15.0,90.0]],
				   [[75.0,90.0],[75.0,144.0],[-15.0,90.0]],
				   [[75.0,90.0],[75.0,144.0],[45.0,162.0],[-45.0,126.0]]]
	River_C=[[[-75.0,108.0],[-45.0,90.0]],]
	""")
	@test identify_bifurcated_river_mouths(river_deltas,
																	 			 cells,
																	 			 lsmask) ==
    Dict{String,Array{CartesianIndex}}("River_A"=>
                                       CartesianIndex[CartesianIndex(49,),CartesianIndex(47,),
    														 											CartesianIndex(46,),CartesianIndex(64,)],
    	 																 "River_C"=>
    	 																 CartesianIndex[CartesianIndex(71,),],
    	 																 "River_B"=>
    	 																 CartesianIndex[CartesianIndex(35,),CartesianIndex(36,),
    														 			 								CartesianIndex(57,)])
end

end
