push!(LOAD_PATH,"/Users/thomasriddick/Documents/workspace/worktrees/feature/improved-run-and-build-structure/src/julia_src/bifurcated_rivermouth_identification_tool")
using Test: @test, @testset
using IdentifyBifurcatedRiverMouths: check_if_line_intersects_cell
using IdentifyBifurcatedRiverMouths: check_if_line_section_intersects_cell 

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
													      	       			 [(lat=0.0,lon=0.0),
													       	        		  (lat=0.0,lon=1.0),
													       	        			(lat=1.0,lon=0.5)]) == false
	@test check_if_line_section_intersects_cell((start_point=(lat=-2.0,lon=0.0),
					       	       											 end_point=(lat=-2.0,lon=-2.0)),
					      	       											 [(lat=0.0,lon=0.0),
					       	       											  (lat=0.0,lon=1.0),
					       	       											  (lat=1.0,lon=0.5)]) == false
	@test check_if_line_section_intersects_cell((start_point=(lat=0.0,lon=2.0),
					       	       											 end_point=(lat=-2.0,lon=-2.0)),
					      	       											 [(lat=0.0,lon=0.0),
					       	        										  (lat=0.0,lon=1.0),
					       	        										  (lat=1.0,lon=0.5)]) == false
	@test check_if_line_section_intersects_cell((start_point=(lat=0.0,lon=2.0),
					       	       											 end_point=(lat=2.0,lon=2.0)),
					      	       											 [(lat=0.0,lon=0.0),
					       	        											(lat=0.0,lon=1.0),
					       	        											(lat=1.0,lon=0.5)]) == false
	@test check_if_line_section_intersects_cell((start_point=(lat=0.5,lon=0.0),
					       	       										   end_point=(lat=0.5,lon=2.0)),
					      	       											 [(lat=0.0,lon=0.0),
					       	        										  (lat=0.0,lon=1.0),
					       	        										  (lat=1.0,lon=0.5)]) == true
	@test check_if_line_section_intersects_cell((start_point=(lat=2.0,lon=0.25),
					       	       											 end_point=(lat=-1.0,lon=0.25)),
					      	       											 [(lat=0.0,lon=0.0),
					       	        											(lat=0.0,lon=1.0),
					       	        											(lat=1.0,lon=0.5)]) == true
	@test check_if_line_section_intersects_cell((start_point=(lat=2.0,lon=0.75),
					       	       											 end_point=(lat=-1.0,lon=0.75)),
					      	       											 [(lat=0.0,lon=0.0),
					       	        											(lat=0.0,lon=1.0),
					       	        											(lat=1.0,lon=0.5)]) == true
	@test check_if_line_section_intersects_cell((start_point=(lat=-2.0,lon=0.75),
					       	       											 end_point=(lat=-1.0,lon=0.75)),
					      	       											 [(lat=0.0,lon=0.0),
					       	        											(lat=0.0,lon=1.0),
					       	        											(lat=1.0,lon=0.5)]) == false
	cell_indices::Array{CartesianIndex} = [ CartesianIndex(i) for i=1:80 ]
	cell_neighbors_int::Array{Int64} = Int64[ #=
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
  cell_neighbors::Array{Int64} =
  	[ CartesianIndex(cell_neighbors_int[i,j]) for i=1:80,j=1:3 ]
	cell_coords::@NamedTuple{lats::Array{Float64},lons::Array{Float64}} =
		(lats=Array{Float64} = Float64[ 75.0 75.0 75.0 75.0 75.0 #=
																	=# 45.0 45.0 45.0 45.0 45.0  45.0 45.0 45.0 45.0 45.0  45.0 45.0 45.0 45.0 45.0 #=
																	=# 15.0 15.0 15.0 15.0 15.0  15.0 15.0 15.0 15.0 15.0  #=
																	=# 15.0 15.0 15.0 15.0 15.0  15.0 15.0 15.0 15.0 15.0 #=
																	=# -15.0 -15.0 -15.0 -15.0 -15.0  -15.0 -15.0 -15.0 -15.0 -15.0 #=
																	=# -15.0 -15.0 -15.0 -15.0 -15.0  -15.0 -15.0 -15.0 -15.0 -15.0 #=
																	=# -45.0 -45.0 -45.0 -45.0 -45.0  -45.0 -45.0 -45.0 -45.0 -45.0  #=
																	=# -45.0 -45.0 -45.0 -45.0 -45.0 #=
 																	=# 75.0 75.0 75.0 75.0 75.0 ],
		 lons=Array{Float64} = Float64[ -144.0 -72.0 0.0 72.0 144.0 #=
		  														 =# -162.0 -144.0 -126.0 -90.0 -72.0 -54.0 -18.0 0.0 54.0 72.0 90.0 126.0 144.0 162.0 #=
		  														 =# -162.0 -144.0 -126.0 -108.0 -90.0 -72.0 -54.0 -36.0 -18.0 0.0 #=
		  														 =# 36.0 54.0 72.0 90.0 108.0 126.0 144.0 162.0 180.0 #=
		  														 =# -144.0 -126.0 -108.0 -90.0 -72.0 -54.0 -36.0 -18.0 0.0 #=
		  														 =# 36.0 54.0 72.0 90.0 108.0 126.0 144.0 162.0 180.0 -162.0 #=
		  														 =# -126.0 -108.0 -90.0 -54.0 -36.0 -18.0 18.0 36.0 54.0 90.0 108.0 126.0 162.0 180.0 -162.0 #=
		  														 =# -108.0 -36.0 36.0 108.0 180.0])
	cell_vertices_lats = Float64[ #=
	      # 1-5
        =# 90.0 60.0 60.0; 90.0 60.0 60.0; 90.0 60.0 60.0; 90.0 60.0 60.0; 90.0 60.0 60.0
        # 6-10
        60.0 30.0 30.0; 60.0 60.0 30.0; 60.0 30.0 30.0; 60.0 30.0 30.0; 60.0 60.0 30.0
        # 11-15
        60.0 30.0 30.0; 60.0 30.0 30.0; 60.0 30.0 30.0; 60.0 60.0 30.0; 60.0 30.0 30.0
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
        0.0 -30.0 -30.0; 0.0 0.0 -30.0; 0.0 -30.0 -30.0; 0.0 0.0 -30.0; 0.0 -30.0 -30.0;
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
	cell_vertices_lons = Float64[ #=
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
	cell_vertices::Array{@NamedTuple{lat::Array{Float64},lon::Array{Float64}}}
	cells::Cells = Cells(cell_indices,cell_neighbors,
	             				 cell_coords,cell_vertices)
	@test find_cells_on_line_section()

end
