push!(LOAD_PATH,"/Users/thomasriddick/Documents/scripts")
using Test: @test, @testset
using IdentifyBifurcatedRiverMouths: check_if_line_intersects_cell
using IdentifyBifurcatedRiverMouths: check_if_line_section_intersects_cell 

@testset "Parameter generation tests" begin
	println(check_if_line_intersects_cell((start_point=(lat=2.0,lon=0.0),
					       end_point=  (lat=2.0,lon=2.0)),
					      [(lat=0.0,lon=0.0),
					       (lat=0.0,lon=1.0),
					       (lat=1.0,lon=0.5)]))
	println(check_if_line_intersects_cell((start_point=(lat=-2.0,lon=0.0),
					       end_point=  (lat=-2.0,lon=2.0)),
					      [(lat=0.0,lon=0.0),
					       (lat=0.0,lon=1.0),
					       (lat=1.0,lon=0.5)]))
	println(check_if_line_intersects_cell((start_point=(lat=0.0,lon=2.0),
					       end_point=  (lat=-2.0,lon=-2.0)),
					      [(lat=0.0,lon=0.0),
					       (lat=0.0,lon=1.0),
					       (lat=1.0,lon=0.5)]))
	println(check_if_line_intersects_cell((start_point=(lat=0.0,lon=2.0),
					       end_point=  (lat=2.0,lon=2.0)),
					      [(lat=0.0,lon=0.0),
					       (lat=0.0,lon=1.0),
					       (lat=1.0,lon=0.5)]))
	println(check_if_line_intersects_cell((start_point=(lat=0.5,lon=0.0),
					       end_point=  (lat=0.5,lon=2.0)),
					      [(lat=0.0,lon=0.0),
					       (lat=0.0,lon=1.0),
					       (lat=1.0,lon=0.5)]))
	println(check_if_line_intersects_cell((start_point=(lat=2.0,lon=0.25),
					       end_point=  (lat=-1.0,lon=0.25)),
					      [(lat=0.0,lon=0.0),
					       (lat=0.0,lon=1.0),
					       (lat=1.0,lon=0.5)]))
	println(check_if_line_intersects_cell((start_point=(lat=2.0,lon=0.75),
					       end_point=  (lat=-1.0,lon=0.75)),
					      [(lat=0.0,lon=0.0),
					       (lat=0.0,lon=1.0),
					       (lat=1.0,lon=0.5)]))
	println(check_if_line_intersects_cell((start_point=(lat=-2.0,lon=0.75),
					       end_point=  (lat=-1.0,lon=0.75)),
					      [(lat=0.0,lon=0.0),
					       (lat=0.0,lon=1.0),
					       (lat=1.0,lon=0.5)]))
	println(check_if_line_section_intersects_cell((start_point=(lat=2.0,lon=0.0),
					       	       end_point=  (lat=2.0,lon=2.0)),
					      	       [(lat=0.0,lon=0.0),
					       	        (lat=0.0,lon=1.0),
					       	        (lat=1.0,lon=0.5)]))
	println(check_if_line_section_intersects_cell((start_point=(lat=-2.0,lon=0.0),
					       	       end_point=  (lat=-2.0,lon=2.0)),
					      	       [(lat=0.0,lon=0.0),
					       	        (lat=0.0,lon=1.0),
					       	        (lat=1.0,lon=0.5)]))
	println(check_if_line_section_intersects_cell((start_point=(lat=0.0,lon=2.0),
					       	       end_point=  (lat=-2.0,lon=-2.0)),
					      	       [(lat=0.0,lon=0.0),
					       	        (lat=0.0,lon=1.0),
					       	        (lat=1.0,lon=0.5)]))
	println(check_if_line_section_intersects_cell((start_point=(lat=0.0,lon=2.0),
					       	       end_point=  (lat=2.0,lon=2.0)),
					      	       [(lat=0.0,lon=0.0),
					       	        (lat=0.0,lon=1.0),
					       	        (lat=1.0,lon=0.5)]))
	println(check_if_line_section_intersects_cell((start_point=(lat=0.5,lon=0.0),
					       	       end_point=  (lat=0.5,lon=2.0)),
					      	       [(lat=0.0,lon=0.0),
					       	        (lat=0.0,lon=1.0),
					       	        (lat=1.0,lon=0.5)]))
	println(check_if_line_section_intersects_cell((start_point=(lat=2.0,lon=0.25),
					       	       end_point=  (lat=-1.0,lon=0.25)),
					      	       [(lat=0.0,lon=0.0),
					       	        (lat=0.0,lon=1.0),
					       	        (lat=1.0,lon=0.5)]))
	println(check_if_line_section_intersects_cell((start_point=(lat=2.0,lon=0.75),
					       	       end_point=  (lat=-1.0,lon=0.75)),
					      	       [(lat=0.0,lon=0.0),
					       	        (lat=0.0,lon=1.0),
					       	        (lat=1.0,lon=0.5)]))
	println(check_if_line_section_intersects_cell((start_point=(lat=-2.0,lon=0.75),
					       	       end_point=  (lat=-1.0,lon=0.75)),
					      	       [(lat=0.0,lon=0.0),
					       	        (lat=0.0,lon=1.0),
					       	        (lat=1.0,lon=0.5)]))

end
