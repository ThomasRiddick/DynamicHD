$:.push(".")
require "cpp_parser.rb"
require "python_parser.rb"
require "julia_parser.rb"
textfile_to_convert="/Users/thomasriddick/Documents/data/temp/test.txt"
textfile_for_converted_text="/Users/thomasriddick/Documents/data/temp/test_converted.txt"
text = IO.read(textfile_to_convert)
#converter = CppToPythonConverter.new(text)
converter = JuliaToFortranConverter.new(text)
#converter = PythonToCppConverter.new(text)
converter.parse
converted_text = converter.convert_julia_tests_to_fortran
#converted_text = converter.convert_cpp_tests_to_python
#converted_text = converter.convert_python_tests_to_cpp
#A few ad-hoc case specific conversions
#converted_text.gsub!(/output\["number_of_lakes"\]/,"number_of_lakes")
#converted_text.gsub!(/output\["lakes_as_array"\]/,"lakes_as_array")
#converted_text.gsub!(/output\["lake_mask"\]/,"lake_mask")
#converted_text.gsub!(/output,alg/,"vector<double>* lakes_as_array")
#converted_text.gsub!(/LatLonEvaluateBasin.evaluate_basins/,"latlon_evaluate_basins")
#converted_text.gsub!(/,\n\s*return_algorithm_object=True/,"")
#converted_text.gsub!(/EXPECT_TRUE\(lakes_as_array == expected_lakes_as_array\);/,
#                     "EXPECT_TRUE(*lakes_as_array == *expected_lakes_as_array);")
#converted_text.gsub!(/EXPECT_TRUE\(lake_mask == expected_lake_mask\);/,
#                     "EXPECT_TRUE(field<bool>(lake_mask,_coarse_grid_params) ==\n" +
 #                    "              field<bool>(expected_lake_mask,_coarse_grid_params));")
#converted_text.gsub!(/double\* expected_lakes_as_array = new double\[\d+\]/,
#                     "vector<double>* expected_lakes_as_array = new vector<double>")
File.open(textfile_for_converted_text,"w") do |f|
  f.write(converted_text)
end
