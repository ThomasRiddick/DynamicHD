$:.push(".")
require "cpp_parser.rb"
textfile_to_convert="/Users/thomasriddick/Documents/data/temp/test.txt"
textfile_for_converted_text="/Users/thomasriddick/Documents/data/temp/test_converted.txt"
text = IO.read(textfile_to_convert)
converter = CppToPythonConverter.new(text)
converter.parse
converted_text = converter.convert_cpp_tests_to_python
File.open(textfile_for_converted_text,"w") do |f|
  f.write(converted_text)
end
